import shutil
import subprocess
import sys
import json
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hdr_engine import process_bracketed_set
from api_orchestrator import EnhancementOrchestrator

app = FastAPI(title="PixelDwell AI Photo Editor")

# ── Directories (created once, NEVER deleted) ─────────────────────────────────
UPLOAD_RAW  = Path("uploads_temp/raw")
CLUSTER_OUT = Path("uploads_temp/clustered")
OUTPUT_DIR  = Path("output_web")

for d in [UPLOAD_RAW, CLUSTER_OUT, OUTPUT_DIR, Path("static")]:
    d.mkdir(parents=True, exist_ok=True)

# Mount ONCE at startup — directories must ALWAYS exist (never rmtree them)
app.mount("/clusters",   StaticFiles(directory=str(CLUSTER_OUT)), name="clusters")
app.mount("/output_web", StaticFiles(directory=str(OUTPUT_DIR)),  name="output_web")
app.mount("/static",     StaticFiles(directory="static"),         name="static")

templates    = Jinja2Templates(directory="templates")
orchestrator = EnhancementOrchestrator()


def _clear_dir(path: Path):
    """Remove contents of a directory WITHOUT deleting the directory itself."""
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload-and-cluster")
async def upload_and_cluster(files: List[UploadFile] = File(...)):
    """
    Phase 1  ·  Save uploads → run cluster_images.py → return cluster cards JSON.
    """
    try:
        # Clear previous run without touching the parent dirs
        _clear_dir(UPLOAD_RAW)
        _clear_dir(CLUSTER_OUT)

        # Save every uploaded file
        saved = 0
        for f in files:
            if not f.filename:
                continue
            dest = UPLOAD_RAW / Path(f.filename).name
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            saved += 1

        if saved == 0:
            return JSONResponse({"error": "No valid image files received."}, status_code=400)

        print(f"[PixelDwell] Saved {saved} files — launching clustering engine...")

        # Run cluster_images.py as a subprocess (inherits venv via sys.executable)
        cmd = [
            sys.executable, "cluster_images.py",
            "--input",  str(UPLOAD_RAW),
            "--output", str(CLUSTER_OUT),
        ]
        result = subprocess.run(cmd, timeout=600, capture_output=False)
        if result.returncode != 0:
            return JSONResponse(
                {"error": "Clustering engine exited with an error. Check terminal for details."},
                status_code=500,
            )

        # Load the JSON written by cluster_images.py
        clusters_json_path = CLUSTER_OUT / "clusters.json"
        if not clusters_json_path.exists():
            return JSONResponse(
                {"error": "Clustering finished, but clusters.json was not produced."},
                status_code=500,
            )

        clusters_data: dict = json.loads(clusters_json_path.read_text(encoding="utf-8"))

        # Build response payload
        response_clusters = []
        for cluster_name, members in clusters_data.items():
            if cluster_name == "noise":
                continue  # Noise images excluded from the result cards

            thumbnails = []
            all_tag_set: list[str] = []

            for member in members:
                img_name = member.get("image", "")
                if not img_name:
                    continue

                thumbnails.append(f"/clusters/{cluster_name}/{img_name}")

                # ── Scene tags (from CLIP scene labeler) ──────────────────
                for tag in member.get("tags", []):
                    if tag and tag not in all_tag_set:
                        all_tag_set.append(tag)

                # ── Object labels (from YOLO-World annotations) ───────────
                for ann in member.get("annotations", []):
                    for lbl_entry in ann.get("labels", []):
                        # labels can be {"label": .., "score": ..} or [label, score]
                        if isinstance(lbl_entry, dict):
                            lbl = lbl_entry.get("label", "")
                        elif isinstance(lbl_entry, (list, tuple)) and lbl_entry:
                            lbl = str(lbl_entry[0])
                        else:
                            lbl = str(lbl_entry)
                        lbl = lbl.strip().title()
                        if lbl and lbl.lower() not in [t.lower() for t in all_tag_set]:
                            all_tag_set.append(lbl)

            if not thumbnails:
                continue

            response_clusters.append({
                "id":         cluster_name,
                "count":      len(thumbnails),
                "thumbnails": thumbnails,
                "tags":       all_tag_set[:20],  # cap at 20 unique labels
            })

        print(f"[PixelDwell] Returning {len(response_clusters)} cluster cards to frontend.")
        return {"clusters": response_clusters}

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/enhance-cluster")
async def enhance_cluster(
    cluster_id: str = Form(...),
    ai_features: List[str] = Form(default=[]),
):
    """
    Phase 2  ·  HDR-fuse the images in one cluster, optionally apply AI features.
    """
    try:
        cluster_dir = CLUSTER_OUT / cluster_id
        if not cluster_dir.exists():
            return JSONResponse({"error": f"Cluster '{cluster_id}' not found."}, status_code=404)

        images = sorted(
            list(cluster_dir.glob("*.jpg")) +
            list(cluster_dir.glob("*.jpeg")) +
            list(cluster_dir.glob("*.png"))
        )
        if not images:
            return JSONResponse({"error": "No images found in that cluster."}, status_code=404)

        out_path = OUTPUT_DIR / f"hdr_{cluster_id}.jpg"
        success  = process_bracketed_set(images, out_path)

        if not success or not out_path.exists():
            return JSONResponse({"error": "HDR fusion failed."}, status_code=500)

        final_path = out_path
        if ai_features:
            final_path = orchestrator.process_image(
                image_path=str(out_path),
                requested_features=ai_features,
                output_dir=str(OUTPUT_DIR),
            )

        return {
            "success":   True,
            "image_url": f"/output_web/{Path(final_path).name}",
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)
