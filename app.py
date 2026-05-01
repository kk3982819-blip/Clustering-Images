import shutil
import subprocess
import sys
import json
import traceback
import logging
from pathlib import Path
from typing import List
import uuid
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hdr_engine import process_bracketed_set
from api_orchestrator import EnhancementOrchestrator
from database import create_job, insert_cluster, insert_image, insert_enhancement, get_latest_job_clusters, get_cluster_details
from full_scene_generator import generate_full_scene_variant

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

@app.get("/job/{job_id}/cluster/{cluster_name}", response_class=HTMLResponse)
async def view_cluster(request: Request, job_id: str, cluster_name: str):
    cluster = get_cluster_details(job_id, cluster_name)
    if not cluster:
        return HTMLResponse("Cluster not found", status_code=404)
    return templates.TemplateResponse(
        request=request, 
        name="cluster.html",
        context={"cluster": cluster}
    )


@app.get("/latest-job")
async def latest_job():
    try:
        clusters = get_latest_job_clusters()
        if clusters:
            return {"clusters": clusters}
        return {"clusters": []}
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)

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
        job_id = str(uuid.uuid4())
        create_job(job_id)

        for cluster_name, members in clusters_data.items():
            if cluster_name == "noise":
                continue  # Noise images excluded from the result cards

            thumbnails = []
            all_tag_set: list[str] = []

            for member in members:
                img_name = member.get("image", "")
                if not img_name:
                    continue

                stem = Path(img_name).stem
                labeled_name = f"{stem}_labeled.jpg"
                labeled_path = CLUSTER_OUT / cluster_name / labeled_name

                thumbnails.append({
                    "original": f"/clusters/{cluster_name}/{img_name}",
                    "labeled": f"/clusters/{cluster_name}/{labeled_name}" if labeled_path.exists() else None,
                    "filename": img_name
                })

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

                # Insert Image into DB
                image_id = str(uuid.uuid4())
                orig_url = f"/clusters/{cluster_name}/{img_name}"
                labeled_url = f"/clusters/{cluster_name}/{labeled_name}" if labeled_path.exists() else ""
                
                cluster_db_id = f"{job_id}_{cluster_name}"
                insert_image(
                    image_id=image_id,
                    cluster_id=cluster_db_id,
                    filename=img_name,
                    original_path=orig_url,
                    labeled_path=labeled_url,
                    annotations=member.get("annotations", [])
                )

            if not thumbnails:
                continue

            tags_to_store = all_tag_set[:20]
            cluster_db_id = f"{job_id}_{cluster_name}"
            insert_cluster(cluster_db_id, job_id, cluster_name, tags_to_store)

            response_clusters.append({
                "id":         cluster_name,
                "job_id":     job_id,
                "count":      len(thumbnails),
                "thumbnails": thumbnails,
                "tags":       tags_to_store,  # cap at 20 unique labels
            })

        print(f"[PixelDwell] Returning {len(response_clusters)} cluster cards to frontend.")
        return {"clusters": response_clusters}

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/enhance-cluster")
async def enhance_cluster(
    cluster_id: str = Form(...),
    job_id: str = Form(default=""),
    ai_features: List[str] = Form(default=[]),
    weather: str = Form(default="sunny"),
):
    """
    Phase 2  ·  HDR-fuse the images in one cluster, optionally apply AI features.
    """
    try:
        cluster_dir = CLUSTER_OUT / cluster_id
        if not cluster_dir.exists():
            return JSONResponse({"error": f"Cluster '{cluster_id}' not found."}, status_code=404)

        # Filter to only pick up CLEAN original photos (exclude _labeled.jpg)
        images = sorted([
            p for p in cluster_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and not p.name.endswith("_labeled.jpg")
            and "_mask_" not in p.name
        ])
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
                params={"weather": weather}
            )
            
            # Log enhancements
            if job_id:
                db_cluster_id = f"{job_id}_{cluster_id}"
                for feature in ai_features:
                    insert_enhancement(
                        enhancement_id=str(uuid.uuid4()),
                        cluster_id=db_cluster_id,
                        feature_name=feature,
                        result_image_path=f"/output_web/{Path(final_path).name}",
                        status="completed"
                    )

        return {
            "success":   True,
            "image_url": f"/output_web/{Path(final_path).name}",
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/enhance-single")
async def enhance_single(
    cluster_id: str = Form(...),
    filename: str = Form(...),
    job_id: str = Form(default=""),
    ai_features: List[str] = Form(default=[]),
    weather: str = Form(default="sunny"),
):
    """
    Phase 2b · Enhance a single image (skip HDR fusion).
    """
    try:
        source_path = CLUSTER_OUT / cluster_id / filename
        if not source_path.exists():
            return JSONResponse({"error": f"Image '{filename}' not found in cluster '{cluster_id}'."}, status_code=404)

        final_path = source_path
        remaining_features = list(ai_features)

        if "sky_replacement" in remaining_features:
            generated_path = OUTPUT_DIR / f"{source_path.stem}_full_scene_sky.jpg"
            final_path = generate_full_scene_variant(
                input_path=source_path,
                output_path=generated_path,
                weather=weather,
            )
            remaining_features = [feature for feature in remaining_features if feature != "sky_replacement"]

        if remaining_features:
            final_path = orchestrator.process_image(
                image_path=str(final_path),
                requested_features=remaining_features,
                output_dir=str(OUTPUT_DIR),
                params={"weather": weather}
            )

        # Log enhancements
        if job_id:
            db_cluster_id = f"{job_id}_{cluster_id}"
            for feature in ai_features:
                insert_enhancement(
                    enhancement_id=str(uuid.uuid4()),
                    cluster_id=db_cluster_id,
                    feature_name=feature,
                    result_image_path=f"/output_web/{Path(final_path).name}",
                    status="completed"
                )

        return {
            "success":   True,
            "image_url": f"/output_web/{Path(final_path).name}",
        }

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


def _resolve_url_to_path(url: str) -> Path:
    """Map a web URL (absolute or relative) back to a local filesystem path."""
    parsed_url = urlparse(url)
    path_str = parsed_url.path
    path_str = path_str.replace("\\", "/")
    
    if path_str.startswith("/clusters/"):
        return CLUSTER_OUT / path_str.replace("/clusters/", "", 1)
    if path_str.startswith("/output_web/"):
        return OUTPUT_DIR / path_str.replace("/output_web/", "", 1)
        
    return Path(path_str)


@app.post("/process-object-removal")
async def process_object_removal(
    image_url: str = Form(...),
    cluster_id: str = Form(...),
    job_id: str = Form(...),
    regions_json: str = Form(...),
):
    """
    Handle removal of one or more objects (clicks or bboxes).
    """
    try:
        local_path = _resolve_url_to_path(image_url)
        if not local_path.exists():
            return JSONResponse({"error": f"Image not found at {local_path}"}, status_code=404)

        regions = json.loads(regions_json)
        logging.info(f"[App] Smart Removal of {len(regions)} regions for {local_path.name}")
        
        final_path = orchestrator.process_image(
            image_path=str(local_path),
            requested_features=["object_removal"],
            output_dir=str(OUTPUT_DIR),
            params={"object_removal": regions}
        )

        return {
            "success": True,
            "image_url": f"/output_web/{Path(final_path).name}",
            "job_id": job_id
        }
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)
