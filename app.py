import shutil
import subprocess
import sys
import json
import os
import traceback
import logging
from pathlib import Path
from typing import Any, List
import uuid
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hdr_engine import process_bracketed_set
from api_orchestrator import EnhancementOrchestrator
from database import create_job, insert_cluster, insert_image, insert_enhancement, get_latest_job_clusters, get_cluster_details
from full_scene_generator import generate_regenerative_sky_variant

app = FastAPI(title="PixelDwell AI Photo Editor")

# ── Directories (created once, NEVER deleted) ─────────────────────────────────
UPLOAD_RAW  = Path("uploads_temp/raw")
CLUSTER_OUT = Path("uploads_temp/clustered")
CLUSTER_STAGING = Path("uploads_temp/clustered_next")
OUTPUT_DIR  = Path("output_web")

for d in [UPLOAD_RAW, CLUSTER_OUT, CLUSTER_STAGING, OUTPUT_DIR, Path("static")]:
    d.mkdir(parents=True, exist_ok=True)

# Mount ONCE at startup — directories must ALWAYS exist (never rmtree them)
app.mount("/clusters",   StaticFiles(directory=str(CLUSTER_OUT)), name="clusters")
app.mount("/output_web", StaticFiles(directory=str(OUTPUT_DIR)),  name="output_web")
app.mount("/static",     StaticFiles(directory="static"),         name="static")

templates    = Jinja2Templates(directory="templates")
orchestrator = EnhancementOrchestrator()
OPENING_LABELS = {"window", "sliding glass door", "glass door"}
OPENING_FALLBACK_LABELS = {"sky"}


def _clear_dir(path: Path):
    """Remove contents of a directory WITHOUT deleting the directory itself."""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _replace_dir_contents(src: Path, dst: Path):
    """Atomically enough for this app: stage elsewhere, then swap live contents."""
    src.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    _clear_dir(dst)
    for child in src.iterdir():
        shutil.move(str(child), str(dst / child.name))


def _load_clusters_payload() -> dict[str, list[dict[str, Any]]]:
    clusters_json_path = CLUSTER_OUT / "clusters.json"
    if not clusters_json_path.exists():
        return {}
    try:
        return json.loads(clusters_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_annotation_labels(annotation: dict[str, Any]) -> list[str]:
    normalized: list[str] = []
    for entry in annotation.get("labels", []):
        label = ""
        if isinstance(entry, dict):
            label = str(entry.get("label", ""))
        elif isinstance(entry, (list, tuple)) and entry:
            label = str(entry[0])
        elif entry is not None:
            label = str(entry)
        label = label.strip().lower()
        if label:
            normalized.append(label)
    return normalized


def _annotation_score(annotation: dict[str, Any]) -> float:
    scores: list[float] = []
    for entry in annotation.get("labels", []):
        if isinstance(entry, dict):
            try:
                scores.append(float(entry.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, (list, tuple)) and len(entry) > 1:
            try:
                scores.append(float(entry[1]))
            except (TypeError, ValueError):
                continue
    return max(scores) if scores else 0.0


def _polygon_from_annotation(annotation: dict[str, Any]) -> list[list[int]]:
    boundary = annotation.get("boundary", [])
    polygon: list[list[int]] = []
    if isinstance(boundary, list):
        for point in boundary:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    polygon.append([int(round(float(point[0]))), int(round(float(point[1])))])
                except (TypeError, ValueError):
                    continue
    if len(polygon) >= 3:
        return polygon

    box = annotation.get("box", [])
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            return []
        if x2 > x1 and y2 > y1:
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return []


def _extract_opening_polygons(
    annotations: list[dict[str, Any]],
    image_size: tuple[int, int] | None,
) -> list[list[list[int]]]:
    image_w, image_h = image_size if image_size is not None else (0, 0)
    image_area = float(max(image_w * image_h, 1))
    primary_polygons: list[list[list[int]]] = []
    fallback_polygons: list[list[list[int]]] = []

    for annotation in annotations:
        labels = set(_extract_annotation_labels(annotation))
        polygon = _polygon_from_annotation(annotation)
        if len(polygon) < 3:
            continue

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area_ratio = (width * height) / image_area if image_area > 0 else 0.0
        width_ratio = width / float(max(image_w, 1)) if image_w > 0 else 0.0
        height_ratio = height / float(max(image_h, 1)) if image_h > 0 else 0.0
        score = _annotation_score(annotation)

        if labels.intersection(OPENING_LABELS):
            if image_w > 0 and image_h > 0:
                if area_ratio < 0.004:
                    continue
                min_width_ratio = 0.045 if score >= 0.10 else 0.06
                if width_ratio < min_width_ratio or height_ratio < 0.12:
                    continue
            if score < 0.02:
                continue
            primary_polygons.append(polygon)
            continue

        if labels.intersection(OPENING_FALLBACK_LABELS):
            if image_w > 0 and image_h > 0:
                if area_ratio < 0.003:
                    continue
                if width_ratio < 0.05 or height_ratio < 0.05:
                    continue
            if score < 0.15:
                continue

            expand_x = int(round(width * 0.10))
            expand_top = int(round(height * 0.05))
            expand_bottom = int(round(height * 0.90))
            x1 = max(0, min(xs) - expand_x)
            y1 = max(0, min(ys) - expand_top)
            x2 = min(max(image_w - 1, 0), max(xs) + expand_x)
            y2 = min(max(image_h - 1, 0), max(ys) + expand_bottom)
            if x2 > x1 and y2 > y1:
                fallback_polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return primary_polygons if primary_polygons else fallback_polygons


def _member_opening_payload(member: dict[str, Any], cluster_id: str) -> tuple[list[list[list[int]]], tuple[int, int] | None]:
    image_name = str(member.get("image", ""))
    if not image_name:
        return [], None

    image_path = CLUSTER_OUT / cluster_id / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        return [], None

    image_h, image_w = image.shape[:2]
    size = (image_w, image_h)
    polygons = _extract_opening_polygons(member.get("annotations", []), size)
    return polygons, size


def _resolve_opening_payload(cluster_id: str, filename: str | None = None) -> tuple[list[list[list[int]]], tuple[int, int] | None]:
    clusters_payload = _load_clusters_payload()
    members = clusters_payload.get(cluster_id, [])
    if not isinstance(members, list):
        return [], None

    preferred_members = members
    if filename:
        exact = [member for member in members if str(member.get("image", "")) == filename]
        if exact:
            preferred_members = exact

    best_polygons: list[list[list[int]]] = []
    best_size: tuple[int, int] | None = None
    best_score = -1.0

    for member in preferred_members:
        polygons, size = _member_opening_payload(member, cluster_id)
        if not polygons or size is None:
            continue
        polygon_score = 0.0
        for polygon in polygons:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            polygon_score += float(max(xs) - min(xs)) * float(max(ys) - min(ys))
        if polygon_score > best_score:
            best_score = polygon_score
            best_polygons = polygons
            best_size = size

    return best_polygons, best_size


def _build_enhanced_detection_preview(final_path: Path, original_image_path: Path | None = None) -> str | None:
    """
    Build a labeled detection preview.
    
    Args:
        final_path: The final enhanced/generated image to display labels on
        original_image_path: The original image to run detection on (before generation).
                           If None, detection runs on final_path instead.
    """
    try:
        from cluster_images import annotate_clustered_image

        # Run detection on original image (before generation), but display on final_path
        detection_input_path = original_image_path if original_image_path else final_path
        
        labeled_path = OUTPUT_DIR / f"{final_path.stem}_lighting_labeled{final_path.suffix}"
        mask_dir = OUTPUT_DIR / "enhancement_lighting_masks" / final_path.stem
        
        logging.info(
            "[Detection] Running detection on original=%s, displaying on final=%s",
            detection_input_path.name,
            final_path.name
        )
        
        # Step 1: Run detection on the ORIGINAL image to get annotations/masks
        _, annotations = annotate_clustered_image(
            image_path=detection_input_path,
            output_path=detection_input_path.with_stem(f"{detection_input_path.stem}_temp_detected"),
            mask_output_dir=mask_dir,
            detection_runtime=None,
            scene_labeler=None,
        )
        
        # Step 2: Draw the same annotations on the FINAL generated image
        if original_image_path and original_image_path != final_path:
            from cluster_images import draw_labeled_boundary, annotation_color
            import json
            
            final_img = cv2.imread(str(final_path))
            if final_img is not None:
                # Reconstruct annotation objects from serialized data
                for ann_data in annotations:
                    try:
                        boundary = ann_data.get("boundary", [])
                        box = ann_data.get("box")
                        labels_data = ann_data.get("labels", [])
                        
                        # Reconstruct label tuples (label, score)
                        labels = []
                        for lbl_entry in labels_data:
                            if isinstance(lbl_entry, dict):
                                labels.append((lbl_entry.get("label", ""), float(lbl_entry.get("score", 0.0))))
                            elif isinstance(lbl_entry, (list, tuple)) and len(lbl_entry) >= 2:
                                labels.append((str(lbl_entry[0]), float(lbl_entry[1])))
                            else:
                                labels.append((str(lbl_entry), 0.0))
                        
                        if labels:
                            draw_labeled_boundary(
                                image=final_img,
                                boundary=boundary,
                                box=box,
                                lines=[f"{label} ({score:.2f})" for label, score in labels],
                                color=(0, 255, 0),  # Use consistent green for all detections
                            )
                    except Exception as e:
                        logging.warning("[Detection] Could not draw annotation: %s", e)
                
                if not cv2.imwrite(str(labeled_path), final_img):
                    raise RuntimeError(f"Could not write labeled output image: {labeled_path}")
        else:
            # No original provided, just use the temp detected image as labeled output
            temp_labeled = detection_input_path.with_stem(f"{detection_input_path.stem}_temp_detected")
            if temp_labeled.exists():
                shutil.move(str(temp_labeled), str(labeled_path))
        
        # Cleanup temp file if it exists
        temp_detected = detection_input_path.with_stem(f"{detection_input_path.stem}_temp_detected")
        if temp_detected.exists():
            try:
                temp_detected.unlink(missing_ok=True)
            except:
                pass
        
        if labeled_path.exists():
            return f"/output_web/{labeled_path.name}"
    except Exception as exc:
        logging.warning("[Enhancement] Could not create lighting detection preview for %s: %s", final_path.name, exc)
        import traceback
        traceback.print_exc()
    return None


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
        if not (CLUSTER_OUT / "clusters.json").exists():
            return {"clusters": []}
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
        _clear_dir(CLUSTER_STAGING)

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
            "--output", str(CLUSTER_STAGING),
        ]
        cluster_timeout_sec = max(60, int(os.environ.get("PIXELDWELL_CLUSTER_TIMEOUT_SEC", "1800")))
        env = os.environ.copy()
        env.setdefault("SUNRAY_CLUSTER_FAST_MODE", "1")
        result = subprocess.run(
            cmd,
            timeout=cluster_timeout_sec,
            capture_output=False,
            env=env,
        )
        if result.returncode != 0:
            return JSONResponse(
                {"error": "Clustering engine exited with an error. Check terminal for details."},
                status_code=500,
            )

        # Load the JSON written by cluster_images.py
        staged_clusters_json_path = CLUSTER_STAGING / "clusters.json"
        if not staged_clusters_json_path.exists():
            return JSONResponse(
                {"error": "Clustering finished, but clusters.json was not produced."},
                status_code=500,
            )

        _replace_dir_contents(CLUSTER_STAGING, CLUSTER_OUT)

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
        if isinstance(exc, subprocess.TimeoutExpired):
            return JSONResponse(
                {"error": f"Clustering exceeded the {exc.timeout} second limit. Increase PIXELDWELL_CLUSTER_TIMEOUT_SEC or reduce the batch size."},
                status_code=504,
            )
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
        logging.info(
            "[EnhanceCluster] cluster_id=%s job_id=%s weather=%s features=%s",
            cluster_id,
            job_id,
            weather,
            list(ai_features),
        )
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

        # ── ORIGINAL HDR path (for sunray detection) ──
        original_hdr_path = out_path
        
        final_path = out_path
        remaining_features = list(ai_features)

        if "sky_replacement" in remaining_features:
            generated_path = OUTPUT_DIR / f"{out_path.stem}_full_scene_sky.jpg"
            opening_polygons, opening_source_size = _resolve_opening_payload(cluster_id=cluster_id)
            logging.info(
                "[EnhanceCluster] sky replacement openings=%d source_size=%s",
                len(opening_polygons),
                opening_source_size,
            )
            final_path = generate_regenerative_sky_variant(
                input_path=out_path,
                output_path=generated_path,
                weather=weather,
                opening_polygons=opening_polygons,
                opening_source_size=opening_source_size,
                fallback_to_composite=False,
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

        # ── DETECTION: Run on original HDR, display on final generated image ──
        labeled_image_url = _build_enhanced_detection_preview(
            final_path=Path(final_path),
            original_image_path=original_hdr_path
        )
        return {
            "success": True,
            "image_url": f"/output_web/{Path(final_path).name}",
            "labeled_image_url": labeled_image_url,
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
        logging.info(
            "[EnhanceSingle] cluster_id=%s filename=%s job_id=%s weather=%s features=%s",
            cluster_id,
            filename,
            job_id,
            weather,
            list(ai_features),
        )
        source_path = CLUSTER_OUT / cluster_id / filename
        if not source_path.exists():
            return JSONResponse({"error": f"Image '{filename}' not found in cluster '{cluster_id}'."}, status_code=404)

        # ── ORIGINAL source path (for sunray detection) ──
        original_source_path = source_path
        
        final_path = source_path
        remaining_features = list(ai_features)

        if "sky_replacement" in remaining_features:
            generated_path = OUTPUT_DIR / f"{source_path.stem}_full_scene_sky.jpg"
            opening_polygons, opening_source_size = _resolve_opening_payload(cluster_id=cluster_id, filename=filename)
            logging.info(
                "[EnhanceSingle] sky replacement openings=%d source_size=%s",
                len(opening_polygons),
                opening_source_size,
            )
            final_path = generate_regenerative_sky_variant(
                input_path=source_path,
                output_path=generated_path,
                weather=weather,
                opening_polygons=opening_polygons,
                opening_source_size=opening_source_size,
                fallback_to_composite=False,
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

        # ── DETECTION: Run on original source, display on final generated image ──
        labeled_image_url = _build_enhanced_detection_preview(
            final_path=Path(final_path),
            original_image_path=original_source_path
        )
        return {
            "success": True,
            "image_url": f"/output_web/{Path(final_path).name}",
            "labeled_image_url": labeled_image_url,
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
