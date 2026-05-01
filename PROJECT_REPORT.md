# Project Report: PixelDwell AI Photo Editing Platform

Report date: April 22, 2026

## Abstract

PixelDwell is an AI-assisted real-estate photo processing platform that organizes, annotates, and enhances property images. The system accepts a batch of uploaded property photos, groups similar images into room or viewpoint clusters, detects visible scene elements, and provides image enhancement actions such as HDR fusion, sky replacement, object removal, decluttering, day-to-dusk conversion, and virtual staging.

The project combines deep learning models with traditional image processing. CLIP embeddings and HDBSCAN are used for semantic and visual clustering. YOLO-World and SAM support open-vocabulary detection and segmentation. OpenCV handles HDR fusion, masking, compositing, relighting, and inpainting-style effects. A FastAPI web application, Jinja templates, and SQLite database provide the user workflow and metadata storage.

## 1. Introduction

Real-estate image workflows often involve large sets of similar photos captured from different rooms, angles, and exposure settings. Manually sorting these images and applying consistent visual improvements is time-consuming. PixelDwell addresses this problem by creating a local computer-vision pipeline that can cluster similar property images, label important scene elements, and apply enhancement operations through a web interface.

The project is designed for local execution and experimentation. Instead of relying on a fully generative image system for every task, it uses a combination of learned visual models, local assets, and image-processing algorithms. This makes the workflow easier to inspect and tune for real-estate photography.

## 2. Problem Statement

The project aims to solve the following problems:

- Property photo sets often contain many visually similar images that need to be grouped by room, scene type, or camera viewpoint.
- Images from the same room can look different because of angle, lighting, exposure, and visible furniture.
- Automated enhancement features require accurate masks for sky, windows, objects, and scene regions.
- Real-estate image enhancements must remain realistic, especially for sky replacement, lighting changes, and HDR fusion.
- The system needs a simple user interface for uploading, reviewing, and enhancing clustered images.

## 3. Objectives

The main objectives of the project are:

- Build a web-based image upload and clustering workflow.
- Group uploaded property photos into visually and semantically related clusters.
- Separate broad room groups into tighter same-corner or same-viewpoint groups.
- Generate labeled preview images with detected scene and object annotations.
- Support enhancement actions for individual images and whole clusters.
- Store job, cluster, image, annotation, and enhancement metadata in SQLite.
- Produce reusable output folders, JSON reports, masks, and enhanced images.

## 4. Scope

The current system covers:

- Uploading `.jpg`, `.jpeg`, and `.png` property images.
- CLIP-based semantic embedding extraction.
- Layout, edge, and color feature extraction.
- Two-stage HDBSCAN clustering.
- Optional strict same-corner plus same-items clustering.
- YOLO-World and SAM-based post-cluster annotation.
- HDR fusion for clustered image sets.
- Local sky replacement using curated sky assets.
- Object removal, decluttering, day-to-dusk, and virtual staging simulation.
- SQLite-backed retrieval of latest jobs and cluster details.
- Web UI pages for cluster overview and cluster-level image review.

The current system does not yet provide:

- Background job execution.
- Real-time upload progress tracking.
- Automatic threshold tuning.
- A formal clustering accuracy evaluation dashboard.
- Dedicated production-grade generative editing service integration.

## 5. Technologies Used

| Area | Technology |
| --- | --- |
| Backend web framework | FastAPI |
| Frontend rendering | HTML and Jinja templates |
| Database | SQLite |
| Semantic image features | CLIP |
| Clustering | HDBSCAN via scikit-learn |
| Detection | YOLO-World and YOLO segmentation models |
| Segmentation | SAM and MobileSAM model files |
| Image processing | OpenCV |
| Machine learning runtime | PyTorch and TorchVision |
| Image I/O | Pillow |
| Output format | Folders, JSON files, images, masks, SQLite rows |

## 6. System Architecture

PixelDwell is organized as a local web application with a batch-processing computer-vision backend.

### 6.1 Main Components

- `app.py`: FastAPI application, upload route, clustering subprocess call, enhancement routes, static file serving, and template rendering.
- `cluster_images.py`: Main clustering engine with feature extraction, two-stage clustering, annotation, mask output, and JSON reporting.
- `database.py`: SQLite schema and helper functions for jobs, clusters, images, and enhancements.
- `hdr_engine.py`: HDR fusion and base enhancement layer using exposure fusion, white balance, CLAHE, and sharpening.
- `mask_generator.py`: Sky and object mask generation using detection, segmentation, and OpenCV cleanup logic.
- `full_scene_generator.py`: Full-scene sky replacement and weather-aware relighting.
- `api_orchestrator.py`: Enhancement routing layer. Advanced effects are currently simulated locally through a mock external API class.
- `templates/index.html`: Upload and cluster overview interface.
- `templates/cluster.html`: Cluster detail and enhancement interface.

### 6.2 Important Folders

| Folder | Purpose |
| --- | --- |
| `input/` | Local input images for command-line clustering tests |
| `uploads_temp/raw/` | Uploaded source images from the web app |
| `uploads_temp/clustered/` | Clustered images, labeled images, masks, and clustering JSON |
| `output_web/` | Final enhanced outputs and generated masks |
| `static/sky_assets/` | Local sky replacement assets |
| `.clip-cache/` | Cached CLIP embeddings |

## 7. Workflow

The high-level workflow is:

```text
User uploads property images
        |
        v
FastAPI saves files into uploads_temp/raw
        |
        v
app.py runs cluster_images.py as a subprocess
        |
        v
CLIP, layout, edge, and color features are extracted
        |
        v
Two-stage HDBSCAN creates semantic and viewpoint clusters
        |
        v
YOLO-World and SAM annotate clustered images
        |
        v
clusters.json, match_scores.json, masks, and image folders are written
        |
        v
app.py stores metadata in SQLite
        |
        v
Frontend displays cluster cards and enhancement options
        |
        v
Enhanced images are saved into output_web
```

## 8. Methodology

### 8.1 Image Upload

The web application exposes `POST /upload-and-cluster`. Uploaded images are saved into `uploads_temp/raw/`. Before a new upload run, the app clears the contents of the temporary raw and clustered folders while keeping the parent folders mounted for static serving.

### 8.2 Feature Extraction

The clustering engine extracts several feature types:

- CLIP embeddings for semantic similarity.
- Grayscale layout vectors for room structure.
- Edge-map vectors for geometry and camera-view cues.
- HSV color histograms for broad color appearance.
- Optional CLIP prompt-signature features in strict mode.

The default feature weights are:

| Feature | Weight |
| --- | ---: |
| Semantic CLIP embedding | 0.45 |
| Layout feature | 0.35 |
| Edge feature | 0.15 |
| Color histogram | 0.05 |

### 8.3 Clustering

The clustering process has two stages.

Stage 1 performs semantic clustering using CLIP embeddings. This separates broad scene groups such as rooms, balconies, windows, kitchens, or empty interiors.

Stage 2 refines each semantic group using a hybrid feature vector. This separates images that may show the same broad scene but from different corners or viewpoints.

The pipeline also includes extra refinement logic:

- Four-image clusters can be split into two likely viewpoint pairs.
- Broad clusters can be further split using viewpoint similarity.
- Over-split groups can be merged back using CLIP centroid similarity.
- Strict mode can require viewpoint, item, and semantic similarity gates.

### 8.4 Detection and Annotation

After clustering, the default post-cluster detector is `world-sam`. It uses YOLO-World-style open-vocabulary detection and SAM segmentation to identify scene elements and produce masks. The output includes labeled images and annotation metadata with labels, scores, bounding boxes, and boundaries.

Detected elements can include:

- Window
- Sky
- Floor
- Wall
- Door
- Balcony
- Tree or outdoor view
- Kitchen cabinet
- Appliance
- Power outlet
- Ceiling light

### 8.5 Enhancement

Enhancements can be applied to a full cluster or a single image.

Cluster enhancement first applies HDR fusion through `hdr_engine.py`. The engine aligns images, performs Mertens exposure fusion, applies white balance, improves local contrast with CLAHE, and sharpens the final output.

Single-image enhancement can apply sky replacement directly through `full_scene_generator.py`. Other effects are routed through `api_orchestrator.py`, where the current implementation simulates external API behavior locally.

Recognized enhancement features include:

- HDR fusion
- Sky replacement
- Object removal
- Declutter
- Day-to-dusk
- Virtual staging
- Sketch overlay
- White balance
- Perspective correction, currently recognized but not fully implemented

## 9. Database Design

The SQLite database file is `pixeldwell.db`.

The schema contains four main tables:

| Table | Purpose |
| --- | --- |
| `jobs` | Stores each upload or clustering job |
| `clusters` | Stores cluster records for a job |
| `images` | Stores image paths, labeled image paths, and annotations |
| `enhancements` | Stores completed enhancement records |

This database lets the frontend retrieve the latest job through `GET /latest-job` and retrieve a single cluster through `GET /job/{job_id}/cluster/{cluster_name}`.

## 10. Inputs and Outputs

### 10.1 Inputs

The system accepts:

- Uploaded property images in `.jpg`, `.jpeg`, or `.png` format.
- User-selected enhancement options.
- Weather presets for sky replacement and relighting.
- Local model weights such as YOLO, SAM, and MobileSAM files.
- Local sky assets from `static/sky_assets/`.

### 10.2 Outputs

The system produces:

- Cluster folders such as `cluster_0`, `cluster_1`, and `noise`.
- `clusters.json` with cluster membership, tags, masks, and annotations.
- `match_scores.json` with pairwise image similarity diagnostics.
- Labeled image previews.
- Segmentation masks.
- Enhanced output images in `output_web/`.
- SQLite metadata records.
- Frontend cluster cards.

## 11. Current Result Summary

The current local sample input contains 12 JPG property images.

The latest clustered output in `uploads_temp/clustered/` produced:

| Group | Image Count | Main Tags |
| --- | ---: | --- |
| `cluster_0` | 2 | floor tiles, empty room, white wall, furnished room, window, balcony |
| `cluster_1` | 2 | floor tiles, empty room, window, white wall, furnished room |
| `cluster_2` | 2 | floor tiles, white wall, balcony, furnished room, outdoor view, empty room |
| `cluster_3` | 3 | floor tiles, white wall, empty room, furnished room |
| `cluster_4` | 2 | floor tiles, white wall, empty room, furnished room |
| `noise` | 1 | furnished room, white wall, floor tiles, empty room |

The `output_web/` folder currently contains generated enhancement outputs and masks, including JPG image outputs and PNG mask/debug files. The sky asset folder contains local assets for clear day, partly cloudy, overcast, golden hour, and sunset sky replacement.

## 12. Strengths

- Uses CLIP semantic features instead of relying only on pixel similarity.
- Adds layout, edge, and color features to improve viewpoint separation.
- Produces inspectable outputs through folders, labels, masks, and JSON reports.
- Supports both cluster-level and single-image enhancement.
- Uses local sky assets and OpenCV compositing for controlled sky replacement.
- Stores metadata in a lightweight SQLite database.
- Keeps the web app simple and easy to run locally.

## 13. Challenges

The major challenges are:

- Real-estate images often have similar walls, floors, and lighting, making clustering difficult.
- Same-room images from different corners can be semantically similar but geometrically different.
- Heavy models such as CLIP, YOLO, and SAM can slow down first-run performance.
- Sky and window masks can fail when windows contain reflections, railings, trees, or bright walls.
- Asset-based sky replacement requires carefully matched lighting and tone adjustment.
- Temporary output folders can be overwritten between upload runs.
- The upload route currently blocks while clustering runs.

## 14. Limitations

The current implementation has the following limitations:

- The clustering subprocess is synchronous and has a fixed timeout.
- Advanced clustering thresholds are available in the CLI but not exposed in the UI.
- Model loading is not fully optimized for repeated web requests.
- The current external enhancement API is mocked in code.
- Noise images are not returned as regular cluster cards.
- Output storage is not fully isolated per job.
- There is no formal accuracy metric for clustering quality yet.

## 15. Future Enhancements

Recommended improvements include:

- Move clustering and enhancement into background jobs.
- Add a job status endpoint and frontend progress display.
- Cache loaded model instances across requests.
- Use per-job directories for raw files, clustered outputs, masks, and final images.
- Expose advanced clustering controls in the UI.
- Add evaluation metrics for cluster quality and mask quality.
- Add dedicated sky assets for night, rain, snow, and storm scenes.
- Improve object-removal quality with a production inpainting model.
- Add user review tools to manually merge or split clusters.
- Add cleanup policies for temporary files and old outputs.

## 16. Conclusion

PixelDwell is a practical AI-assisted platform for real-estate image organization and enhancement. The project successfully combines semantic clustering, visual feature engineering, object detection, segmentation, local image processing, and a web-based workflow.

The strongest part of the system is its end-to-end pipeline: users can upload property images, receive organized clusters, inspect labeled scene elements, and apply enhancement actions. The current design is suitable for experimentation and local deployment. With background processing, stronger model caching, per-job output isolation, and improved enhancement models, the project can evolve into a more production-ready real-estate photo editing tool.

## 17. Key Project Files

| File | Role |
| --- | --- |
| `app.py` | Web routes, upload handling, clustering subprocess, enhancement endpoints |
| `cluster_images.py` | Main clustering, feature extraction, annotation, and output generation engine |
| `database.py` | SQLite schema and data access helpers |
| `hdr_engine.py` | HDR fusion and base image enhancement |
| `mask_generator.py` | Sky and object mask generation |
| `full_scene_generator.py` | Full-scene sky replacement and relighting |
| `api_orchestrator.py` | Enhancement routing and simulated external processing |
| `templates/index.html` | Main upload and cluster-card UI |
| `templates/cluster.html` | Cluster detail and enhancement UI |
| `requirements.txt` | Python dependencies |
