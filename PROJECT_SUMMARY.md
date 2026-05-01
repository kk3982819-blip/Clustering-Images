# Project Summary

## Overview

This project is an AI-assisted real-estate photo processing platform.

It helps organize and enhance property images by:

- uploading property photos
- clustering similar images together
- detecting visible room and scene elements
- generating labeled preview images
- applying enhancements such as HDR fusion, sky replacement, decluttering, day-to-dusk, and staging

The system combines computer vision, image clustering, object detection, segmentation, and local image compositing.

## Main Workflow

```text
User uploads property photos
        |
        v
Images are saved to uploads_temp/raw
        |
        v
cluster_images.py groups similar photos
        |
        v
Clusters are written to uploads_temp/clustered
        |
        v
app.py reads clusters.json and stores metadata
        |
        v
Frontend displays cluster cards
        |
        v
User applies image enhancements
        |
        v
Enhanced results are saved to output_web
```

## Core Features

### 1. Image Clustering

The project groups uploaded photos using:

- CLIP semantic embeddings
- layout features
- edge features
- color histogram features
- two-stage HDBSCAN clustering

This helps separate images by room, viewpoint, and visual similarity.

### 2. Object and Scene Detection

After clustering, the system can detect and annotate visible elements using:

- YOLO-World
- YOLO fallback detection
- SAM segmentation

Detected items can include:

- sky
- window
- wall
- floor
- door
- balcony
- trees
- room fixtures
- furniture/appliances

### 3. Sky Replacement

Sky replacement is local and asset-based.

It uses:

- YOLO/SAM/OpenCV to create sky/window masks
- local sky images from `static/sky_assets`
- OpenCV blending
- weather-aware lighting and tone adjustment

It does not use a generative sky model.

### 4. HDR and Enhancements

The app supports enhancement actions such as:

- HDR fusion
- sky replacement
- object removal / declutter
- day-to-dusk
- virtual staging

Enhanced outputs are written to:

```text
output_web/
```

## Key Technologies

- FastAPI
- HTML/Jinja templates
- SQLite
- OpenCV
- CLIP
- HDBSCAN
- YOLO-World
- SAM
- PyTorch

## Important Folders

```text
uploads_temp/raw/        uploaded source images
uploads_temp/clustered/  clustered images and clusters.json
output_web/              final enhanced outputs
static/sky_assets/       sky replacement assets
.clip-cache/             cached CLIP embeddings
```

## Main Outputs

The system produces:

- cluster folders
- `clusters.json`
- `match_scores.json`
- labeled debug images
- segmentation masks
- enhanced images
- SQLite metadata records
- frontend cluster cards

## Main Challenges

The hardest parts of the project are:

- correctly clustering visually similar real-estate rooms
- separating same-room different-corner photos
- avoiding over-clustering and under-clustering
- generating accurate sky/window masks
- making sky replacement look realistic
- matching ground lighting to selected weather
- handling slow AI model loading and inference
- managing temporary files and per-job outputs

## Current Limitations

- Clustering runs as a blocking subprocess.
- Heavy models can make the first run slow.
- Sky replacement depends on local sky assets.
- Night/rain/snow realism would improve with dedicated assets.
- Advanced clustering settings are available in CLI but not exposed in UI.
- Temporary upload folders are cleared between new upload runs.

## Summary

In short, this project takes raw real-estate images, clusters them by visual and semantic similarity, annotates important scene elements, and lets the user apply AI-assisted image enhancements. It is built as a local computer-vision pipeline rather than a fully generative image system.

