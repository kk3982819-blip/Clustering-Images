# Project Inputs, Outputs, and Challenges

## Project Overview

This project is an AI-assisted real-estate image processing tool.

It accepts property photos, groups similar images into clusters, detects visible scene elements, and allows enhancement actions such as HDR fusion, sky replacement, decluttering, and staging.

The current system is mainly built with:

- FastAPI for the web backend.
- HTML templates for the frontend.
- CLIP for semantic image similarity.
- HDBSCAN for clustering.
- YOLO-World and SAM for object/window/sky mask detection.
- OpenCV for image processing, compositing, HDR, and relighting.
- SQLite for storing job, cluster, image, and enhancement metadata.

## Inputs

### 1. Uploaded Property Images

Users upload real-estate photos through the web UI.

Supported formats:

```text
.jpg
.jpeg
.png
```

Uploaded images are saved into:

```text
uploads_temp/raw/
```

These images are the primary input for the clustering workflow.

### 2. User Enhancement Selections

After clustering, the user can choose enhancement options from the UI.

Current enhancement inputs include:

- `Sky Replace`
- selected weather type, such as `sunny`, `cloudy`, `foggy`, `snowy`, `sunset`, `night`
- `Declutter`
- `Day-to-Dusk`
- `Staging`
- selected image or full cluster

These choices are sent as form data to FastAPI routes such as:

```text
POST /enhance-single
POST /enhance-cluster
```

### 3. Local Model Files

The project depends on local model weights.

Important files:

```text
yolov8l-world.pt
yolov8x-seg.pt
sam_b.pt
mobile_sam.pt
yolov8n-seg.pt
```

Main usage:

- YOLO-World detects real-estate elements and window/opening regions.
- SAM generates masks for detected regions.
- CLIP generates semantic embeddings for clustering.

### 4. Local Sky Assets

Sky replacement uses local image assets, not a generative model.

Current assets:

```text
static/sky_assets/clear_day.jpg
static/sky_assets/partly_cloudy.jpg
static/sky_assets/overcast.jpg
static/sky_assets/golden_hour.jpg
static/sky_assets/sunset.jpg
```

Weather selections are mapped to these assets.

### 5. Configuration Parameters

`cluster_images.py` supports multiple CLI parameters.

Important parameters:

```text
--input
--output
--model
--batch-size
--min-cluster-size
--min-samples
--cluster-epsilon
--semantic-weight
--layout-weight
--edge-weight
--color-weight
--post-cluster-detector
```

The web app currently uses mostly default values.

## Outputs

### 1. Clustered Image Folders

After upload and clustering, images are written into:

```text
uploads_temp/clustered/
```

Typical output structure:

```text
uploads_temp/clustered/
  clusters.json
  match_scores.json
  cluster_0/
    image_1.jpg
    image_1_labeled.jpg
  cluster_1/
    image_2.jpg
    image_2_labeled.jpg
  noise/
  masks/
```

### 2. `clusters.json`

The main clustering result is:

```text
uploads_temp/clustered/clusters.json
```

It contains:

- cluster names
- image filenames
- scene tags
- annotation metadata
- mask paths

`app.py` reads this file to build frontend cluster cards.

### 3. Labeled Debug Images

The post-cluster detector creates labeled versions of images.

Example:

```text
uploads_temp/clustered/cluster_0/image_1_labeled.jpg
```

These show detected elements such as:

- window
- sky
- floor
- wall
- door
- balcony
- tree
- refrigerator
- power outlet

### 4. Mask Files

When detection/segmentation runs, masks can be saved under:

```text
uploads_temp/clustered/masks/
```

Sky replacement also generates sky masks in:

```text
output_web/
```

Example:

```text
output_web/image_full_scene_sky_skymask.png
```

### 5. Enhanced Images

Final enhancement outputs are written to:

```text
output_web/
```

Examples:

```text
output_web/hdr_cluster_0.jpg
output_web/image_full_scene_sky.jpg
output_web/image_sky_replacement.jpg
```

These images are served through:

```text
/output_web/<filename>
```

### 6. Database Records

The app writes metadata to:

```text
pixeldwell.db
```

Stored entities:

- jobs
- clusters
- images
- enhancements

The database lets the frontend retrieve the latest job and individual cluster details.

### 7. Frontend Cluster Cards

The UI receives a JSON response containing:

- cluster id
- job id
- image count
- thumbnail URLs
- detected tags

These are rendered as interactive cluster cards.

## Main Project Challenges

### 1. Clustering Similar Real-Estate Images

Real-estate images can look very similar.

Challenges:

- multiple rooms with white walls and tiled floors
- same room photographed from different corners
- different rooms with similar layout
- exposure-bracketed shots of the same view

The project addresses this using:

- CLIP semantic embeddings
- layout features
- edge features
- color histograms
- two-stage HDBSCAN clustering
- same-corner refinement

### 2. Avoiding Over-Clustering and Under-Clustering

The system must balance two opposite problems:

- over-clustering: splitting the same room into too many clusters
- under-clustering: merging different rooms into one cluster

This is handled with:

- semantic clustering first
- viewpoint refinement second
- merge-back logic for over-split clusters
- threshold tuning

### 3. Slow Model Loading and Inference

YOLO, SAM, and CLIP are heavy models.

Challenges:

- first run can be slow
- CPU execution is slower than CUDA
- post-cluster detection adds processing time
- sky mask generation can be slow because it uses object detection and segmentation

Possible improvements:

- cache loaded model instances
- cache generated masks
- run clustering/enhancement in background workers
- expose a fast mode without post-cluster detection

### 4. Sky Mask Accuracy

Sky replacement depends on accurate masks.

Challenges:

- windows may include railings, trees, or balcony grills
- bright white walls can be mistaken for sky
- foggy/cloudy skies are harder to detect by color
- glass reflections can confuse the mask

The current system uses:

- YOLO-World for window/opening detection
- SAM for segmentation
- OpenCV treeline/edge logic
- mask cleanup and solidification

### 5. Realistic Sky Replacement

The sky replacement is asset-based, not generative.

Challenges:

- limited sky variety
- night currently reuses an existing asset
- weather realism depends on local sky image quality
- indoor lighting must match the selected sky

Possible improvements:

- add dedicated assets for night, snow, rain, and storm
- improve color harmonization
- add reflection/light-wrap around window frames
- separate sky replacement presets from lighting presets

### 6. Realistic Ground Lighting

Ground/room lighting should not look the same for every weather type.

Challenges:

- sunny and sunset should show visible rays or floor light
- cloudy/foggy/rainy should not show strong sunlight beams
- night should be darker and cooler
- snow can create brighter cool reflected light

The current implementation uses weather-specific lighting values and a whitelist for weather types that should receive the ground beam effect.

### 7. File and State Management

The app uses temporary folders:

```text
uploads_temp/raw/
uploads_temp/clustered/
output_web/
```

Challenges:

- previous run data is cleared on upload
- generated outputs can accumulate
- database records can point to files that are later replaced or deleted

Possible improvements:

- use per-job directories
- add cleanup policies
- store durable output paths by job id

### 8. Synchronous Backend Workflow

The current upload route runs clustering as a blocking subprocess.

Challenge:

- large uploads can block the request
- timeout is fixed at 600 seconds
- user cannot see detailed progress

Possible improvements:

- background task queue
- job status endpoint
- progress events over WebSocket or polling
- cancel/retry support

## Summary

The project input is mainly property image uploads plus user-selected enhancement options.

The project output is clustered image groups, labeled annotations, masks, enhanced images, and database-backed UI metadata.

The hardest parts are:

- grouping visually similar real-estate images correctly
- creating accurate sky/window masks
- making sky replacement and ground lighting look realistic
- managing slow AI model inference
- keeping outputs organized per job

