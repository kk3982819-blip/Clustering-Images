# Clustering Workflow

## Purpose

This document explains how uploaded property images are grouped into clusters in the current app.

The clustering pipeline is implemented mainly in:

- `app.py`
- `cluster_images.py`
- `database.py`
- `templates/index.html`
- `templates/cluster.html`

## High-Level Flow

```text
User uploads images
        |
        v
app.py /upload-and-cluster
        |
        | saves files to uploads_temp/raw
        v
cluster_images.py subprocess
        |
        | CLIP embeddings + visual features
        v
two-stage HDBSCAN clustering
        |
        | optional YOLO-World + SAM annotations
        v
uploads_temp/clustered/
        |
        | clusters.json + cluster folders
        v
app.py reads clusters.json
        |
        | stores job/cluster/image metadata in SQLite
        v
Frontend receives cluster cards
```

## 1. Upload Entry Point

File: `app.py`

Route:

```text
POST /upload-and-cluster
```

The route accepts uploaded image files from the frontend.

Steps:

1. Clears previous temporary inputs:

```text
uploads_temp/raw/
uploads_temp/clustered/
```

2. Saves uploaded files into:

```text
uploads_temp/raw/
```

3. Runs the clustering engine as a subprocess:

```python
python cluster_images.py --input uploads_temp/raw --output uploads_temp/clustered
```

The subprocess timeout is currently `600` seconds.

## 2. Clustering Engine

File: `cluster_images.py`

Main function:

```python
main()
```

Default model:

```text
CLIP ViT-B/32
```

Default clustering method:

```text
Two-stage HDBSCAN
```

Default post-cluster detector:

```text
world-sam
```

Because `app.py` does not pass `--post-cluster-detector none`, the default `world-sam` annotation stage runs.

## 3. Image Discovery

`cluster_images.py` scans the input folder for supported images.

Supported extensions:

```text
.jpg
.jpeg
.png
```

If no images are found, the script exits.

## 4. CLIP Embedding Extraction

Each image is converted into a semantic embedding using CLIP.

Default:

```text
ViT-B/32
```

Embeddings are cached under:

```text
.clip-cache/
```

These CLIP embeddings represent high-level visual/semantic similarity, such as whether two images show the same room type or scene.

## 5. Visual Feature Extraction

The pipeline also extracts lower-level visual features from each image:

- layout features
- edge/structure features
- color histogram features

These are used to distinguish images that may be semantically similar but visually from different angles or corners.

## 6. Feature Combination

`cluster_images.py` combines CLIP and visual features using weighted values.

Default weights:

```text
semantic_weight = 0.45
layout_weight   = 0.35
edge_weight     = 0.15
color_weight    = 0.05
```

The combined vector is used for viewpoint/same-corner refinement.

## 7. First-Stage Clustering: Semantic Grouping

The first stage clusters images using CLIP semantic embeddings.

Method:

```text
HDBSCAN
```

Purpose:

- group images that likely show the same room or scene type
- separate clearly unrelated images

Default parameters:

```text
min_cluster_size = 2
min_samples      = 1
cluster_epsilon  = 0.0
```

Images that do not fit strongly into a cluster are labeled as:

```text
noise
```

## 8. Second-Stage Clustering: Same-Corner Refinement

Inside each semantic cluster, the pipeline refines groups using the combined hybrid feature vector.

Purpose:

- split one broad room group into tighter viewpoint groups
- keep same-corner / same-angle photos together
- avoid grouping different room corners just because they are semantically similar

This stage also uses HDBSCAN.

## 9. Refinement Helpers

The pipeline includes extra correction logic:

- `maybe_split_quad_cluster(...)`
  - tries to split four-image clusters into two viewpoint pairs when appropriate

- `maybe_refine_broad_viewpoint_cluster(...)`
  - breaks very broad room clusters into tighter subclusters

- `merge_semantic_subclusters(...)`
  - merges subclusters back when CLIP centroid similarity says they are still the same semantic scene

This prevents both over-grouping and over-splitting.

## 10. Optional Strict Mode

`cluster_images.py` supports:

```text
--strict-same-corner-items
```

When enabled, it builds extra CLIP prompt-signature features from room/item prompts.

This helps ensure that images stay together only when both viewpoint and visible items are similar.

The current `app.py` route does not enable this flag by default.

## 11. Post-Cluster Detection And Annotation

Default:

```text
--post-cluster-detector world-sam
```

Models used:

- `yolov8l-world.pt`
  - prompted open-vocabulary detection

- `yolov8x-seg.pt`
  - fallback detector

- `sam_b.pt`
  - segmentation masks for detected objects/regions

This stage creates:

- labeled debug images
- object/scene annotations
- optional mask files

The labels are later shown as chips in the frontend cluster cards.

## 12. Output Folder Structure

Output root:

```text
uploads_temp/clustered/
```

Typical structure:

```text
uploads_temp/clustered/
  clusters.json
  match_scores.json
  cluster_0/
    image_1.jpg
    image_1_labeled.jpg
    image_2.jpg
    image_2_labeled.jpg
  cluster_1/
    ...
  noise/
    ...
  masks/
    cluster_0/
      ...
```

## 13. `clusters.json`

The main handoff file is:

```text
uploads_temp/clustered/clusters.json
```

It contains cluster payloads like:

```json
{
  "cluster_0": [
    {
      "image": "example.jpg",
      "tags": ["sky", "window"],
      "mask_dir": "masks/cluster_0/example",
      "mask_files": [],
      "annotations": []
    }
  ],
  "noise": []
}
```

`app.py` reads this file after the subprocess completes.

## 14. Database Handoff

File: `database.py`

After reading `clusters.json`, `app.py` creates a job and inserts:

- job row
- cluster rows
- image rows
- image annotations

Database file:

```text
pixeldwell.db
```

Important functions:

- `create_job(...)`
- `insert_cluster(...)`
- `insert_image(...)`
- `get_latest_job_clusters(...)`
- `get_cluster_details(...)`

## 15. Frontend Response

After database insertion, `app.py` returns cluster cards to the frontend.

Each cluster card includes:

- cluster id
- job id
- image count
- thumbnail URLs
- detected tags/labels

Original images are served from:

```text
/clusters/<cluster_name>/<image_name>
```

Labeled images are served from:

```text
/clusters/<cluster_name>/<image_stem>_labeled.jpg
```

## 16. Workflow Summary

```text
1. User uploads images.
2. app.py saves them to uploads_temp/raw.
3. app.py runs cluster_images.py.
4. cluster_images.py extracts CLIP embeddings.
5. cluster_images.py extracts layout/edge/color features.
6. CLIP features create semantic clusters.
7. Hybrid features refine same-corner/viewpoint groups.
8. YOLO-World/SAM annotate clustered images.
9. Clustered files and clusters.json are written.
10. app.py reads clusters.json.
11. app.py stores job, cluster, and image metadata.
12. Frontend renders cluster cards.
```

## Current Limitations

- Clustering runs as a blocking subprocess from `app.py`.
- The timeout is fixed at 600 seconds.
- Heavy model loading can make first runs slow.
- The app route uses default clustering settings; advanced CLI flags are not exposed in the UI.
- `noise` images are excluded from returned cluster cards.
- Existing `uploads_temp/raw` and `uploads_temp/clustered` contents are cleared before each new upload run.

## Useful CLI Command

Run clustering manually:

```powershell
python cluster_images.py --input uploads_temp/raw --output uploads_temp/clustered
```

Run without post-cluster detection:

```powershell
python cluster_images.py --input uploads_temp/raw --output uploads_temp/clustered --post-cluster-detector none
```

