# Image Clustering Architecture

## Purpose

This project clusters visually related images from one input folder into:

- semantic groups such as the same room or same scene family
- tighter same-corner groups inside those semantic groups
- optional strict groups that also try to keep visible items consistent

The implementation is centered in `cluster_images.py`.

## Top-Level Design

The pipeline is a single-process batch workflow:

1. discover images from the input folder
2. load CLIP and compute semantic image embeddings
3. extract hand-crafted visual features from each image
4. optionally extract CLIP prompt-signature features for strict mode
5. combine semantic and visual features into one hybrid embedding
6. run two-stage clustering
7. write clustered copies plus JSON outputs

## Main Components

### 1. Input and Runtime

- `parse_args()`
  Defines thresholds, feature weights, strict mode, device, and I/O paths.
- `resolve_device()`
  Selects `cpu`, `cuda`, or auto mode.
- `discover_images()`
  Collects supported image files from the input directory.
- `load_rgb_image()`
  Opens images safely and normalizes them to RGB.

## 2. Semantic Feature Extraction

- `load_clip_module()` and `load_clip_runtime()`
  Load CLIP either from the local `.vendor` folder or from the environment.
- `embed_images()`
  Encodes each image into a normalized CLIP embedding.

Role in the system:

- CLIP provides scene-level meaning
- it is the first-stage clustering signal
- it also acts as a semantic guardrail in strict mode

## 3. Visual Structure Extraction

`extract_visual_features()` builds a visual feature vector per image from:

- layout features via `image_to_layout_vector()`
- edge-map features via `image_to_edge_vector()`
- color histogram features via `image_to_color_histogram()`

These features are used to separate images that are semantically similar but taken from different viewpoints.

## 4. Viewpoint Matching Features

The second-stage refinement uses additional viewpoint cues:

- opening profiles via `image_to_opening_profile()`
- ORB descriptors via `image_to_orb_descriptors()`
- ORB local match score via `orb_similarity_score()`

`viewpoint_similarity_matrix()` combines them with this weighted formula:

```text
0.35 * layout
+ 0.25 * edge
+ 0.25 * opening profile
+ 0.15 * ORB local match score
```

Role in the system:

- layout and edge features capture broad geometry
- opening profiles help with windows and doors
- ORB helps match local structures and repeated details

## 5. Strict Item Similarity Features

If `--strict-same-corner-items` is enabled:

- `extract_clip_item_features()` computes prompt-signature vectors
- the prompts come from `STRICT_ITEM_PROMPTS`
- the image is compared against fixed text prompts such as kitchen, living room, sofa, window, sliding glass door, and similar items

This creates a CLIP-based proxy for visible room/item content.

## 6. Feature Fusion

`combine_features()` merges:

- CLIP semantic embedding
- layout vector
- edge vector
- color histogram

using normalized weights from the CLI:

- `semantic-weight`
- `layout-weight`
- `edge-weight`
- `color-weight`

The result is a normalized hybrid embedding used by the default second-stage clustering path.

## 7. Clustering Engine

### Stage 1: Semantic Clustering

`cluster_embeddings()` runs HDBSCAN on CLIP embeddings only.

Purpose:

- group images that belong to the same broad scene or room
- avoid mixing unrelated scenes before viewpoint refinement

### Stage 2A: Default Same-Corner Refinement

In the non-strict path, `cluster_same_corner_groups()` does one of the following inside each semantic cluster:

- cluster hybrid embeddings with HDBSCAN
- for large broad clusters, split them further with `maybe_refine_broad_viewpoint_cluster()`
- for 4-image cases, try a dedicated pair-splitting heuristic with `maybe_split_quad_cluster()`
- merge over-split groups back together with `merge_semantic_subclusters()` if CLIP centroid similarity is high

### Stage 2B: Strict Same-Corner Plus Same-Items Refinement

If strict mode is enabled, `strict_same_corner_item_clusters()` builds a strict similarity matrix using:

```text
0.50 * viewpoint similarity
+ 0.30 * item similarity
+ 0.20 * semantic similarity
```

A pair is only allowed to connect if all three gates pass:

- viewpoint similarity >= `view-similarity-threshold`
- item similarity >= `item-similarity-threshold`
- semantic similarity >= `semantic-similarity-floor`

This is why strict mode is better for near-duplicate or same-corner grouping, but can also create more noise.

## 8. Outputs

`reset_output_dir()` clears and recreates the output directory.

`copy_clustered_images()` writes:

- `cluster_0`, `cluster_1`, and similar folders
- `noise/` for label `-1`
- `clusters.json` with final cluster membership

`write_match_scores()` writes `match_scores.json`, which includes:

- per-image match lists
- match percentages
- semantic, hybrid, viewpoint, and item percentages
- `same_cluster`
- `passes_strict_thresholds` in strict mode

## Main Data Objects

- `image_paths`
  Discovered input files.
- `clip_embeddings`
  Semantic CLIP vectors.
- `visual_features`
  Layout, edge, and color features.
- `item_features`
  Prompt-signature vectors in strict mode.
- `embeddings`
  Final hybrid embeddings after feature fusion.
- `labels`
  Final cluster labels, with `-1` meaning noise.

## Execution Sequence

The `main()` function runs the workflow in this order:

1. parse arguments and configure logging
2. resolve input, output, cache, and device
3. discover images
4. compute CLIP embeddings
5. compute visual features
6. optionally compute strict-mode item features
7. align feature arrays if unreadable images were skipped
8. build hybrid embeddings
9. run `cluster_same_corner_groups()`
10. write clustered copies and JSON reports

## Architectural Strengths

- clear separation between semantic grouping and viewpoint refinement
- combines learned features and hand-crafted features
- supports both balanced and strict clustering styles
- writes interpretable diagnostics through `match_scores.json`

## Architectural Limitations

- threshold behavior is dataset-sensitive
- strict mode uses hard gates, which can reject valid detail views
- prompt signatures are fixed and domain-biased
- viewpoint refinement recomputes image-derived features multiple times
- no built-in evaluation harness or auto-tuning loop
