# Implementation Plan

## 1. Objective

This project clusters images from one input folder into visually meaningful groups.

The current implementation is designed to produce:

- broad semantic groups such as the same room or scene family
- tighter same-corner groups inside those semantic groups
- optional strict groups that also try to keep visible items consistent
- a `noise` bucket for images that do not fit any stable cluster

The implementation is centered in `cluster_images.py`.

## 2. Current Tech Stack

The pipeline is implemented as a Python batch script with a small number of supporting markdown docs.

Main libraries and tools:

- `torch`
- `clip` from OpenAI CLIP
- `Pillow`
- `opencv-python`
- `scikit-learn`
- `numpy`

Main models and algorithms:

- CLIP image embeddings for scene-level semantics
- CLIP text prompt signatures for optional strict item-aware mode
- OpenCV ORB descriptors for local viewpoint matching
- HDBSCAN for the main clustering stages
- Agglomerative clustering for refinement and strict grouping

## 3. High-Level Architecture

The system is implemented as a single-process end-to-end pipeline:

1. parse arguments and runtime settings
2. discover input images
3. load CLIP
4. compute semantic embeddings
5. compute hand-crafted visual features
6. optionally compute CLIP item-signature features
7. fuse features into hybrid embeddings
8. run stage-1 semantic clustering
9. run stage-2 refinement inside each semantic cluster
10. write output folders and JSON diagnostics

This is a pipeline design, not a service-based design. There is no database, API, or background worker. The script reads a folder, processes it, and writes a new output folder.

## 4. Main Entry Point

The workflow starts in `main()`.

`main()` is responsible for:

- parsing CLI arguments
- resolving device and directories
- discovering images
- building semantic and visual features
- invoking clustering
- resetting the output directory
- writing result folders and JSON files

This is the execution order of the current implementation:

1. `parse_args()`
2. `setup_logging()`
3. `resolve_device()`
4. `discover_images()`
5. `embed_images()`
6. `extract_visual_features()`
7. optionally `extract_clip_item_features()`
8. feature alignment if some images were unreadable
9. `combine_features()`
10. `cluster_same_corner_groups()`
11. `copy_clustered_images()`
12. `write_match_scores()`

## 5. Runtime and Input Layer

### 5.1 Argument parsing

`parse_args()` defines the full runtime contract for the script:

- input folder
- output folder
- CLIP model name
- batch size
- clustering thresholds
- feature weights
- strict mode controls
- device selection

This keeps the implementation tunable without changing code for every experiment.

### 5.2 Device resolution

`resolve_device()` selects:

- `cuda` if available and requested or auto-detected
- otherwise `cpu`

This allows the same code to run on a laptop or GPU machine.

### 5.3 Image discovery

`discover_images()` scans the input directory and keeps only supported extensions:

- `.jpg`
- `.jpeg`
- `.png`

The current code expects a flat folder of images. It does not recursively walk nested subfolders.

### 5.4 Image loading

`load_rgb_image()`:

- opens the image safely
- applies EXIF orientation correction
- converts to RGB
- returns `None` if the image cannot be read

This avoids pipeline crashes from bad files.

## 6. CLIP Loading Strategy

The project uses `load_clip_module()` and `load_clip_runtime()` to load CLIP.

Current behavior:

- it first tries the vendored `.vendor/clip` package
- if that vendored package is unavailable or unreadable, it falls back to the installed `clip` package in the Python environment

This makes the project more portable across different local setups.

The default model is:

- `ViT-B/32`

That model can be overridden from the CLI with `--model`.

## 7. Feature Extraction Design

The implementation separates features into three groups:

- semantic features
- visual structure features
- optional item-aware features

### 7.1 Semantic features with CLIP

`embed_images()`:

- loads CLIP
- preprocesses each valid image
- encodes it with `model.encode_image(...)`
- concatenates batch outputs
- L2-normalizes the final embedding matrix

Purpose:

- capture room-level or scene-level meaning
- keep semantically unrelated images apart in stage 1

This is the strongest high-level signal in the system.

### 7.2 Visual structure features

`extract_visual_features()` computes multiple handcrafted features per image:

- layout vector from grayscale downsampled image
- edge vector from edge-filtered grayscale image
- color histogram from HSV image

These features are used because CLIP alone is often too semantic. In real-estate or interior datasets, two corners of the same room can be semantically similar while still needing separate clusters.

### 7.3 Viewpoint-specific cues

For same-corner refinement, the implementation computes extra viewpoint signals:

- opening profile
- ORB descriptors
- ORB local match score

Functions involved:

- `image_to_opening_profile()`
- `image_to_orb_descriptors()`
- `orb_similarity_score()`
- `viewpoint_similarity_matrix()`

The viewpoint similarity matrix is computed as:

```text
0.35 * layout similarity
0.25 * edge similarity
0.25 * opening profile similarity
0.15 * ORB similarity
```

Purpose:

- separate different corners of the same room
- reward similar geometry, openings, and local structures

### 7.4 Strict item-aware features

If `--strict-same-corner-items` is enabled, the project also computes CLIP prompt-signature item vectors using `extract_clip_item_features()`.

This works by:

1. tokenizing a fixed list of room and furniture prompts
2. encoding those prompts with CLIP text encoder
3. encoding images with CLIP image encoder
4. measuring image-to-prompt similarity
5. turning those similarities into a soft distribution

This creates an item-aware representation used only in strict mode.

Purpose:

- distinguish views that are semantically similar but contain different visible items
- make same-corner grouping stricter

## 8. Feature Fusion

The project does not cluster directly on raw CLIP plus raw handcrafted arrays.

Instead, it combines them into a hybrid embedding through `combine_features()`.

Default weights:

- semantic: `0.45`
- layout: `0.35`
- edge: `0.15`
- color: `0.05`

The fused vector is normalized again before clustering.

Why this was implemented:

- semantic features give broad scene meaning
- visual features help distinguish different viewpoints
- weighted fusion gives a tunable balance between those two goals

## 9. Clustering Strategy

The project uses a two-stage clustering design.

This is an important implementation decision.

The pipeline does not try to solve everything in one clustering pass. Instead, it first creates broad semantic groups, then refines within those groups.

### 9.1 Stage 1: Semantic clustering

`cluster_embeddings()` runs HDBSCAN on CLIP embeddings only.

Current HDBSCAN configuration:

- metric: `euclidean`
- `min_cluster_size`: CLI-driven
- `min_samples`: CLI-driven
- `cluster_selection_epsilon`: CLI-driven
- `cluster_selection_method`: `eom`
- `n_jobs=1`

Purpose:

- find broad same-room or same-scene groups
- leave uncertain images as noise
- avoid mixing unrelated scenes before viewpoint refinement

If no stable semantic clusters are found, the pipeline falls back to one-stage HDBSCAN on hybrid embeddings.

### 9.2 Stage 2: Refinement inside each semantic cluster

`cluster_same_corner_groups()` implements the second stage.

This function loops through each semantic cluster and then chooses one of two refinement paths:

- default path
- strict path

## 10. Default Refinement Path

The default path is used when `--strict-same-corner-items` is not enabled.

Inside each semantic cluster, the implementation does the following:

1. run local HDBSCAN on the hybrid embeddings
2. if a 4-image cluster is detected, try a special pair-splitting heuristic
3. if a local cluster is large and broad, try agglomerative viewpoint-based splitting
4. merge subclusters back if semantic centroids are very similar
5. assign final labels

### 10.1 Local hybrid clustering

The code runs `cluster_embeddings()` again on the hybrid vectors for the images inside the current semantic group.

Purpose:

- split a broad room-level group into tighter viewpoint groups

### 10.2 Special 4-image split heuristic

`maybe_split_quad_cluster()` handles the common case where one room group contains exactly four images and may actually be two clean pairs.

It uses opening profile and ORB matching to decide whether:

- pair A belongs together
- pair B belongs together
- the split is sufficiently better than competing pairings

This heuristic exists because small groups can be unstable under generic clustering.

### 10.3 Broad viewpoint refinement

`maybe_refine_broad_viewpoint_cluster()` is used when a local cluster has at least five images.

This function:

- computes a viewpoint similarity matrix
- converts similarity into distance
- runs `AgglomerativeClustering`
- uses `linkage="complete"`
- uses a threshold derived from `view_similarity_threshold`

Purpose:

- break a broad room cluster into smaller same-corner groups

### 10.4 Semantic merge-back step

After splitting, `merge_semantic_subclusters()` may merge subclusters back together.

This function:

- computes a CLIP centroid for each candidate subcluster
- computes centroid-to-centroid cosine similarity
- merges groups whose semantic similarity is at least `semantic_merge_threshold`

Purpose:

- undo over-splitting when two subclusters are actually the same room view family

This step is powerful, but it is also one of the most sensitive parts of the implementation.

Example of current behavior:

- if `semantic_merge_threshold` is too low, visually distinct corner groups may merge back into one large cluster
- if it is too high, valid same-room subgroups may remain unnecessarily split

## 11. Strict Refinement Path

The strict path is used when `--strict-same-corner-items` is enabled.

This path is designed for tighter same-corner and same-item consistency.

The implementation:

1. optionally uses the 4-image pair heuristic first
2. otherwise computes a strict pairwise similarity matrix
3. applies hard gates on viewpoint, item, and semantic similarity
4. clusters with complete-link agglomerative clustering
5. converts undersized groups to noise

### 11.1 Hard gating

Two images are only allowed to remain connected if all three conditions pass:

- viewpoint similarity >= `view_similarity_threshold`
- item similarity >= `item_similarity_threshold`
- semantic similarity >= `semantic_similarity_floor`

### 11.2 Strict similarity formula

If the gates pass, the strict similarity score is:

```text
0.50 * viewpoint similarity
+ 0.30 * item similarity
+ 0.20 * semantic similarity
```

If the gates do not pass, the pair score becomes `0.0`.

### 11.3 Strict clustering

The strict similarity matrix is converted to distance and clustered using:

- `AgglomerativeClustering`
- `metric="precomputed"`
- `linkage="complete"`
- `distance_threshold = 1 - strict_cluster_threshold`

Purpose:

- keep only strongly consistent same-corner groups
- reject mixed groups more aggressively

Tradeoff:

- strict mode usually creates cleaner groups
- strict mode also tends to produce more noise

## 12. Why Both HDBSCAN and Agglomerative Were Used

This was implemented as a hybrid approach because the two algorithms solve different problems.

### HDBSCAN role

Used for:

- broad semantic grouping
- local hybrid grouping without a fixed cluster count
- natural noise detection

Why it was chosen:

- the number of clusters is not known in advance
- cluster sizes vary across datasets
- some images should remain noise rather than being forced into a bad cluster

### Agglomerative role

Used for:

- viewpoint-based refinement of broad local groups
- strict mode clustering with complete-link logic

Why it was chosen:

- it can operate on a custom precomputed distance matrix
- it allows direct threshold-based splitting
- complete-link is useful when all members of a group must be mutually compatible

This is not a conflict. It is a staged design where HDBSCAN handles broad discovery and agglomerative handles stricter refinement.

## 13. Thresholds and Defaults

### 13.1 Main clustering thresholds

- `min-cluster-size = 2`
- `min-samples = 1`
- `cluster-epsilon = 0.0`

### 13.2 Feature weights

- `semantic-weight = 0.45`
- `layout-weight = 0.35`
- `edge-weight = 0.15`
- `color-weight = 0.05`

### 13.3 Default-path refinement thresholds

- `view-similarity-threshold = 0.34`
- `semantic-merge-threshold = 0.98`
- `view-max-cluster-size = None`

### 13.4 Strict-path thresholds

- `item-similarity-threshold = 0.84`
- `strict-cluster-threshold = 0.56`
- `semantic-similarity-floor = 0.90`

## 14. Internal Heuristics

The project also contains several internal thresholds and heuristics that are not exposed as CLI flags.

Examples:

- ORB match distance filter `< 42`
- opening profile brightness threshold `> 210`
- blue opening heuristic using channel offsets
- 4-image pair-splitting quality checks inside `maybe_split_quad_cluster()`

These heuristics matter because some cluster behavior comes from them, not only from the user-facing thresholds.

## 15. Data Alignment Strategy

One implementation detail that matters in practice:

- an image may fail to load in one branch
- a feature row may be missing for that image

The code aligns valid image paths across branches before clustering so that:

- every row in the semantic embedding matrix
- every row in the visual feature matrix
- every row in the optional item feature matrix

still refers to the same image.

This prevents index corruption when unreadable images are skipped.

## 16. Output Design

After clustering, `reset_output_dir()` recreates the output directory and the project writes:

- `cluster_0`, `cluster_1`, and similar folders
- `noise/`
- `clusters.json`
- `match_scores.json`

### 16.1 `clusters.json`

Contains:

- final cluster ids
- filenames per cluster
- final noise list

Purpose:

- simple machine-readable result file

### 16.2 `match_scores.json`

Contains:

- per-image ranked matches
- same-cluster flag
- semantic score
- hybrid score
- viewpoint score
- optional item score in strict mode
- optional strict-threshold pass/fail flag

Purpose:

- explain why the model grouped or did not group two images
- support threshold tuning and debugging

## 17. Current Implementation Strengths

- clear separation between semantic grouping and viewpoint refinement
- combines learned and handcrafted signals
- tunable through CLI without code edits
- supports both balanced and strict behaviors
- provides diagnostics through `match_scores.json`

## 18. Current Implementation Weaknesses

- threshold behavior is dataset-sensitive
- semantic merge-back can over-merge visually distinct subclusters
- strict mode can over-produce noise
- viewpoint refinement recomputes image-derived features multiple times
- there is no automated evaluation harness for tuning
- there is no automatic threshold search or per-dataset adaptation

## 19. Example of an Important Current Tuning Issue

One observed failure mode in this project is:

- viewpoint refinement correctly splits a broad room group
- semantic merge-back then joins those subgroups again because CLIP centroid similarity is very high

This means:

- the split logic may be correct
- the merge threshold may still be too permissive

This is why `semantic-merge-threshold` is one of the most important tuning controls in the current implementation.

## 20. Run Commands

Basic run:

```powershell
python cluster_images.py --input .\input --output .\output
```

Run a specific input into a new output folder:

```powershell
python cluster_images.py --input .\input --output .\new_output_folder
```

Use a stricter semantic merge threshold:

```powershell
python cluster_images.py --input .\input --output .\new_output_folder --semantic-merge-threshold 0.98
```

Strict item-aware mode:

```powershell
python cluster_images.py --input .\input --output .\output_strict_items --strict-same-corner-items
```

## 21. Suggested Next Implementation Steps

If this project is extended, the next logical implementation steps are:

1. add an evaluation harness with known expected cluster labels
2. log cluster-level reasons for merges and splits
3. make merge-back use both semantic and viewpoint compatibility
4. cache repeated viewpoint features to reduce recomputation
5. add contact sheets or HTML inspection reports
6. add a threshold-sweep script for faster tuning

## 22. Summary

The project was implemented as a practical hybrid clustering pipeline:

- CLIP for semantics
- handcrafted features for viewpoint structure
- HDBSCAN for broad discovery and noise handling
- agglomerative clustering for tighter refinement
- JSON outputs for explainability and tuning

That design is appropriate for interior-photo or real-estate style datasets where:

- room meaning matters
- camera viewpoint matters
- cluster count is unknown
- some images should remain as noise
