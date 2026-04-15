# Improvement Plan

## Goal

Improve clustering quality across different datasets while keeping the current pipeline understandable and tunable.

## Current Weak Spots

### 1. Thresholds are dataset-sensitive

Symptoms:

- thresholds that work for real-estate interiors can be too strict for office images
- new datasets may create too much noise or over-splitting

Cause:

- the pipeline uses fixed thresholds for viewpoint, item, semantic, and clustering stages

Recommended solution:

- add named presets such as `strict`, `balanced`, and `loose`
- optionally add domain presets such as `real_estate`, `office`, and `generic`
- store thresholds in one config object instead of only CLI flags

Expected impact:

- easier reuse across datasets
- less manual trial and error

## 2. Hard strict gates can reject valid same-room detail shots

Symptoms:

- a detail shot can be semantically close to a room cluster but still become noise
- one low pairwise score can block clustering because complete-link is strict

Cause:

- strict mode requires all three gates to pass
- complete-link agglomerative clustering is sensitive to weakest-link pairs

Recommended solution:

- add a post-pass to reassign noise images to the nearest cluster when:
  - semantic similarity is high enough
  - item similarity is high enough
  - average cluster compatibility is strong enough
- consider average-link or graph-based clustering instead of only complete-link

Expected impact:

- fewer false-noise cases
- better handling of detail shots and partial views

## 3. Fixed prompt list is biased toward the original domain

Symptoms:

- strict item signatures work better for real-estate interiors than for office scenes
- some datasets may be described poorly by the current prompt list

Cause:

- `STRICT_ITEM_PROMPTS` is a fixed list
- prompt coverage is narrow

Recommended solution:

- make prompt sets configurable by domain
- add an office prompt set and a generic indoor scene prompt set
- optionally load prompts from a `.json` or `.yaml` file

Expected impact:

- better item similarity for non-real-estate data
- cleaner strict-mode clustering across domains

## 4. ORB is helpful but brittle for low-texture or exposure-shifted images

Symptoms:

- ORB contributes little when walls are plain, lighting is poor, or blur is present
- local match quality varies a lot across datasets

Cause:

- ORB is a hand-crafted local matcher and depends on detectable keypoints

Recommended solution:

- keep ORB as a supporting signal, not the dominant one
- optionally add another local/structural signal such as SSIM, DINO features, or learned local descriptors
- make the ORB contribution weight configurable if needed

Expected impact:

- more stable viewpoint matching
- reduced dependence on local texture

## 5. Repeated image loading and repeated feature work increase runtime

Symptoms:

- the same images are reopened in multiple stages
- viewpoint matrices recompute derived image features repeatedly

Cause:

- feature extraction is scattered across separate functions without reuse of precomputed data

Recommended solution:

- introduce an in-memory per-image feature cache
- precompute layout, edges, opening profiles, and ORB descriptors once
- reuse cached features across refinement and scoring

Expected impact:

- lower runtime
- simpler profiling and easier debugging

## 6. There is no formal evaluation harness

Symptoms:

- threshold tuning is manual and visual
- it is hard to compare runs objectively

Cause:

- no ground-truth dataset, metrics, or benchmark script

Recommended solution:

- create a small labeled benchmark set
- track metrics such as cluster purity, pairwise precision, pairwise recall, and noise rate
- add a comparison script for multiple threshold profiles

Expected impact:

- more reliable tuning
- faster iteration on algorithm changes

## 7. Output is good for debugging but not yet optimized for human review

Symptoms:

- JSON is informative but slow to scan for large runs
- users need to open images manually to inspect cluster quality

Cause:

- current output focuses on raw cluster folders and JSON

Recommended solution:

- generate contact sheets per cluster
- optionally generate an HTML summary page
- include cluster-level summaries such as representative image and top near-miss candidates

Expected impact:

- faster manual QA
- easier threshold comparison

## 8. Output reset is destructive

Symptoms:

- rerunning a job deletes the existing output folder

Cause:

- `reset_output_dir()` removes the target directory entirely before writing

Recommended solution:

- add an optional `--overwrite` flag
- if overwrite is disabled, create timestamped output folders automatically

Expected impact:

- safer experiment workflow
- less accidental loss of previous runs

## Recommended Delivery Order

### Phase 1: Small, high-value changes

1. add presets for `strict`, `balanced`, and `loose`
2. add domain-specific prompt sets
3. add timestamped output folders or an explicit overwrite flag
4. generate contact sheets for cluster review

## Phase 2: Quality improvements

1. add noise reassignment based on nearest-cluster compatibility
2. add cached per-image feature computation
3. add evaluation scripts and benchmark inputs

## Phase 3: Larger model and algorithm upgrades

1. test average-link or graph-based strict clustering
2. add better learned local features
3. add room-type classification or detector-assisted scene understanding

## Concrete Solution Ideas

### A. Preset system

Example:

```text
strict:
  view_similarity_threshold = 0.34
  semantic_similarity_floor = 0.90
  item_similarity_threshold = 0.84

balanced:
  view_similarity_threshold = 0.28
  semantic_similarity_floor = 0.86
  item_similarity_threshold = 0.82

loose:
  view_similarity_threshold = 0.25
  semantic_similarity_floor = 0.84
  item_similarity_threshold = 0.80
```

### B. Noise reassignment rule

For each noise image:

1. compute its compatibility with every non-noise cluster
2. use average of top `k` member similarities, not only the single best pair
3. reassign only if:
   - semantic similarity is above a guardrail
   - item similarity is above a guardrail in strict mode
   - average viewpoint compatibility is high enough

### C. Evaluation script

Produce a report such as:

- total images
- clusters formed
- noise count
- pairwise precision
- pairwise recall
- per-cluster representative image

## Recommendation

If the goal is to improve quality quickly without a major rewrite, the best next step is:

1. add presets
2. add noise reassignment
3. add contact sheets

That combination gives better control, fewer false-noise results, and easier validation with relatively low engineering cost.
