# Sunray Detection Plan

## Objective

Build a detector that can identify **visible sunrays / light shafts** in arbitrary images, not just bright sun patches on indoor floors.

The target output should be:

- `ray_mask`: binary or soft mask of visible rays
- `ray_alpha`: confidence or opacity map for compositing
- `light_source`: estimated sun / window / bright-source origin
- `ray_direction_field`: dominant ray directions
- `confidence`: image-level confidence score

## Current Repo Situation

The existing [`sunray_detector.py`](./sunray_detector.py) is a useful heuristic baseline, but it is specialized:

- it assumes an opening such as a window or sky region is detectable
- it builds a downward "sunlight corridor"
- it searches mostly for **bright floor patches** under that corridor
- it is stronger for indoor scenes than for outdoor crepuscular rays

That means it does **not** yet solve the more general problem of detecting sunrays in "any image".

## What Makes General Sunray Detection Hard

Sunrays are not a single visual pattern. They can appear as:

- indoor shafts from windows or doors
- outdoor crepuscular rays through clouds
- partial rays hidden by haze, fog, smoke, or dust
- overexposed backlit streaks with weak edges
- soft volumetric beams with no sharp boundaries

Common failure cases:

- bright floor reflections mistaken as rays
- white walls or roads mistaken as shafts
- cloud streaks confused with radial rays
- lens flare confused with atmospheric light beams
- no visible solar disk, so the source must be inferred indirectly

## Detection Principle

A robust system should combine four signals:

1. **Source evidence**
   There must be a plausible bright source or opening.
2. **Directional structure**
   Rays usually align along a shared direction or fan out radially from a source.
3. **Photometric consistency**
   Rays are often brighter, lower contrast, and slightly veiled compared with nearby background.
4. **Occlusion consistency**
   Rays often alternate with darker gaps caused by trees, clouds, window frames, or other blockers.

## Baseline Pipeline

This is the first pipeline I would implement in this repo.

### 1. Input Normalization

- Resize image to a working resolution, for example longest side `1024`
- Keep original image for final upsampling
- Convert to:
  - RGB / BGR
  - LAB
  - HSV
  - grayscale
- Compute:
  - gradient magnitude
  - structure tensor or oriented gradients
  - local contrast map
  - haze / veiling-light proxy from low-frequency luminance

### 2. Scene Typing

Classify the image into coarse scene types:

- `indoor_opening`
- `outdoor_sky`
- `ambiguous`

This does not need a heavy classifier at first. A rule-based version can use:

- sky-mask presence
- large bright top-region area
- window / opening detections
- horizon-like segmentation

Why this matters:

- indoor rays are often bounded by openings and hit floors or walls
- outdoor rays are often radial and pass through cloud gaps

### 3. Light Source Estimation

Estimate one or more candidate source points.

Possible strategies:

- bright-region detection in upper image area
- saturated highlight clustering
- sky-window centroid estimation
- vanishing-point-like back-projection from oriented bright streaks
- radial voting from gradient normals

Output:

- `N` source hypotheses with scores
- source type: `sun_visible`, `sun_hidden`, `window`, `unknown_bright_source`

### 4. Ray Candidate Generation

Generate ray proposals from each source hypothesis using multiple detectors.

#### A. Radial Orientation Energy

- convert image to polar or log-polar coordinates around the source
- in that coordinate system, rays become near-vertical elongated structures
- apply multi-scale ridge / line filters
- map results back to image space

#### B. Directional Filter Bank

- use steerable filters or Gabor-like filters at many orientations
- keep elongated bright responses aligned with source-to-pixel direction

#### C. Line / Wedge Proposals

- use Hough-like line voting or wedge-shaped region proposals
- allow a ray to widen as distance from source increases

Candidate output:

- soft proposal maps
- a list of candidate wedges or ray regions

### 5. Ray Validation

For each candidate region, compute features such as:

- mean brightness increase over local neighborhood
- reduction in local texture contrast
- directional coherence
- radial alignment with source
- widening consistency with distance
- alternation with shadow gaps
- intersection with sky or opening regions
- translucency score

Reject candidates that look like:

- plain specular highlights
- floor reflections with no radial geometry
- clouds with no source consistency
- lens flare artifacts with circular symmetry only

### 6. Mask Fusion

Fuse all candidate maps into one soft mask.

Recommended fusion order:

1. source-conditioned candidate maps
2. scene prior
3. photometric confidence
4. morphological cleanup
5. edge-aware smoothing

The output at this stage should be a **soft alpha mask**, not only a hard binary mask.

### 7. Refinement

Refine with:

- guided filter or bilateral refinement
- connected-component filtering
- confidence thresholding
- optional CRF or graph-cut refinement if boundaries matter

### 8. Final Outputs

Return:

- `mask_soft.png`
- `mask_binary.png`
- `overlay.jpg`
- `debug_source_points.jpg`
- `debug_orientation_map.jpg`
- `debug_radial_votes.jpg`
- JSON metadata with source position, confidence, and dominant angles

## Suggested New Pipeline On Top Of The Baseline

The most practical upgrade is a **pretrained-first hybrid pipeline**, not a fully custom model from day one.

## Recommended Pretrained Stack

### Primary Model

- `SegFormer-B0` or `SegFormer-B2` fine-tuned for `sunray` vs `background`

Reason:

- lightweight and realistic to fine-tune in this repo
- good semantic segmentation backbone
- easier to integrate than heavier universal segmentation models

### Auxiliary Geometry Prior

- `Depth Anything V2` for relative depth and atmospheric-depth cues

Reason:

- rays often become more visible through haze and distance
- depth helps separate volumetric shafts from flat bright surfaces

### Optional Mask Refinement

- `SAM` or `SAM 2`

Reason:

- useful for refining boundaries after detection
- useful for annotation tooling and manual correction
- not a replacement for the actual ray detector

### Optional Future Backbone

- `DINOv2` features for a later custom head if SegFormer hits a ceiling

### Not My First Choice Here

- `Mask2Former`

Reason:

- stronger but heavier
- more operationally complex
- not the fastest path for this repo

### Important Note

Pretrained weights reduce the amount of data you need, but they do **not** eliminate the need for task-specific labels. There is no mainstream off-the-shelf model that reliably outputs a `sunray mask` directly.

## Pretrained-First Hybrid Pipeline

This is the pipeline I would build next.

### Stage A. Heuristic Prior Generation

Use the existing geometric logic to produce priors:

- opening / sky candidates
- light-source hypotheses
- corridor or radial ray prior
- bright-region map
- local haze / veiling-light proxy

This stage stays useful even after adding a pretrained model.

### Stage B. Depth Prior Extraction

Run `Depth Anything V2` to estimate a relative depth map.

Use depth-derived features such as:

- depth gradient
- far-region confidence
- atmospheric softness prior
- depth-consistent ray expansion score

Goal:

- suppress false positives from flat highlights
- strengthen true volumetric shafts that extend through space

### Stage C. Primary Ray Segmentation

Fine-tune `SegFormer` on ray masks.

Recommended target:

- start with binary segmentation: `sunray` vs `background`
- predict a probability map
- convert to soft alpha in post-processing

Recommended training input for the first version:

- RGB image only

Why:

- simplest way to reuse pretrained weights cleanly
- avoids custom architecture surgery in the first iteration

### Stage D. Prior Fusion

Fuse the SegFormer probability map with the handcrafted priors:

- source consistency prior
- radial alignment prior
- sky / opening prior
- depth prior
- floor prior for indoor scenes only

This fusion can be rule-based at first:

- weighted averaging
- confidence gating
- connected-component rescoring

### Stage E. SAM-Based Refinement

Use `SAM` only after candidate detection:

- prompt with bounding boxes around ray regions
- prompt with positive points inside high-confidence regions
- refine coarse segmentation boundaries

This is best treated as:

- a refinement step
- an annotation-assist step
- a pseudo-label cleanup tool

### Stage F. Final Outputs

Return:

- `ray_prob.png`
- `ray_mask.png`
- `ray_alpha.png`
- `overlay.jpg`
- `debug_depth.png`
- `debug_source_points.jpg`
- `debug_prior_fusion.jpg`

## Why This Pipeline Is Better

Compared with the current implementation, this new pipeline:

- uses pretrained visual representations instead of relying only on handcrafted thresholds
- keeps the strong geometric priors already present in the repo
- handles both indoor and outdoor scenes better
- still supports hidden-light-source inference
- is easier to train than a fully custom model from scratch
- can grow into a stronger multi-input model later

## Practical Implementation Plan For This Repo

I would implement this in phases instead of replacing `sunray_detector.py` immediately.

### Phase 1. Build a Pretrained-First Baseline

Create:

- `light_source_estimator.py`
- `ray_feature_extractor.py`
- `depth_prior.py`
- `sunray_pipeline.py`

Tasks:

- separate source estimation from mask generation
- add outdoor mode
- add `Depth Anything V2` inference support
- output debug priors and soft masks

Goal:

- keep a hybrid baseline that is easy to debug before any fine-tuning

### Phase 2. Bootstrap Labels

Create:

- `tools/bootstrap_sunray_labels.py`
- `tools/review_sunray_labels.py`

Tasks:

- generate pseudo-labels from the current heuristic detector
- improve them with depth prior and manual correction
- optionally use SAM to refine rough masks during annotation

Goal:

- create a small but high-quality training set quickly

### Phase 3. Fine-Tune SegFormer

Create:

- `models/segformer_sunray.py`
- `train_segformer_sunray.py`
- `infer_segformer_sunray.py`

Tasks:

- fine-tune `SegFormer-B0` or `SegFormer-B2` for binary ray segmentation
- train on real labels plus curated synthetic data
- export probability maps for downstream fusion

Goal:

- replace threshold-heavy detection with a pretrained segmentation model

### Phase 4. Add Prior Fusion and SAM Refinement

Tasks:

- fuse SegFormer output with source and depth priors
- optionally refine strong candidates with SAM
- preserve current fallback behavior
- add flags such as:
  - `mode=legacy_floor_patch`
  - `mode=pretrained_baseline`
  - `mode=segformer_fused`
  - `mode=segformer_fused_sam`

Goal:

- safe migration without breaking current workflows

### Phase 5. Optional Upgrade Path

Only after the above is working:

- modify the segmentation model to consume extra prior channels
- replace rule-based fusion with a learned fusion head
- test DINOv2 or a custom multi-input architecture if needed

## Training Data Strategy

If you want strong performance, data quality will matter more than model size.

### Data Sources

- real indoor photos with window shafts
- outdoor sky photos with crepuscular rays
- fog / haze / smoke scenes
- negative samples with bright reflections, clouds, and lens flare

### Label Bootstrapping

Recommended approach:

1. run the current heuristic detector on many images
2. use those masks as weak pseudo-labels
3. refine the best samples manually
4. use SAM-assisted cleanup where masks are close but messy
5. train SegFormer on the cleaned subset first

This is the fastest way to get a pretrained model working without building a dataset from scratch.

### Synthetic Augmentation

Generate synthetic rays by:

- choosing a source point
- creating wedge-shaped alpha beams
- modulating opacity with Perlin noise or smooth turbulence
- inserting occlusion gaps
- compositing with depth-aware or segmentation-aware attenuation

Synthetic data is especially useful because hand-labeling soft ray masks is expensive.

### Hard-Negative Mining

Actively collect failures such as:

- glossy floor reflections
- white roads and sidewalks
- bright cloud streaks
- window glare
- lens flare

This matters because false positives are likely to dominate before the fine-tuned model matures.

## Evaluation Plan

Measure:

- IoU for binary ray mask
- Dice / F1 for ray region overlap
- MAE on soft alpha map if available
- source-point localization error
- false positive rate on reflection-heavy scenes

Create separate validation buckets:

- indoor clear shafts
- indoor weak shafts
- outdoor crepuscular rays
- haze / smoke scenes
- hard negatives

## Risks

- source estimation failure will break downstream geometry
- soft rays often have ambiguous boundaries
- reflections and overexposed highlights are the main false positives
- small datasets will make end-to-end learning unstable

Because of that, a hybrid system is the safest approach.

## Recommendation

The right path is:

1. keep the current detector as a specialized indoor fallback
2. use a pretrained segmentation model instead of training a custom detector from scratch
3. use `SegFormer` as the primary detector
4. use `Depth Anything V2` as an auxiliary prior
5. use `SAM` only for refinement and annotation support
6. evaluate on indoor, outdoor, haze, and hard-negative splits separately

In short:

- current code = "detect sunlit patch below opening"
- new baseline = "pretrained SegFormer + geometric priors"
- best practical pipeline = "SegFormer + Depth prior + heuristic fusion + optional SAM refinement"
