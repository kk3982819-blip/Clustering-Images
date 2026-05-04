# PixelDwell Generation Options A, B, C

Date: 2026-04-23

Scope:
This document explains three realistic paths for making PixelDwell generate clearer full-scene outputs with better lighting and shadow fit:

- Option A: image-to-image generation
- Option B: hybrid relighting pipeline
- Option C: full 3D ray tracing pipeline

It is written against the current PixelDwell codebase, where sky replacement is handled mainly through `mask_generator.py`, `full_scene_generator.py`, `app.py`, and `api_orchestrator.py`.

## Current State

The current system is strongest at:

- sky/window masking
- local sky compositing
- weather-aware tone adjustment
- basic room relighting
- deterministic processing around an original photograph

The current system is weakest at:

- physically correct shadow casting
- realistic global illumination
- full-scene regeneration
- geometry-aware lighting
- premium virtual staging realism

That is why the question of "can ray tracing be used?" depends on which problem PixelDwell is actually solving.

## Executive Summary

### Option A: Image-to-Image Generation

Best when:
- you want a new photorealistic image quickly
- you want cleaner outputs than the current OpenCV compositing path
- you can accept plausible shadows instead of physically exact shadows

Main tradeoff:
- stronger visual polish, weaker physical correctness

### Option B: Hybrid Relighting

Best when:
- you want to preserve the original property photo
- you want better shadow fit than simple 2D compositing
- you want the most practical upgrade path from the current PixelDwell architecture

Main tradeoff:
- more engineering than Option A, but still not true ray tracing

### Option C: Full 3D Ray Tracing

Best when:
- you want physically correct shadows and light transport
- you want premium virtual staging, renovation previews, or full CGI-quality room generation

Main tradeoff:
- highest quality, highest complexity, highest cost

## Option A: Image-to-Image Generation

### What It Is

This option uses the input property photo as a conditioning image and sends it through a generative image-edit pipeline. The model regenerates the scene while trying to preserve layout, perspective, and major structure.

Typical inputs:

- original image
- window/sky mask
- optional depth map
- prompt such as:
  "real-estate interior photo, clean bright natural lighting, balanced exposure, realistic exterior sky, crisp details, premium listing quality"

Typical outputs:

- a refreshed full image
- a variation set if needed

### How It Would Fit PixelDwell

This would sit naturally behind the existing orchestration layer.

Likely flow:

```text
cluster.html
    |
    v
app.py
    |
    v
generation_orchestrator.py
    |
    +-- mask_generator.py
    +-- depth/control map builder
    +-- external image-edit API or local generative model
    |
    v
output_web/generated_full_scene_*.jpg
```

### Suggested New Modules

- `generation_orchestrator.py`
- `prompt_builder.py`
- `control_map_builder.py`
- `image_edit_adapter.py`
- `quality_scorer.py`

### Strengths

1. Fastest path to visually impressive output.
2. Can improve overall clarity, tone, and atmosphere in one step.
3. Handles difficult cases better than simple cut-and-paste compositing in some scenes.
4. Naturally aligns with the PDF's advanced AI enhancement direction.

### Weaknesses

1. Shadows are plausible, not physically guaranteed.
2. Models can drift from the original photo.
3. Window frames, balconies, reflections, and furniture edges can hallucinate.
4. Output consistency across a batch must be explicitly managed.

### When To Use It

- premium "enhance this room" mode
- marketing-grade hero images
- difficult images where deterministic sky compositing keeps failing

### Best Technical Stack

- external image editing API or hosted diffusion pipeline
- optional ControlNet-style conditioning with depth/edge maps
- prompt templates constrained for real-estate photography

### PixelDwell Recommendation For Option A

Use this as a premium or fallback pipeline, not as the first replacement for the current deterministic sky replacement path.

## Option B: Hybrid Relighting Pipeline

### What It Is

This keeps the original photo as the base image, but upgrades lighting realism through geometry-aware approximation.

Instead of fully regenerating the room, it estimates:

- room depth
- floor/wall/ceiling regions
- window geometry
- sun direction
- brightness spill
- cast shadow regions

Then it applies:

- window pull recovery
- improved indoor relighting
- synthesized but controlled floor and wall shadows
- sky replacement with stronger lighting match

### Why This Fits PixelDwell Best

It builds directly on what the current codebase already does:

- `mask_generator.py` already finds openings and masks
- `full_scene_generator.py` already composites sky and applies relighting
- the app already routes enhancement requests cleanly

This means Option B is the best engineering continuation of the current product.

### Suggested New Modules

- `depth_estimator.py`
- `room_region_estimator.py`
- `window_pull.py`
- `light_estimator.py`
- `shadow_synthesizer.py`
- `batch_consistency.py`
- `quality_scorer.py`

### Suggested Flow

```text
input image
    |
    +-- opening mask
    +-- floor/wall/ceiling segmentation
    +-- monocular depth estimation
    +-- sun direction estimation
    |
    v
window pull recovery
    |
    v
shadow and spill synthesis
    |
    v
sky compositing
    |
    v
consistency grading + fallback
    |
    v
final output
```

### Strengths

1. Preserves the original listing photo.
2. More controllable than full generative regeneration.
3. Better production fit for real-estate workflows.
4. Lower hallucination risk than Option A.
5. Much cheaper and simpler than full 3D ray tracing.

### Weaknesses

1. Still not physically exact.
2. Requires better region estimation and light modeling than the current project has.
3. Hard scenes with mirrored glass or complex furniture still remain difficult.

### When To Use It

- default production sky replacement
- day-to-dusk quality upgrade
- premium lighting correction while keeping the real room unchanged

### Best Technical Stack

- OpenCV for compositing and controlled image operations
- monocular depth model
- segmentation for floor/wall/ceiling/openings
- learned or heuristic sun-direction estimation
- synthetic shadow projection using geometry cues

### PixelDwell Recommendation For Option B

This should be the main next-generation architecture for PixelDwell's sky replacement and lighting pipeline.

## Option C: Full 3D Ray Tracing

### What It Is

This approach reconstructs the room as a 3D scene and renders it with physically based lighting.

Required components:

- room geometry reconstruction
- camera pose estimation
- floor, wall, ceiling, and opening dimensions
- material assignment
- sun and sky lighting setup
- ray-traced or path-traced rendering

Possible tools:

- Blender with Cycles
- Unreal Engine path tracer
- Omniverse or other physically based renderers

### What Ray Tracing Actually Solves

Ray tracing is the right tool when you need:

- physically consistent shadows
- realistic bounce light
- material-aware reflections
- staged furniture interacting correctly with light
- renovation previews or CGI-quality output

### Why It Is Not A Simple Drop-In Upgrade

The current PixelDwell system starts from a single 2D photograph. Ray tracing needs much more:

- scene geometry
- object depth
- material estimates
- calibrated light direction
- camera parameters

Without those, ray tracing cannot produce trustworthy shadows.

### Suggested New Modules

- `room_reconstruction.py`
- `camera_solver.py`
- `material_estimator.py`
- `blender_scene_builder.py`
- `renderer_service.py`
- `render_compositor.py`
- `asset_library_manager.py`

### Strengths

1. Best shadow realism.
2. Best physical lighting quality.
3. Best path for high-end virtual staging and renovation rendering.
4. Best long-term premium feature set.

### Weaknesses

1. Highest development complexity.
2. Slowest runtime.
3. Highest infrastructure cost.
4. Hardest to maintain.
5. Requires 3D scene reconstruction accuracy that PixelDwell does not currently have.

### When To Use It

- premium virtual staging
- renovation previews
- new-furniture insertion with correct light and shadow interaction
- premium marketing renders

### PixelDwell Recommendation For Option C

Use this only as a separate premium rendering path, not as the default sky replacement path.

## Comparison Matrix

| Dimension | Option A: Image-to-Image | Option B: Hybrid Relighting | Option C: 3D Ray Tracing |
| --- | --- | --- | --- |
| Visual polish | High | Medium to high | Very high |
| Shadow realism | Medium | Medium to high | Very high |
| Physical correctness | Low to medium | Medium | Very high |
| Hallucination risk | High | Low to medium | Low |
| Keeps original photo intact | Partial | Strong | Variable |
| Engineering complexity | Medium | Medium to high | Very high |
| Runtime cost | Medium to high | Medium | High |
| Best fit for current PixelDwell code | Medium | Very high | Low |
| Best for sky replacement | Good | Best | Overkill |
| Best for virtual staging | Medium | Good | Best |

## Recommended Strategy For PixelDwell

### Recommended order

1. Build Option B first.
2. Add Option A as a premium enhancement or fallback path.
3. Reserve Option C for future premium staging and renovation features.

### Why

Option B is the best match for the existing system and business need.

It improves:

- sky replacement realism
- shadow fit
- lighting coherence
- production safety

without forcing PixelDwell into a full 3D pipeline too early.

Option A should then be added for:

- hero-image generation
- difficult scenes that defeat deterministic compositing
- premium marketing output

Option C should only be pursued when PixelDwell is ready to support:

- premium staged interiors
- renovation visualization
- full-scene CGI workflows

## Recommended Concrete Roadmap

### Phase 1: Upgrade Current Pipeline Into Option B

Add:

- window pull stage
- depth estimation
- floor/wall/ceiling segmentation
- sun direction estimation
- shadow synthesis
- quality scoring
- cluster-level consistency controls

Expected result:
- better shadow fit
- fewer beam and spill artifacts
- more realistic sky replacement

### Phase 2: Add Option A As Premium Generation

Add:

- prompt-controlled image editing pipeline
- control maps from masks, depth, and edges
- output ranking and quality filtering

Expected result:
- premium listing visuals
- high-end image enhancement mode

### Phase 3: Build Option C Only For Premium 3D Features

Add:

- room reconstruction
- 3D camera matching
- material estimation
- physically based rendering service

Expected result:
- premium staging and renovation rendering

## Final Recommendation

If the goal is:

"make the whole image look very clear and make the shadow fit properly"

then the best PixelDwell answer is:

- Do not jump straight to full ray tracing for standard sky replacement.
- Build Option B as the primary production architecture.
- Add Option A for premium generative upgrades.
- Keep Option C for a later premium 3D rendering product line.

In one line:

Option B is the best next architecture, Option A is the best premium shortcut, and Option C is the best long-term premium rendering path.
