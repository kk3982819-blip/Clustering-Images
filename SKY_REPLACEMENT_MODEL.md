# Sky Replacement Architecture Plan

## Purpose

This document explains how sky replacement is implemented in the current application.

The system is not using a generative image model for the sky. It replaces the sky by:

1. Detecting the visible sky/window region.
2. Loading a matching local sky image asset.
3. Blending that asset into the detected mask.
4. Applying weather-aware room and ground lighting.

## High-Level Architecture

```text
templates/cluster.html
        |
        | selected image + ai_features=sky_replacement + weather
        v
app.py /enhance-single
        |
        | calls generate_full_scene_variant(...)
        v
full_scene_generator.py
        |
        | calls generate_sky_mask(...)
        v
mask_generator.py
        |
        | YOLO-World + SAM + OpenCV mask cleanup
        v
sky mask PNG
        |
        v
OpenCV sky compositing + weather lighting
        |
        v
output_web/*_full_scene_sky.jpg
```

## Frontend Entry Point

File: `templates/cluster.html`

The UI exposes:

- `Sky Replace` checkbox.
- Weather dropdown with values like `sunny`, `cloudy`, `foggy`, `snowy`, `sunset`, `night`, etc.
- `Enhance Selected Image Only` button.

The frontend sends the selected weather value as form data:

```text
weather=<selected_weather>
ai_features=sky_replacement
```

## Backend Entry Point

File: `app.py`

Route:

```text
POST /enhance-single
```

Relevant behavior:

1. Finds the selected image inside `uploads_temp/clustered/<cluster_id>/`.
2. Checks whether `sky_replacement` is selected.
3. Builds an output path:

```text
output_web/<source_stem>_full_scene_sky.jpg
```

4. Calls:

```python
generate_full_scene_variant(
    input_path=source_path,
    output_path=generated_path,
    weather=weather,
)
```

5. Returns the generated image URL to the frontend.

## Mask Generation

File: `mask_generator.py`

Function:

```python
generate_sky_mask(image_path, output_mask_path)
```

Models used:

- `yolov8l-world.pt`
  - Loaded through `ultralytics.YOLO`.
  - Detects windows, sliding glass doors, and glass doors.

- `sam_b.pt`
  - Loaded through `ultralytics.SAM`.
  - Segments detected openings.

OpenCV is then used to:

- Extract likely sky pixels inside openings.
- Use edge/treeline logic to avoid replacing trees or lower outdoor objects.
- Remove small noise components.
- Solidify the final mask so sky panes are not patchy.

Output:

```text
output_web/<source_stem>_full_scene_sky_skymask.png
```

## Sky Asset Selection

File: `full_scene_generator.py`

Mapping:

```python
SKY_ASSET_MAP = {
    "sunny": "clear_day",
    "partly_cloudy": "partly_cloudy",
    "cloudy": "overcast",
    "foggy": "overcast",
    "rainy": "overcast",
    "drizzling": "overcast",
    "windy": "partly_cloudy",
    "dramatic": "sunset",
    "sunset": "golden_hour",
    "night": "golden_hour",
    "snowy": "overcast",
}
```

Assets are loaded from:

```text
static/sky_assets/
```

Current assets:

- `clear_day.jpg`
- `partly_cloudy.jpg`
- `overcast.jpg`
- `golden_hour.jpg`
- `sunset.jpg`

## Sky Compositing

File: `full_scene_generator.py`

Core steps:

1. Load input image with OpenCV.
2. Generate/load sky mask.
3. Load selected sky asset.
4. Resize/crop the sky asset to cover the image dimensions.
5. Convert the mask into a feathered alpha matte.
6. Blend the new sky into the original image:

```python
result = image * (1 - alpha) + sky_layer * alpha
```

This keeps replacement local to the detected sky/window area.

## Weather-Aware Lighting

File: `full_scene_generator.py`

Function:

```python
_apply_room_relighting(...)
```

The system applies two types of weather adjustment:

1. Overall weather tone:
   - exposure
   - contrast
   - saturation
   - color tint

2. Ground/room light ray effect:
   - enabled only for selected direct-light weather types.

Current ground-light enabled weather:

```python
GROUND_LIGHT_WEATHERS = {
    "sunny",
    "clear",
    "sunset",
    "golden_hour",
    "dusk",
    "night",
    "rainbow",
}
```

Weather such as `cloudy`, `foggy`, `rainy`, `drizzling`, `snowy`, and `dramatic` gets the overall tone but not the ground beam/spill effect.

## Orchestrated API Path

File: `api_orchestrator.py`

There is also a `sky_replacement` path inside `EnhancementOrchestrator`.

Important note:

`NanoBananaAPI` is a local mock/simulation wrapper. It is not calling a real external AI model.

The orchestrator path also uses:

- `mask_generator.generate_sky_mask(...)`
- local sky assets
- OpenCV compositing

## Data Flow Summary

```text
User selects weather
        |
Frontend sends form data
        |
FastAPI receives /enhance-single
        |
full_scene_generator.generate_full_scene_variant
        |
mask_generator.generate_sky_mask
        |
YOLO-World detects openings
        |
SAM segments openings
        |
OpenCV cleans and solidifies mask
        |
OpenCV blends sky asset
        |
Weather lighting/tone applied
        |
Output image written to output_web
```

## Current Limitations

- No generative sky model is used.
- Night currently maps to `golden_hour.jpg`; a dedicated night sky asset would improve realism.
- Mask generation can be slow because YOLO and SAM are heavy.
- Cluster HDR enhancement and single-image enhancement do not use exactly the same sky replacement route.
- Weather realism depends heavily on the quality of the local sky assets.

## Recommended Future Improvements

1. Add dedicated assets:
   - `night.jpg`
   - `snowy.jpg`
   - `rainy.jpg`
   - `dramatic_storm.jpg`

2. Cache masks by source image path to avoid recomputing YOLO/SAM output.

3. Cache loaded YOLO/SAM model instances instead of reloading them per request.

4. Make cluster and single-image sky replacement use the same `full_scene_generator.py` path.

5. Add a preview mode to test weather lighting without regenerating masks.

