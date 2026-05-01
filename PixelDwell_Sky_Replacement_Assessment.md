# PixelDwell Sky Replacement Assessment

Date: 2026-04-23

Source requirement document reviewed:
`PixelDwell AI Photo Editing Platform-technical approach document (1) (1).pdf`

Scope:
This assessment checks whether the current implementation satisfies the PDF requirements for sky replacement specifically, then lists the most important improvements.

## Overall Verdict

Partially satisfies the PDF requirement for sky replacement.

The current system already implements a usable sky replacement pipeline inside the web app, but it does not fully match the Phase 1 approach described in the PDF. The biggest gap is architectural: the PDF places sky replacement under API-based advanced enhancements, while the current implementation is mainly a local OpenCV plus asset-compositing solution with a mock API wrapper.

## What Already Matches The PDF Well

1. Sky replacement exists as a working feature in the application.
   The feature is exposed in the UI and flows through the FastAPI backend into generated output images.

2. The implementation is modular.
   The current design is separated across `app.py`, `mask_generator.py`, `full_scene_generator.py`, and `api_orchestrator.py`, which aligns with the PDF's modular architecture direction.

3. The solution uses a hybrid computer-vision pipeline.
   Window detection, sky-mask generation, cleanup, compositing, and relighting are handled in distinct stages using YOLO-World, SAM, and OpenCV.

4. The output is inspectable.
   The system writes both enhanced images and generated masks to disk, which supports debugging and quality review.

5. Weather-aware behavior is implemented.
   The current system changes sky asset selection and room relighting based on selected weather, which fits the PDF's real-estate enhancement goal.

## Where It Does Not Fully Satisfy The PDF Yet

1. It does not implement the PDF's intended Phase 1 API-based sky replacement.
   The PDF classifies sky replacement under advanced API-driven enhancements, but the current implementation mainly uses local sky assets and OpenCV compositing. The `NanoBananaAPI` path is a local mock/simulation, not a real external AI service.

2. It does not provide a true "window pull and sky view enhancement" stage as a separate deterministic correction step.
   The PDF explicitly lists this under algorithmic local processing. The current code focuses more on mask generation and replacement than on recovering detail in blown-out windows before or alongside replacement.

3. Sky diversity is limited.
   The current setup uses a small curated set of sky images. This works for demos, but it limits realism, repeatability across many listings, and match quality for time of day, cloud density, and horizon style.

4. Quality control is weak.
   There is no explicit pass/fail scoring for the sky mask, compositing edge quality, reflection realism, or lighting consistency before returning the final image.

5. Batch consistency is limited.
   When several images from the same property are processed, there is no explicit mechanism to keep sky style, horizon feel, color temperature, and relighting fully consistent across the whole set.

6. Failure handling is limited.
   If mask quality is weak or the scene is difficult, the system does not yet have a strong fallback strategy such as "skip replacement", "downgrade to window pull only", or "request manual review".

7. Some visual artifacts can still remain in difficult scenes.
   Based on the current code and representative tests, difficult cases still include bright threshold areas, glass reflections, window trim contamination, railings, trees close to the frame, and indoor light spill mismatch.

8. There are no formal evaluation metrics.
   The PDF emphasizes output quality and scalability, but the current sky replacement path does not yet define measurable quality metrics such as mask precision, boundary error, rejection rate, or human review score.

## Sky Replacement Status Against PDF Expectations

Architecture fit: Partial
Reason:
The feature is modular and integrated, but it does not yet match the PDF's intended real API-based enhancement layer for advanced sky replacement.

Functional fit: Good
Reason:
The feature works end-to-end and includes opening-aware masking, compositing, and weather-aware relighting.

Output-quality fit: Partial
Reason:
The current results can be good, but realism still depends heavily on mask quality, asset selection, and relighting correctness.

Scalability fit: Partial
Reason:
The pipeline structure is scalable, but the absence of formal quality checks, true API integration, and batch consistency control limits production readiness.

## Highest-Value Improvements

### Priority 1: Replace The Mock API With A Real Advanced Sky Service Or An In-House Model

Why:
This is the biggest mismatch with the PDF. The PDF expects sky replacement to sit in the advanced AI enhancement layer during Phase 1.

What to do:
- Connect `api_orchestrator.py` to a real external image enhancement API, or
- Route sky replacement to a dedicated in-house model later while keeping the same orchestration contract.

Expected benefit:
- Better realism
- Better match to the PDF
- Easier future swap from external service to internal model

### Priority 2: Add A Dedicated Window Pull Stage Before Replacement

Why:
The PDF separately calls out "window pull and sky view enhancement" as a local deterministic capability.

What to do:
- Recover window highlights before sky replacement
- Normalize blown-out glass areas
- Preserve more believable transition between frame, glass, and sky

Expected benefit:
- Cleaner blends
- More realistic window interiors
- Better handling of overexposed source images

### Priority 3: Expand The Sky Asset System Into A Matched Sky Catalog

Why:
Five static sky assets are not enough for broad real-estate production.

What to do:
- Build a larger asset library tagged by weather, time of day, horizon height, sun direction, cloud density, and color temperature
- Choose assets using scene metadata and image heuristics

Expected benefit:
- Better scene matching
- Less repeated-looking output
- More realistic batch delivery

### Priority 4: Add Quality Scoring And Automatic Fallback Rules

Why:
A production pipeline should know when not to trust its own mask or blend.

What to do:
- Score mask confidence
- Score edge contamination near frames and railings
- Score indoor/outdoor color mismatch
- Skip or downgrade the effect when confidence is low

Expected benefit:
- Fewer bad outputs
- Better reliability
- Safer batch automation

### Priority 5: Improve Glass And Reflection Modeling

Why:
Real estate windows are difficult because they contain reflections, mullions, railings, trees, and mixed exposures.

What to do:
- Add reflection-aware blending
- Add better frame and mullion protection
- Use per-pane modeling where possible
- Reduce false sky spread near trim and balcony structures

Expected benefit:
- Better realism around openings
- Fewer obvious replacement artifacts

### Priority 6: Enforce Cross-Image Consistency For A Property Set

Why:
A listing usually contains multiple images. Buyers notice inconsistency immediately.

What to do:
- Carry shared weather/time-of-day settings across the selected cluster
- Reuse compatible sky choice and color profile for related views
- Add property-level enhancement presets

Expected benefit:
- More professional listing output
- Better visual coherence across all photos

### Priority 7: Add Evaluation Metrics And A Review Dataset

Why:
Without measurement, quality improvements will stay subjective.

What to do:
- Create a validation set of indoor scenes with windows
- Track mask quality, blend quality, and reviewer scores
- Log failure categories such as reflections, trees, railings, and overexposed frames

Expected benefit:
- Better iteration speed
- More defensible quality claims
- Easier prioritization of future work

## Practical Conclusion

If the question is "does the current code satisfy the PDF for sky replacement?", the correct answer is:

It satisfies the feature at a functional prototype level, but not yet at the full architectural and production-quality level implied by the PDF.

In practical terms:
- Yes, the project has a real sky replacement implementation.
- Yes, the pipeline structure is compatible with the PDF's direction.
- No, it is not yet fully aligned with the PDF's Phase 1 advanced-AI expectation.
- No, it does not yet have the quality-control and production-hardening pieces needed for strong real-estate deployment.

## Recommended Final Positioning

The most accurate project statement would be:

"PixelDwell currently has a working modular sky replacement prototype based on detection, segmentation, local sky compositing, and weather-aware relighting. It partially satisfies the technical approach PDF, but still needs true AI-service integration, stronger window-pull preprocessing, larger sky matching assets or model-backed generation, quality scoring, and batch consistency controls to fully meet the intended production-grade requirement."
