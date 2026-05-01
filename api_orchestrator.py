import logging
import os
import time
import uuid
from pathlib import Path
from restoration.pipeline import RestorationPipeline
from restoration.config import CONFIG


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NanoBananaAPI:
    """
    Mock external enhancement service used for advanced effects that are not
    handled locally.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NANO_BANANA_API_KEY", "mock_key_123")
        self.endpoint = "https://api.nanobanana.com/v1/enhance"

    def submit_job(self, image_path, feature, params=None):
        job_id = str(uuid.uuid4())
        logging.info(
            "[NanoBananaAPI] Uploading %s for feature '%s' (Job: %s)",
            Path(image_path).name,
            feature,
            job_id,
        )
        time.sleep(0.5)
        return job_id

    def fetch_result(self, job_id, output_path, input_path, mask_path=None, feature=None, params=None):
        params = params or {}
        logging.info("[NanoBananaAPI] Polling status for Job %s...", job_id)

        import cv2
        import numpy as np

        img = cv2.imread(str(input_path))
        if img is None:
            logging.error("[NanoBananaAPI] Could not load input image.")
            return False

        if feature in ["object_removal", "declutter"] and mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                logging.info("[NanoBananaAPI] Executing Professional AI Inpainting for %s...", feature)

                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                structural = cv2.inpaint(img, mask, 15, cv2.INPAINT_NS)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                inner_mask = cv2.erode(mask, kernel, iterations=1)
                final_inpaint = cv2.inpaint(structural, inner_mask, 3, cv2.INPAINT_TELEA)

                noise = np.random.normal(0, 1, final_inpaint.shape).astype(np.uint8)
                final_result = cv2.addWeighted(final_inpaint, 0.98, noise, 0.02, 0)

                mask_2d = mask.squeeze()
                if mask_2d.ndim == 3:
                    mask_2d = mask_2d[:, :, 0]

                result = img.copy()
                result[mask_2d > 0] = final_result[mask_2d > 0]
                cv2.imwrite(str(output_path), result)
                return True

        if feature in ["day_to_dusk", "dusk_to_night"]:
            logging.info("[NanoBananaAPI] Simulating %s transformation...", feature)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
            hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.75, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            glow = cv2.GaussianBlur(result, (0, 0), 15)
            result = cv2.addWeighted(result, 0.85, glow, 0.15, 0)
            cv2.imwrite(str(output_path), result)
            return True

        if feature == "sky_replacement":
            try:
                from full_scene_generator import generate_regenerative_sky_variant

                generate_regenerative_sky_variant(
                    input_path=input_path,
                    output_path=output_path,
                    weather=params.get("weather", "sunny"),
                    mask_path=mask_path,
                )
                return True
            except Exception as exc:
                logging.error("[SkyReplace] Generative replacement failed; falling back to compositor: %s", exc)

        if feature == "sky_replacement" and mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                weather = params.get("weather", "sunny")
                img_h, img_w = img.shape[:2]

                # ── Load local curated sky asset ──────────────────────────
                sky_asset_map = {
                    "sunny": "clear_day", "clear": "clear_day",
                    "sunrise": "golden_hour",
                    "partly_cloudy": "partly_cloudy", "high_wisps": "partly_cloudy",
                    "cloudy": "overcast", "foggy": "overcast",
                    "rainy": "overcast", "drizzling": "overcast",
                    "windy": "partly_cloudy", "dramatic": "partly_cloudy",
                    "night": "golden_hour",
                    "sunset": "golden_hour", "rainbow": "partly_cloudy",
                    "snowy": "overcast",
                }
                asset_name = sky_asset_map.get(weather, "partly_cloudy")
                asset_path = Path("static/sky_assets") / f"{asset_name}.jpg"

                if not asset_path.exists():
                    logging.warning("[SkyReplace] Asset %s not found, using partly_cloudy", asset_path)
                    asset_path = Path("static/sky_assets/partly_cloudy.jpg")

                sky_src = cv2.imread(str(asset_path))
                if sky_src is None:
                    logging.error("[SkyReplace] Could not load sky asset %s", asset_path)
                    cv2.imwrite(str(output_path), img)
                    return True

                result = img.copy()

                # ── Clean up mask ─────────────────────────────────────────
                mask = (mask > 127).astype(np.uint8) * 255
                cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)

                # Remove tiny noise components
                component_count, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                min_sky_area = max(200, int(img_h * img_w * 0.0005))
                clean_mask = np.zeros_like(mask)
                for cc_label in range(1, component_count):
                    if stats[cc_label, cv2.CC_STAT_AREA] >= min_sky_area:
                        clean_mask[labels_cc == cc_label] = 255

                if np.count_nonzero(clean_mask) == 0:
                    cv2.imwrite(str(output_path), result)
                    logging.info("[SkyReplace] Empty sky mask; saved base image.")
                    return True

                # ── Analyze scene color temperature ───────────────────────
                # Sample indoor pixels (areas NOT in mask) for color matching
                indoor_mask = cv2.bitwise_not(clean_mask)
                indoor_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(float)
                indoor_pixels = indoor_lab[indoor_mask > 0]
                if len(indoor_pixels) > 100:
                    scene_l_mean = indoor_pixels[:, 0].mean()
                    scene_a_mean = indoor_pixels[:, 1].mean()
                    scene_b_mean = indoor_pixels[:, 2].mean()
                else:
                    scene_l_mean, scene_a_mean, scene_b_mean = 180.0, 128.0, 128.0

                # ── Process each connected sky region ─────────────────────
                # Each region gets the FULL sky asset resized independently to
                # its own bounding box. This guarantees the gradient always runs
                # top-to-bottom correctly regardless of region position or size.
                region_count, region_labels, region_stats, _ = cv2.connectedComponentsWithStats(
                    clean_mask, connectivity=8
                )

                for region_id in range(1, region_count):
                    region_area = region_stats[region_id, cv2.CC_STAT_AREA]
                    if region_area < min_sky_area:
                        continue

                    rx = max(0, region_stats[region_id, cv2.CC_STAT_LEFT])
                    ry = max(0, region_stats[region_id, cv2.CC_STAT_TOP])
                    rw = min(region_stats[region_id, cv2.CC_STAT_WIDTH], img_w - rx)
                    rh = min(region_stats[region_id, cv2.CC_STAT_HEIGHT], img_h - ry)
                    if rw < 5 or rh < 5:
                        continue

                    region_mask = (region_labels[ry:ry+rh, rx:rx+rw] == region_id).astype(np.uint8) * 255

                    # Erode to pull sky away from frame edges
                    erode_px = max(3, int(min(rw, rh) * 0.04)) | 1
                    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
                    region_core = cv2.erode(region_mask, erode_kernel, iterations=1)
                    if np.count_nonzero(region_core) < max(50, region_area // 8):
                        region_core = region_mask.copy()

                    # ── Resize full sky asset to this region's exact dimensions ──
                    # Using LANCZOS4 for high-quality downsampling.
                    sky_fit = cv2.resize(sky_src, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

                    # ── Phase 3: Color harmonization via LAB (DISABLED to preserve exact reference colors) ──────────────
                    # sky_lab = cv2.cvtColor(sky_fit, cv2.COLOR_BGR2LAB).astype(float)
                    # shift_strength = 0.40 if weather not in ["sunset", "dusk"] else 0.15
                    # sky_a_mean = sky_lab[:, :, 1].mean()
                    # sky_b_mean = sky_lab[:, :, 2].mean()
                    # sky_lab[:, :, 1] += (scene_a_mean - sky_a_mean) * shift_strength
                    # sky_lab[:, :, 2] += (scene_b_mean - sky_b_mean) * shift_strength
                    # 
                    # brightness_mult = 1.15 if weather not in ["sunset", "dusk"] else 1.05
                    # target_L = min(240.0, scene_l_mean * brightness_mult)
                    # sky_l_mean = sky_lab[:, :, 0].mean()
                    # if sky_l_mean > 0:
                    #     sky_lab[:, :, 0] *= (target_L / sky_l_mean)
                    # sky_lab = np.clip(sky_lab, 0, 255).astype(np.uint8)
                    # sky_fit = cv2.cvtColor(sky_lab, cv2.COLOR_LAB2BGR)

                    # ── Phase 4: Glass simulation ─────────────────────────
                    # 4a. Slight desaturation (DISABLED)
                    # sky_hsv = cv2.cvtColor(sky_fit, cv2.COLOR_BGR2HSV).astype(float)
                    # sky_hsv[:, :, 1] *= 0.92  # 8% desaturation
                    # sky_fit = cv2.cvtColor(np.clip(sky_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

                    # 4b. Subtle noise to break up perfect smoothness
                    rng = np.random.default_rng(region_id)
                    noise = rng.normal(0, 1.5, sky_fit.shape).astype(np.float32)
                    sky_fit = np.clip(sky_fit.astype(float) + noise, 0, 255).astype(np.uint8)

                    # 4c. Faint reflection overlay (indoor scene flipped, 4% opacity)
                    scene_roi = result[ry:ry+rh, rx:rx+rw]
                    reflection = cv2.flip(scene_roi, 1)  # horizontal flip
                    reflection_blur = cv2.GaussianBlur(reflection, (31, 31), 10)
                    sky_fit = cv2.addWeighted(sky_fit, 0.96, reflection_blur, 0.04, 0)

                    # 4d. Slight blur at edges (glass distortion)
                    sky_fit = cv2.GaussianBlur(sky_fit, (3, 3), 0.8)

                    # ── Phase 5: Edge-aware feathering ────────────────────
                    dist = cv2.distanceTransform(region_core, cv2.DIST_L2, 5)
                    # Use a TIGHT fixed transition (8px)
                    if dist.max() > 0:
                        alpha = np.clip(dist / 8.0, 0.0, 1.0)
                    else:
                        alpha = region_core.astype(float) / 255.0

                    # Guided filter for edge-aware feathering
                    try:
                        guide = cv2.cvtColor(scene_roi, cv2.COLOR_BGR2GRAY)
                        alpha_u8 = (alpha * 255).astype(np.uint8)
                        alpha = cv2.ximgproc.guidedFilter(
                            guide, alpha_u8, radius=max(3, int(min(rw, rh) * 0.03)),
                            eps=900
                        ).astype(float) / 255.0
                    except AttributeError:
                        # Fallback if ximgproc not available
                        feather_k = max(5, int(min(rw, rh) * 0.08)) | 1
                        alpha = cv2.GaussianBlur(
                            (alpha * 255).astype(np.uint8), (feather_k, feather_k), 0
                        ).astype(float) / 255.0

                    alpha[region_mask == 0] = 0.0
                    alpha_3ch = np.expand_dims(alpha, axis=2)

                    # ── Composite ─────────────────────────────────────────
                    roi = result[ry:ry+rh, rx:rx+rw].astype(float)
                    blend = roi * (1.0 - alpha_3ch) + sky_fit.astype(float) * alpha_3ch
                    result[ry:ry+rh, rx:rx+rw] = np.clip(blend, 0, 255).astype(np.uint8)

                    logging.info(
                        "[SkyReplace] Composited region %d/%d (%dx%d) with glass sim + guided feather",
                        region_id, region_count - 1, rw, rh,
                    )

                cv2.imwrite(str(output_path), result)
                logging.info("[SkyReplace] Realistic sky replacement saved to %s", output_path)
                return True

        if feature in ["virtual_staging", "sketch_overlay"]:
            logging.info("[NanoBananaAPI] Simulating %s generation...", feature)
            result = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            cv2.imwrite(str(output_path), result)
            return True

        if feature == "regenerate":
            logging.info("[NanoBananaAPI] Regenerating image with AI model...")
            try:
                from image_regeneration_model import get_regeneration_model
                model = get_regeneration_model()
                prompt = params.get("prompt", "realistic high quality photo")
                strength = params.get("strength", 0.5)
                regenerated = model.regenerate_image(input_path, mask_path, prompt, strength)
                regenerated.save(output_path)
                logging.info("[Regenerate] Image regenerated and saved to %s", output_path)
                return True
            except Exception as e:
                logging.error("[Regenerate] Failed to regenerate image: %s", e)
                # Fallback to copy
                import shutil
                shutil.copy2(input_path, output_path)
                return True

        import shutil

        shutil.copy2(input_path, output_path)
        return True


class EnhancementOrchestrator:
    """
    Route requested enhancements to local or external-style processing.
    """

    def __init__(self):
        self.ai_api = NanoBananaAPI()
        self.restoration_pipeline = RestorationPipeline(CONFIG)
        self.supported_features = {
            "white_balance": "LOCAL",
            "hdr": "LOCAL",
            "perspective_correction": "LOCAL",
            "sky_replacement": "API",
            "declutter": "API",
            "object_removal": "API",
            "virtual_staging": "API",
            "day_to_dusk": "API",
            "sketch_overlay": "API",
            "regenerate": "API",
            "restoration": "LOCAL",  # New feature
        }

    def process_image(self, image_path, requested_features, output_dir, params=None):
        params = params or {}
        current_image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("--- Starting Orchestrator for %s ---", current_image_path.name)

        for feature in requested_features:
            if feature not in self.supported_features:
                logging.warning("Feature '%s' is unknown. Skipping.", feature)
                continue

            execution_mode = self.supported_features[feature]

            if execution_mode == "LOCAL":
                logging.info("[%s] Routing to LOCAL execution.", feature)
                if feature == "restoration":
                    step_filename = f"{current_image_path.stem}_{feature}.jpg"
                    step_path = output_dir / step_filename
                    try:
                        from PIL import Image
                        image = Image.open(current_image_path)
                        mask = None
                        if "mask_path" in params:
                            mask_path = Path(params["mask_path"])
                            if mask_path.exists():
                                mask = Image.open(mask_path)
                        restored = self.restoration_pipeline.full_restore(image, mask=mask)
                        restored.save(step_path)
                        current_image_path = step_path
                        logging.info("[%s] Restoration completed.", feature)
                    except Exception as e:
                        logging.error("[%s] Restoration failed: %s", feature, e)
                continue

            if execution_mode == "API":
                logging.info("[%s] Routing to EXTERNAL API execution.", feature)
                step_filename = f"{current_image_path.stem}_{feature}.jpg"
                step_path = output_dir / step_filename

                mask_path = None
                if feature == "sky_replacement":
                    mask_filename = f"{current_image_path.stem}_skymask.png"
                    mask_path = output_dir / mask_filename
                    try:
                        from mask_generator import generate_sky_mask

                        logging.info("[%s] Generating strict sky boundary mask...", feature)
                        success = generate_sky_mask(current_image_path, mask_path)
                        if not success:
                            logging.warning("[%s] Failed to detect sky boundary. Fallback to API default.", feature)
                            mask_path = None
                    except Exception as exc:
                        logging.error("[%s] Mask generation error: %s", feature, exc)

                elif feature in ["object_removal", "declutter"]:
                    points = params.get(feature)
                    if points:
                        mask_filename = f"{current_image_path.stem}_objmask.png"
                        mask_path = output_dir / mask_filename
                        try:
                            from mask_generator import generate_object_mask

                            logging.info("[%s] Generating interactive object mask at %s...", feature, points)
                            success = generate_object_mask(current_image_path, mask_path, points)
                            if not success:
                                mask_path = None
                        except Exception as exc:
                            logging.error("[%s] Object mask generation error: %s", feature, exc)

                job_id = self.ai_api.submit_job(current_image_path, feature, params=params)
                success = self.ai_api.fetch_result(
                    job_id,
                    step_path,
                    current_image_path,
                    mask_path=mask_path,
                    feature=feature,
                    params=params,
                )

                if success:
                    current_image_path = step_path

        return current_image_path


if __name__ == "__main__":
    test_image = "test_hdr_output.jpg"
    Path(test_image).touch()

    orchestrator = EnhancementOrchestrator()
    final_output = orchestrator.process_image(
        test_image,
        requested_features=["white_balance", "sky_replacement", "day_to_dusk"],
        output_dir="output_final",
    )
    print(f"Workflow Complete. Final result at: {final_output}")
