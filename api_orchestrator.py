import os
import time
import json
import uuid
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NanoBananaAPI:
    """
    Mock implementation of the Google Nano Banana API (or similar external Generative AI service).
    This handles features that are too complex/heavy for local OpenCV execution:
    - Sky Replacement
    - Object Removal / Decluttering
    - Virtual Staging
    - Day-to-Dusk Conversion
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NANO_BANANA_API_KEY", "mock_key_123")
        self.endpoint = "https://api.nanobanana.com/v1/enhance"

    def submit_job(self, image_path, feature, params=None):
        """
        Submits an image for AI processing.
        """
        job_id = str(uuid.uuid4())
        logging.info(f"[NanoBananaAPI] Uploading {Path(image_path).name} for feature '{feature}' (Job: {job_id})")
        # Mock network delay
        time.sleep(1)
        return job_id

    def fetch_result(self, job_id, output_path, input_path, mask_path=None, feature=None):
        """
        Polls for result and downloads it to the output_path.
        Simulates cloud processing for various Advanced AI features.
        """
        logging.info(f"[NanoBananaAPI] Polling status for Job {job_id}...")
        time.sleep(1) 
        
        import cv2
        import numpy as np
        import urllib.request
        
        # Load input image
        img = cv2.imread(str(input_path))
        if img is None:
            logging.error("[NanoBananaAPI] Could not load input image.")
            return False

        # ── Feature-Specific Simulation ──────────────────────────────────────
        
        # A. Object Removal / Decluttering (High-Fidelity AI Inpainting)
        if feature in ["object_removal", "declutter"] and mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                logging.info(f"[NanoBananaAPI] Executing Professional AI Inpainting for {feature}...")
                
                # 1. Prepare Mask (Ensure full coverage)
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # ── PRO APPROACH: Multi-Pass Inpainting ──────────────────────
                # Pass 1: Structural Fill (Large Radius)
                structural = cv2.inpaint(img, mask, 15, cv2.INPAINT_NS)
                
                # Pass 2: Detail Refinement (Small Radius)
                # We use a slightly eroded version of the mask for the second pass
                # to focus on the transitions.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                inner_mask = cv2.erode(mask, kernel, iterations=1)
                final_inpaint = cv2.inpaint(structural, inner_mask, 3, cv2.INPAINT_TELEA)
                
                # ── Realism: Texture Injection ──────────────────────────────
                # Pure inpainting looks "flat". We add a tiny amount of noise
                # back to the inpainted area to match the original image grain.
                noise = np.random.normal(0, 1, final_inpaint.shape).astype(np.uint8)
                final_result = cv2.addWeighted(final_inpaint, 0.98, noise, 0.02, 0)
                
                # Apply only to masked regions (Ensure mask is 2D for boolean indexing)
                mask_2d = mask.squeeze()
                if mask_2d.ndim == 3: mask_2d = mask_2d[:, :, 0]
                
                result = img.copy()
                result[mask_2d > 0] = final_result[mask_2d > 0]
                
                cv2.imwrite(str(output_path), result)
                return True

        # B. Day-to-Dusk / Dusk-to-Night (Global Transform)
        if feature in ["day_to_dusk", "dusk_to_night"]:
            logging.info(f"[NanoBananaAPI] Simulating {feature} transformation...")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
            # Shift hue toward orange/gold, increase saturation, lower value
            hsv[:,:,0] = (hsv[:,:,0] + 5) % 180 
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255) 
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 0.75, 0, 255) 
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Add a slight "twilight" glow
            glow = cv2.GaussianBlur(result, (0, 0), 15)
            result = cv2.addWeighted(result, 0.85, glow, 0.15, 0)
            cv2.imwrite(str(output_path), result)
            return True

        # C. Sky Replacement (High-Fidelity Compositing)
        if feature == "sky_replacement" and mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Fetch a clean high-res professional sky (Pure Blue Sky with Fluffy Clouds)
                sky_url = "https://images.unsplash.com/photo-1597200381847-30ec200eeb9a?q=82&w=2000&auto=format&fit=crop"
                try:
                    req = urllib.request.urlopen(sky_url)
                    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    sky_src = cv2.imdecode(arr, -1)
                    if sky_src is not None:
                        result = img.copy()
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 100: continue
                            x, y, w, h = cv2.boundingRect(cnt)
                            
                            # ── Aspect-Ratio Preservation ────────────────────
                            target_aspect = w / max(h, 1)
                            src_h, src_w = sky_src.shape[:2]
                            src_aspect = src_w / max(src_h, 1)
                            
                            if target_aspect > src_aspect:
                                crop_w = src_w
                                crop_h = int(src_w / target_aspect)
                                cx, cy = 0, (src_h - crop_h) // 4 # Horizon bias: show more top sky
                            else:
                                crop_h = src_h
                                crop_w = int(src_h * target_aspect)
                                cx, cy = (src_w - crop_w) // 2, 0
                            
                            sky_fit = cv2.resize(sky_src[max(0,cy):min(src_h,cy+crop_h), max(0,cx):min(src_w,cx+crop_w)], (w, h), interpolation=cv2.INTER_LANCZOS4)
                            
                            # ── Realism: Light Matching ──────────────────────
                            # Slightly boost sky exposure to look like "outdoors"
                            sky_fit = cv2.convertScaleAbs(sky_fit, alpha=1.1, beta=10)
                            
                            # ── Professional Feathering (Tightened) ──────────
                            local_mask = np.zeros(mask.shape, dtype=np.uint8)
                            cv2.drawContours(local_mask, [cnt], -1, 255, -1)
                            region_mask = local_mask[y:y+h, x:x+w]
                            
                            # Erode slightly to ensure we don't bleed onto the wall
                            # This is the "solution" to remove the extra part bleeding
                            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            region_mask = cv2.erode(region_mask, kernel_erode, iterations=1)
                            
                            # Soften edges slightly for realism (but keep it tight)
                            feather_k = int(max(3, min(w, h) // 40) * 2) | 1
                            alpha = cv2.GaussianBlur(region_mask, (feather_k, feather_k), 0)
                            alpha = alpha.astype(float) / 255.0
                            alpha_3ch = np.expand_dims(alpha, axis=2)
                            
                            roi = result[y:y+h, x:x+w].astype(float)
                            blend = roi * (1 - alpha_3ch) + sky_fit.astype(float) * alpha_3ch
                            result[y:y+h, x:x+w] = np.clip(blend, 0, 255).astype(np.uint8)
                        
                        cv2.imwrite(str(output_path), result)
                        logging.info(f"[NanoBananaAPI] Realistic sky replacement saved to {output_path}")
                        return True
                except Exception as e:
                    logging.error(f"[NanoBananaAPI] Sky fitting failed: {e}")

        # D. Virtual Staging / Renovation Sketch
        if feature in ["virtual_staging", "sketch_overlay"]:
            logging.info(f"[NanoBananaAPI] Simulating {feature} generation...")
            # For demonstration, we boost details to simulate "staging" injection
            result = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            cv2.imwrite(str(output_path), result)
            return True

        # If no specific simulation matched or failed, copy input to output
        import shutil
        shutil.copy2(input_path, output_path)
        return True


class EnhancementOrchestrator:
    """
    The brain that decides whether to process locally or send to API.
    Based on Section 5 "Enhancement Decision Logic" from the design doc.
    """
    def __init__(self):
        self.ai_api = NanoBananaAPI()
        
        # Mapping features to execution environment
        self.supported_features = {
            "white_balance": "LOCAL",
            "hdr": "LOCAL",
            "perspective_correction": "LOCAL",
            "sky_replacement": "API",
            "declutter": "API",
            "object_removal": "API",
            "virtual_staging": "API",
            "day_to_dusk": "API",
            "sketch_overlay": "API"
        }

    def process_image(self, image_path, requested_features, output_dir, params=None):
        """
        Takes a base image and dynamically routes enhancement requests.
        params: Optional dict for feature-specific data (e.g. {'object_removal': [[x,y]]})
        """
        params = params or {}
        current_image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Starting Orchestrator for {current_image_path.name} ---")
        
        for feature in requested_features:
            if feature not in self.supported_features:
                logging.warning(f"Feature '{feature}' is unknown. Skipping.")
                continue
                
            execution_mode = self.supported_features[feature]
            
            if execution_mode == "LOCAL":
                logging.info(f"[{feature}] Routing to LOCAL execution.")
                # Local CV operations would be called here (e.g. from hdr_engine module)
                pass 
                
            elif execution_mode == "API":
                logging.info(f"[{feature}] Routing to EXTERNAL API execution.")
                # Prepare a temporary output name for this step
                step_filename = f"{current_image_path.stem}_{feature}.jpg"
                step_path = output_dir / step_filename
                
                # If feature is sky_replacement or object_removal, handle masks
                mask_path = None
                if feature == "sky_replacement":
                    mask_filename = f"{current_image_path.stem}_skymask.png"
                    mask_path = output_dir / mask_filename
                    try:
                        from mask_generator import generate_sky_mask
                        logging.info(f"[{feature}] Generating strict sky boundary mask...")
                        success = generate_sky_mask(current_image_path, mask_path)
                        if not success:
                            logging.warning(f"[{feature}] Failed to detect sky boundary. Fallback to API default.")
                            mask_path = None
                    except Exception as e:
                        logging.error(f"[{feature}] Mask generation error: {e}")
                
                elif feature in ["object_removal", "declutter"]:
                    points = params.get(feature)
                    if points:
                        mask_filename = f"{current_image_path.stem}_objmask.png"
                        mask_path = output_dir / mask_filename
                        try:
                            from mask_generator import generate_object_mask
                            logging.info(f"[{feature}] Generating interactive object mask at {points}...")
                            success = generate_object_mask(current_image_path, mask_path, points)
                            if not success: mask_path = None
                        except Exception as e:
                            logging.error(f"[{feature}] Object mask generation error: {e}")
                
                # Execute API Call
                job_id = self.ai_api.submit_job(current_image_path, feature)
                success = self.ai_api.fetch_result(job_id, step_path, current_image_path, mask_path=mask_path, feature=feature)
                
                if success:
                    # The output of this step becomes the input for the next feature
                    current_image_path = step_path
                    
        return current_image_path

if __name__ == "__main__":
    # Test the orchestrator
    test_image = "test_hdr_output.jpg"
    Path(test_image).touch()
    
    orchestrator = EnhancementOrchestrator()
    final_output = orchestrator.process_image(
        test_image, 
        requested_features=["white_balance", "sky_replacement", "day_to_dusk"],
        output_dir="output_final"
    )
    
    print(f"Workflow Complete. Final result at: {final_output}")
