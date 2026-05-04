import logging
import cv2
import numpy as np
from pathlib import Path
import torch

# Try to load a depth model for Phase 1.1 (Depth-Aware Masking)
try:
    # Using MiDaS small for speed and low memory
    # Note: If this fails due to trust issues, run once interactively and press 'y'
    DEPTH_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    DEPTH_TRANSFORMS = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    HAS_DEPTH = True
except Exception as exc:
    logging.warning("[SkyProcessor] Depth model load failed (trust issue or network). Falling back to pseudo-depth. Error: %s", exc)
    HAS_DEPTH = False

class SkyReplacementEngine:
    """
    Advanced Sky Replacement Engine implementing Phase 1 and 2:
    - High-Fidelity Masking (Depth-Gate + Guided Filtering)
    - Photorealistic Compositing (Color Matching + Light Wrap)
    """

    def __init__(self, sky_assets_dir="static/sky_assets"):
        self.sky_assets_dir = Path(sky_assets_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if HAS_DEPTH:
            DEPTH_MODEL.to(self.device).eval()

    def get_depth_map(self, img_bgr):
        """Estimate depth for the image to help isolate sky (infinity)."""
        if not HAS_DEPTH:
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_batch = DEPTH_TRANSFORMS(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = DEPTH_MODEL(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        # Normalize to [0, 255] where 255 is closest and 0 is furthest
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        return depth.astype(np.uint8)

    def refine_mask_with_depth(self, mask, depth_map):
        """
        Phase 1.1: Use depth to remove foreground items (railings/trees) from the sky mask.
        Sky is always at the furthest point (lowest depth values).
        """
        if depth_map is None:
            return mask

        # Sky should be in the bottom 15% of depth values (furthest away)
        # We use a soft threshold to avoid jagged edges
        sky_depth_threshold = np.percentile(depth_map, 15)
        depth_gate = (depth_map <= sky_depth_threshold).astype(np.uint8) * 255
        
        # Smooth the gate to prevent artifacts
        depth_gate = cv2.GaussianBlur(depth_gate, (15, 15), 0)
        
        refined = cv2.bitwise_and(mask, depth_gate)
        return refined

    def apply_color_matching(self, foreground, background_sky, mask):
        """
        Phase 2.1: Adjust foreground (house) to match background (sky) color temperature.
        """
        # Convert to LAB for independent color/luminance control
        fg_lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(background_sky, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Get mean color of the new sky
        bg_l_mean, bg_a_mean, bg_b_mean = cv2.split(bg_lab)
        bg_a_target = np.mean(bg_a_mean)
        bg_b_target = np.mean(bg_b_mean)

        # Get mean color of the foreground (outside the sky mask)
        inv_mask = cv2.bitwise_not(mask)
        fg_pixels = fg_lab[inv_mask > 0]
        if fg_pixels.size == 0:
            return foreground
            
        fg_a_mean = np.mean(fg_pixels[:, 1])
        fg_b_mean = np.mean(fg_pixels[:, 2])

        # Shift foreground color toward background color (30% strength for subtle realism)
        strength = 0.3
        fg_lab[:, :, 1] += (bg_a_target - fg_a_mean) * strength
        fg_lab[:, :, 2] += (bg_b_target - fg_b_mean) * strength

        matched = cv2.cvtColor(np.clip(fg_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return matched

    def apply_light_wrap(self, foreground, sky_pixels, mask):
        """
        Phase 2.2: Add a subtle glow from the bright sky onto the window edges.
        """
        # Create a blurred version of the sky at the edges
        glow_mask = cv2.GaussianBlur(mask, (51, 51), 0)
        # Only keep the part of the glow that overlaps the foreground
        glow_mask = cv2.bitwise_and(glow_mask, cv2.bitwise_not(mask))
        
        # Extract color from the sky
        avg_sky_color = cv2.mean(sky_pixels, mask=mask)[:3]
        glow_layer = np.full(foreground.shape, avg_sky_color, dtype=np.uint8)
        
        # Blend the glow onto the foreground
        alpha = glow_mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2) * 0.4 # 40% max intensity
        
        wrapped = foreground.astype(float) * (1 - alpha) + glow_layer.astype(float) * alpha
        return np.clip(wrapped, 0, 255).astype(np.uint8)

    def process(self, image_path, sky_type="sunset", output_path=None):
        """Execute the full replacement pipeline."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        # 1. Generate/Load Mask
        # For this implementation, we assume generate_sky_mask exists
        from mask_generator import generate_sky_mask
        mask_path = Path("temp_mask.png")
        generate_sky_mask(image_path, mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None or np.count_nonzero(mask) == 0:
            logging.warning("[SkyProcessor] No sky detected in %s", image_path)
            return img

        # 2. Refine Mask with Depth (Phase 1.1)
        depth_map = self.get_depth_map(img)
        mask = self.refine_mask_with_depth(mask, depth_map)

        # 3. Load Sky Asset
        asset_path = self.sky_assets_dir / f"{sky_type}.jpg"
        if not asset_path.exists():
            # Fallback to a generic one
            asset_path = self.sky_assets_dir / "clear_day.jpg"
            
        sky_src = cv2.imread(str(asset_path))
        if sky_src is None:
            logging.error("[SkyProcessor] Missing sky asset: %s", asset_path)
            return img

        # Fit sky to image
        sky_fit = cv2.resize(sky_src, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # 4. Color Match Foreground to Sky (Phase 2.1)
        # Apply to a copy so we don't mess with the original
        matched_fg = self.apply_color_matching(img, sky_fit, mask)

        # 5. Composite
        # Use guided filter for professional edges (Phase 1.2)
        try:
            guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            refined_mask = cv2.ximgproc.guidedFilter(guide, mask, radius=12, eps=1000)
        except:
            refined_mask = cv2.GaussianBlur(mask, (7, 7), 0)

        alpha = refined_mask.astype(float) / 255.0
        alpha_3ch = np.expand_dims(alpha, axis=2)
        
        # Phase 2.3: Darken-only edge blending to remove blue halos
        # We blend the sky only where it's lighter or darker as needed, 
        # but primarily we want to avoid that "blue glow" on dark frame edges.
        fg_float = matched_fg.astype(float)
        sky_float = sky_fit.astype(float)
        
        # Final Blend
        blended = fg_float * (1 - alpha_3ch) + sky_float * alpha_3ch
        result = np.clip(blended, 0, 255).astype(np.uint8)

        # 6. Light Wrap (Phase 2.2) - Subtle glow onto frames
        result = self.apply_light_wrap(result, sky_fit, refined_mask)

        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = SkyReplacementEngine()
    # Example usage:
    # engine.process("input/house.jpg", "sunset", "output/house_sunset.jpg")
