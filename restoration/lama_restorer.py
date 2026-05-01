import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .base_restorer import BaseRestorer

logger = logging.getLogger(__name__)


class LamaRestorer(BaseRestorer):
    def _load_model(self):
        if not Path(self.model_path).exists():
            logger.warning("LaMa model path %s is missing; using OpenCV inpaint fallback.", self.model_path)
            self.model = None
            return

        try:
            from diffusers import AutoPipelineForInpainting

            self.model = AutoPipelineForInpainting.from_pretrained(self.model_path).to(self.device)
            logger.info("LaMa-compatible inpainting model loaded from %s.", self.model_path)
        except Exception as exc:
            logger.warning("Could not load inpainting model from %s; using OpenCV fallback: %s", self.model_path, exc)
            self.model = None

    def restore(self, image, mask, **kwargs):
        """
        Inpaint the image using LaMa.
        image: PIL Image
        mask: PIL Image (binary mask, white for areas to inpaint)
        """
        # Ensure mask is binary
        mask_np = np.array(mask.convert("L"))
        if mask_np.max() <= 1:
            mask_np = mask_np * 255
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np)

        if self.model is None:
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            inpainted = cv2.inpaint(image_bgr, mask_np, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

        # Run inpainting
        prompt = kwargs.get("prompt", "")
        try:
            result = self.model(image=image.convert("RGB"), mask_image=mask_pil, prompt=prompt)
        except TypeError:
            result = self.model(image=image.convert("RGB"), mask_image=mask_pil)
        return result.images[0]
