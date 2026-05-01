import logging

import numpy as np
import torch
from PIL import Image
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from .base_restorer import BaseRestorer

logger = logging.getLogger(__name__)


class EsrganRestorer(BaseRestorer):
    def _load_model(self):
        # Real-ESRGAN uses RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['params'], strict=True)
        model.eval()
        model.to(self.device)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False  # Use full precision
        )
        logger.info("Real-ESRGAN model loaded.")

    def restore(self, image, scale=4, **kwargs):
        """
        Upscale the image using Real-ESRGAN.
        image: PIL Image
        scale: Upscaling factor (default 4)
        """
        # Convert PIL to numpy
        img_np = np.array(image)

        # Upsample
        output, _ = self.upsampler.enhance(img_np, outscale=scale)

        # Convert back to PIL
        return Image.fromarray(output)
