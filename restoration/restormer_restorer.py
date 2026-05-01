import logging

import torch
import torchvision.transforms as transforms
from .base_restorer import BaseRestorer

logger = logging.getLogger(__name__)


class RestormerRestorer(BaseRestorer):
    def _load_model(self):
        from basicsr.archs.restormer_arch import Restormer

        # Load Restormer model (assuming it's a .pth file)
        self.model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['params'])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Restormer model loaded.")

    def restore(self, image, **kwargs):
        """
        Restore the image using Restormer (general enhancement).
        image: PIL Image
        """
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess
        output = output.squeeze(0).cpu()
        output = (output + 1) / 2  # Denormalize
        output = torch.clamp(output, 0, 1)
        to_pil = transforms.ToPILImage()
        return to_pil(output)
