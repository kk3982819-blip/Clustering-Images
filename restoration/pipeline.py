import logging
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)


RESTORER_SPECS = {
    "lama": ("lama_restorer", "LamaRestorer", "lama_path"),
    "deepfill": ("deepfill_restorer", "DeepFillRestorer", "deepfill_path"),
    "restormer": ("restormer_restorer", "RestormerRestorer", "restormer_path"),
    "mprnet": ("mprnet_restorer", "MPRNetRestorer", "mprnet_path"),
    "esrgan": ("esrgan_restorer", "EsrganRestorer", "esrgan_path"),
}


class RestorationPipeline:
    def __init__(self, config):
        self.config = config
        self.models = {}

    def _get_model(self, name):
        """Load restoration models on first use so app startup stays lightweight."""
        if name in self.models:
            return self.models[name]

        module_name, class_name, config_key = RESTORER_SPECS[name]
        model_path = self.config[config_key]

        if name != "lama" and not Path(model_path).exists():
            raise FileNotFoundError(f"{name} model file not found at {model_path}")

        module = import_module(f".{module_name}", package=__package__)
        restorer_cls = getattr(module, class_name)
        self.models[name] = restorer_cls(model_path)
        return self.models[name]

    def _try_restore(self, primary, fallback, *args, **kwargs):
        """Try primary model, fallback on failure."""
        try:
            result = self._get_model(primary).restore(*args, **kwargs)
            if self._is_quality_ok(result):
                return result
            else:
                logger.warning(f"{primary} output quality low, switching to {fallback}")
        except Exception as e:
            logger.warning(f"{primary} failed: {e}, switching to {fallback}")

        try:
            return self._get_model(fallback).restore(*args, **kwargs)
        except Exception as e:
            logger.error(f"{fallback} also failed: {e}")
            raise RuntimeError("Restoration failed")

    def _is_quality_ok(self, image):
        """Simple quality check (placeholder)."""
        # Implement PSNR or SSIM check here
        return True  # For now, assume OK

    def inpaint(self, image, mask):
        """Inpaint with fallback."""
        return self._try_restore('lama', 'deepfill', image, mask=mask)

    def restore(self, image):
        """General restoration with fallback."""
        return self._try_restore('restormer', 'mprnet', image)

    def upscale(self, image, scale=4):
        """Upscale (no fallback yet)."""
        return self._get_model('esrgan').restore(image, scale=scale)

    def full_restore(self, image, mask=None, upscale=True):
        """Full pipeline: inpaint if mask, restore, upscale."""
        if mask:
            try:
                image = self.inpaint(image, mask)
            except Exception as e:
                logger.warning("Inpainting skipped: %s", e)
        try:
            image = self.restore(image)
        except Exception as e:
            logger.warning("General restoration skipped: %s", e)
        if upscale:
            try:
                image = self.upscale(image)
            except Exception as e:
                logger.warning("Upscaling skipped: %s", e)
        return image
