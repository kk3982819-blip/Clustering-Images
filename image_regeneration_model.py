import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

DEFAULT_REGEN_MODEL_ID = "SG161222/RealVisXL_V4.0_Lightning" # High-realism photorealistic model
PLACEHOLDER_MODEL_IDS = {
    "",
    "your-local-or-hf-inpaint-model",
    "your-local-or-hf-model",
    "your-model-id",
}


class ImageRegenerationModel:
    """
    Image regeneration model using Stable Diffusion Inpainting for realistic image enhancement.
    """

    def __init__(self, model_id=None):
        configured_model_id = model_id or os.environ.get("PIXELDWELL_REGEN_MODEL_ID", DEFAULT_REGEN_MODEL_ID)
        configured_model_id = configured_model_id.strip()
        if configured_model_id in PLACEHOLDER_MODEL_IDS:
            logging.warning(
                "Ignoring placeholder PIXELDWELL_REGEN_MODEL_ID=%r; using %s",
                configured_model_id,
                DEFAULT_REGEN_MODEL_ID,
            )
            configured_model_id = DEFAULT_REGEN_MODEL_ID

        model_id = configured_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading inpainting model %s on %s...", model_id, self.device)
        self._use_cpu_offload = False
        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        variant = os.environ.get("PIXELDWELL_REGEN_VARIANT")
        if variant:
            load_kwargs["variant"] = variant
        elif self.device == "cuda" and model_id == DEFAULT_REGEN_MODEL_ID:
            load_kwargs["variant"] = "fp16"

        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, **load_kwargs)
        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            offload_setting = os.environ.get("PIXELDWELL_REGEN_CPU_OFFLOAD", "auto").strip().lower()
            self._use_cpu_offload = offload_setting in {"1", "true", "yes", "on"} or (
                offload_setting == "auto" and total_vram_gb < 8.0
            )

        if self._use_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            logging.info("Enabling model CPU offload for low-VRAM CUDA device.")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                logging.info("xFormers attention is unavailable; continuing without it: %s", exc)

    @staticmethod
    def _resize_for_inpainting(image, mask, max_side):
        width, height = image.size
        if max(width, height) <= max_side:
            new_width, new_height = width, height
        else:
            scale = max_side / float(max(width, height))
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))

        new_width = max(64, (new_width // 8) * 8)
        new_height = max(64, (new_height // 8) * 8)
        if (new_width, new_height) == (width, height):
            return image, mask

        return (
            image.resize((new_width, new_height), Image.LANCZOS),
            mask.resize((new_width, new_height), Image.NEAREST),
        )

    def regenerate_image(
        self,
        image_path,
        mask_path=None,
        prompt="realistic photo",
        strength=0.75,
        negative_prompt=None,
        guidance_scale=7.5,
        num_inference_steps=28,
        max_side=None,
        seed=None,
    ):
        """
        Regenerate the image using inpainting to make changes look natural.

        Args:
            image_path: Path to the input image
            mask_path: Path to the mask indicating areas to regenerate (optional)
            prompt: Text prompt for regeneration
            strength: How much to regenerate (0-1)

        Returns:
            PIL Image of the regenerated image
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        if mask_path and Path(mask_path).exists():
            # Load mask
            mask = Image.open(mask_path).convert("L")
            # Resize mask to match image
            mask = mask.resize(image.size, Image.NEAREST)
        else:
            # Create a full mask if no mask provided
            mask = Image.new("L", image.size, 255)

        if max_side is None:
            max_side = int(os.environ.get("PIXELDWELL_REGEN_MAX_SIDE", "1024"))
        image, mask = self._resize_for_inpainting(image, mask, max_side=max_side)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        autocast_ctx = torch.autocast("cuda") if self.device == "cuda" else nullcontext()

        # Generate
        with autocast_ctx:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        return result

# Global instance
_regeneration_model = None

def get_regeneration_model():
    global _regeneration_model
    if _regeneration_model is None:
        _regeneration_model = ImageRegenerationModel()
    return _regeneration_model
