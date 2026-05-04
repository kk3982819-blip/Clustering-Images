from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    HAS_TRANSFORMERS = True
except ImportError:
    SegformerConfig = None
    SegformerForSemanticSegmentation = None
    HAS_TRANSFORMERS = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class SunraySample:
    image_path: Path
    mask_path: Path


@dataclass(slots=True)
class LoadedSunrayModel:
    model: nn.Module
    architecture: str
    input_size: int
    device: str
    is_transformer: bool
    weights_path: str | None


class TinySunraySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.head(decoded)


class SunraySegmentationDataset(Dataset):
    def __init__(self, samples: list[SunraySample], image_size: int = 512) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path))
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {sample.image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {sample.mask_path}")

        pixel_values = preprocess_image_tensor(image, self.image_size)
        resized_mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        labels = torch.from_numpy((resized_mask > 127).astype(np.float32)).unsqueeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }


def preferred_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_image_tensor(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _build_tiny_model(device: str, input_size: int, weights_path: str | None = None) -> LoadedSunrayModel:
    model = TinySunraySegNet().to(device)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return LoadedSunrayModel(
        model=model,
        architecture="tiny_fallback",
        input_size=input_size,
        device=device,
        is_transformer=False,
        weights_path=weights_path,
    )


def _build_transformer_model(
    device: str,
    input_size: int,
    weights_path: str | None,
    backbone: str,
    local_files_only: bool,
) -> LoadedSunrayModel | None:
    if not HAS_TRANSFORMERS:
        return None

    model: nn.Module
    if weights_path and Path(weights_path).is_dir():
        model = SegformerForSemanticSegmentation.from_pretrained(weights_path, local_files_only=True)
    elif weights_path and Path(weights_path).is_file():
        checkpoint = torch.load(weights_path, map_location=device)
        if checkpoint.get("architecture") != "segformer":
            return None
        config = SegformerConfig(**checkpoint["config"])
        model = SegformerForSemanticSegmentation(config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=1,
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only,
            )
        except Exception:
            config = SegformerConfig(num_labels=1)
            model = SegformerForSemanticSegmentation(config)

    model.to(device)
    model.eval()
    return LoadedSunrayModel(
        model=model,
        architecture="segformer",
        input_size=input_size,
        device=device,
        is_transformer=True,
        weights_path=weights_path or backbone,
    )


def load_sunray_model(
    weights_path: str | Path | None = None,
    backbone: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    device: str | None = None,
    input_size: int = 512,
    prefer_transformer: bool = True,
    local_files_only: bool = True,
) -> LoadedSunrayModel:
    device = device or preferred_device()
    weights_str = str(weights_path) if weights_path is not None else None

    if prefer_transformer:
        transformer_model = _build_transformer_model(
            device=device,
            input_size=input_size,
            weights_path=weights_str,
            backbone=backbone,
            local_files_only=local_files_only,
        )
        if transformer_model is not None:
            return transformer_model

    return _build_tiny_model(device=device, input_size=input_size, weights_path=weights_str)


def predict_probability_map(bundle: LoadedSunrayModel, image_bgr: np.ndarray) -> np.ndarray:
    original_h, original_w = image_bgr.shape[:2]
    pixel_values = preprocess_image_tensor(image_bgr, bundle.input_size).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        if bundle.is_transformer:
            outputs = bundle.model(pixel_values=pixel_values)
            logits = outputs.logits
        else:
            logits = bundle.model(pixel_values)
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
        probabilities = torch.sigmoid(logits[:, :1]).squeeze().detach().cpu().numpy()

    return probabilities.astype(np.float32)


def save_sunray_model(bundle: LoadedSunrayModel, output_dir: str | Path, extra_metadata: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "architecture": bundle.architecture,
        "input_size": bundle.input_size,
        "weights_path": bundle.weights_path,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    if bundle.is_transformer and HAS_TRANSFORMERS:
        bundle.model.save_pretrained(output_dir)
    else:
        torch.save(
            {
                "architecture": bundle.architecture,
                "input_size": bundle.input_size,
                "state_dict": bundle.model.state_dict(),
            },
            output_dir / "model.pt",
        )

    (output_dir / "sunray_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def discover_sunray_samples(dataset_root: str | Path) -> list[SunraySample]:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return []

    samples: list[SunraySample] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            samples.append(SunraySample(image_path=image_path, mask_path=mask_path))
    return samples


def split_samples(samples: list[SunraySample], val_fraction: float = 0.2) -> tuple[list[SunraySample], list[SunraySample]]:
    if not samples:
        return [], []
    val_count = max(1, int(round(len(samples) * val_fraction))) if len(samples) > 1 else 0
    train_count = max(1, len(samples) - val_count) if len(samples) > 1 else len(samples)
    return samples[:train_count], samples[train_count:]
