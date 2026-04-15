from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"

import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps
from sklearn.cluster import AgglomerativeClustering, HDBSCAN

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ORB_EXTRACTOR = cv2.ORB_create(1200)
ORB_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
YOLO_MODEL_NAME = "yolov8n-seg.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25
SEGMENTATION_MIN_CONTOUR_AREA = 300
_YOLO_MODEL: object | None = None
SCENE_LABEL_PROMPTS = {
    "sky": ["sky", "blue sky"],
    "green grass": ["grass", "green field"],
    "trees": ["trees"],
    "window": ["window", "a window", "sliding glass door"],
    "balcony": ["balcony", "a balcony"],
    "outdoor view": ["outdoor view"],
    "white wall": ["white wall", "a white painted wall"],
    "floor tiles": ["floor tiles", "tile floor"],
}
SCENE_LABEL_THRESHOLDS = {
    "sky": 0.20,
    "green grass": 0.20,
    "trees": 0.20,
    "window": 0.21,
    "balcony": 0.22,
    "outdoor view": 0.20,
    "white wall": 0.22,
    "floor tiles": 0.22,
}
SCENE_LABEL_MAX_AREA_RATIO = {
    "sky": 0.18,
    "green grass": 0.18,
    "trees": 0.18,
    "window": 0.28,
    "balcony": 0.30,
    "outdoor view": 0.30,
    "white wall": 0.26,
    "floor tiles": 0.60,
}
SCENE_LABEL_COLORS = {
    "sky": (235, 180, 70),
    "green grass": (80, 200, 80),
    "trees": (30, 150, 60),
    "window": (255, 170, 0),
    "balcony": (255, 140, 0),
    "outdoor view": (255, 120, 60),
    "white wall": (200, 200, 200),
    "floor tiles": (0, 180, 255),
}
STRICT_ITEM_PROMPTS = [
    "a real estate interior photo of an empty room",
    "a real estate interior photo of a furnished room",
    "a real estate interior photo of a bedroom",
    "a real estate interior photo of a living room",
    "a real estate interior photo of a hall",
    "a real estate interior photo of a kitchen",
    "a real estate interior photo of a bathroom",
    "a real estate interior photo of a balcony",
    "a bed",
    "a bedside table",
    "a sofa",
    "a television",
    "a dining table",
    "chairs",
    "a kitchen island",
    "kitchen cabinets",
    "a countertop",
    "a stove",
    "a sink",
    "a refrigerator",
    "a wardrobe",
    "a toilet",
    "a bathtub",
    "a shower",
    "a window",
    "a sliding glass door",
]
FLAG_LABELS = [
    "grass",
    "green field",
    "sky",
    "blue sky",
    "window",
    "balcony",
    "trees",
    "outdoor view",
    "empty room",
    "furnished room",
    "floor tiles",
    "white wall",
]


Boundary = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SceneRegionCandidate:
    box: tuple[int, int, int, int]
    boundary: Boundary
    allowed_labels: tuple[str, ...]


@dataclass
class SceneLabelerRuntime:
    model: object
    preprocess: object
    device: str
    text_features: torch.Tensor
    label_to_prompt_indices: dict[str, list[int]]


@dataclass(frozen=True)
class SceneLabelDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]
    boundary: Boundary


@dataclass
class SceneDetectionGroup:
    box: tuple[int, int, int, int]
    boundary: Boundary
    labels: list[tuple[str, float]]


@dataclass(frozen=True)
class ImageAnnotation:
    source: str
    box: tuple[int, int, int, int]
    boundary: Boundary
    labels: tuple[tuple[str, float], ...]


def load_clip_module() -> tuple[object, str]:
    clip_init = VENDOR_DIR / "clip" / "__init__.py"
    if clip_init.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "clip",
                clip_init,
                submodule_search_locations=[str(clip_init.parent)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load CLIP package from {clip_init}")

            clip = importlib.util.module_from_spec(spec)
            sys.modules["clip"] = clip
            spec.loader.exec_module(clip)
            if not hasattr(clip, "load"):
                raise RuntimeError(f"Loaded CLIP package from {clip_init}, but clip.load is missing.")
            return clip, str(clip_init)
        except (OSError, PermissionError):
            sys.modules.pop("clip", None)

    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "CLIP is not installed. Install dependencies from requirements.txt or install it into .vendor."
        ) from exc

    if not hasattr(clip, "load"):
        raise RuntimeError("Imported clip module does not expose clip.load. Check your CLIP installation.")
    return clip, getattr(clip, "__file__", "installed package")


def load_clip_runtime(
    model_name: str,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[object, object, object]:
    clip, clip_source = load_clip_module()
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded CLIP from %s", clip_source)
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root=str(cache_dir))
    model.eval()
    return clip, model, preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster images from an input folder using CLIP + HDBSCAN.")
    parser.add_argument("--input", default="input", help="Input image folder.")
    parser.add_argument("--output", default="output", help="Output folder for clustered images.")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=1, help="HDBSCAN min_samples.")
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon. Higher values merge nearby clusters more aggressively.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.45,
        help="Weight for CLIP semantic features.",
    )
    parser.add_argument(
        "--layout-weight",
        type=float,
        default=0.35,
        help="Weight for grayscale layout features. Increase this to favor same-corner clustering.",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.15,
        help="Weight for edge-map features. Increase this to favor similar geometry and room structure.",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=0.05,
        help="Weight for color histogram features.",
    )
    parser.add_argument(
        "--view-max-cluster-size",
        type=int,
        default=None,
        help="Optional cap for second-stage same-corner clusters. Lower values force large room groups to split.",
    )
    parser.add_argument(
        "--view-similarity-threshold",
        type=float,
        default=0.34,
        help="Minimum pairwise viewpoint similarity used when refining broad same-room clusters into tighter same-corner groups.",
    )
    parser.add_argument(
        "--semantic-merge-threshold",
        type=float,
        default=0.98,
        help="Merge back same-room subclusters when their CLIP centroid similarity is above this threshold.",
    )
    parser.add_argument(
        "--strict-same-corner-items",
        action="store_true",
        help="Use stricter second-stage clustering that requires both close viewpoint similarity and close item similarity.",
    )
    parser.add_argument(
        "--item-similarity-threshold",
        type=float,
        default=0.84,
        help="Minimum CLIP prompt-signature similarity required for images to remain in the same strict cluster.",
    )
    parser.add_argument(
        "--strict-cluster-threshold",
        type=float,
        default=0.56,
        help="Minimum combined strict similarity used by complete-link clustering in strict same-corner+items mode.",
    )
    parser.add_argument(
        "--semantic-similarity-floor",
        type=float,
        default=0.90,
        help="Minimum CLIP image embedding similarity required before two images can be grouped in strict mode.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("cluster_images")


def resolve_device(raw_device: str) -> str:
    if raw_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return raw_device


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb_image(image_path: Path) -> Image.Image | None:
    try:
        with Image.open(image_path) as image:
            normalized = ImageOps.exif_transpose(image)
            return normalized.convert("RGB")
    except (OSError, ValueError):
        return None


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def image_to_layout_vector(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    grayscale = image.resize(size, Image.Resampling.BICUBIC).convert("L")
    vector = np.asarray(grayscale, dtype=np.float32).reshape(-1)
    vector -= vector.mean()
    return l2_normalize(vector.reshape(1, -1))[0]


def image_to_edge_vector(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    edge_image = image.resize(size, Image.Resampling.BICUBIC).convert("L").filter(ImageFilter.FIND_EDGES)
    vector = np.asarray(edge_image, dtype=np.float32).reshape(-1)
    vector -= vector.mean()
    return l2_normalize(vector.reshape(1, -1))[0]


def image_to_color_histogram(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV").resize((48, 48), Image.Resampling.BICUBIC), dtype=np.float32)
    histograms: list[np.ndarray] = []

    channel_bins = [(0.0, 255.0, 12), (0.0, 255.0, 6), (0.0, 255.0, 6)]
    for channel_index, (low, high, bins) in enumerate(channel_bins):
        channel = hsv[:, :, channel_index].reshape(-1)
        histogram, _ = np.histogram(channel, bins=bins, range=(low, high), density=False)
        histograms.append(histogram.astype(np.float32))

    return l2_normalize(np.concatenate(histograms).reshape(1, -1))[0]


def image_to_opening_profile(image: Image.Image) -> np.ndarray:
    resized = image.resize((64, 64), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    bright_mask = array.mean(axis=2) > 210.0
    blue_mask = (blue > green + 8.0) & (blue > red + 8.0) & (blue > 120.0)
    opening_mask = (bright_mask | blue_mask).astype(np.float32)

    y_coords, _ = np.mgrid[0:64, 0:64]
    weight = np.ones_like(opening_mask, dtype=np.float32)
    weight[y_coords < 40] += 0.8
    weight[y_coords < 24] += 0.5
    weighted_mask = opening_mask * weight

    feature = np.concatenate(
        [
            weighted_mask.mean(axis=0),
            weighted_mask.mean(axis=1),
            np.array(
                [
                    weighted_mask[:, :21].mean(),
                    weighted_mask[:, 21:43].mean(),
                    weighted_mask[:, 43:].mean(),
                ],
                dtype=np.float32,
            ),
        ]
    )
    feature -= feature.mean()
    return l2_normalize(feature.reshape(1, -1))[0]


def image_to_orb_descriptors(image: Image.Image) -> np.ndarray | None:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, descriptors = ORB_EXTRACTOR.detectAndCompute(gray, None)
    return descriptors


def orb_similarity_score(descriptors_a: np.ndarray | None, descriptors_b: np.ndarray | None) -> float:
    if descriptors_a is None or descriptors_b is None or not len(descriptors_a) or not len(descriptors_b):
        return 0.0

    matches = ORB_MATCHER.match(descriptors_a, descriptors_b)
    if not matches:
        return 0.0

    good_matches = [match for match in matches if match.distance < 42]
    denominator = max(min(len(descriptors_a), len(descriptors_b)), 1)
    return float(len(good_matches) / denominator)


def viewpoint_similarity_matrix(image_paths: list[Path], logger: logging.Logger) -> np.ndarray | None:
    images: list[Image.Image] = []
    for image_path in image_paths:
        image = load_rgb_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image during viewpoint refinement: %s", image_path)
            return None
        images.append(image)

    layout_features = [image_to_layout_vector(image, size=(24, 24)) for image in images]
    edge_features = [image_to_edge_vector(image, size=(24, 24)) for image in images]
    opening_features = [image_to_opening_profile(image) for image in images]
    orb_descriptors = [image_to_orb_descriptors(image) for image in images]

    size = len(image_paths)
    similarity = np.eye(size, dtype=np.float32)
    for first_index in range(size):
        for second_index in range(first_index + 1, size):
            score = (
                0.35 * float(np.dot(layout_features[first_index], layout_features[second_index]))
                + 0.25 * float(np.dot(edge_features[first_index], edge_features[second_index]))
                + 0.25 * float(np.dot(opening_features[first_index], opening_features[second_index]))
                + 0.15 * orb_similarity_score(orb_descriptors[first_index], orb_descriptors[second_index])
            )
            similarity[first_index, second_index] = score
            similarity[second_index, first_index] = score

    return similarity


def maybe_split_quad_cluster(image_paths: list[Path], logger: logging.Logger) -> list[list[int]] | None:
    if len(image_paths) != 4:
        return None

    images: list[Image.Image] = []
    for image_path in image_paths:
        image = load_rgb_image(image_path)
        if image is None:
            return None
        images.append(image)

    opening_profiles = [image_to_opening_profile(image) for image in images]
    orb_descriptors = [image_to_orb_descriptors(image) for image in images]

    similarity = np.zeros((4, 4), dtype=np.float32)
    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            opening_similarity = float(np.dot(opening_profiles[first_index], opening_profiles[second_index]))
            local_similarity = orb_similarity_score(orb_descriptors[first_index], orb_descriptors[second_index])
            similarity_score = (0.7 * opening_similarity) + (0.3 * local_similarity)
            similarity[first_index, second_index] = similarity_score
            similarity[second_index, first_index] = similarity_score

    pairings = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]
    scored_pairings: list[tuple[float, float, list[tuple[int, int]]]] = []
    for pairing in pairings:
        scores = [float(similarity[left, right]) for left, right in pairing]
        scored_pairings.append((sum(scores), min(scores), pairing))

    scored_pairings.sort(key=lambda item: item[0], reverse=True)
    best_score, best_min_pair, best_pairing = scored_pairings[0]
    second_best_score = scored_pairings[1][0]

    if best_score < 0.65 or best_min_pair < 0.25 or (best_score - second_best_score) < 0.10:
        return None

    logger.info(
        "Refined 4-image room cluster into 2 pairs using viewpoint matching (best=%.3f, second=%.3f)",
        best_score,
        second_best_score,
    )
    return [[left, right] for left, right in best_pairing]


def maybe_refine_broad_viewpoint_cluster(
    image_paths: list[Path],
    similarity_threshold: float,
    logger: logging.Logger,
) -> tuple[list[list[int]], list[int]] | None:
    if len(image_paths) < 5:
        return None

    similarity = viewpoint_similarity_matrix(image_paths, logger)
    if similarity is None:
        return None

    distance = 1.0 - similarity
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=max(0.0, 1.0 - similarity_threshold),
    )
    labels = model.fit_predict(distance)

    clusters: list[list[int]] = []
    noise_indices: list[int] = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        members = [index for index, label in enumerate(labels) if int(label) == cluster_id]
        if len(members) >= 2:
            clusters.append(members)
        else:
            noise_indices.extend(members)

    if len(clusters) <= 1:
        return None

    logger.info(
        "Refined broad viewpoint cluster of %s images into %s subclusters with %s noise images using threshold %.2f",
        len(image_paths),
        len(clusters),
        len(noise_indices),
        similarity_threshold,
    )
    return clusters, noise_indices


def strict_same_corner_item_clusters(
    image_paths: list[Path],
    clip_embeddings: np.ndarray,
    item_features: np.ndarray,
    min_cluster_size: int,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    strict_cluster_threshold: float,
    logger: logging.Logger,
) -> tuple[list[list[int]], list[int]] | None:
    if len(image_paths) == 0:
        return None

    viewpoint_similarity = viewpoint_similarity_matrix(image_paths, logger)
    if viewpoint_similarity is None:
        return None

    semantic_similarity = np.clip(clip_embeddings @ clip_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    item_similarity = np.clip(item_features @ item_features.T, -1.0, 1.0).astype(np.float32, copy=False)

    size = len(image_paths)
    strict_similarity = np.eye(size, dtype=np.float32)
    for first_index in range(size):
        for second_index in range(first_index + 1, size):
            view_score = float(viewpoint_similarity[first_index, second_index])
            item_score = float(item_similarity[first_index, second_index])
            semantic_score = float(semantic_similarity[first_index, second_index])
            if (
                view_score < view_similarity_threshold
                or item_score < item_similarity_threshold
                or semantic_score < semantic_similarity_floor
            ):
                score = 0.0
            else:
                score = (
                    0.50 * view_score
                    + 0.30 * item_score
                    + 0.20 * semantic_score
                )
            strict_similarity[first_index, second_index] = score
            strict_similarity[second_index, first_index] = score

    if len(image_paths) < 2:
        return None

    distance = 1.0 - strict_similarity
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=max(0.0, 1.0 - strict_cluster_threshold),
    )
    labels = model.fit_predict(distance)

    clusters: list[list[int]] = []
    noise_indices: list[int] = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        members = [index for index, label in enumerate(labels) if int(label) == cluster_id]
        if len(members) >= min_cluster_size:
            clusters.append(members)
        else:
            noise_indices.extend(members)

    logger.info(
        "Strict same-corner+items refinement produced %s clusters and %s noise images from %s semantic images",
        len(clusters),
        len(noise_indices),
        len(image_paths),
    )
    return clusters, noise_indices


def merge_semantic_subclusters(
    semantic_groups: list[dict],
    clip_embeddings: np.ndarray,
    semantic_merge_threshold: float,
    logger: logging.Logger,
) -> list[np.ndarray]:
    if len(semantic_groups) <= 1:
        return [group["indices"] for group in semantic_groups]

    parent = list(range(len(semantic_groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    centroids: list[np.ndarray] = []
    for group in semantic_groups:
        group_embeddings = clip_embeddings[group["indices"]]
        centroid = group_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        centroids.append(centroid)

    for first_index in range(len(semantic_groups)):
        if semantic_groups[first_index]["frozen"]:
            continue
        for second_index in range(first_index + 1, len(semantic_groups)):
            if semantic_groups[second_index]["frozen"]:
                continue
            similarity = float(np.dot(centroids[first_index], centroids[second_index]))
            if similarity >= semantic_merge_threshold:
                logger.info(
                    "Merging same-room subclusters with CLIP centroid similarity %.3f (threshold %.2f)",
                    similarity,
                    semantic_merge_threshold,
                )
                union(first_index, second_index)

    merged_groups: dict[int, list[np.ndarray]] = {}
    for group_index, group in enumerate(semantic_groups):
        merged_groups.setdefault(find(group_index), []).append(group["indices"])

    return [np.concatenate(group_parts).astype(int, copy=False) for group_parts in merged_groups.values()]


def extract_visual_features(image_paths: list[Path], logger: logging.Logger) -> tuple[np.ndarray, list[Path]]:
    features: list[np.ndarray] = []
    valid_paths: list[Path] = []

    for image_path in image_paths:
        image = load_rgb_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image during visual feature extraction: %s", image_path)
            continue

        layout = image_to_layout_vector(image, size=(24, 24))
        edges = image_to_edge_vector(image, size=(24, 24))
        color = image_to_color_histogram(image)
        features.append(np.concatenate([layout, edges, color]).astype(np.float32, copy=False))
        valid_paths.append(image_path)

    if not features:
        return np.empty((0, 0), dtype=np.float32), []

    return np.stack(features), valid_paths


def combine_features(
    clip_embeddings: np.ndarray,
    visual_features: np.ndarray,
    semantic_weight: float,
    layout_weight: float,
    edge_weight: float,
    color_weight: float,
) -> np.ndarray:
    if len(clip_embeddings) != len(visual_features):
        raise ValueError("Feature sets must have the same number of rows.")

    visual_layout_dim = 24 * 24
    visual_edge_dim = 24 * 24
    layout_features = visual_features[:, :visual_layout_dim]
    edge_features = visual_features[:, visual_layout_dim : visual_layout_dim + visual_edge_dim]
    color_features = visual_features[:, visual_layout_dim + visual_edge_dim :]

    raw_weights = np.array([semantic_weight, layout_weight, edge_weight, color_weight], dtype=np.float32)
    if np.all(raw_weights <= 0):
        raise ValueError("At least one feature weight must be greater than zero.")

    normalized_weights = raw_weights / raw_weights.sum()
    weighted_parts = [
        clip_embeddings * normalized_weights[0],
        layout_features * normalized_weights[1],
        edge_features * normalized_weights[2],
        color_features * normalized_weights[3],
    ]
    return l2_normalize(np.concatenate(weighted_parts, axis=1).astype(np.float32, copy=False))


def get_image_tags(
    image_feature: torch.Tensor,
    text_features: torch.Tensor,
    labels: list[str],
    top_k: int = 4,
) -> tuple[list[str], list[float]]:
    if image_feature.ndim == 1:
        image_feature = image_feature.unsqueeze(0)

    similarity = (image_feature @ text_features.T).squeeze()
    top_k = min(top_k, len(labels))
    values, indices = similarity.topk(top_k)
    tag_indices = indices.tolist()
    tags = [labels[index] for index in tag_indices]
    scores = [float(score) for score in values.tolist()]
    return tags, scores


def embed_images(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[Path], list[list[str]]]:
    clip, clip_source = load_clip_module()

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded CLIP from %s", clip_source)
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root=str(cache_dir))
    model.eval()

    embedded_paths: list[Path] = []
    batches: list[np.ndarray] = []
    image_tags: list[list[str]] = []

    with torch.no_grad():
        text_tokens = clip.tokenize(FLAG_LABELS).to(device)
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            batch_valid_paths = []

            for image_path in batch_paths:
                image = load_rgb_image(image_path)
                if image is None:
                    logger.warning("Skipping unreadable image: %s", image_path)
                    continue
                batch_tensors.append(preprocess(image))
                batch_valid_paths.append(image_path)

            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_features = model.encode_image(batch_tensor).float()
            normalized_batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            for image_feature in normalized_batch_features:
                tags, _ = get_image_tags(image_feature, text_features, FLAG_LABELS)
                image_tags.append(tags)

            batches.append(batch_features.cpu().numpy().astype(np.float32, copy=False))
            embedded_paths.extend(batch_valid_paths)
            logger.info("Embedded %s/%s images", len(embedded_paths), len(image_paths))

    if not batches:
        return np.empty((0, 0), dtype=np.float32), [], []

    return l2_normalize(np.concatenate(batches, axis=0).astype(np.float32, copy=False)), embedded_paths, image_tags


def extract_clip_item_features(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    clip, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )

    with torch.no_grad():
        prompt_tokens = clip.tokenize(STRICT_ITEM_PROMPTS).to(device)
        prompt_features = model.encode_text(prompt_tokens).float()
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    valid_paths: list[Path] = []
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            batch_valid_paths = []
            for image_path in batch_paths:
                image = load_rgb_image(image_path)
                if image is None:
                    logger.warning("Skipping unreadable image during item signature extraction: %s", image_path)
                    continue
                batch_tensors.append(preprocess(image))
                batch_valid_paths.append(image_path)

            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(device)
            image_features = model.encode_image(batch_tensor).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ prompt_features.T
            prompt_distribution = torch.softmax(similarity * 10.0, dim=1)
            batches.append(prompt_distribution.cpu().numpy().astype(np.float32, copy=False))
            valid_paths.extend(batch_valid_paths)

    if not batches:
        return np.empty((0, 0), dtype=np.float32), []

    return l2_normalize(np.concatenate(batches, axis=0).astype(np.float32, copy=False)), valid_paths


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_epsilon: float,
    max_cluster_size: int | None = None,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int)

    clusterer = HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=max(0.0, cluster_epsilon),
        cluster_selection_method="eom",
        max_cluster_size=max_cluster_size,
        n_jobs=1,
        copy=True,
    )
    return clusterer.fit_predict(embeddings).astype(int, copy=False)


def cluster_same_corner_groups(
    image_paths: list[Path],
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    min_cluster_size: int,
    min_samples: int,
    cluster_epsilon: float,
    view_max_cluster_size: int | None,
    view_similarity_threshold: float,
    semantic_merge_threshold: float,
    strict_same_corner_items: bool,
    item_similarity_threshold: float,
    strict_cluster_threshold: float,
    semantic_similarity_floor: float,
    logger: logging.Logger,
) -> np.ndarray:
    if len(clip_embeddings) == 0:
        return np.array([], dtype=int)

    semantic_labels = cluster_embeddings(
        embeddings=clip_embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_epsilon=0.0,
    )
    final_labels = np.full(len(clip_embeddings), -1, dtype=int)
    next_label = 0

    semantic_cluster_ids = sorted({int(label) for label in semantic_labels if int(label) != -1})
    if not semantic_cluster_ids:
        logger.info("No stable first-stage semantic clusters found; falling back to one-stage hybrid clustering.")
        return cluster_embeddings(
            embeddings=hybrid_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_epsilon=cluster_epsilon,
            max_cluster_size=view_max_cluster_size,
        )

    for semantic_cluster_id in semantic_cluster_ids:
        semantic_indices = np.where(semantic_labels == semantic_cluster_id)[0]
        semantic_count = len(semantic_indices)
        logger.info("First-stage semantic cluster %s contains %s images", semantic_cluster_id, semantic_count)

        if semantic_count < min_cluster_size:
            continue

        if strict_same_corner_items:
            if item_features is None or len(item_features) != len(clip_embeddings):
                raise ValueError("Strict same-corner+items mode requires item features for every embedded image.")

            refined_pairs = maybe_split_quad_cluster(
                [image_paths[index] for index in semantic_indices],
                logger,
            )
            if refined_pairs is not None:
                for pair_indices in refined_pairs:
                    cluster_indices = semantic_indices[pair_indices]
                    final_labels[cluster_indices] = next_label
                    logger.info(
                        "Semantic cluster %s -> strict paired cluster %s contains %s images",
                        semantic_cluster_id,
                        next_label,
                        len(cluster_indices),
                    )
                    next_label += 1
                continue

            strict_refinement = strict_same_corner_item_clusters(
                image_paths=[image_paths[index] for index in semantic_indices],
                clip_embeddings=clip_embeddings[semantic_indices],
                item_features=item_features[semantic_indices],
                min_cluster_size=min_cluster_size,
                view_similarity_threshold=view_similarity_threshold,
                item_similarity_threshold=item_similarity_threshold,
                semantic_similarity_floor=semantic_similarity_floor,
                strict_cluster_threshold=strict_cluster_threshold,
                logger=logger,
            )
            if strict_refinement is None:
                continue

            strict_clusters, strict_noise = strict_refinement
            for strict_cluster in strict_clusters:
                cluster_indices = semantic_indices[strict_cluster]
                final_labels[cluster_indices] = next_label
                logger.info(
                    "Semantic cluster %s -> strict cluster %s contains %s images",
                    semantic_cluster_id,
                    next_label,
                    len(cluster_indices),
                )
                next_label += 1
            if strict_noise:
                final_labels[semantic_indices[strict_noise]] = -1
            continue

        semantic_groups: list[dict] = []
        local_labels = cluster_embeddings(
            embeddings=hybrid_embeddings[semantic_indices],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_epsilon=cluster_epsilon,
            max_cluster_size=view_max_cluster_size,
        )
        local_cluster_ids = sorted({int(label) for label in local_labels if int(label) != -1})

        non_noise_count = int(np.sum(local_labels != -1))
        if not local_cluster_ids or (len(local_cluster_ids) == 1 and non_noise_count == semantic_count):
            refined_pairs = maybe_split_quad_cluster(
                [image_paths[index] for index in semantic_indices],
                logger,
            )
            if refined_pairs is not None:
                for pair_indices in refined_pairs:
                    semantic_groups.append(
                        {
                            "indices": semantic_indices[pair_indices].astype(int, copy=False),
                            "frozen": True,
                        }
                    )

        if semantic_groups:
            pass
        elif not local_cluster_ids:
            logger.info(
                "Semantic cluster %s could not be split by viewpoint; keeping it as one cluster.",
                semantic_cluster_id,
            )
            semantic_groups.append({"indices": semantic_indices.astype(int, copy=False), "frozen": False})
        else:
            for local_cluster_id in local_cluster_ids:
                local_mask = local_labels == local_cluster_id
                local_indices = semantic_indices[local_mask]

                refined_large_cluster = maybe_refine_broad_viewpoint_cluster(
                    [image_paths[index] for index in local_indices],
                    similarity_threshold=view_similarity_threshold,
                    logger=logger,
                )
                if refined_large_cluster is not None:
                    refined_subclusters, refined_noise = refined_large_cluster
                    for refined_group in refined_subclusters:
                        semantic_groups.append(
                            {
                                "indices": local_indices[refined_group].astype(int, copy=False),
                                "frozen": False,
                            }
                        )
                    if refined_noise:
                        final_labels[local_indices[refined_noise]] = -1
                    continue

                semantic_groups.append({"indices": local_indices.astype(int, copy=False), "frozen": False})

        merged_groups = merge_semantic_subclusters(
            semantic_groups=semantic_groups,
            clip_embeddings=clip_embeddings,
            semantic_merge_threshold=semantic_merge_threshold,
            logger=logger,
        )

        for merged_group in merged_groups:
            final_labels[merged_group] = next_label
            logger.info(
                "Semantic cluster %s -> viewpoint cluster %s contains %s images",
                semantic_cluster_id,
                next_label,
                len(merged_group),
            )
            next_label += 1

    return final_labels


def similarity_to_percent(score: float) -> float:
    return round(float(np.clip(score, 0.0, 1.0)) * 100.0, 2)


def build_match_scores_payload(
    image_paths: list[Path],
    labels: np.ndarray,
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    strict_same_corner_items: bool,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    logger: logging.Logger,
) -> dict:
    if not image_paths:
        return {"mode": "strict" if strict_same_corner_items else "default", "images": []}

    semantic_similarity = np.clip(clip_embeddings @ clip_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    hybrid_similarity = np.clip(hybrid_embeddings @ hybrid_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    viewpoint_similarity = viewpoint_similarity_matrix(image_paths, logger)
    if viewpoint_similarity is None:
        viewpoint_similarity = np.eye(len(image_paths), dtype=np.float32)
    else:
        viewpoint_similarity = np.clip(viewpoint_similarity, -1.0, 1.0).astype(np.float32, copy=False)

    item_similarity: np.ndarray | None = None
    if item_features is not None and len(item_features) == len(image_paths):
        item_similarity = np.clip(item_features @ item_features.T, -1.0, 1.0).astype(np.float32, copy=False)

    images_payload: list[dict] = []
    for image_index, image_path in enumerate(image_paths):
        matches: list[dict] = []
        for other_index, other_path in enumerate(image_paths):
            if image_index == other_index:
                continue

            semantic_score = float(np.clip(semantic_similarity[image_index, other_index], 0.0, 1.0))
            hybrid_score = float(np.clip(hybrid_similarity[image_index, other_index], 0.0, 1.0))
            viewpoint_score = float(np.clip(viewpoint_similarity[image_index, other_index], 0.0, 1.0))
            item_score = None
            if item_similarity is not None:
                item_score = float(np.clip(item_similarity[image_index, other_index], 0.0, 1.0))

            if strict_same_corner_items and item_score is not None:
                match_score = (
                    0.50 * viewpoint_score
                    + 0.30 * item_score
                    + 0.20 * semantic_score
                )
                passes_thresholds = (
                    viewpoint_score >= view_similarity_threshold
                    and item_score >= item_similarity_threshold
                    and semantic_score >= semantic_similarity_floor
                )
            else:
                match_score = (
                    0.55 * hybrid_score
                    + 0.30 * viewpoint_score
                    + 0.15 * semantic_score
                )
                passes_thresholds = None

            match_payload = {
                "image": other_path.name,
                "match_percent": similarity_to_percent(match_score),
                "same_cluster": bool(int(labels[image_index]) == int(labels[other_index]) and int(labels[image_index]) != -1),
                "semantic_percent": similarity_to_percent(semantic_score),
                "hybrid_percent": similarity_to_percent(hybrid_score),
                "viewpoint_percent": similarity_to_percent(viewpoint_score),
            }
            if item_score is not None:
                match_payload["item_percent"] = similarity_to_percent(item_score)
            if passes_thresholds is not None:
                match_payload["passes_strict_thresholds"] = passes_thresholds

            matches.append(match_payload)

        matches.sort(key=lambda item: item["match_percent"], reverse=True)
        images_payload.append(
            {
                "image": image_path.name,
                "cluster_id": int(labels[image_index]),
                "matches": matches,
            }
        )

    return {
        "mode": "strict" if strict_same_corner_items else "default",
        "images": images_payload,
    }


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_yolo_model(logger: logging.Logger) -> object:
    global _YOLO_MODEL

    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install it with `pip install ultralytics`.")
    if "-seg" not in YOLO_MODEL_NAME:
        raise RuntimeError(f"YOLO model must be a segmentation model. Got: {YOLO_MODEL_NAME}")

    if _YOLO_MODEL is None:
        logger.info("Loading YOLO model %s", YOLO_MODEL_NAME)
        try:
            _YOLO_MODEL = YOLO(YOLO_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                "Could not load YOLO segmentation model `yolov8n-seg.pt`. Ultralytics will download it automatically "
                "when the weight is not cached locally. Download the weight once or place `yolov8n-seg.pt` in the working directory, "
                "then rerun the clustering command."
            ) from exc

    return _YOLO_MODEL


def load_scene_labeler_runtime(
    model_name: str,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> SceneLabelerRuntime:
    clip, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )

    prompt_texts: list[str] = []
    label_to_prompt_indices: dict[str, list[int]] = {}
    for label, prompts in SCENE_LABEL_PROMPTS.items():
        label_to_prompt_indices[label] = list(range(len(prompt_texts), len(prompt_texts) + len(prompts)))
        prompt_texts.extend(prompts)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompt_texts).to(device)
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return SceneLabelerRuntime(
        model=model,
        preprocess=preprocess,
        device=device,
        text_features=text_features,
        label_to_prompt_indices=label_to_prompt_indices,
    )


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    return x1, y1, x2, y2


def box_to_boundary(box: tuple[int, int, int, int]) -> Boundary:
    x1, y1, x2, y2 = box
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def normalize_boundary_points(
    points: np.ndarray | list[tuple[float, float]] | list[list[float]],
    width: int,
    height: int,
) -> Boundary:
    normalized: list[tuple[int, int]] = []
    for point in points:
        x = max(0, min(int(round(float(point[0]))), width - 1))
        y = max(0, min(int(round(float(point[1]))), height - 1))
        candidate = (x, y)
        if normalized and normalized[-1] == candidate:
            continue
        normalized.append(candidate)

    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()
    if len(normalized) < 3:
        return ()
    return tuple(normalized)


def contour_to_boundary(contour: np.ndarray, width: int, height: int) -> Boundary:
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(2.0, perimeter * 0.01)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = approx.reshape(-1, 2) if len(approx) >= 3 else contour.reshape(-1, 2)
    boundary = normalize_boundary_points(polygon, width, height)
    if boundary:
        return boundary

    x, y, w, h = cv2.boundingRect(contour)
    fallback_box = clamp_box(x, y, x + w - 1, y + h - 1, width, height)
    if fallback_box is None:
        return ()
    return box_to_boundary(fallback_box)


def boundary_to_cv_points(boundary: Boundary) -> np.ndarray:
    return np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))


def serialize_box(box: tuple[int, int, int, int]) -> list[int]:
    return [int(value) for value in box]


def serialize_boundary(boundary: Boundary) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in boundary]


def serialize_labels(labels: tuple[tuple[str, float], ...] | list[tuple[str, float]]) -> list[dict[str, object]]:
    return [{"label": label, "score": round(float(score), 4)} for label, score in labels]


def annotation_color(annotation: ImageAnnotation) -> tuple[int, int, int]:
    if annotation.source == "yolo":
        return (0, 255, 0)

    primary_label = annotation.labels[0][0] if annotation.labels else ""
    return SCENE_LABEL_COLORS.get(primary_label, (255, 170, 0))


def serialize_annotation(annotation: ImageAnnotation) -> dict[str, object]:
    return {
        "source": annotation.source,
        "box": serialize_box(annotation.box),
        "boundary": serialize_boundary(annotation.boundary),
        "labels": serialize_labels(annotation.labels),
    }


def mask_to_scene_candidates(
    mask: np.ndarray,
    width: int,
    height: int,
    allowed_labels: tuple[str, ...],
    min_area_ratio: float,
    pad_ratio: float = 0.02,
    max_candidates: int = 4,
) -> list[SceneRegionCandidate]:
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    if mask_u8.ndim != 2:
        raise ValueError("Scene candidate masks must be single-channel.")

    kernel_w = max(3, width // 80)
    kernel_h = max(3, height // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(1, int(width * height * min_area_ratio))
    pad_x = max(4, int(width * pad_ratio))
    pad_y = max(4, int(height * pad_ratio))

    candidates: list[SceneRegionCandidate] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:max_candidates]:
        if cv2.contourArea(contour) < min_area:
            continue

        boundary = contour_to_boundary(contour, width, height)
        if not boundary:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        box = clamp_box(x - pad_x, y - pad_y, x + w - 1 + pad_x, y + h - 1 + pad_y, width, height)
        if box is not None:
            candidates.append(SceneRegionCandidate(box=box, boundary=boundary, allowed_labels=allowed_labels))

    return candidates


def generate_scene_region_candidates(image: np.ndarray) -> list[SceneRegionCandidate]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)

    row_grid = np.arange(height, dtype=np.int16).reshape(-1, 1)
    top_region = row_grid < int(height * 0.72)
    upper_region = row_grid < int(height * 0.86)
    mid_upper_region = (row_grid > int(height * 0.12)) & upper_region

    sky_mask = (blue > green + 12) & (blue > red + 18) & (blue > 105) & top_region
    vegetation_mask = (green > red + 10) & (green > blue + 5) & (green > 70)
    view_seed_mask = sky_mask | vegetation_mask
    view_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 28), max(9, height // 24)))
    expanded_view_mask = cv2.dilate(view_seed_mask.astype(np.uint8) * 255, view_kernel, iterations=1) > 0
    wall_mask = (value > 190) & (saturation < 32) & mid_upper_region & ~expanded_view_mask

    candidates: list[SceneRegionCandidate] = []
    candidates.extend(
        mask_to_scene_candidates(
            expanded_view_mask,
            width=width,
            height=height,
            allowed_labels=("window", "balcony", "outdoor view"),
            min_area_ratio=0.01,
            max_candidates=4,
        )
    )
    candidates.extend(
        mask_to_scene_candidates(
            sky_mask,
            width=width,
            height=height,
            allowed_labels=("sky", "outdoor view"),
            min_area_ratio=0.005,
            max_candidates=5,
        )
    )
    candidates.extend(
        mask_to_scene_candidates(
            vegetation_mask,
            width=width,
            height=height,
            allowed_labels=("green grass", "trees", "outdoor view"),
            min_area_ratio=0.004,
            max_candidates=6,
        )
    )
    candidates.extend(
        mask_to_scene_candidates(
            wall_mask,
            width=width,
            height=height,
            allowed_labels=("white wall",),
            min_area_ratio=0.05,
            max_candidates=4,
        )
    )

    fixed_boxes = [
        (clamp_box(0, int(height * 0.55), width - 1, height - 1, width, height), ("floor tiles",)),
        (clamp_box(0, int(height * 0.65), int(width * 0.58), height - 1, width, height), ("floor tiles",)),
        (clamp_box(int(width * 0.42), int(height * 0.65), width - 1, height - 1, width, height), ("floor tiles",)),
    ]

    for box, allowed_labels in fixed_boxes:
        if box is not None:
            candidates.append(
                SceneRegionCandidate(
                    box=box,
                    boundary=box_to_boundary(box),
                    allowed_labels=allowed_labels,
                )
            )

    deduped_candidates: list[SceneRegionCandidate] = []
    seen: set[tuple[tuple[int, int, int, int], Boundary, tuple[str, ...]]] = set()
    for candidate in candidates:
        key = (candidate.box, candidate.boundary, candidate.allowed_labels)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    return deduped_candidates


def compute_box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def detect_scene_regions(
    image_rgb: Image.Image,
    image_bgr: np.ndarray,
    scene_labeler: SceneLabelerRuntime,
) -> list[SceneDetectionGroup]:
    image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    candidates = generate_scene_region_candidates(image_bgr)
    if not candidates:
        return []

    crop_tensors: list[torch.Tensor] = []
    valid_candidates: list[SceneRegionCandidate] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box
        crop = image_rgb.crop((x1, y1, x2 + 1, y2 + 1))
        if crop.width < 12 or crop.height < 12:
            continue
        crop_tensors.append(scene_labeler.preprocess(crop))
        valid_candidates.append(candidate)

    if not crop_tensors:
        return []

    batch_tensor = torch.stack(crop_tensors).to(scene_labeler.device)
    with torch.no_grad():
        crop_features = scene_labeler.model.encode_image(batch_tensor).float()
        crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarity = crop_features @ scene_labeler.text_features.T

    label_detections: list[SceneLabelDetection] = []
    for row, candidate in zip(similarity, valid_candidates):
        x1, y1, x2, y2 = candidate.box
        area_ratio = float((x2 - x1) * (y2 - y1)) / image_area
        for label in candidate.allowed_labels:
            prompt_indices = scene_labeler.label_to_prompt_indices[label]
            score = float(row[prompt_indices].max().item())
            if score >= SCENE_LABEL_THRESHOLDS[label] and area_ratio <= SCENE_LABEL_MAX_AREA_RATIO[label]:
                label_detections.append(
                    SceneLabelDetection(
                        label=label,
                        score=score,
                        box=candidate.box,
                        boundary=candidate.boundary,
                    )
                )

    kept_detections: list[SceneLabelDetection] = []
    for label in SCENE_LABEL_PROMPTS:
        per_label = sorted(
            (detection for detection in label_detections if detection.label == label),
            key=lambda detection: detection.score,
            reverse=True,
        )
        for detection in per_label:
            if any(compute_box_iou(detection.box, kept.box) > 0.45 for kept in kept_detections if kept.label == label):
                continue
            kept_detections.append(detection)

    merged_groups: list[SceneDetectionGroup] = []
    for detection in sorted(kept_detections, key=lambda item: item.score, reverse=True):
        matched_group = None
        for group in merged_groups:
            if compute_box_iou(detection.box, group.box) > 0.82:
                matched_group = group
                break

        if matched_group is None:
            merged_groups.append(
                SceneDetectionGroup(
                    box=detection.box,
                    boundary=detection.boundary,
                    labels=[(detection.label, detection.score)],
                )
            )
            continue

        if detection.label not in {label for label, _ in matched_group.labels}:
            matched_group.labels.append((detection.label, detection.score))

    for group in merged_groups:
        group.labels.sort(key=lambda item: item[1], reverse=True)

    return merged_groups


def draw_labeled_boundary(
    image: np.ndarray,
    boundary: Boundary,
    box: tuple[int, int, int, int],
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    x1, y1, _, _ = box
    image_height, image_width = image.shape[:2]
    short_side = min(image_height, image_width)
    boundary_thickness = max(2, int(round(short_side / 900)))
    draw_boundary = boundary if boundary else box_to_boundary(box)
    cv2.polylines(
        image,
        [boundary_to_cv_points(draw_boundary)],
        isClosed=True,
        color=color,
        thickness=boundary_thickness,
        lineType=cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = min(1.3, max(0.9, short_side / 2400.0))
    text_thickness = max(2, int(round(short_side / 1200)))
    outline_thickness = text_thickness + 2
    line_gap = max(10, int(round(short_side / 220)))
    pad_x = max(12, int(round(short_side / 180)))
    pad_y = max(10, int(round(short_side / 220)))

    text_sizes = [cv2.getTextSize(line, font, font_scale, text_thickness)[0] for line in lines]
    text_width = max((size[0] for size in text_sizes), default=0)
    text_height = sum(size[1] for size in text_sizes) + line_gap * max(0, len(text_sizes) - 1)
    panel_width = text_width + (pad_x * 2)
    panel_height = text_height + (pad_y * 2)

    bg_x1 = min(max(0, x1), max(0, image_width - panel_width - 1))
    bg_y1 = y1 - panel_height - 10
    if bg_y1 < 0:
        bg_y1 = y1 + 10
    bg_y1 = min(max(0, bg_y1), max(0, image_height - panel_height - 1))
    bg_x2 = min(image_width - 1, bg_x1 + panel_width)
    bg_y2 = min(image_height - 1, bg_y1 + panel_height)

    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (24, 24, 24), thickness=-1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, max(2, boundary_thickness))

    text_y = bg_y1 + pad_y + text_sizes[0][1]
    for line, size in zip(lines, text_sizes):
        origin = (bg_x1 + pad_x, text_y)
        cv2.putText(
            image,
            line,
            origin,
            font,
            font_scale,
            (16, 16, 16),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            origin,
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        text_y += size[1] + line_gap


def draw_segmentation_boundaries(
    image_path: Path,
    output_path: Path,
    mask_output_dir: Path,
    yolo_model: object,
) -> list[str]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for segmentation: {image_path}")

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    results = yolo_model(str(image_path), verbose=False)
    saved_masks: list[str] = []
    mask_index = 0

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        for index, mask in enumerate(masks):
            mask_u8 = (mask * 255).astype(np.uint8)
            mask_name = f"{image_path.stem}_mask_{mask_index:03d}.png"
            mask_path = mask_output_dir / mask_name
            if not cv2.imwrite(str(mask_path), mask_u8):
                raise RuntimeError(f"Could not write segmentation mask: {mask_path}")
            saved_masks.append(mask_name)
            mask_index += 1

            if index >= len(boxes):
                continue

            conf = float(boxes.conf[index]) if getattr(boxes, "conf", None) is not None else 1.0
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cls = int(boxes.cls[index])
            label = result.names[cls]

            for contour in contours:
                if cv2.contourArea(contour) < SEGMENTATION_MIN_CONTOUR_AREA:
                    continue

                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                x, y = contour[0][0]
                cv2.putText(
                    image,
                    label,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write segmented output image: {output_path}")

    return saved_masks


def copy_clustered_images(
    image_paths: list[Path],
    labels: np.ndarray,
    image_tags: list[list[str]],
    output_dir: Path,
    yolo_model: object,
) -> dict:
    if len(image_paths) != len(image_tags):
        raise ValueError("Image tags must align with image paths.")

    result: dict[str, list[dict[str, object]]] = {}
    masks_root_dir = output_dir / "masks"
    masks_root_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(int(label) for label in labels))
    for label in unique_labels:
        members = [
            (path, image_tags[index])
            for index, (path, cluster_label) in enumerate(zip(image_paths, labels))
            if int(cluster_label) == label
        ]
        if not members:
            continue

        if label == -1:
            noise_dir = output_dir / "noise"
            noise_dir.mkdir(parents=True, exist_ok=True)
            noise_payload: list[dict[str, object]] = []
            for image_path, tags in members:
                image_output_path = noise_dir / image_path.name
                image_mask_dir = masks_root_dir / "noise" / image_path.stem
                saved_masks = draw_segmentation_boundaries(image_path, image_output_path, image_mask_dir, yolo_model)
                noise_payload.append(
                    {
                        "image": image_path.name,
                        "tags": tags,
                        "mask_dir": str(image_mask_dir.relative_to(output_dir)),
                        "mask_files": saved_masks,
                    }
                )
            result["noise"] = noise_payload
            continue

        cluster_dir = output_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        cluster_payload: list[dict[str, object]] = []
        for image_path, tags in members:
            image_output_path = cluster_dir / image_path.name
            image_mask_dir = masks_root_dir / f"cluster_{label}" / image_path.stem
            saved_masks = draw_segmentation_boundaries(image_path, image_output_path, image_mask_dir, yolo_model)
            cluster_payload.append(
                {
                    "image": image_path.name,
                    "tags": tags,
                    "mask_dir": str(image_mask_dir.relative_to(output_dir)),
                    "mask_files": saved_masks,
                }
            )

        result[f"cluster_{label}"] = cluster_payload

    (output_dir / "clusters.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_match_scores(
    image_paths: list[Path],
    labels: np.ndarray,
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    strict_same_corner_items: bool,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    payload = build_match_scores_payload(
        image_paths=image_paths,
        labels=labels,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=hybrid_embeddings,
        item_features=item_features,
        strict_same_corner_items=strict_same_corner_items,
        view_similarity_threshold=view_similarity_threshold,
        item_similarity_threshold=item_similarity_threshold,
        semantic_similarity_floor=semantic_similarity_floor,
        logger=logger,
    )
    (output_dir / "match_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    cache_dir = Path(".clip-cache").resolve()
    device = resolve_device(args.device)

    logger.info("Reading images from %s", input_dir)
    image_paths = discover_images(input_dir)
    if not image_paths:
        raise SystemExit("No supported images found in input folder.")

    logger.info("Found %s images", len(image_paths))
    clip_embeddings, embedded_paths, image_tags = embed_images(
        image_paths=image_paths,
        model_name=args.model,
        batch_size=max(1, args.batch_size),
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )

    visual_features, visual_paths = extract_visual_features(image_paths, logger)
    item_features: np.ndarray | None = None
    if args.strict_same_corner_items:
        logger.info("Extracting CLIP prompt signatures for strict same-corner+items mode")
        item_features, item_paths = extract_clip_item_features(
            image_paths=embedded_paths,
            model_name=args.model,
            batch_size=max(1, args.batch_size),
            device=device,
            cache_dir=cache_dir,
            logger=logger,
        )
        if embedded_paths != item_paths:
            path_set = set(embedded_paths) & set(item_paths)
            embedded_indices = [index for index, path in enumerate(embedded_paths) if path in path_set]
            item_indices = [index for index, path in enumerate(item_paths) if path in path_set]
            clip_embeddings = clip_embeddings[embedded_indices]
            embedded_paths = [embedded_paths[index] for index in embedded_indices]
            image_tags = [image_tags[index] for index in embedded_indices]
            item_features = item_features[item_indices]

    if embedded_paths != visual_paths:
        path_set = set(embedded_paths) & set(visual_paths)
        embedded_indices = [index for index, path in enumerate(embedded_paths) if path in path_set]
        visual_indices = [index for index, path in enumerate(visual_paths) if path in path_set]
        clip_embeddings = clip_embeddings[embedded_indices]
        visual_features = visual_features[visual_indices]
        embedded_paths = [embedded_paths[index] for index in embedded_indices]
        image_tags = [image_tags[index] for index in embedded_indices]
        if item_features is not None:
            item_features = item_features[embedded_indices]

    embeddings = combine_features(
        clip_embeddings=clip_embeddings,
        visual_features=visual_features,
        semantic_weight=args.semantic_weight,
        layout_weight=args.layout_weight,
        edge_weight=args.edge_weight,
        color_weight=args.color_weight,
    )
    logger.info(
        "Using feature weights semantic=%.2f layout=%.2f edge=%.2f color=%.2f",
        args.semantic_weight,
        args.layout_weight,
        args.edge_weight,
        args.color_weight,
    )

    logger.info("Clustering %s embedded images with two-stage HDBSCAN", len(embedded_paths))
    labels = cluster_same_corner_groups(
        image_paths=embedded_paths,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=embeddings,
        item_features=item_features,
        min_cluster_size=max(2, args.min_cluster_size),
        min_samples=max(1, args.min_samples),
        cluster_epsilon=args.cluster_epsilon,
        view_max_cluster_size=args.view_max_cluster_size,
        view_similarity_threshold=args.view_similarity_threshold,
        semantic_merge_threshold=args.semantic_merge_threshold,
        strict_same_corner_items=args.strict_same_corner_items,
        item_similarity_threshold=args.item_similarity_threshold,
        strict_cluster_threshold=args.strict_cluster_threshold,
        semantic_similarity_floor=args.semantic_similarity_floor,
        logger=logger,
    )

    yolo_model = load_yolo_model(logger)
    reset_output_dir(output_dir)
    for image_path, tags in zip(embedded_paths, image_tags):
        print(f"{image_path.name} -> {tags}")

    result = copy_clustered_images(embedded_paths, labels, image_tags, output_dir, yolo_model)
    write_match_scores(
        image_paths=embedded_paths,
        labels=labels,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=embeddings,
        item_features=item_features,
        strict_same_corner_items=args.strict_same_corner_items,
        view_similarity_threshold=args.view_similarity_threshold,
        item_similarity_threshold=args.item_similarity_threshold,
        semantic_similarity_floor=args.semantic_similarity_floor,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Clusters written to %s", output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
