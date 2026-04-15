from __future__ import annotations

import argparse
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


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ORB_EXTRACTOR = cv2.ORB_create(1200)
ORB_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        default=0.95,
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


def embed_images(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[Path]]:
    clip, clip_source = load_clip_module()

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded CLIP from %s", clip_source)
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root=str(cache_dir))
    model.eval()

    embedded_paths: list[Path] = []
    batches: list[np.ndarray] = []

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
            batch_features = model.encode_image(batch_tensor).float().cpu().numpy()
            batches.append(batch_features)
            embedded_paths.extend(batch_valid_paths)
            logger.info("Embedded %s/%s images", len(embedded_paths), len(image_paths))

    if not batches:
        return np.empty((0, 0), dtype=np.float32), []

    return l2_normalize(np.concatenate(batches, axis=0).astype(np.float32, copy=False)), embedded_paths


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


def copy_clustered_images(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
) -> dict:
    clusters_payload: list[dict] = []
    noise_images: list[str] = []

    unique_labels = sorted(set(int(label) for label in labels))
    for label in unique_labels:
        members = [path for path, cluster_label in zip(image_paths, labels) if int(cluster_label) == label]
        if not members:
            continue

        if label == -1:
            noise_dir = output_dir / "noise"
            noise_dir.mkdir(parents=True, exist_ok=True)
            for image_path in members:
                shutil.copy2(image_path, noise_dir / image_path.name)
                noise_images.append(image_path.name)
            continue

        cluster_dir = output_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for image_path in members:
            shutil.copy2(image_path, cluster_dir / image_path.name)

        clusters_payload.append(
            {
                "cluster_id": label,
                "images": [path.name for path in members],
            }
        )

    result = {
        "clusters": clusters_payload,
        "noise": noise_images,
    }
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
    clip_embeddings, embedded_paths = embed_images(
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
            item_features = item_features[item_indices]

    if embedded_paths != visual_paths:
        path_set = set(embedded_paths) & set(visual_paths)
        embedded_indices = [index for index, path in enumerate(embedded_paths) if path in path_set]
        visual_indices = [index for index, path in enumerate(visual_paths) if path in path_set]
        clip_embeddings = clip_embeddings[embedded_indices]
        visual_features = visual_features[visual_indices]
        embedded_paths = [embedded_paths[index] for index in embedded_indices]
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

    reset_output_dir(output_dir)
    result = copy_clustered_images(embedded_paths, labels, output_dir)
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
