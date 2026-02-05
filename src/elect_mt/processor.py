from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from .config import OUTPUT_VARIANTS, SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ProcessResult:
    input_path: Path
    outputs: list[Path]


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


_segmentation_model: object | None = None


def _get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is not None:
        return _segmentation_model

    try:
        import mediapipe as mp
        _segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        logger.info("Loaded MediaPipe SelfieSegmentation model.")
    except Exception as exc:
        logger.warning("Segmentation model unavailable: %s", exc)
        _segmentation_model = None

    return _segmentation_model


def blur_background_only(bgr: np.ndarray, model) -> np.ndarray:
    if model is None:
        return bgr  # skip blur if model unavailable

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = model.process(rgb)

    if result.segmentation_mask is None:
        logger.warning("Segmentation mask missing; skipping background blur.")
        return bgr

    mask = result.segmentation_mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_3 = np.stack([mask] * 3, axis=-1)

    foreground = mask_3  # person remains sharp
    background = 1.0 - foreground  # blur background

    blurred = cv2.GaussianBlur(bgr, (91, 91), 0)

    output = bgr.astype(np.float32) * foreground + blurred.astype(np.float32) * background
    return np.clip(output, 0, 255).astype(np.uint8)


def _to_blur(bgr: np.ndarray, model=None) -> np.ndarray:
    if model is None:
        model = _get_segmentation_model()
    return blur_background_only(bgr, model)


def _to_sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(bgr, -1, kernel)


def _to_thermal(bgr: np.ndarray) -> np.ndarray:
    gray = _to_gray(bgr)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def _to_sepia(bgr: np.ndarray) -> np.ndarray:
    bgr_float = bgr.astype(np.float32)
    transform = np.array([[0.131, 0.534, 0.272],
                          [0.168, 0.686, 0.349],
                          [0.189, 0.769, 0.393]], dtype=np.float32)
    sepia = cv2.transform(bgr_float, transform)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def _to_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def process_image_variants(bgr: np.ndarray, model=None) -> dict[str, np.ndarray]:
    return {
        "blur": _to_blur(bgr, model),
        "sharpen": _to_sharpen(bgr),
        "thermal": _to_thermal(bgr),
        "sepia": _to_sepia(bgr),
        "clahe": _to_clahe(bgr),
    }


def process_file(input_path: Path, output_dir: Path, *, overwrite: bool = False, model=None) -> ProcessResult | None:
    if not is_image_path(input_path):
        return None

    _ensure_dir(output_dir)

    try:
        bgr = _load_bgr(input_path)
    except ValueError as exc:
        logger.warning("Skipping unreadable image: %s (%s)", input_path, exc)
        return None

    variants = process_image_variants(bgr, model=model)
    outputs: list[Path] = []

    for variant_name in OUTPUT_VARIANTS:
        if variant_name not in variants:
            continue
        output_path = output_dir / f"{input_path.stem}_{variant_name}.png"
        if output_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        ok = cv2.imwrite(str(output_path), variants[variant_name])
        if not ok:
            raise IOError(f"Failed to write output image: {output_path}")
        outputs.append(output_path)

    return ProcessResult(input_path=input_path, outputs=outputs)


def iter_image_files(input_dir: Path, *, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in input_dir.rglob("*") if is_image_path(p))
        return
    yield from (p for p in input_dir.glob("*") if is_image_path(p))


def process_directory(input_dir: Path, output_dir: Path, *, recursive: bool = False, overwrite: bool = False) -> list[ProcessResult]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    model = _get_segmentation_model()  # load model once
    results: list[ProcessResult] = []

    for image_path in sorted(iter_image_files(input_dir, recursive=recursive)):
        result = process_file(image_path, output_dir, overwrite=overwrite, model=model)
        if result is not None:
            results.append(result)
    return results
