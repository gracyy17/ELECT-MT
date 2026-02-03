from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # noqa: BLE001
    mp = None

from .config import OUTPUT_VARIANTS, SUPPORTED_IMAGE_EXTENSIONS


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


_segmentation_model = None


def _get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is not None:
        return _segmentation_model
    if mp is None:
        raise RuntimeError("mediapipe is required for background blur. Install mediapipe and retry.")
    _segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return _segmentation_model


def _to_blur(bgr: np.ndarray) -> np.ndarray:
    model = _get_segmentation_model()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = model.process(rgb)
    if result.segmentation_mask is None:
        raise RuntimeError("Failed to segment person for background blur.")

    mask = result.segmentation_mask
    mask_3 = np.stack([mask, mask, mask], axis=-1)
    foreground = (mask_3 > 0.5).astype(np.float32)

    blurred = cv2.GaussianBlur(bgr, (21, 21), 0)
    bgr_f = bgr.astype(np.float32)
    blended = bgr_f * foreground + blurred.astype(np.float32) * (1.0 - foreground)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _to_sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    return cv2.filter2D(bgr, -1, kernel)


def _to_thermal(bgr: np.ndarray) -> np.ndarray:
    gray = _to_gray(bgr)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def _to_sepia(bgr: np.ndarray) -> np.ndarray:
    bgr_float = bgr.astype(np.float32)
    transform = np.array(
        [
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393],
        ],
        dtype=np.float32,
    )
    sepia = cv2.transform(bgr_float, transform)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def process_image_variants(bgr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "blur": _to_blur(bgr),
        "sharpen": _to_sharpen(bgr),
        "thermal": _to_thermal(bgr),
        "sepia": _to_sepia(bgr),
    }


def process_file(
    input_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> ProcessResult | None:
    if not is_image_path(input_path):
        return None

    _ensure_dir(output_dir)

    bgr = _load_bgr(input_path)
    variants = process_image_variants(bgr)

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


def process_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    overwrite: bool = False,
) -> list[ProcessResult]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    results: list[ProcessResult] = []
    for image_path in sorted(iter_image_files(input_dir, recursive=recursive)):
        result = process_file(image_path, output_dir, overwrite=overwrite)
        if result is not None:
            results.append(result)
    return results
