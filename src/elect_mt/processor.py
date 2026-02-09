from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import importlib
import os
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
_yolo_model: Any | None = None


def _get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is not None:
        return _segmentation_model

    try:
        import mediapipe as mp
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            logger.warning("MediaPipe solutions module is unavailable.")
            _segmentation_model = None
            return _segmentation_model

        selfie_segmentation = getattr(solutions, "selfie_segmentation", None)
        if selfie_segmentation is None:
            logger.warning("MediaPipe selfie_segmentation is unavailable.")
            _segmentation_model = None
            return _segmentation_model

        _segmentation_model = selfie_segmentation.SelfieSegmentation(model_selection=1)
        logger.info("Loaded MediaPipe SelfieSegmentation model.")
    except Exception as exc:
        logger.warning("Segmentation model unavailable: %s", exc)
        _segmentation_model = None

    return _segmentation_model


def _get_yolo_model() -> Any | None:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    try:
        module = importlib.import_module("ultralytics")
        yolo_ctor = getattr(module, "YOLO", None)
        if yolo_ctor is None:
            raise AttributeError("ultralytics.YOLO not found")
        _yolo_model = yolo_ctor("yolov8n.pt")
        logger.info("Loaded YOLOv8n model for person detection.")
    except Exception as exc:
        logger.warning("YOLO model unavailable: %s", exc)
        _yolo_model = None

    return _yolo_model


def blur_background_only(bgr: np.ndarray, model) -> np.ndarray:
    seg_mask = _safe_process_segmentation(bgr, model)
    if seg_mask is None:
        logger.warning("Segmentation mask missing; applying full blur.")
        return cv2.GaussianBlur(bgr, (31, 31), 0)

    mask = seg_mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = np.clip((mask - 0.1) / 0.9, 0.0, 1.0)
    mask_3 = np.stack([mask] * 3, axis=-1)

    foreground = mask_3 
    background = 1.0 - foreground  # blur background

    blurred = cv2.GaussianBlur(bgr, (91, 91), 0)

    output = bgr.astype(np.float32) * foreground + blurred.astype(np.float32) * background
    return np.clip(output, 0, 255).astype(np.uint8)


def _safe_process_segmentation(bgr: np.ndarray, model) -> np.ndarray | None:
    def _grabcut_mask() -> np.ndarray | None:
        try:
            h, w = bgr.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            return fg.astype(np.float32)
        except Exception as exc:
            logger.warning("GrabCut fallback failed; skipping. (%s)", exc)
            return None

    if model is None:
        return _grabcut_mask()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    try:
        result = model.process(rgb)
    except Exception as exc:
        logger.warning("Segmentation processing failed; skipping. (%s)", exc)
        return None

    mask = getattr(result, "segmentation_mask", None)
    if mask is None:
        return _grabcut_mask()
    return mask


def _to_blur(bgr: np.ndarray, model=None) -> np.ndarray:
    if model is None:
        model = _get_segmentation_model()
    return blur_background_only(bgr, model)


def _to_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _to_background_removal(bgr: np.ndarray, model=None) -> np.ndarray:
    if model is None:
        model = _get_segmentation_model()

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

    seg_mask = _safe_process_segmentation(bgr, model)
    if seg_mask is None:
        logger.warning("Segmentation mask missing; background removal will be opaque.")
        bgra[:, :, 3] = 255
        return bgra

    mask = seg_mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = np.clip((mask - 0.1) / 0.9, 0.0, 1.0)
    alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    bgra[:, :, 3] = alpha
    return bgra


def _to_color_identification(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    banner_h = min(100, max(60, h // 6))
    flat = bgr.reshape(-1, 3)
    total = flat.shape[0]
    if total > 10000:
        step = max(1, total // 10000)
        idx = np.arange(0, total, step)
    else:
        idx = np.arange(total)
    ys = (idx // w).astype(np.int32)
    xs = (idx % w).astype(np.int32)
    visible_mask = ys >= banner_h
    if np.any(visible_mask):
        idx = idx[visible_mask]
        ys = ys[visible_mask]
        xs = xs[visible_mask]
    sample = flat[idx]

    sample = np.asarray(sample, dtype=np.float32)
    k = 8
    try:
        _criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        best_labels = np.zeros((sample.shape[0], 1), dtype=np.int32)
        _compactness, labels, centers = cv2.kmeans(
            sample, k, best_labels, _criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k)
        order = np.argsort(counts)[::-1]
        centers = centers[order]
        label_map = {int(src): int(dst) for dst, src in enumerate(order)}
        ordered_labels = np.array([label_map[int(l)] for l in labels], dtype=np.int32)
    except Exception:
        centers = np.array([sample.mean(axis=0)], dtype=np.float32)
        ordered_labels = np.zeros(sample.shape[0], dtype=np.int32)

    out = bgr.copy()
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)

    gap = 6
    swatch_w = max(10, (w - 20 - gap * (k - 1)) // k)
    x = 10
    for i, center in enumerate(centers):
        b, g, r = [int(v) for v in np.clip(center, 0, 255)]
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        swatch_top = 22
        swatch_bottom = banner_h - 10
        cv2.rectangle(out, (x, swatch_top), (x + swatch_w, swatch_bottom), (b, g, r), -1)
        text_x = x
        text_y = 14
        arrow_start = (text_x + 2, text_y + 4)
        cluster_mask = ordered_labels == i
        if np.any(cluster_mask):
            cluster_pixels = sample[cluster_mask]
            diffs = cluster_pixels - center
            dists = np.sum(diffs * diffs, axis=1)
            best_idx = int(np.argmin(dists))
            cx = int(xs[cluster_mask][best_idx])
            cy = int(ys[cluster_mask][best_idx])
            arrow_end = (cx, cy)
        else:
            arrow_end = (x + swatch_w // 2, (swatch_top + swatch_bottom) // 2)
        cv2.arrowedLine(out, arrow_start, arrow_end, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.1)
        cv2.putText(
            out,
            hex_code,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x += swatch_w + gap
    return out


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    return float(inter) / float(min(area_a, area_b))


def _suppress_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]], *, overlap_threshold: float = 0.85
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    def _area(box: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    ordered = sorted(boxes, key=_area)  # keep tighter boxes; drop big duplicates
    kept: list[tuple[int, int, int, int]] = []
    for candidate in ordered:
        if any(_overlap_ratio(candidate, existing) >= overlap_threshold for existing in kept):
            continue
        kept.append(candidate)

    return kept


def _to_object_detection_count(bgr: np.ndarray, model=None) -> np.ndarray:
    if model is None:
        model = _get_segmentation_model()

    draw_boxes: list[tuple[int, int, int, int]] = []
    count = 0

    haar_dir = getattr(getattr(cv2, "data", object()), "haarcascades", "")
    cascade_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if not face_cascade.empty():
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        draw_boxes = [(int(x), int(y), int(x + fw), int(y + fh)) for (x, y, fw, fh) in faces]
        draw_boxes = _suppress_overlapping_boxes(draw_boxes)
        count = len(draw_boxes)

    out = bgr.copy()
    for (x1, y1, x2, y2) in draw_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(
        out,
        f"persons: {count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def process_image_variants(bgr: np.ndarray, model=None) -> dict[str, np.ndarray]:
    return {
        "background_blur": _to_blur(bgr, model),
        "background_removal": _to_background_removal(bgr, model),
        "color_identification": _to_color_identification(bgr),
        "person_detection_count": _to_object_detection_count(bgr, model),
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
