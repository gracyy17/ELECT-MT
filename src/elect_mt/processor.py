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
        _segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
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


def _safe_process_segmentation(bgr: np.ndarray, model) -> np.ndarray | None:
    if model is None:
        return None

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
        return None
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


def _to_object_detection_count(bgr: np.ndarray, model=None) -> np.ndarray:
    if model is None:
        model = _get_segmentation_model()

    draw_boxes: list[tuple[int, int, int, int]] = []

    yolo = _get_yolo_model()
    if yolo is not None:
        try:
            predict = getattr(yolo, "predict", None)
            if predict is None:
                raise AttributeError("YOLO model has no predict method")
            results = predict(bgr, verbose=False, conf=0.25)
            if results:
                result = results[0]
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is not None:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()
                    person_class = None
                    for key, value in names.items():
                        if value == "person":
                            person_class = int(key)
                            break
                    if person_class is None:
                        person_class = 0
                    person_mask = classes == person_class
                    count = int(np.sum(person_mask))
                    out = bgr.copy()
                    for box in xyxy[person_mask]:
                        x1, y1, x2, y2 = [int(v) for v in box]
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
        except Exception as exc:
            logger.warning("YOLO inference failed; falling back. (%s)", exc)

    hog = cv2.HOGDescriptor()
    people_detector = getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector", None)
    if people_detector is None:
        logger.warning("HOG people detector is unavailable.")
    else:
        hog.setSVMDetector(people_detector())

    def _nms(boxes, scores, iou_thresh=0.6):
        if not boxes:
            return []
        order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        kept = []

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / max(1e-6, area_a + area_b - inter)

        while order:
            i = order.pop(0)
            kept.append(boxes[i])
            order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
        return kept

    count = 0

    if people_detector is not None:
        h, w = bgr.shape[:2]
        scale = 1.0
        max_side = max(h, w)

        if max_side < 600:
            scale = 800 / max_side
        elif max_side > 1200:
            scale = 1200 / max_side

        resized = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else bgr

        rects, weights = hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )

        boxes, scores = [], []
        min_h = int(resized.shape[0] * 0.30)

        for i, (x, y, rw, rh) in enumerate(rects):
            score = float(weights[i]) if i < len(weights) else 1.0
            if score < 0.8:
                continue
            if rh < min_h:
                continue
            aspect = rh / max(1, rw)
            if not (1.5 <= aspect <= 4.0):
                continue

            boxes.append((x, y, x + rw, y + rh))
            scores.append(score)

        draw_boxes = _nms(boxes, scores, 0.6)
        count = len(draw_boxes)

    if count == 0:
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
            draw_boxes = [(x, y, x + fw, y + fh) for (x, y, fw, fh) in faces]
            count = len(draw_boxes)

    if count == 0:
        seg_mask = _safe_process_segmentation(bgr, model)
        if seg_mask is not None:
            mask = (seg_mask > 0.6).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            num_labels, _ = cv2.connectedComponents(mask)
            count = max(0, num_labels - 1)

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
        "object_detection_count": _to_object_detection_count(bgr, model),
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
