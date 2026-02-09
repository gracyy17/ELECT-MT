from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from elect_mt.processor import iter_image_files, is_image_path, process_directory, process_file


def _write_test_image(path: Path) -> np.ndarray:
    rng = np.random.default_rng(123)
    bgr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    ok = cv2.imwrite(str(path), bgr)
    assert ok
    return bgr


def test_process_file_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Use a lossless input format so pixel values round-trip reliably.
    src = input_dir / "sample.png"
    _write_test_image(src)

    result = process_file(src, output_dir)
    assert result is not None

    expected = {
        output_dir / "sample_background_blur.png",
        output_dir / "sample_background_removal.png",
        output_dir / "sample_color_identification.png",
        output_dir / "sample_person_detection_count.png",
    }
    assert set(result.outputs) == expected
    for p in expected:
        assert p.exists()
        assert p.stat().st_size > 0

    bg_removed = cv2.imread(str(output_dir / "sample_background_removal.png"), cv2.IMREAD_UNCHANGED)
    assert bg_removed is not None
    assert bg_removed.ndim == 3
    assert bg_removed.shape[2] in (3, 4)


def test_process_directory_ignores_non_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    (input_dir / "note.txt").write_text("not an image", encoding="utf-8")

    results = process_directory(input_dir, output_dir)
    assert results == []


def test_is_image_path_and_iter_image_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    img = input_dir / "a.png"
    txt = input_dir / "note.txt"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    txt.write_text("not an image", encoding="utf-8")

    assert is_image_path(img)
    assert not is_image_path(txt)

    images = list(iter_image_files(input_dir))
    assert images == [img]


def test_process_file_skips_corrupt_image(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    corrupt = input_dir / "bad.png"
    corrupt.write_text("not really an image", encoding="utf-8")

    result = process_file(corrupt, output_dir)
    assert result is None
