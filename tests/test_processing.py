from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from elect_mt.processor import process_directory, process_file


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
    original = _write_test_image(src)

    result = process_file(src, output_dir)
    assert result is not None

    expected = {
        output_dir / "sample_blur.png",
        output_dir / "sample_sharpen.png",
        output_dir / "sample_thermal.png",
        output_dir / "sample_sepia.png",
    }
    assert set(result.outputs) == expected
    for p in expected:
        assert p.exists()
        assert p.stat().st_size > 0

    thermal = cv2.imread(str(output_dir / "sample_thermal.png"), cv2.IMREAD_COLOR)
    assert thermal is not None
    assert thermal.ndim == 3 and thermal.shape[2] == 3


def test_process_directory_ignores_non_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    (input_dir / "note.txt").write_text("not an image", encoding="utf-8")

    results = process_directory(input_dir, output_dir)
    assert results == []
