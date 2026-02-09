import os
import subprocess
import sys
import tempfile
import cv2
import numpy as np


def create_test_image(path):
    """Create a small valid image for testing."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def run_batch(input_dir, output_dir):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "elect_mt",
            "batch",
            "--input",
            input_dir,
            "--output",
            output_dir,
            "--overwrite",
        ],
        check=True,
    )


# -------------------- POSITIVE TESTS --------------------

def test_features_generate_outputs():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        img_path = os.path.join(input_dir, "test.jpg")
        create_test_image(img_path)

        run_batch(input_dir, output_dir)

        outputs = os.listdir(output_dir)
        keywords = ["color", "count", "blur", "removal"]

        for key in keywords:
            assert any(key in name.lower() for name in outputs), \
                f"Missing output for feature: {key}"


def test_outputs_are_not_empty():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        img_path = os.path.join(input_dir, "test.jpg")
        create_test_image(img_path)

        run_batch(input_dir, output_dir)

        for f in os.listdir(output_dir):
            path = os.path.join(output_dir, f)
            assert os.path.getsize(path) > 0


# -------------------- NEGATIVE TESTS --------------------

def test_non_image_files_are_ignored():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        fake = os.path.join(input_dir, "fake.txt")
        with open(fake, "w") as f:
            f.write("not an image")

        run_batch(input_dir, output_dir)

        assert len(os.listdir(output_dir)) == 0


def test_empty_input_folder():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        run_batch(input_dir, output_dir)

        assert len(os.listdir(output_dir)) == 0


# -------------------- EDGE CASES --------------------

def test_multiple_images_processed():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        for i in range(5):
            img_path = os.path.join(input_dir, f"img{i}.jpg")
            create_test_image(img_path)

        run_batch(input_dir, output_dir)

        assert len(os.listdir(output_dir)) >= 5


def test_supported_image_formats():
    with tempfile.TemporaryDirectory() as input_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        formats = ["jpg", "png", "bmp", "jpeg"]

        for fmt in formats:
            img_path = os.path.join(input_dir, f"sample.{fmt}")
            create_test_image(img_path)

        run_batch(input_dir, output_dir)

        assert len(os.listdir(output_dir)) > 0
