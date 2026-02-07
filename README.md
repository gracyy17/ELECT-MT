# ELECT-MT — Automated Image Processing (Python + OpenCV)

ELECT-MT processes images from an `input/` directory and writes processed outputs to `output/`.
When running **Watch** mode, it automatically applies image processing as soon as new images are added to `input/`.

It supports two modes:
- **Batch**: process everything currently in `input/`
- **Watch**: monitor `input/` and process new images as they arrive

## Tools & Technologies
- Python 3
- OpenCV (`opencv-python`)
- MediaPipe (person segmentation for background blur)
- watchdog (filesystem watch)
- pytest
- GitHub Actions (CI)

## Quickstart (Windows / PowerShell)

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.
.venv\Scripts\Activate.ps1
```

### 2) Install the project
Editable install (recommended for development):
```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

For consistent local/CI results, keep `yolov8n.pt` in the repo root (used by YOLO person counting).
OpenCV is pinned to 4.9.x to avoid NumPy>=2 conflicts in CI.

### 3) Run batch processing
Using the console script:
```powershell
elect-mt batch --input input --output output --overwrite
```

Or via module execution:
```powershell
python -m elect_mt batch --input input --output output --overwrite
```

### 4) Run watch mode
```powershell
elect-mt watch --input input --output output
```

## How Image Processing Works
For each image, the app generates five outputs:
- `blur` (background-only blur via MediaPipe segmentation)
- `sharpen`
- `thermal` (colormap)
- `sepia`
- `clahe` (contrast limited adaptive histogram equalization)

Each output is saved as `inputname_filter.png` in the `output/` folder.

## CLI

After `pip install -e .`, the `elect-mt` command is available.

### Batch
```powershell
elect-mt batch --input input --output output
```

Options:
- `--recursive`: process images in subfolders
- `--overwrite`: replace existing outputs

### Watch
```powershell
elect-mt watch --input input --output output
```

Options:
- `--recursive`: watch subfolders
- `--overwrite`: replace existing outputs

## How CI Works
On every push or pull request, GitHub Actions runs:
1) Checkout repository
2) Set up Python
3) Install dependencies
4) Run pytest

Workflow file: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Inputs

Put images into `input/`.

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

## Output Naming
For an input file `cat.jpg`, outputs are written as:
- `cat_blur.png`
- `cat_sharpen.png`
- `cat_thermal.png`
- `cat_sepia.png`
- `cat_clahe.png`

## Testing
```powershell
python -m pytest
```

## Git hygiene (important)

Do not commit virtual environments.

This repo’s `.gitignore` ignores `.venv/`, `.venv*/`, `venv/`, and `venv*/` to prevent accidentally pushing large binary files (which GitHub will reject).

## Screenshots / Examples
Add your own output screenshots here after running the app.

