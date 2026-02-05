# ELECT-MT — Automated Image Processing (Python + OpenCV)

This project automatically processes images dropped into an `input/` directory and writes processed outputs to `output/`.

## Tools & Technologies
- Python 3
- OpenCV (`opencv-python`)
- MediaPipe (person segmentation for background blur)
- watchdog (filesystem watch)
- pytest
- GitHub Actions (CI)

## How Image Processing Works
For each image, the app generates five outputs:
- `blur` (background-only blur via MediaPipe segmentation)
- `sharpen`
- `thermal` (colormap)
- `sepia`
- `clahe` (contrast limited adaptive histogram equalization)

Each output is saved as `inputname_filter.png` in the `output/` folder.

## How CI Works
On every push or pull request, GitHub Actions runs:
1) Checkout repository
2) Set up Python
3) Install dependencies
4) Run pytest

Workflow file: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Run Locally

### 1) Create a virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
pip install -e .
```

### 3) Put images into `input/`
Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

### 4) Run batch processing
```powershell
python -m elect_mt batch --input input --output output
```

### 5) Run watch processing
```powershell
python -m elect_mt watch --input input --output output
```

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

## Screenshots / Examples
Add your own output screenshots here after running the app.

