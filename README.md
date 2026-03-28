# KeypointDetector

## Overview

KeypointDetector (`kp_detection`) is a Python package that provides a unified interface for keypoint detection using various OpenCV-backed algorithms.  
For more details, see [src/kp_detection/README.md](src/kp_detection/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode so changes to the source take effect immediately:

```bash
pip install -e .
```

Dependencies (numpy, opencv-contrib-python) are installed automatically.  
To install only the dependencies without the package, use:

```bash
pip install -r requirements.txt
```

## Example

After installing the package, import it from any directory:

```python
import cv2

from kp_detection import KPDetectionMethod, KPDetectionResult

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

method = KPDetectionMethod.SIFT
params = method.parameter_class(method=method)
detector = method.detector_class(params=params)
result: KPDetectionResult = detector.detect(image)
# Use result according to KPDetectionResult / ArrayKPDetectionResult (see package README).
```

`KPDetectionMethod` maps each variant to a parameter dataclass and a detector class via `parameter_class` and `detector_class` (see `method.py`). For Harris or Shi–Tomasi, `parameter_class` resolves to `HarrisParameters` or `ShiTomashiParameters` automatically when you pass the matching `method`.
