"""
Computer Vision Preprocessing & Image Utilities.
Provides CLAHE contrast enhancement, focus sharpness scoring,
saliency contour cropping, and base64 payload converters.
"""

import base64
import io
from typing import Tuple, Optional
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


def enhance_contrast_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB Luminance channel.
    Significantly improves text readability on medicine boxes and low-contrast circuit traces.
    """
    if cv2 is None or img_bgr is None:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def compute_sharpness_score(img_bgr: np.ndarray) -> float:
    """
    Calculates the variance of the Laplacian to evaluate image focus / blurriness.
    Values > 100 indicate crisp focus; < 50 indicates motion blur or unfocused camera.
    """
    if cv2 is None or img_bgr is None:
        return 120.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(round(score, 1))


def crop_salient_object(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Localizes the primary foreground object (medicine blister, plate, circuit board)
    using edge gradient energy and contour bounding.
    """
    if cv2 is None or img_bgr is None:
        return img_bgr, None

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, None

    # Filter out tiny noise contours
    valid_cnts = [c for c in contours if cv2.contourArea(c) > (w * h * 0.04)]
    if not valid_cnts:
        return img_bgr, None

    # Pick largest salient contour
    largest_cnt = max(valid_cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_cnt)

    # Pad bounding box slightly
    pad = 15
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    crop = img_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2 - x1, y2 - y1)


def ndarray_to_base64(img_bgr: np.ndarray, format: str = "JPEG") -> str:
    """Converts OpenCV BGR image array into a base64 encoded string for API requests."""
    if cv2 is not None:
        success, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            return base64.b64encode(encoded).decode("utf-8")

    # Fallback via PIL
    if Image is not None:
        rgb = img_bgr[:, :, ::-1] if len(img_bgr.shape) == 3 else img_bgr
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Pure raw bytes fallback
    return base64.b64encode(img_bgr.tobytes()).decode("utf-8")
