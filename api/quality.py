"""Photo quality gate: reject blurry or dark photos before they reach training data.

Thresholds were picked by scoring 300 validation photos (all passed), plus
blurred and darkened copies of the same photos (all rejected):
  sharpness (Laplacian variance): real photos 1st percentile 72.6, blurred copies max ~5
  brightness (mean gray, 0-255):  real photos 1st percentile 48,  darkened copies 95th pct 30
"""
import os

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "30"))
DARK_THRESHOLD = float(os.getenv("DARK_THRESHOLD", "35"))


def check_quality(image_bytes: bytes) -> dict:
    """Returns {"ok", "reason", "sharpness", "brightness"}. reason is None, "dark", "blurry" or "unreadable"."""
    import cv2  # imported here, not at startup, to keep server boot memory low (Render free tier: 512 MB)
    import numpy as np
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "reason": "unreadable", "sharpness": 0.0, "brightness": 0.0}

    # Score at a fixed size so phone photos and small photos are judged the same way
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Check dark first: a very dark photo also looks "blurry" (few edges)
    if brightness < DARK_THRESHOLD:
        reason = "dark"
    elif sharpness < BLUR_THRESHOLD:
        reason = "blurry"
    else:
        reason = None
    return {"ok": reason is None, "reason": reason,
            "sharpness": round(sharpness, 1), "brightness": round(brightness, 1)}
