"""Color-only features: HSV H-S histogram and RGB histogram.

HSV H-S 2D histogram is the Phase-1 baseline: hue+saturation capture the
"color" of a scene while deliberately ignoring brightness (V), which varies
with lighting/exposure and is not a reliable similarity cue for aerial tiles.
"""
import cv2
import numpy as np

H_BINS = 30
S_BINS = 32
R_BINS = G_BINS = B_BINS = 16


def _load_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def hsv_hist(path: str) -> np.ndarray:
    """Normalized 2D Hue-Saturation histogram, flattened to 1D (L1-normalized)."""
    bgr = _load_bgr(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def rgb_hist(path: str) -> np.ndarray:
    """Normalized 3D RGB histogram, flattened to 1D (L1-normalized)."""
    bgr = _load_bgr(path)
    hist = cv2.calcHist(
        [bgr], [0, 1, 2], None, [B_BINS, G_BINS, R_BINS],
        [0, 256, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity in [0, 1]: sum of per-bin minimums of two normalized histograms."""
    return float(np.minimum(a, b).sum())
