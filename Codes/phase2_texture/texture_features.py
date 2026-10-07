"""Texture-only features: LBP, GLCM (Haralick), and Gabor filter bank.

All methods work on a resized grayscale image so GLCM/Gabor stay fast
across 3000 images while remaining comparable to each other.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

RESIZE = (128, 128)

# --- LBP ---
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"  # produces LBP_POINTS + 2 bins

# --- GLCM ---
GLCM_LEVELS = 32  # grey levels after quantization
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# --- Gabor ---
GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_FREQS = [0.1, 0.25, 0.4]
_GABOR_KERNELS = [
    gabor_kernel(frequency=f, theta=t) for t in GABOR_THETAS for f in GABOR_FREQS
]


def _load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, RESIZE, interpolation=cv2.INTER_AREA)


def lbp_hist(path: str) -> np.ndarray:
    """Normalized histogram of uniform LBP codes (rotation/illumination-robust texture)."""
    gray = _load_gray(path)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def glcm_features(path: str) -> np.ndarray:
    """Haralick texture descriptors averaged over distances, per angle, per property."""
    gray = _load_gray(path)
    quantized = (gray.astype(np.float64) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quantized = np.clip(quantized, 0, GLCM_LEVELS - 1)

    glcm = graycomatrix(
        quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        feats.append(vals.mean(axis=0))  # average over distances -> (n_angles,)
    return np.concatenate(feats)  # len = len(PROPS) * len(ANGLES)


def gabor_features(path: str) -> np.ndarray:
    """Mean + std of the response magnitude for each filter in the Gabor bank."""
    gray = _load_gray(path).astype(np.float64) / 255.0
    feats = []
    for kernel in _GABOR_KERNELS:
        real = ndi.convolve(gray, np.real(kernel), mode="wrap")
        imag = ndi.convolve(gray, np.imag(kernel), mode="wrap")
        mag = np.sqrt(real**2 + imag**2)
        feats.append(mag.mean())
        feats.append(mag.std())
    return np.array(feats)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
