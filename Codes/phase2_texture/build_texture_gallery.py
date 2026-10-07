"""Precompute and cache LBP, GLCM, and Gabor texture features for every gallery image."""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import gallery_index
import texture_features as tf

CACHE_DIR = Path(__file__).resolve().parent / "cache"

EXTRACTORS = {
    "lbp": tf.lbp_hist,
    "glcm": tf.glcm_features,
    "gabor": tf.gabor_features,
}


def build(method: str):
    extract = EXTRACTORS[method]
    df = gallery_index()
    feats = []
    for path in tqdm(df["path"], desc=f"Extracting {method.upper()} features"):
        feats.append(extract(path))
    feats = np.vstack(feats)

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(CACHE_DIR / f"gallery_{method}_feats.npy", feats)
    df[["id", "path", "scene_class"]].to_csv(CACHE_DIR / "gallery_ids.csv", index=False)
    print(f"Saved {feats.shape} -> cache/gallery_{method}_feats.npy")


if __name__ == "__main__":
    for method in ("lbp", "glcm", "gabor"):
        build(method)
