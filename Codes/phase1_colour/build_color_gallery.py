"""Precompute and cache color histograms for every gallery image."""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import gallery_index
import color_features as cf

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def build(space: str):
    assert space in ("hsv", "rgb")
    extract = cf.hsv_hist if space == "hsv" else cf.rgb_hist

    df = gallery_index()
    feats = []
    for path in tqdm(df["path"], desc=f"Extracting {space.upper()} histograms"):
        feats.append(extract(path))
    feats = np.vstack(feats)

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(CACHE_DIR / f"gallery_{space}_feats.npy", feats)
    df[["id", "path", "scene_class"]].to_csv(CACHE_DIR / "gallery_ids.csv", index=False)
    print(f"Saved {feats.shape} -> cache/gallery_{space}_feats.npy")


if __name__ == "__main__":
    for space in ("hsv", "rgb"):
        build(space)
