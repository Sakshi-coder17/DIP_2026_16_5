"""Phase 2: texture-only CBIR. Query a single test image against the gallery
using LBP, GLCM, or Gabor similarity, show Top-K, and save a figure.

LBP is a normalized histogram -> compared with histogram intersection.
GLCM/Gabor are real-valued descriptors on different scales -> z-score
normalized using gallery statistics, then compared with cosine similarity.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import query_index
import texture_features as tf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

EXTRACTORS = {"lbp": tf.lbp_hist, "glcm": tf.glcm_features, "gabor": tf.gabor_features}
IS_HISTOGRAM = {"lbp": True, "glcm": False, "gabor": False}


def load_gallery(method: str):
    feats = np.load(CACHE_DIR / f"gallery_{method}_feats.npy")
    ids = pd.read_csv(CACHE_DIR / "gallery_ids.csv")
    return ids, feats


def gallery_stats(feats: np.ndarray):
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve(query_path: str, method: str, k: int = 5):
    extract = EXTRACTORS[method]
    q_feat = extract(query_path)
    ids, gallery_feats = load_gallery(method)

    if IS_HISTOGRAM[method]:
        sims = np.minimum(gallery_feats, q_feat).sum(axis=1)
    else:
        mean, std = gallery_stats(gallery_feats)
        g_norm = (gallery_feats - mean) / std
        q_norm = (q_feat - mean) / std
        sims = np.array([cosine_sim(q_norm, g) for g in g_norm])

    order = np.argsort(-sims)[:k]
    results = ids.iloc[order].copy()
    results["similarity"] = sims[order]
    return results.reset_index(drop=True)


def show_results(query_id: str, query_path: str, results: pd.DataFrame, method: str):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(3 * (len(results) + 1), 3.5))

    q_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    axes[0].imshow(q_img)
    axes[0].set_title(f"QUERY\n{query_id}")
    axes[0].axis("off")

    for i, row in results.iterrows():
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"#{i+1} {row['id']}\nsim={row['similarity']:.3f}\n{row['scene_class']}")
        axes[i + 1].axis("off")

    fig.suptitle(f"Texture baseline ({method.upper()}) - Top-{len(results)} for {query_id}")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"texture_{method}_{query_id}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", default=None)
    parser.add_argument("--method", default="lbp", choices=["lbp", "glcm", "gabor"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    q_df = query_index()
    row = q_df.iloc[0] if args.query_id is None else q_df[q_df["id"] == args.query_id].iloc[0]

    print(f"Query: {row['id']}  (scene_class={row['scene_class']})")
    results = retrieve(row["path"], args.method, args.k)
    print(results[["id", "scene_class", "similarity"]].to_string(index=False))
    show_results(row["id"], row["path"], results, args.method)
