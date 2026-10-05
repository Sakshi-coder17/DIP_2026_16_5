"""Phase 1: color-only CBIR. Query a single test image against the gallery,
rank by HSV (or RGB) histogram similarity, show Top-K, and save a figure."""
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
import color_features as cf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_gallery(space: str):
    feats = np.load(CACHE_DIR / f"gallery_{space}_feats.npy")
    ids = pd.read_csv(CACHE_DIR / "gallery_ids.csv")
    return ids, feats


def retrieve(query_path: str, space: str, k: int = 5):
    extract = cf.hsv_hist if space == "hsv" else cf.rgb_hist
    q_feat = extract(query_path)

    ids, gallery_feats = load_gallery(space)
    sims = np.array([cf.histogram_intersection(q_feat, g) for g in gallery_feats])

    order = np.argsort(-sims)[:k]
    results = ids.iloc[order].copy()
    results["similarity"] = sims[order]
    return results.reset_index(drop=True)


def show_results(query_id: str, query_path: str, results: pd.DataFrame, space: str):
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

    fig.suptitle(f"Color baseline ({space.upper()}) - Top-{len(results)} for {query_id}")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"color_{space}_{query_id}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", default=None, help="e.g. airport_5 (default: first test image)")
    parser.add_argument("--space", default="hsv", choices=["hsv", "rgb"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    q_df = query_index()
    if args.query_id is None:
        row = q_df.iloc[0]
    else:
        row = q_df[q_df["id"] == args.query_id].iloc[0]

    print(f"Query: {row['id']}  (scene_class={row['scene_class']})")
    results = retrieve(row["path"], args.space, args.k)
    print(results[["id", "scene_class", "similarity"]].to_string(index=False))
    show_results(row["id"], row["path"], results, args.space)
