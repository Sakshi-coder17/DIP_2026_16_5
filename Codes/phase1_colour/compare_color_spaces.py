"""Quantitative RGB vs HSV comparison for the Phase-1 color baseline.

Relevance proxy for this early phase: same scene_class (folder) as the query.
(Proper multi-label-based evaluation with mAP/NDCG comes in a later phase.)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import query_index
import color_features as cf
from query_color import load_gallery

K = 5


def precision_at_k(space: str, q_df: pd.DataFrame) -> float:
    extract = cf.hsv_hist if space == "hsv" else cf.rgb_hist
    ids, gallery_feats = load_gallery(space)
    gallery_classes = ids["scene_class"].values

    precisions = []
    for _, row in tqdm(q_df.iterrows(), total=len(q_df), desc=f"Evaluating {space.upper()}"):
        q_feat = extract(row["path"])
        sims = np.minimum(gallery_feats, q_feat).sum(axis=1)
        top_k = np.argsort(-sims)[:K]
        hits = (gallery_classes[top_k] == row["scene_class"]).sum()
        precisions.append(hits / K)
    return float(np.mean(precisions))


if __name__ == "__main__":
    q_df = query_index()
    results = {}
    for space in ("hsv", "rgb"):
        results[space] = precision_at_k(space, q_df)

    print("\n=== Phase 1: Color baseline, Precision@5 (same-scene-class proxy) ===")
    for space, p in results.items():
        print(f"{space.upper():5s}: {p:.4f}")
