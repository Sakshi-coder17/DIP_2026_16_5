"""Quantitative comparison of LBP vs GLCM vs Gabor texture baselines.

Relevance proxy: same scene_class as the query (same proxy as Phase 1, for
a fair side-by-side; the real multi-label evaluation comes in a later phase).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import query_index
import texture_features as tf
from query_texture import EXTRACTORS, IS_HISTOGRAM, load_gallery, gallery_stats, cosine_sim

K = 5


def precision_at_k(method: str, q_df: pd.DataFrame) -> float:
    extract = EXTRACTORS[method]
    ids, gallery_feats = load_gallery(method)
    gallery_classes = ids["scene_class"].values

    if not IS_HISTOGRAM[method]:
        mean, std = gallery_stats(gallery_feats)
        g_norm = (gallery_feats - mean) / std
        g_unit = g_norm / np.linalg.norm(g_norm, axis=1, keepdims=True)

    precisions = []
    for _, row in tqdm(q_df.iterrows(), total=len(q_df), desc=f"Evaluating {method.upper()}"):
        q_feat = extract(row["path"])
        if IS_HISTOGRAM[method]:
            sims = np.minimum(gallery_feats, q_feat).sum(axis=1)
        else:
            q_norm = (q_feat - mean) / std
            q_unit = q_norm / (np.linalg.norm(q_norm) or 1.0)
            sims = g_unit @ q_unit
        top_k = np.argsort(-sims)[:K]
        hits = (gallery_classes[top_k] == row["scene_class"]).sum()
        precisions.append(hits / K)
    return float(np.mean(precisions))


if __name__ == "__main__":
    q_df = query_index()
    results = {}
    for method in ("lbp", "glcm", "gabor"):
        results[method] = precision_at_k(method, q_df)

    print("\n=== Phase 2: Texture baselines, Precision@5 (same-scene-class proxy) ===")
    for method, p in results.items():
        print(f"{method.upper():6s}: {p:.4f}")
