# Phase 2 — Texture-Only CBIR Baselines

Part of **Explainable Multi-Label Content-Based Image Retrieval for Aerial Images using Classical Computer Vision**.
No ML/DL — classical image processing only.

## What this phase does

Given a query image, extract a texture descriptor and compare it against every gallery image,
returning the Top-K most similar images. Three classical texture methods are implemented,
all computed on a resized 128×128 grayscale image for speed and comparability:

- **LBP** (uniform, radius=2, 16 points) → 18-bin histogram, compared via histogram intersection.
- **GLCM / Haralick** (distances [1,2], 4 angles, 6 properties: contrast, dissimilarity,
  homogeneity, energy, correlation, ASM) → 24-dim vector.
- **Gabor filter bank** (4 orientations × 3 frequencies) → mean+std response per filter → 24-dim vector.

GLCM and Gabor are real-valued (not histograms), so they're z-score normalized using gallery
statistics and compared with **cosine similarity**. LBP stays a histogram, compared with
**histogram intersection**, same as Phase 1's color features.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Indexes gallery/query images and joins the 17-label semantic annotations |
| `texture_features.py` | LBP / GLCM / Gabor extraction |
| `build_texture_gallery.py` | Precomputes and caches all three feature sets for the 2400 gallery images |
| `query_texture.py` | Retrieves and visualizes Top-K results for one query image |
| `compare_texture_methods.py` | Quantitative LBP vs GLCM vs Gabor comparison over all 600 queries |

## How to run

```bash
pip install opencv-python numpy pandas scikit-image scipy matplotlib tqdm
python build_texture_gallery.py                       # builds cache/ (run once)
python query_texture.py --query_id airport_85 --method lbp --k 5
python compare_texture_methods.py                      # full quantitative comparison
```

## Result (Precision@5, same-scene-class proxy, 600 queries)

| Method | Precision@5 |
|---|---|
| LBP | 0.2540 |
| GLCM | 0.2470 |
| Gabor | 0.2690 |
| *(Phase 1 recap)* HSV | 0.3897 |
| *(Phase 1 recap)* RGB | 0.4203 |

Texture alone is noticeably weaker than color alone on this dataset — many aerial classes
(e.g. parking lots, storage tanks, industrial areas) share similar micro-texture statistics,
so texture mainly acts as a **disambiguator** for color-confused cases rather than a strong
standalone cue. This motivates the multi-feature fusion done in a later phase. Qualitatively,
Gabor's top-5 for an Airport query pulled Bridge/Playground/Stadium/Square — all generic
"open paved/structured area" textures, a useful example for the explainability phase: a
texture-only match can look visually plausible while being semantically wrong.
