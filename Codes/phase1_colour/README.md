# Phase 1 — Color-Only CBIR Baseline

Part of **Explainable Multi-Label Content-Based Image Retrieval for Aerial Images using Classical Computer Vision**.
No ML/DL — classical image processing only.

## What this phase does

Given a query image from `images/images_test`, extract a color histogram, compare it against
every gallery image in `images/images_tr` using histogram intersection, and return the Top-K
most similar images.

Two color spaces are implemented and compared:

- **HSV** — 2D Hue-Saturation histogram (30×32 bins). Brightness (V) is deliberately dropped
  since it's lighting/exposure dependent, not a reliable similarity cue.
- **RGB** — 3D histogram (16×16×16 bins).

Similarity metric: **histogram intersection** (`sum(min(hist_a, hist_b))`), bounded in [0, 1].

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Indexes gallery/query images and joins the 17-label semantic annotations |
| `color_features.py` | HSV / RGB histogram extraction + histogram intersection |
| `build_color_gallery.py` | Precomputes and caches histograms for all 2400 gallery images |
| `query_color.py` | Retrieves and visualizes Top-K results for one query image |
| `compare_color_spaces.py` | Quantitative HSV vs RGB comparison over all 600 queries |

## How to run

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib tqdm
python build_color_gallery.py                        # builds cache/ (run once)
python query_color.py --query_id airport_85 --space hsv --k 5
python compare_color_spaces.py                        # full quantitative comparison
```

## Result (Precision@5, same-scene-class proxy, 600 queries)

| Color space | Precision@5 |
|---|---|
| HSV (H-S only) | 0.3897 |
| RGB | 0.4203 |

RGB slightly outperforms HSV on this dataset — aerial tiles here have fairly uniform outdoor
lighting, so dropping brightness in HSV didn't pay off the way it typically does on
illumination-variant datasets. Both are weak used alone, which is expected: color is one of
four planned visual cues (color, texture, shape, spatial) that get fused in later phases, and
the relevance proxy here (same folder class) is a placeholder — real evaluation uses the
17-label semantic ground truth, added in a later phase.
