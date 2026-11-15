# Explainable Multi-Label CBIR for Aerial Images using Classical Computer Vision

**ECE 501 Digital Image Processing, Ahmedabad University.** Project No. 16, Group No. 05.
Team: Sakshi Patel, Virja Shah, Khushi Shah, Dhruvi Doshi.

This is a content-based image retrieval (CBIR) system for multi-label aerial images. Given a query tile, it returns the five most similar gallery images and explains each match (which visual cue and which image regions produced it). It also flags results that may be unreliable.

All features are handcrafted Digital Image Processing descriptors implemented from first principles in NumPy. The project uses **no OpenCV, no scikit-image and no deep learning**.

**Pipeline:** query image → pre-processing → 11 descriptors (colour, texture, shape, spatial) → per-cue similarity → learned weighted fusion → top-5 ranking → explanation (per-cue breakdown + heatmaps) → reliability check (return or flag).

---

## Highlights

- **11 handcrafted descriptors**:
  - Colour: RGB and HSV histograms.
  - Texture: LBP, GLCM (Haralick) and a Gabor filter bank.
  - Shape: HOG, edge histogram and Hu moments.
  - Spatial layout: grid colour moments, grid intensity and an MPEG-7-style colour layout descriptor (DCT).
- **Learned score fusion**: the final score is S = Σ w_c · S_c. The 11 weights are tuned on a separate validation split, so the test set is never used for tuning.
- **Explainability**: each match gets an exact per-cue score breakdown and occlusion-based region heatmaps. A deletion test confirms that the heatmaps track the score (30/30 pairs).
- **Uncertainty**: a reliability score built from score margins and cue disagreement flags doubtful results and queries, each with reason tags.
- **Statistically tested evaluation**: Precision@K, Hit@5 and mAP, for the same scene class and for multi-label Jaccard thresholds 0.5 to 0.8, with paired-bootstrap confidence intervals.

## Methodology (brief)

| Phase | Method | Key details |
|---|---|---|
| 0. Dataset & protocol | Multi-label AID [1, 2] | 3,000 RGB images (600×600), 30 scene classes, 17 object labels (5.15 per image). 2,400 gallery / 600 query; 600 gallery images (20 per class, seed 42) held out for tuning. |
| 1. Colour | RGB and HSV histograms [3] | RGB: 16 levels per channel → 4,096 bins. H–S: 30×32 = 960 bins (V dropped). L1-normalised, histogram intersection. |
| 2. Texture | LBP [4], GLCM [5], Gabor [6] | Uniform LBP (59 bins). GLCM on 32 grey levels, d ∈ {1, 2}, 4 directions, 5 Haralick features (40-D). Gabor bank of 4 orientations × 3 frequencies via FFT, mean and std of response (24-D). |
| 3. Shape | HOG [7], edge histogram [8], Hu moments [9] | HOG: 16×16 cells, 9 bins, 2×2 blocks, L2-Hys (1,764-D). Sobel edge density on a 4×4 grid (16-D). 7 log-scaled Hu invariants. |
| 4. Spatial layout | Grid colour moments [10], grid intensity, CLD [8] | Mean and std of H, S, V on a 4×4 grid (96-D). Mean luminance per cell (16-D). 8×8 luma blocks → 2-D DCT-II, 4×4 low-frequency coefficients (16-D). |
| Similarity | Per-cue scores in [0, 1] | Histograms: intersection Σ min(q, g). Vectors: gallery z-score, then (1 + cos)/2. |
| 5A. Fusion | Learned weighted sum | S = Σ w_c S_c with w_c ≥ 0 and Σ w_c = 1. Weights maximise ½(mAP + P@10) on the validation split, using random simplex starts plus coordinate ascent (metrics are rank-based, so there is no gradient). |
| 5B. Explainability | Score breakdown + occlusion [11] | Contributions w_c S_c sum exactly to S. Heatmaps: blank each cell of an 8×8 grid in both images, re-extract, re-score; maps are centred to remove the grey-patch bias. Checked with a deletion test. |
| 5C. Uncertainty | Margins + cue disagreement | Top-1 margin, margin to the next rank, score stand-out, spread of cue percentiles, top-5 agreement → NumPy logistic model fitted on 1,800 non-tuned gallery queries. The least reliable 30% are flagged, with reason tags. |
| 5D. Evaluation | P@K, Hit@5, mAP | Same-class and Jaccard ≥ 0.5 / 0.6 / 0.7 / 0.8 relevance; paired bootstrap (1,000 resamples) for confidence intervals. |

## Results (600 held-out queries vs 2,400 gallery images)

| System | Same-class P@5 | Same-class mAP | Jaccard ≥ 0.5 P@5 | Jaccard ≥ 0.8 P@5 |
|---|---|---|---|---|
| Chance (random ranking) | 3.3% | ≈3.3% | 50.0% | 13.8% |
| Colour only (Phase 1, RGB) | 42.03% | 18.79% | 81.63% | 40.13% |
| Equal-weight 4-cue fusion (Phase 4) | 50.10% | 25.55% | 84.67% | 43.20% |
| **Tuned 11-cue fusion (final)** | **56.50%** | **29.24%** | **86.23%** | **45.43%** |

- **Gain over Phase 4**: +6.4 points same-class P@5 (95% CI [4.7, 8.1]). Every Jaccard threshold also improves significantly.
- **Learned weights**: texture 0.43 (LBP alone 0.35), shape 0.24 (HOG 0.21), spatial 0.17, colour 0.16. Hu moments receive a weight of 0.
- **Reliability flags**: flagged queries reach 31.4% P@5 against 67.2% for unflagged ones. Answering only the 30% of queries rated most reliable gives 84.6% P@5.

### Single descriptors (same-class, %)

| Descriptor | Dim. | P@5 | Hit@5 | mAP | Jaccard ≥ 0.5 P@5 | Learned weight |
|---|---|---|---|---|---|---|
| RGB histogram | 4096 | 42.03 | 69.00 | 18.79 | 81.63 | 0.094 |
| HSV histogram | 960 | 38.87 | 64.67 | 17.32 | 80.07 | 0.063 |
| LBP | 59 | 31.47 | 63.50 | 13.87 | 73.07 | 0.349 |
| GLCM | 40 | 24.97 | 58.83 | 15.01 | 69.17 | 0.072 |
| Gabor | 24 | 26.83 | 60.17 | 15.41 | 73.07 | 0.013 |
| HOG | 1764 | 27.57 | 57.17 | 11.43 | 62.47 | 0.205 |
| Edge histogram | 16 | 13.23 | 38.67 | 7.26 | 64.03 | 0.038 |
| Hu moments | 7 | 11.03 | 36.33 | 7.07 | 63.90 | 0.000 |
| Grid colour moments | 96 | 34.63 | 64.83 | 19.59 | 76.43 | 0.090 |
| Grid intensity | 16 | 11.63 | 32.83 | 7.10 | 52.00 | 0.038 |
| CLD | 16 | 14.73 | 38.00 | 8.89 | 60.70 | 0.040 |

### Fusion progression (%)

| System | P@5 | Hit@5 | mAP | Jaccard ≥ 0.5 P@5 | Jaccard ≥ 0.8 P@5 |
|---|---|---|---|---|---|
| Colour (RGB) | 42.03 | 69.00 | 18.79 | 81.63 | 40.13 |
| + Texture | 48.57 | 75.17 | 25.60 | 83.33 | 41.70 |
| + Shape | 50.97 | 78.17 | 26.16 | 84.70 | 42.50 |
| + Spatial, equal weights (Phase 4) | 50.10 | 78.50 | 25.55 | 84.67 | 43.20 |
| 4 cues, tuned | 50.23 | 78.83 | 25.31 | 84.50 | 43.17 |
| 11 cues, equal | 50.57 | 79.83 | 25.70 | 84.53 | 43.03 |
| **11 cues, tuned (final)** | **56.50** | **84.17** | **29.24** | **86.23** | **45.43** |

### Final system vs Phase 4 at every relevance level (%, gain with 95% bootstrap CI)

| Relevance | Chance P@5 | P@5 Phase 4 | P@5 final | P@5 gain | mAP Phase 4 | mAP final | mAP gain |
|---|---|---|---|---|---|---|---|
| Same class | 3.3 | 50.10 | 56.50 | +6.40 [4.73, 8.07] | 25.55 | 29.24 | +3.69 [2.91, 4.49] |
| Jaccard ≥ 0.5 | 50.0 | 84.67 | 86.23 | +1.57 [0.47, 2.73] | 64.55 | 66.73 | +2.18 [1.61, 2.77] |
| Jaccard ≥ 0.6 | 36.3 | 71.33 | 73.60 | +2.27 [0.97, 3.63] | 50.82 | 54.02 | +3.20 [2.67, 3.74] |
| Jaccard ≥ 0.7 | 24.6 | 58.17 | 60.80 | +2.63 [1.13, 4.20] | 38.08 | 41.28 | +3.20 [2.70, 3.73] |
| Jaccard ≥ 0.8 | 13.8 | 43.20 | 45.43 | +2.23 [0.73, 3.77] | 26.18 | 28.53 | +2.34 [1.81, 2.85] |

### Explainability and reliability

| Check | Result |
|---|---|
| Deletion test: score change after blanking 4 of 16 cells (30 pairs) | most supportive −0.069, random −0.083, least supportive −0.098; ordering holds in 30/30 pairs |
| Flagged vs unflagged results (same-class precision) | 37.9% vs 64.5% (30% flagged, AUC 0.713) |
| Flagged vs unflagged queries (P@5) | 31.4% vs 67.2% (30% flagged, AUC 0.782) |
| Selective retrieval: queries answered → P@5 | 100% → 56.5%, 70% → 67.2%, 50% → 75.4%, 30% → 84.6% |
| Per-class change vs Phase 4 (95% CI) | 17 classes significantly better, 3 significantly worse, 10 within noise |

## Repository structure

```
midsem/
├── phase_0/        Dataset indexing, label loading, validation split (dataset.py)
├── phase_1/        Colour: RGB / HSV histograms
├── phase_2/        Texture: LBP, GLCM, Gabor
├── phase_3/        Shape: HOG, edge histogram, Hu moments
├── phase_4/        Spatial layout: grid moments, grid intensity, CLD; 4-cue equal fusion
├── phase_5a/       Weight tuning and score fusion (fusion_core.py, weight_tuning.py)
├── phase_5b/       Explainability: per-cue breakdown, occlusion heatmaps, deletion test
├── phase_5c/       Uncertainty: indicators, reliability model, flags
├── phase_5d/       Final evaluation: P@K, mAP, Jaccard 0.5-0.8, bootstrap, per-class
├── final_system/   One command: ranking + scores + explanation + reliability flags
├── results_charts/ Summary charts and CSVs for all phases
├── final_deliverables/  Final report (PDF/HTML/LaTeX), presentation, explanation guide
├── weekly_reports/ Week 6 and Week 7 progress reports
└── report/         Mid-semester report and presentation
```

Each `phase_*` folder has its own `README.md` with the method, equations, results and commands, plus `cache/` (precomputed features) and `outputs/` (tables, JSON results, figures for all 30 classes).

## Setup

**Requirements:** Python 3.11+ with `numpy`, `pandas`, `pillow` and `matplotlib`. `python-pptx` is needed only to rebuild the presentation.

```bash
pip install numpy pandas pillow matplotlib
```

**Dataset:** the multi-label AID dataset [1, 2]. Place it in the folder that **contains** this repository:

```
<parent>/
├── images/
│   ├── images_tr/<ClassName>/*.jpg     # 2,400 gallery images
│   └── images_test/<ClassName>/*.jpg   # 600 query images
├── multilabel.csv                       # 17 binary labels per image
└── midsem/                              # this repository
```

## How to run

The Phase 0–4 feature caches are included. Rerun `phase_N/build_*_gallery.py` only if you change a descriptor.

```bash
# Phase 5A: per-cue similarities and weight tuning
python phase_5a/build_cue_sims.py          # ~1.5 min
python phase_5a/weight_tuning.py           # ~25 min (writes phase_5a/final_weights.json)

# Phase 5C: uncertainty model
python phase_5c/build_calibration_sims.py  # ~2.5 min
python phase_5c/run_uncertainty.py

# Phase 5D: final evaluation
python phase_5d/evaluate.py

# Phase 5B: explanations
python phase_5b/explain_query.py --query_id airport_85 --k 5 --grid 8
python phase_5b/faithfulness_check.py

# Final system for one query (ranking + explanation + flags)
python final_system/cbir_system.py --query_id airport_85
python final_system/cbir_system.py --one_per_class --grid 4   # one query per class

# Summary charts for all phases
python generate_phase5_charts.py
```

### Output

`final_system/cbir_system.py` writes a figure and a JSON file for each query. They contain the top-5 images, the fused and per-cue scores, the region heatmaps, and the reliability of each result and of the query, with reason tags.

## Limitations

- **Chance levels:** a random ranking already scores 50% P@5 at Jaccard ≥ 0.5, so read the multi-label numbers against their chance level.
- **Per-class results:** 17 classes improve significantly and 3 get worse (StorageTanks, Beach, Airport). One global weight vector is a compromise across classes.
- **Similar classes and rotation:** visually similar classes (residential densities, farmland/meadow) remain hard. HOG and the grid descriptors are not rotation invariant.
- **Reliability:** query-level flags are useful (AUC 0.78). Result-level flags are no better than the fused score alone (AUC 0.71), and the reliability score is over-confident on held-out queries, so treat it as a relative score.
- **Heatmaps:** they explain which regions change the score, not what the scene contains.

## References

1. Y. Hua, L. Mou, X. X. Zhu, "Relation network for multilabel aerial image classification," *IEEE TGRS*, 2020.
2. G.-S. Xia et al., "AID: A benchmark data set for performance evaluation of aerial scene classification," *IEEE TGRS*, 2017.
3. M. J. Swain, D. H. Ballard, "Color indexing," *IJCV*, 1991.
4. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, 2002.
5. R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural features for image classification," *IEEE TSMC*, 1973.
6. B. S. Manjunath, W. Y. Ma, "Texture features for browsing and retrieval of image data," *IEEE TPAMI*, 1996.
7. N. Dalal, B. Triggs, "Histograms of oriented gradients for human detection," *CVPR*, 2005.
8. B. S. Manjunath et al., "Color and texture descriptors," *IEEE TCSVT*, 2001.
9. M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Trans. Inf. Theory*, 1962.
10. M. Stricker, M. Orengo, "Similarity of color images," *Proc. SPIE*, 1995.
11. M. D. Zeiler, R. Fergus, "Visualizing and understanding convolutional networks," *ECCV*, 2014.
