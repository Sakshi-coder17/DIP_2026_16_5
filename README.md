# DIP_2026_16_5
# Explainable Multi-Label CBIR for Aerial Images using Classical Computer Vision

**ECE 501 Digital Image Processing, Ahmedabad University.** Project No. 16, Group No. 05.
Team: Sakshi Patel, Virja Shah, Khushi Shah, Dhruvi Doshi.

This is a content-based image retrieval (CBIR) system for multi-label aerial images. Given a query tile, it returns the five most similar gallery images and explains each match (which visual cue and which image regions produced it). It also flags results that may be unreliable.

All features are handcrafted Digital Image Processing descriptors implemented from first principles in NumPy. The project uses **no OpenCV, no scikit-image and no deep learning**.

![System pipeline](results_charts/cbir_architecture.png)

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

![Results across phases](results_charts/precision_progression_phase1_to_5.png)

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

### Example output

`final_system/cbir_system.py` produces a figure and a JSON file for each query. They contain the top-5 images, the fused and per-cue scores, the region heatmaps, and the reliability of each result and of the query, with reason tags.

![Final system output](final_system/outputs/cbir_airport_85.png)

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
