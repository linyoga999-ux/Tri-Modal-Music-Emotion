# Tri-Modal-Music-Emotion
# Tri-Modal Music Emotion

Official code repository for the paper:

> *A Multimodal Acoustic-Subjective-Physiological Framework for Affective 
> Characterization of Contemporary Music Heritage*  
> Submitted to **npj Heritage Science** (2026).

## Overview

This repository contains the analysis pipeline for the Tri-modal Stacking 
framework described in the paper. The framework integrates three feature 
streams — 79 handcrafted acoustic descriptors, 14 culturally-localized 
subjective perception ratings, and 12 autonomic physiological features — 
through a LightGBM + CatBoost + Ridge stacking ensemble, with SHAP-based 
modality attribution.

## Repository structure

| File | Purpose |
|------|---------|
| `step2_train_main_v9.py` | Stacking training pipeline (10-fold CV across all unimodal / dual / tri-modal configurations) |
| `step3_shap_analysis.py` | SHAP-via-surrogate attribution and modality-aggregate analysis |
| `step4_make_figures.py` | Figure generation pipeline |

## Data

The feature matrices, fold assignments, and ground-truth ratings 
associated with this study are archived (restricted access during peer 
review) at Zenodo:

**DOI:** [10.5281/zenodo.20130942](https://doi.org/10.5281/zenodo.20130942)

Access to the underlying data is restricted during the peer-review 
process and will be opened upon acceptance of the manuscript. Reviewers 
and editors may request access via the corresponding author.

The underlying 600-track audio corpus consists of canonical contemporary 
works subject to third-party copyright and is therefore not redistributable.

## Requirements

- Python 3.10+
- `lightgbm`, `catboost`, `scikit-learn`, `shap`, `numpy`, `pandas`, 
  `matplotlib`, `librosa`

## Citation

Citation details will be added upon publication.

## License

The code in this repository is released under the MIT License. The 
associated data on Zenodo is governed by the access conditions specified 
in that record.
