# Notebooks

## Executive Summary

This folder contains the full, reproducible pipeline for the Inpatient Length-of-Stay Prediction project: a single Jupyter notebook covering everything from raw data acquisition through final model deployment preparation.

## Contents

- [`Length_of_Stay_v6_new_build.ipynb`](Length_of_Stay_v6_new_build.ipynb) — the complete pipeline (16 sections): environment setup, data cleaning, exploratory analysis, model family comparison, preprocessing and feature experiments, loss function comparison and blending, hyperparameter optimization, final evaluation, feature importance, and deployment artifact generation. See [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for a detailed section-by-section explanation, or the top-level [README](../README.md) for a project overview.

**Execution note:** designed for full sequential execution (`Runtime → Run all`) or resumption from a prior full run — every computationally heavy step is checkpointed. Running an individual cell in isolation, without the earlier cells having run in the same session, is not supported.

---

*Part of the [Inpatient Length-of-Stay Prediction](https://github.com/DiaaAldein/Inpatient-Analysis-Predicting-Length-of-Stay) project — Diaa Aldein Alsayed Ibrahim Osman ([LinkedIn](https://www.linkedin.com/in/diaa-ibrahim-data/)).*
