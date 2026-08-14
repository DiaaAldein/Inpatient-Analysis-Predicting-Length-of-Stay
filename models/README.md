# Model Artifacts & Deployment Files

## Executive Summary

This folder contains the final, deployable artifacts of the Inpatient Length-of-Stay Prediction project: the trained model pipelines, the deployment configuration that keeps the live app's inputs in sync with the training data, and the Streamlit application itself. Everything here is generated directly by the project notebook (see [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md), Sections 15–16) — nothing in this folder is hand-written or maintained separately from the modeling pipeline.

## Contents

- [Files in This Folder](#files-in-this-folder)
- [Running the App Locally](#running-the-app-locally)
- [A Note on Large Files](#a-note-on-large-files)

## Files in This Folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application. Loads both model pipelines and `deployment_config.json` at startup; every input field is populated dynamically from the config, never hard-coded. |
| `deployment_config.json` | Every valid value for each categorical feature (extracted directly from training data), the fixed clinical ordinal scales, the final blend weight, and the model file names. Regenerated automatically whenever the notebook is re-run. |
| `requirements.txt` | Python dependencies for running `app.py`, with exact versions confirmed from the environment that trained the models — used by Streamlit Community Cloud, which looks for a dependency file alongside the entrypoint script. |
| `final_pipeline_L2_log1p.joblib` | The trained CatBoost model (L2 loss on a `log1p`-transformed target) plus its fitted preprocessor, bundled together. |
| `final_pipeline_Tweedie_regularized.joblib` | The trained CatBoost model (regularized Tweedie loss) plus its fitted preprocessor. Blended with the model above (60/40) to produce the final prediction. |

## Running the App Locally

```bash
cd models
pip install -r requirements.txt
streamlit run app.py
```

## A Note on Large Files

The two `.joblib` model files are moderate in size (~16 MB each) and are committed directly to this repository — no Git LFS is required. See the top-level [README](../README.md) for the full project overview, methodology, and results.
