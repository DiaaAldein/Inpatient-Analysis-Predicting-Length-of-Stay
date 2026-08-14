
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# FIX: relative paths like "deployment_config.json" resolve against the
# CURRENT WORKING DIRECTORY, which is not guaranteed to be app.py's own
# folder - Streamlit Community Cloud, for one, runs scripts from the
# repository root, not from inside models/. Building every path from
# app.py's own location instead makes this correct regardless of where
# or how the app is launched from.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# FIX: joblib does not pickle a FunctionTransformer's function body - it saves a
# reference to the function's NAME and the module it was defined in, and looks it
# up again at load time. `full_preprocessor` (Section 6) uses
# FunctionTransformer(apply_ordinal_maps), defined in the notebook's own
# environment - that function does not exist in this standalone script's
# namespace, so loading the pickled pipeline fails with an AttributeError unless
# an identically-named function is defined here too. Kept byte-for-byte identical
# to Section 6's definition, since any behavioral difference here would silently
# change what the deployed model actually computes.
ordinal_maps = {
    'age_group': {'18 to 29': 1, '30 to 49': 2, '50 to 69': 3, '70 or Older': 4},
    'apr_severity_of_illness_description': {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4},
    'apr_risk_of_mortality': {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4},
}

def apply_ordinal_maps(X):
    X = X.copy()
    for col, mapping in ordinal_maps.items():
        X[col] = X[col].map(mapping)
    return X

# Load the deployment manifest (Section 15) and both final model pipelines - all
# three files are expected to sit alongside app.py.
with open(os.path.join(_APP_DIR, "deployment_config.json"), encoding="utf-8") as f:
    CONFIG = json.load(f)

l2_pipeline = joblib.load(os.path.join(_APP_DIR, CONFIG["model_files"]["L2_log1p"]))
tweedie_pipeline = joblib.load(os.path.join(_APP_DIR, CONFIG["model_files"]["Tweedie_regularized"]))

FEATURE_COLUMNS = CONFIG["feature_columns"]
CATEGORICAL_VALUES = CONFIG["categorical_values"]
ORDINAL_ORDER = CONFIG["ordinal_scale_order"]
MDC_TO_DRG_MAP = CONFIG["mdc_to_drg_map"]
BLEND_WEIGHT = CONFIG["blend_weight_l2_log1p"]


def prediction(inputs: dict) -> float:
    """Builds a single-row DataFrame matching the training feature columns,
    runs both final pipelines, and returns the blended prediction in days."""
    df = pd.DataFrame(columns=FEATURE_COLUMNS)
    for col, value in inputs.items():
        df.at[0, col] = value

    X_l2_enc = l2_pipeline["preprocessor"].transform(df)
    l2_pred_days = np.expm1(l2_pipeline["model"].predict(X_l2_enc))[0]

    X_tweedie_enc = tweedie_pipeline["preprocessor"].transform(df)
    tweedie_pred_days = tweedie_pipeline["model"].predict(X_tweedie_enc)[0]

    return BLEND_WEIGHT * l2_pred_days + (1 - BLEND_WEIGHT) * tweedie_pred_days


def Main():
    st.title("Inpatient Length-of-Stay Prediction")
    st.caption(
        f"Blended CatBoost model - test set R\u00b2={CONFIG['headline_test_metrics']['r2']:.3f}, "
        f"MAE={CONFIG['headline_test_metrics']['mae_days']:.2f} days"
    )
    st.markdown(
        "Predicts a hospital patient's expected **length of stay (in days)** from "
        "information available at admission. Trained on 2015 New York (SPARCS) "
        "inpatient records, **adult patients (18+) only**. Intended to support "
        "hospital bed-planning and resource-allocation workflows - **not** a "
        "clinical decision tool for individual patients. See the "
        "[project README](https://github.com/DiaaAldein/Inpatient-Analysis-Predicting-Length-of-Stay) "
        "for full methodology and known limitations."
    )

    st.subheader("Clinical Information")
    apr_mdc_description = st.selectbox(
        "APR MDC (major diagnostic category)",
        CATEGORICAL_VALUES["apr_mdc_description"],
        help="The broad clinical category (e.g., 'Diseases of the Circulatory "
             "System') the admission falls under. Selecting this first narrows "
             "the APR DRG options below to only those that are valid for this "
             "category.",
    )
    apr_drg_description = st.selectbox(
        "APR DRG (diagnosis-related group)",
        MDC_TO_DRG_MAP[apr_mdc_description],
        help="A more specific clinical grouping within the selected MDC above "
             "(e.g., a particular type of surgery or condition). Options are "
             "filtered to those actually observed for the selected MDC in the "
             "training data.",
    )
    ccs_diagnosis_description = st.selectbox(
        "Diagnosis", CATEGORICAL_VALUES["ccs_diagnosis_description"])
    ccs_procedure_description = st.selectbox(
        "Procedure", CATEGORICAL_VALUES["ccs_procedure_description"])
    apr_medical_surgical_description = st.selectbox(
        "Medical / Surgical",
        CATEGORICAL_VALUES["apr_medical_surgical_description"],
        help="Whether the admission was primarily medical (treated without a "
             "major procedure) or surgical.",
    )
    apr_severity_of_illness_description = st.selectbox(
        "Severity of Illness",
        ORDINAL_ORDER["apr_severity_of_illness_description"],
        help="Clinical severity scale, from Minor to Extreme. The strongest "
             "single predictor of length of stay found during this project's "
             "exploratory analysis.",
    )
    apr_risk_of_mortality = st.selectbox(
        "Risk of Mortality",
        ORDINAL_ORDER["apr_risk_of_mortality"],
        help="A related but distinct APR-based scale estimating risk of death, "
             "from Minor to Extreme.",
    )

    st.subheader("Admission Information")
    type_of_admission = st.selectbox(
        "Type of Admission", CATEGORICAL_VALUES["type_of_admission"])
    emergency_department_indicator = st.selectbox(
        "Emergency Department Admission", CATEGORICAL_VALUES["emergency_department_indicator"])
    payment_typology_1 = st.selectbox(
        "Primary Payment Type", CATEGORICAL_VALUES["payment_typology_1"])

    st.subheader("Patient Information")
    age_group = st.selectbox(
        "Age Group",
        ORDINAL_ORDER["age_group"],
        help="The model was trained on adult patients (18+) only - pediatric "
             "admissions are out of scope and not supported by this tool.",
    )
    gender = st.selectbox("Gender", CATEGORICAL_VALUES["gender"])

    if st.button("Predict Length of Stay"):
        inputs = {
            "ccs_diagnosis_description": ccs_diagnosis_description,
            "ccs_procedure_description": ccs_procedure_description,
            "apr_drg_description": apr_drg_description,
            "apr_mdc_description": apr_mdc_description,
            "apr_medical_surgical_description": apr_medical_surgical_description,
            "apr_severity_of_illness_description": apr_severity_of_illness_description,
            "apr_risk_of_mortality": apr_risk_of_mortality,
            "type_of_admission": type_of_admission,
            "emergency_department_indicator": emergency_department_indicator,
            "payment_typology_1": payment_typology_1,
            "age_group": age_group,
            "gender": gender,
        }
        result_days = prediction(inputs)
        st.success(f"Predicted length of stay: **{result_days:.1f} days**")
        st.caption(
            "This is a research/portfolio model, not a clinical decision tool - "
            "see the project README for known limitations."
        )

    st.divider()
    st.caption("Built by Diaa Aldein Alsayed Ibrahim Osman")


Main()
