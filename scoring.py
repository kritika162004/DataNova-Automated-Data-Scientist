"""
scoring.py - use the trained model on NEW applicants (raw rows, same columns as the uploaded CSV).

    bundle = save_scoring_bundle(state, out_dir)       # called by the pipeline after modelling
    out = score_applicants(new_df, bundle)             # DataFrame: default_probability, risk_band, flagged

The preprocessing is the SAME fitted object used for training (imputer/scaler/encoder fitted on
train rows only), so scores are consistent with the test results in the report.
"""
import os

import joblib
import numpy as np
import pandas as pd

from data_prep import coerce_numeric_strings, clean_feature_names


def save_scoring_bundle(state: dict, out_dir: str = "reports") -> str:
    pre = state["preprocessor"]
    model_bundle = joblib.load(state["model_path"])
    bundle = {
        "preprocessor": pre,
        "input_columns": list(pre.feature_names_in_),
        "target_col": state.get("target_col"),
        "dropped_cols": list(state.get("dropped_cols", [])),
        "model": model_bundle["model"],
        "features": model_bundle["features"],
        "threshold": model_bundle["threshold"],
        "model_name": model_bundle["model_name"],
    }
    path = os.path.join(out_dir, "scoring_bundle.joblib")
    joblib.dump(bundle, path)
    return path


def score_applicants(df_raw: pd.DataFrame, bundle_path: str) -> pd.DataFrame:
    b = joblib.load(bundle_path) if isinstance(bundle_path, str) else bundle_path
    df = coerce_numeric_strings(df_raw.copy())
    for c in b["input_columns"]:                       # tolerate missing columns: the imputer fills them
        if c not in df.columns:
            df[c] = np.nan
    X = df[b["input_columns"]].copy()
    for c in X.columns[X.dtypes == bool]:
        X[c] = X[c].astype(int)
    Xt = b["preprocessor"].transform(X)
    Xt.columns = clean_feature_names(Xt.columns)
    proba = b["model"].predict_proba(Xt[b["features"]])[:, 1]
    thr = b["threshold"]
    band = np.where(proba >= thr, "High", np.where(proba >= thr * 0.6, "Medium", "Low"))
    out = pd.DataFrame({"default_probability": np.round(proba, 4), "risk_band": band,
                        "flagged_as_default": proba >= thr}, index=df_raw.index)
    return out
