"""
Mock backend: fake-but-correctly-shaped outputs for every agent (Task 4.1 mock data).
Field names match Section 6 of the task brief, so the UI can be built and tested
before Members 1-3 finish. Replace by real code in pipeline_adapter.py.
"""
import os
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- sample data
def make_mock_df(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.5, n)
    loan = rng.lognormal(9.5, 0.6, n)
    credit = np.clip(rng.normal(650, 70, n), 300, 850)
    dti = np.clip(loan / income * 10 + rng.normal(0, 3, n), 0, 60)
    late = rng.poisson(0.6, n)
    risk = -0.012 * (credit - 650) + 0.05 * dti + 0.4 * late - 1.0
    default = (rng.random(n) < 1 / (1 + np.exp(-(risk - 2.0)))).astype(int)
    df = pd.DataFrame({
        "loan_id": np.arange(10000, 10000 + n),
        "age": rng.integers(21, 70, n),
        "income": income.round(0),
        "loan_amount": loan.round(0),
        "credit_score": credit.round(0),
        "debt_to_income": dti.round(2),
        "employment_years": rng.integers(0, 30, n),
        "num_open_accounts": rng.integers(1, 15, n),
        "late_payments": late,
        "loan_purpose": rng.choice(["car", "home", "education", "business"], n),
        "home_ownership": rng.choice(["rent", "own", "mortgage"], n),
        "recoveries": np.where(default == 1, rng.uniform(0, 5000, n), 0).round(0),
        "default": default,
    })
    df.loc[rng.choice(n, 40, replace=False), "income"] = np.nan
    return df


# --------------------------------------------------------------- M1: profiler
def mock_profile(df: pd.DataFrame) -> dict:
    target = "default" if "default" in df.columns else df.columns[-1]
    ratio = None
    if df[target].nunique() == 2:
        vc = df[target].value_counts()
        ratio = f"{vc.min() / vc.sum() * 100:.1f}% default rate (1:{int(vc.max() / vc.min())} ratio)"
    miss = df.isna().sum()
    fixes = [{"id": f"{c}|missing_value", "default": True,
              "description": f"{c}: impute with median",
              "raw": {"column": c, "issue_type": "missing_value", "suggested_action": "impute with median",
                      "reason": f"{int(m)} missing values; the median is robust to skew."}}
             for c, m in miss.items() if m > 0]
    leak = []
    if "recoveries" in df.columns:
        leak.append({"col": "recoveries",
                     "reason": "Only known after a loan defaults - would leak the outcome"})
    if "loan_id" in df.columns:
        leak.append({"col": "loan_id", "reason": "Identifier with no predictive meaning"})
    return {
        "n_rows": len(df), "n_cols": df.shape[1],
        "missing_total": int(miss.sum()),
        "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(miss[c]),
                     "unique": int(df[c].nunique())} for c in df.columns],
        "proposed_target": target,
        "target_reason": f"'{target}' is binary (0/1) and its name suggests loan default.",
        "leakage_candidates": leak,
        "suggested_fixes": fixes, "warnings": [],
        "imbalance_ratio": ratio, "domain": "finance (credit risk)",
        "summary_text": (f"The dataset has {len(df)} borrowers and {df.shape[1]} columns. "
                         f"Class balance: {ratio}. Accuracy alone would be misleading."),
    }


# -------------------------------------------------------------- charts helper
def _chart(path, title, kind="bar"):
    rng = np.random.default_rng(abs(hash(title)) % 2**32)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if kind == "hist":
        ax.hist(rng.normal(0, 1, 400), bins=25, color="#2b6cb0")
    elif kind == "line":
        ax.plot(np.sort(rng.random(30)), np.sort(rng.random(30)), color="#2b6cb0")
        ax.plot([0, 1], [0, 1], "--", color="grey")
    else:
        ax.bar(list("ABCDE"), rng.random(5), color="#2b6cb0")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------- M1/M2/M3 pipeline
def mock_run_pipeline(df, target_col, leakage_cols, fixes, settings, progress_callback=None):
    out_dir = tempfile.mkdtemp(prefix="ads_mock_")
    feats_all = [c for c in df.columns if c != target_col and c not in leakage_cols]
    n_iter = int(settings["iterations"])

    # PSO (Member 2) -- progress_callback(iteration, total_iterations, best_fitness)
    rng = np.random.default_rng(1)
    f, history = 0.45, []
    for i in range(n_iter):
        f = max(0.18, f - rng.uniform(0.0, 0.02))
        history.append(round(f, 4))
        time.sleep(0.03)
        if progress_callback:
            progress_callback(i + 1, n_iter, history[-1])
    k = max(2, len(feats_all) // 2)
    selected = list(rng.choice(feats_all, k, replace=False))

    comparison = [
        {"method": "PSO (ours)", "n_features": k, "auc": 0.842, "f1": 0.512},
        {"method": "All features", "n_features": len(feats_all), "auc": 0.838, "f1": 0.505},
        {"method": "Mutual information", "n_features": k, "auc": 0.811, "f1": 0.471},
        {"method": "RFE", "n_features": k, "auc": 0.826, "f1": 0.489},
        {"method": "Genetic Algorithm", "n_features": k + 1, "auc": 0.835, "f1": 0.498},
    ]
    shap = sorted(({"feature": s, "mean_abs_shap": round(float(rng.uniform(0.02, 0.4)), 3)}
                   for s in selected), key=lambda d: -d["mean_abs_shap"])

    return {
        "raw_df": df, "target_col": target_col, "leakage_cols": leakage_cols,
        "imbalance_ratio": 11.5,
        "eda_charts": [_chart(os.path.join(out_dir, "eda1.png"), "Income distribution", "hist"),
                       _chart(os.path.join(out_dir, "eda2.png"), "Default rate by purpose"),
                       _chart(os.path.join(out_dir, "eda3.png"), "Credit score distribution", "hist")],
        "selected_features": selected, "pso_history": history,
        "pso_runtime": round(n_iter * 0.03 + 0.4, 2),
        "baseline_feature_sets": {"All features": feats_all, "Mutual information": selected[:k],
                                  "RFE": selected[:k], "Genetic Algorithm": selected},
        "best_model_name": "XGBoost", "model_path": os.path.join(out_dir, "model.pkl"),
        "metrics": {"auc": 0.842, "f1": 0.512, "precision": 0.58, "recall": 0.46, "accuracy": 0.91},
        "comparison_table": comparison, "shap_summary": shap,
        "model_charts": [_chart(os.path.join(out_dir, "m1.png"), "ROC curve", "line"),
                         _chart(os.path.join(out_dir, "m2.png"), "SHAP importance"),
                         _chart(os.path.join(out_dir, "m3.png"), "Confusion matrix")],
    }


# ------------------------------------------------------------------- reporter
def mock_report(state, style):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    path = os.path.join(tempfile.mkdtemp(prefix="ads_rep_"), f"report_{style.lower()}.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 780, f"Credit-risk report ({style}) - MOCK")
    c.setFont("Helvetica", 11)
    m = state["metrics"]
    c.drawString(60, 750, f"AUC {m['auc']}  F1 {m['f1']}  Model: {state['best_model_name']}")
    c.save()
    return path


# ------------------------------------------------------------------------ Q&A
def mock_answer(state, question, history=None):
    q = question.lower()
    if "better" in q or "all features" in q:
        t = {r["method"]: r for r in state["comparison_table"]}
        p, a = t["PSO (ours)"], t["All features"]
        return (f"PSO reached AUC {p['auc']} with {p['n_features']} features, versus "
                f"{a['auc']} with all {a['n_features']} features.")
    if "kept" in q or "why" in q:
        return "Selected features: " + ", ".join(state["selected_features"]) + "."
    return "I can only answer from the stored results, and I don't have that information."
