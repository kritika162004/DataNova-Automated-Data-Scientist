"""
modeling.py  -  Member 3 stage: model selection, training, testing, metrics,
baseline comparison, SHAP explainability and model charts (Visualizer pass 2).

Reads from the shared LangGraph state:
    X_train, X_test (DataFrames), y_train, y_test
    selected_features        (list of names, from Member 2's PSO)
    baseline_feature_sets    (dict: method name -> feature list, from Member 2)
    pso_history              (best fitness per iteration, from Member 2)

Writes to the state (names match what reporter.py already reads):
    best_model_name, model_path, metrics, comparison_table,
    shap_summary, model_charts, cv_results

Rules this module follows (so results are trustworthy):
  * Model selection uses cross-validation on TRAIN data only.
  * The decision threshold is chosen from out-of-fold TRAIN predictions.
  * The test set is used only for final evaluation, never for any choice.
  * Imbalance is handled with class weights (no resampling, so no leakage).
"""
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

warnings.filterwarnings("ignore")

SEED = 42
PRIMARY = "#1F3A5F"
ACCENT = "#2A7F8E"
MUTED = "#B8C4CC"
WARN = "#C8553D"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_df(X, cols=None, name="X"):
    """Return a DataFrame. Bare arrays are only accepted together with column names."""
    if isinstance(X, pd.DataFrame):
        return X
    if cols is None:
        raise ValueError(
            f"{name} is a bare array with no column names, so PSO feature names cannot be matched. "
            "Pass DataFrames, or put the column names in state['feature_cols'].")
    return pd.DataFrame(X, columns=list(cols))


def _check_target(y, name):
    """Target must be binary 0/1 with 1 = default. Fail loudly instead of silently inverting."""
    y = np.asarray(y).ravel()
    vals = np.unique(y)
    if len(vals) != 2 or not set(vals.tolist()) <= {0, 1}:
        raise ValueError(
            f"{name} must contain exactly the values 0 and 1 (1 = default); found {vals.tolist()[:5]}. "
            "Encode the target before modeling, e.g. y = (loan_status == 'Charged Off').astype(int).")
    return y.astype(int)


def _pos_weight(y):
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    return (len(y) - pos) / pos


def _candidates(y_train, fast=False):
    """Candidate models with small hyperparameter search spaces."""
    spw = _pos_weight(y_train)
    cands = {
        "Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=SEED)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=SEED),
            {"n_estimators": [100, 200] if not fast else [100],
             "max_depth": [4, 6, 10, None],
             "min_samples_leaf": [1, 5, 10]},
        ),
    }
    if HAS_XGB:
        cands["XGBoost"] = (
            XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          n_jobs=-1, random_state=SEED, verbosity=0),
            {"n_estimators": [100, 200] if not fast else [100],
             "max_depth": [3, 4, 6],
             "learning_rate": [0.03, 0.1, 0.2],
             "subsample": [0.8, 1.0]},
        )
    return cands


def _best_threshold(y_true, proba):
    """Threshold that maximises F1 (used on out-of-fold TRAIN predictions)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), f
    return best_t


def _evaluate(model, X_tr, y_tr, X_te, y_te, cv_splits=5):
    """Fit on train, choose threshold from OOF train predictions, test once."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    thr = _best_threshold(y_tr, oof)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    return {
        "proba": proba, "pred": pred, "threshold": thr, "train_time": train_time,
        "AUC": roc_auc_score(y_te, proba),
        "PR_AUC": average_precision_score(y_te, proba),
        "F1": f1_score(y_te, pred, zero_division=0),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "Accuracy": accuracy_score(y_te, pred),
    }


# --------------------------------------------------------------------------
# 3.2 model selection (train data only)
# --------------------------------------------------------------------------
def select_model(X_train, y_train, fast=False, cv_splits=5, n_iter=6):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    results, best = [], None
    for name, (est, grid) in _candidates(y_train, fast).items():
        n_combo = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(
            est, grid, n_iter=min(n_iter, n_combo), scoring="roc_auc", cv=skf,
            random_state=SEED, n_jobs=1, refit=False,
        )
        search.fit(X_train, y_train)
        row = {
            "Model": name,
            "CV_AUC": round(float(search.best_score_), 4),
            "CV_AUC_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
            "Best_Params": search.best_params_,
        }
        results.append(row)
        if best is None or row["CV_AUC"] > best[0]["CV_AUC"]:
            best = (row, est, search.best_params_)
    row, est, params = best
    est.set_params(**params)
    return row["Model"], est, results


# --------------------------------------------------------------------------
# 3.6 explainability
# --------------------------------------------------------------------------
def explain_with_shap(model, model_name, X_train, X_test, features, top_k=8,
                      max_rows=500):
    """Returns (shap_summary list[str], shap_values, X_sample) or ([], None, None)."""
    try:
        import shap
        Xs = X_test[features].iloc[:max_rows]
        if model_name == "Logistic Regression":
            scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
            Xt = pd.DataFrame(scaler.transform(Xs), columns=features, index=Xs.index)
            bg = pd.DataFrame(scaler.transform(X_train[features].iloc[:100]),
                              columns=features)
            vals = shap.LinearExplainer(clf, bg).shap_values(Xt)
        else:
            vals = shap.TreeExplainer(model).shap_values(Xs)
        if isinstance(vals, list):          # older shap: [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                  # newer shap: (n, features, classes)
            vals = vals[:, :, 1]

        mean_abs = np.abs(vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        summary = []
        for i in order:
            col = Xs.iloc[:, i].to_numpy(dtype=float)
            corr = np.corrcoef(col, vals[:, i])[0, 1] if col.std() > 0 else 0.0
            direction = "raise" if corr >= 0 else "lower"
            summary.append(
                f"{features[i]} (mean |SHAP| {mean_abs[i]:.3f}; higher values {direction} default risk)"
            )
        return summary, vals, Xs
    except Exception as e:
        print(f"[modeling] SHAP skipped: {e}")
        return [], None, None


# --------------------------------------------------------------------------
# 3.7 charts (Visualizer pass 2)
# --------------------------------------------------------------------------
def _style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRIMARY, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def make_charts(out_dir, y_test, final, pso_history, comparison_table,
                shap_vals, shap_X):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, final["proba"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(fpr, tpr, color=PRIMARY, lw=2, label=f"Model (AUC = {final['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    _style(ax, "ROC Curve (test set)"); ax.legend(frameon=False)
    p = os.path.join(out_dir, "roc_curve.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # Confusion matrix
    cm = confusion_matrix(y_test, final["pred"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["No default", "Default"]); ax.set_yticks([0, 1], ["No default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)", fontsize=12, fontweight="bold", color=PRIMARY, loc="left")
    p = os.path.join(out_dir, "confusion_matrix.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # PSO convergence
    if pso_history is not None and len(pso_history) > 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(range(1, len(pso_history) + 1), pso_history, color=ACCENT, lw=2, marker="o", ms=3)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness (lower is better)")
        _style(ax, "PSO Convergence")
        p = os.path.join(out_dir, "pso_convergence.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        paths.append(p)

    # Method comparison
    if comparison_table:
        names = [r["Method"] for r in comparison_table]
        aucs = [r["ROC_AUC"] for r in comparison_table]
        colors = [PRIMARY if "PSO" in n else MUTED for n in names]
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bars = ax.bar(names, aucs, color=colors)
        for b, r in zip(bars, comparison_table):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{r['ROC_AUC']:.3f}\n({r['Num_Features']} feats)", ha="center", va="bottom", fontsize=8)
        lo = max(0.0, min(aucs) - 0.05)
        ax.set_ylim(lo, min(1.0, max(aucs) + 0.06)); ax.set_ylabel("Test ROC-AUC")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        _style(ax, "Feature Selection Methods Compared")
        p = os.path.join(out_dir, "method_comparison.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        paths.append(p)

    # SHAP importance
    if shap_vals is not None:
        imp = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(imp)[::-1][:10][::-1]
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.barh([shap_X.columns[i] for i in order], imp[order], color=ACCENT)
        ax.set_xlabel("Mean |SHAP value|")
        _style(ax, "")
        fig.suptitle("Top Default-Risk Drivers (SHAP)", x=0.02, ha="left", fontsize=12, fontweight="bold", color=PRIMARY)
        p = os.path.join(out_dir, "shap_importance.png"); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        paths.append(p)

    return paths


# --------------------------------------------------------------------------
# main entry point: LangGraph node
# --------------------------------------------------------------------------
def modeling_node(state, out_dir="reports", fast=False):
    cols = state.get("feature_cols")
    X_train = _as_df(state["X_train"], cols, "X_train")
    X_test = _as_df(state["X_test"], cols, "X_test")
    y_train = _check_target(state["y_train"], "y_train")
    y_test = _check_target(state["y_test"], "y_test")

    all_feats = list(X_train.columns)
    wanted = state.get("selected_features")
    if wanted:
        missing = [f for f in wanted if f not in all_feats]
        if len(missing) == len(wanted):
            raise ValueError(
                "None of state['selected_features'] exist in X_train. "
                f"Selected: {list(wanted)[:5]}... X_train columns: {all_feats[:5]}...")
        if missing:
            print(f"[modeling] WARNING: {len(missing)} selected feature(s) not in X_train and ignored: {missing}")
        selected = [f for f in wanted if f in all_feats]
    else:
        print("[modeling] WARNING: no selected_features in state, using all features")
        selected = all_feats

    # 3.2 model selection on PSO-selected features, TRAIN data only
    best_name, best_est, cv_results = select_model(X_train[selected], y_train, fast=fast)

    # 3.3 / 3.4 final training, single test evaluation
    from sklearn.base import clone
    final_model = clone(best_est)
    final = _evaluate(final_model, X_train[selected], y_train, X_test[selected], y_test)

    metrics = {
        "AUC": round(final["AUC"], 4), "PR_AUC": round(final["PR_AUC"], 4),
        "F1": round(final["F1"], 4), "Precision": round(final["Precision"], 4),
        "Recall": round(final["Recall"], 4), "Accuracy": round(final["Accuracy"], 4),
        "Threshold": round(final["threshold"], 2),
        "Train_Time_s": round(final["train_time"], 2),
        "Num_Features": len(selected),
    }

    # 3.5 baseline comparison: same model type, same split, same settings
    methods = {"All Features": all_feats}
    for name, feats in (state.get("baseline_feature_sets") or {}).items():
        kept = [f for f in feats if f in all_feats]
        if not kept:
            print(f"[modeling] WARNING: baseline '{name}' has no features present in X_train, skipped")
            continue
        if len(kept) < len(feats):
            print(f"[modeling] WARNING: baseline '{name}' lost {len(feats) - len(kept)} unknown feature(s)")
        methods[name] = kept
    methods["PSO (Ours)"] = selected

    comparison_table = []
    for name, feats in methods.items():
        if name == "PSO (Ours)":
            r = final  # already computed, avoid refitting
        else:
            r = _evaluate(clone(best_est), X_train[feats], y_train, X_test[feats], y_test)
        comparison_table.append({
            "Method": name, "Num_Features": len(feats),
            "ROC_AUC": round(r["AUC"], 3), "F1": round(r["F1"], 3),
            "Precision": round(r["Precision"], 3), "Recall": round(r["Recall"], 3),
            "Train_Time_s": round(r["train_time"], 2),
        })

    # 3.6 SHAP
    shap_summary, shap_vals, shap_X = explain_with_shap(
        final_model, best_name, X_train, X_test, selected)

    # 3.7 charts
    model_charts = make_charts(out_dir, y_test, final, state.get("pso_history"),
                               comparison_table, shap_vals, shap_X)

    # 3.8 save model
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "credit_model.joblib")
    joblib.dump({"model": final_model, "features": selected,
                 "threshold": final["threshold"], "model_name": best_name}, model_path)

    state.update({
        "best_model_name": best_name, "model_path": model_path,
        "metrics": metrics, "comparison_table": comparison_table,
        "shap_summary": shap_summary, "model_charts": model_charts,
        "cv_results": cv_results,
    })
    return state
