"""
data_prep.py  -  Member 1 stage: data understanding, human checkpoint, preprocessing, EDA charts.

Typical use (one call, output goes straight into the modeling stage):

    import pandas as pd
    from data_prep import prepare_for_modeling
    df = pd.read_csv("loans.csv")
    state.update(prepare_for_modeling(df))          # auto-approves the profiler's suggestions
    # or, after the user confirms things in Streamlit:
    state.update(prepare_for_modeling(df, target_col="loan_status",
                                      cols_to_drop=[...], target_mapping={"Charged Off": 1, "Fully Paid": 0}))

Keys written to the state (names match modeling.py and reporter.py):
    X_train, X_test      DataFrames (column names kept), cleaned, imputed, scaled / one-hot encoded
    y_train, y_test      0/1 Series, 1 = default
    feature_cols         list of final feature names
    target_col, leakage_cols, imbalance_ratio
    eda_charts           list of PNG paths (Visualizer pass 1)
    profile_summary, dropped_cols, target_mapping, rows_removed, finance_gate, preprocessor

Rules this module follows:
  * the train/test split happens BEFORE any imputer / scaler / encoder is fitted (no leakage)
  * the target is always converted to 0/1 here, so later stages never see text labels
  * leakage columns are found by name AND by a statistical check (one feature that almost
    perfectly predicts the outcome is almost certainly leakage)
"""
import os
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LEAKAGE_AUC = 0.95          # a single feature this predictive on credit default is suspicious
MAX_CATEGORY_LEVELS = 30    # more levels than this -> one-hot would explode, suggest dropping
MAX_MISSING_FRACTION = 0.60

NAVY, TEAL, WARN, MUTED = "#1F3A5F", "#2A7F8E", "#C8553D", "#B8C4CC"

CREDIT_KEYWORDS = [
    "income", "inc", "loan", "credit", "fico", "debt", "dti", "default", "delinq", "revol",
    "interest", "int_rate", "installment", "borrower", "mortgage", "grade", "emp_length",
    "term", "home_ownership", "purpose", "open_acc", "pub_rec", "util",
]
# exact name (checked first) or, for names of 5+ letters, contained in the column name
TARGET_NAMES = ["is_default", "default", "loan_status", "bad_loan", "defaulted", "delinquent",
                "charged_off", "target", "class", "risk", "label", "outcome", "y"]
LEAKAGE_NAME_PATTERNS = [
    "recover", "collection", "total_rec_", "total_pymnt", "last_pymnt", "next_pymnt",
    "out_prncp", "last_credit_pull", "settlement", "hardship", "chargeoff", "charged_off", "late_fee",
]
# target value words
_UNKNOWN_WORDS = ("current", "in grace", "issued", "active", "pending")
_GOOD_WORDS = ("fully paid", "paid off", "no default", "non-default", "not default", "good")
_BAD_WORDS = ("charged off", "chargedoff", "default", "late", "delinquent", "bad", "fail")


# ----------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------
_NUM_STRIP = re.compile(r"[,$%\s]|months?|years?", re.I)


def coerce_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text like '13.56%', ' 36 months', '$1,200' into numbers (only when >=95% convert)."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        nn = s.dropna()
        if nn.empty:
            continue
        sample = nn.astype(str).head(2000).str.replace(_NUM_STRIP, "", regex=True)
        if pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.95:
            out[c] = pd.to_numeric(s.astype(str).str.replace(_NUM_STRIP, "", regex=True), errors="coerce")
    return out


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """How well one numeric column alone ranks the outcome (0.5 = useless, 1.0 = perfect)."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x.fillna(x.median())
    if x.nunique() < 2 or y.nunique() < 2:
        return 0.5
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def _sample(df: pd.DataFrame, n: int = 50000) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=SEED)


# ----------------------------------------------------------------------------------------
# Task 1.1  finance gate
# ----------------------------------------------------------------------------------------
def finance_gate(df: pd.DataFrame) -> dict:
    """Does this CSV look like credit / loan data?"""
    cols = [str(c).lower() for c in df.columns]
    matched = [c for c in cols if any(k in c for k in CREDIT_KEYWORDS)]
    score = len(matched) / max(len(cols), 1)
    is_finance = score >= 0.25 or len(matched) >= 4
    return {
        "is_finance": bool(is_finance),
        "confidence": "High" if score >= 0.5 else ("Medium" if is_finance else "Low"),
        "matched_columns": matched,
        "reason": f"{len(matched)} of {len(cols)} columns look credit-related: {matched[:6]}",
    }


# ----------------------------------------------------------------------------------------
# target handling  (always ends as 0/1 with 1 = default)
# ----------------------------------------------------------------------------------------
def _classify_value(v) -> int | None:
    """1 = bad (default), 0 = good, None = unknown or undecidable (by words in the label)."""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if not isinstance(v, str):
        return None
    t = v.strip().lower()
    if any(w in t for w in _UNKNOWN_WORDS):
        return None
    if any(w in t for w in _GOOD_WORDS):
        return 0
    if any(w in t for w in _BAD_WORDS):
        return 1
    return None


def encode_target(series: pd.Series, mapping: dict | None = None):
    """
    Convert a target column to 0/1 (1 = default).  Returns (y, mapping, n_unlabeled).
    Rows whose label is unknown (e.g. 'Current' loans) get NaN and are dropped by the caller.
    `mapping` may be passed by the user: {original value: 1 | 0 | None}.
    """
    s = series
    values = list(s.dropna().unique())
    if mapping is None:
        if len(values) < 2:
            raise ValueError(f"Target '{s.name}' has fewer than 2 distinct values, cannot predict it.")
        if set(values) <= {0, 1, True, False}:
            mapping = {v: int(v) for v in values}
            n_one = int((s == 1).sum())
            name = str(s.name).lower()
            explicit = any(w in name for w in ("default", "bad", "charged", "delinq", "fraud"))
            # Credit defaults are the minority class.  If '1' is the majority and the column is not
            # explicitly named after a bad outcome (e.g. German Credit: credit_risk 1 = GOOD), flip it.
            if not explicit and n_one > 0.5 * s.notna().sum():
                mapping = {1: 0, 0: 1}
        else:
            cls = {v: _classify_value(v) for v in values}
            bad = [v for v, c in cls.items() if c == 1]
            good = [v for v, c in cls.items() if c == 0]
            if len(values) == 2:
                if len(bad) == 1:
                    mapping = {bad[0]: 1, [v for v in values if v != bad[0]][0]: 0}
                elif len(good) == 1:
                    mapping = {good[0]: 0, [v for v in values if v != good[0]][0]: 1}
                else:  # no wording hints: the rarer class is the default
                    counts = s.value_counts()
                    minority = counts.idxmin()
                    mapping = {minority: 1, [v for v in values if v != minority][0]: 0}
            elif bad and good:
                mapping = {v: c for v, c in cls.items()}   # None entries are dropped later
            else:
                raise ValueError(
                    f"Cannot tell which values of '{s.name}' mean default. Values: {values[:8]}. "
                    "Pass target_mapping={'<value>': 1 (default) or 0, ...}; use None to drop a value.")
    y = s.map(lambda v: mapping.get(v, None) if not (isinstance(v, float) and np.isnan(v)) else None)
    y = pd.to_numeric(y, errors="coerce")
    if y.dropna().nunique() < 2:
        raise ValueError(f"After mapping, target '{s.name}' has only one class. Check target_mapping: {mapping}")
    n_unlabeled = int(y.isna().sum())
    return y, mapping, n_unlabeled


def _target_candidates(df: pd.DataFrame) -> list:
    scored = []
    for c in df.columns:
        low = str(c).lower()
        rank = None
        for i, name in enumerate(TARGET_NAMES):
            if low == name:
                rank = i
                break
        if rank is None:
            for i, name in enumerate(TARGET_NAMES):
                if len(name) >= 4 and name in low:
                    rank = 100 + i
                    break
        if rank is not None and 2 <= df[c].nunique(dropna=True) <= 12:
            scored.append((rank, c))
    return [c for _, c in sorted(scored, key=lambda t: t[0])]


# ----------------------------------------------------------------------------------------
# Task 1.2  smart profiler
# ----------------------------------------------------------------------------------------
def smart_profiler(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """Propose the target, find leakage / id / unusable columns, measure imbalance."""
    df = coerce_numeric_strings(df)
    n = len(df)
    notes = []

    # ---- target
    candidates = _target_candidates(df)
    y = mapping = None
    target = target_col
    tried = [target_col] if target_col else candidates
    for cand in tried:
        if cand not in df.columns:
            notes.append(f"Target '{cand}' not found in the data.")
            continue
        try:
            y, mapping, n_unlabeled = encode_target(df[cand])
            target = cand
            break
        except ValueError as e:
            notes.append(str(e))
    if y is None:
        target = None
        n_unlabeled = 0

    imbalance_pct, imbalance_ratio = None, "Unknown"
    if y is not None:
        yy = y.dropna().astype(int)
        pos, neg = int((yy == 1).sum()), int((yy == 0).sum())
        imbalance_pct = 100.0 * pos / max(pos + neg, 1)
        imbalance_ratio = f"{imbalance_pct:.1f}% default rate (1:{max(round(neg / max(pos, 1)), 1)} ratio)"
        if imbalance_pct > 50:
            notes.append("Class 1 is the majority. Check that 1 really means default.")
        if mapping and set(mapping) <= {0, 1} and mapping.get(1) == 0 and mapping.get(0) == 1:
            notes.append(f"'{target}' looked inverted (1 was the majority class), so 0/1 were flipped "
                         "so that 1 = default. Please confirm this.")
        if n_unlabeled:
            notes.append(f"{n_unlabeled} rows have an unknown outcome (e.g. 'Current' loans) and will be removed.")

    # ---- drop suggestions
    leakage, reasons, drops = [], {}, {}
    others = [c for c in df.columns if c != target]

    for c in others:                                   # leakage by name
        low = str(c).lower()
        hit = next((p for p in LEAKAGE_NAME_PATTERNS if p in low), None)
        if hit:
            leakage.append(c)
            reasons[c] = f"name looks like a post-loan field ('{hit}')"

    if y is not None:                                  # leakage by statistics
        valid = y.notna()
        sub, yv = _sample(df[valid]), y[valid].astype(int)
        yv = yv.loc[sub.index]
        for c in others:
            if c in leakage or not _is_numeric(sub[c]):
                continue
            try:
                auc = _single_feature_auc(sub[c], yv)
            except Exception:
                continue
            if auc >= LEAKAGE_AUC:
                leakage.append(c)
                reasons[c] = f"suspected leakage: this one column alone separates the outcome (AUC {auc:.2f})"

    for c in leakage:
        drops[c] = reasons[c]

    id_cols = []
    for c in others:
        if c in drops:
            continue
        low = str(c).lower()
        s = df[c]
        if re.fullmatch(r"(id|index|unnamed: ?0|.*_id|id_.*|memberid|member_id|url)", low):
            id_cols.append(c)
            drops[c] = "identifier column"
        elif n >= 100 and s.nunique() / n >= 0.98 and (
                s.dtype == object or pd.api.types.is_string_dtype(s)
                or (pd.api.types.is_integer_dtype(s) and s.nunique() == n)):
            id_cols.append(c)
            drops[c] = "nearly every value is unique (looks like an identifier)"

    for c in others:
        if c in drops:
            continue
        s = df[c]
        miss = s.isna().mean()
        if s.nunique(dropna=True) <= 1:
            drops[c] = "constant column"
        elif miss > MAX_MISSING_FRACTION:
            drops[c] = f"{miss:.0%} of values are missing"
        elif not _is_numeric(s):
            if s.nunique() > MAX_CATEGORY_LEVELS:
                drops[c] = f"{s.nunique()} distinct text values (too many to one-hot encode)"
            else:
                sample = s.dropna().astype(str).head(200)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean()
                    except Exception:
                        parsed = 0.0
                if parsed >= 0.9 and sample.str.contains(r"\d").mean() > 0.9:
                    drops[c] = "date column (not used as a feature)"

    keep = [c for c in others if c not in drops]
    numeric_cols = [c for c in keep if _is_numeric(df[c])]
    categorical_cols = [c for c in keep if c not in numeric_cols]
    missing = (df[keep].isna().mean() * 100).round(1)
    missing = {c: float(v) for c, v in missing.items() if v > 0}

    return {
        "target_col": target,
        "target_candidates": candidates,
        "target_mapping": {str(k): v for k, v in (mapping or {}).items()},
        "imbalance_pct": imbalance_pct,
        "imbalance_ratio": imbalance_ratio,
        "leakage_cols": leakage,
        "leakage_reasons": reasons,
        "id_cols": id_cols,
        "suggested_drops": drops,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "recommended_encoding": {c: "one-hot" for c in categorical_cols},
        "missing_values_pct": missing,
        "rows_with_unknown_outcome": int(n_unlabeled),
        "total_rows": n,
        "total_columns": len(df.columns),
        "warnings": notes,
    }


# ----------------------------------------------------------------------------------------
# Task 1.3  human checkpoint
# ----------------------------------------------------------------------------------------
def human_checkpoint_apply(df: pd.DataFrame, target_col: str, cols_to_drop=(),
                           target_mapping: dict | None = None, keep_cols=()):
    """
    Apply the user's decisions: drop the approved columns, convert the target to 0/1 and remove
    rows with an unknown outcome.  Returns (clean_df, info).  The target column cannot be dropped.
    Call smart_profiler(clean_df, target_col) afterwards to re-profile.
    """
    df = coerce_numeric_strings(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:10]}...")
    y, mapping, _ = encode_target(df[target_col], target_mapping)

    drops = [c for c in cols_to_drop if c in df.columns and c != target_col and c not in set(keep_cols)]
    clean = df.drop(columns=drops).copy()
    clean[target_col] = y
    before = len(clean)
    clean = clean[clean[target_col].notna()].copy()
    clean[target_col] = clean[target_col].astype(int)
    info = {
        "dropped_cols": drops,
        "ignored_drop_requests": [c for c in cols_to_drop if c not in df.columns or c == target_col],
        "rows_removed": before - len(clean),
        "target_mapping": {str(k): v for k, v in mapping.items()},
    }
    return clean.reset_index(drop=True), info


# ----------------------------------------------------------------------------------------
# Task 1.4  preprocessing + split (fit on TRAIN only)
# ----------------------------------------------------------------------------------------

def clean_feature_names(cols) -> list:
    """XGBoost rejects [ ] < in feature names (e.g. 'status_... < 100 DM'). Make names safe + unique."""
    out, seen = [], {}
    for c in cols:
        n = str(c).replace("<=", "_le_").replace(">=", "_ge_").replace("<", "_lt_").replace(">", "_gt_")
        n = n.replace("[", "_").replace("]", "_")
        n = re.sub(r"[^0-9A-Za-z_]+", "_", n)
        n = re.sub(r"_+", "_", n).strip("_") or "feature"
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n}_{k}")
    return out

def _split_and_transform(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    df = coerce_numeric_strings(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    y, _, _ = encode_target(df[target_col])             # no-op if already 0/1
    keep = y.notna()
    X = df.drop(columns=[target_col])[keep].copy()
    y = y[keep].astype(int)

    if y.value_counts().min() < 2:
        raise ValueError("A class has fewer than 2 rows, so a stratified split is impossible.")
    for c in X.columns[X.dtypes == bool]:
        X[c] = X[c].astype(int)

    num_cols = [c for c in X.columns if _is_numeric(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    if cat_cols:                                         # plain object columns with real NaN
        X[cat_cols] = X[cat_cols].astype(object)
        X[cat_cols] = X[cat_cols].where(X[cat_cols].notna(), np.nan)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="pandas")

    Xtr = pre.fit_transform(X_tr)                        # fitted on TRAIN only
    Xte = pre.transform(X_te)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    Xtr.columns = Xte.columns = clean_feature_names(Xtr.columns)
    return (Xtr, Xte, y_tr.reset_index(drop=True), y_te.reset_index(drop=True),
            list(Xtr.columns), pre)


def preprocess_and_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                         random_state: int = SEED):
    """Returns X_train, X_test, y_train, y_test, feature_cols  (DataFrames / 0-1 Series)."""
    return _split_and_transform(df, target_col, test_size, random_state)[:5]


# ----------------------------------------------------------------------------------------
# Task 1.5  EDA charts (Visualizer pass 1)  -  matplotlib, so no Chrome / kaleido needed
# ----------------------------------------------------------------------------------------
def _style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_eda_charts(df: pd.DataFrame, target_col: str, output_dir: str = "charts") -> list:
    """Charts for the cleaned data (target already 0/1): balance, default rate by category,
    distributions by outcome, correlations, missing values."""
    os.makedirs(output_dir, exist_ok=True)
    df = _sample(coerce_numeric_strings(df), 100000)
    y = df[target_col].astype(int)
    paths = []

    # 1) class balance
    counts = y.value_counts().reindex([0, 1]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(["No default", "Default"], counts.values, color=[NAVY, WARN])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100 * v / counts.sum():.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Loans")
    _style(ax, "Class Balance")
    paths.append(_save(fig, os.path.join(output_dir, "eda_class_balance.png")))

    # 2) default rate by category (two most informative categoricals)
    overall = y.mean()
    scored = []
    for c in df.columns:
        if c == target_col or _is_numeric(df[c]) or not (2 <= df[c].nunique() <= 12):
            continue
        g = df.groupby(c)[target_col].agg(["mean", "size"])
        g = g[g["size"] >= 30]
        if len(g) >= 2:
            scored.append((g["mean"].max() - g["mean"].min(), c, g.sort_values("mean")))
    for _, c, g in sorted(scored, key=lambda t: t[0], reverse=True)[:2]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh([str(i) for i in g.index], g["mean"] * 100, color=TEAL, height=0.6)
        ax.axvline(overall * 100, color=WARN, ls="--", lw=1.2)
        for i, v in enumerate(g["mean"] * 100):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel(f"Default rate (%)   |   dashed line = overall rate ({overall:.1%})")
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
        ax.set_title(f"Default Rate by {c}", fontsize=12, fontweight="bold", color=NAVY, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(c))
        paths.append(_save(fig, os.path.join(output_dir, f"eda_default_rate_by_{safe}.png")))

    # 3) distributions of the most informative numeric columns, split by outcome
    num_cols = [c for c in df.columns if c != target_col and _is_numeric(df[c]) and df[c].nunique() > 5]
    if num_cols:
        sub = _sample(df, 30000)
        ranked = sorted(num_cols, key=lambda c: _single_feature_auc(sub[c], sub[target_col].astype(int)),
                        reverse=True)[:4]
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))
        for ax, c in zip(axes.ravel(), ranked):
            x = df[c].astype(float)
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            bins = np.linspace(lo, hi, 30) if hi > lo else 30
            for val, color, name in ((0, NAVY, "No default"), (1, WARN, "Default")):
                ax.hist(x[(y == val)].dropna().clip(lo, hi), bins=bins, density=True,
                        histtype="stepfilled", alpha=0.45, color=color, label=name)
            ax.set_title(c, fontsize=10, fontweight="bold", color=NAVY, loc="left")
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
        for ax in axes.ravel()[len(ranked):]:
            ax.axis("off")
        axes.ravel()[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Top Numeric Features by Outcome", fontsize=12, fontweight="bold", color=NAVY, x=0.02, ha="left")
        paths.append(_save(fig, os.path.join(output_dir, "eda_distributions.png")))

    # 4) correlation heatmap (features most related to the outcome)
    if len(num_cols) >= 2:
        corr_t = df[num_cols].corrwith(y).abs().sort_values(ascending=False)
        cols = list(corr_t.index[:9]) + [target_col]
        cm = df[cols].corr()
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols)), cols, fontsize=8)
        for (i, j), v in np.ndenumerate(cm.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Correlation (features most related to default)", fontsize=12,
                     fontweight="bold", color=NAVY, loc="left")
        paths.append(_save(fig, os.path.join(output_dir, "eda_correlation.png")))

    # 5) missing values
    miss = (df.drop(columns=[target_col]).isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(10)[::-1]
    if len(miss):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh(list(miss.index), miss.values, color=TEAL, height=0.5 if len(miss) < 4 else 0.7)
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Missing values (%)")
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
        ax.set_title("Missing Values", fontsize=12, fontweight="bold", color=NAVY, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        paths.append(_save(fig, os.path.join(output_dir, "eda_missing_values.png")))

    return paths


# ----------------------------------------------------------------------------------------
# one call for the whole stage
# ----------------------------------------------------------------------------------------
def prepare_for_modeling(df: pd.DataFrame, target_col: str | None = None, cols_to_drop=None,
                         keep_cols=(), target_mapping: dict | None = None, test_size: float = 0.2,
                         random_state: int = SEED, make_charts: bool = True,
                         chart_dir: str = "charts", require_finance: bool = False) -> dict:
    """
    Run the whole Member 1 stage.  Returns a dict to merge into the LangGraph state.
    With no arguments it auto-approves the profiler's suggestions; the UI passes the user's
    confirmed target_col / cols_to_drop / target_mapping instead.
    """
    df = coerce_numeric_strings(df)

    gate = finance_gate(df)
    if not gate["is_finance"]:
        msg = f"This does not look like credit / loan data ({gate['reason']})."
        if require_finance:
            raise ValueError(msg)
        print(f"[data_prep] WARNING: {msg}")

    prof = smart_profiler(df, target_col)
    target = target_col or prof["target_col"]
    if target is None:
        raise ValueError("Could not find a target column. Pass target_col=... " + " ".join(prof["warnings"]))

    drops = list(prof["suggested_drops"]) if cols_to_drop is None else list(cols_to_drop)
    clean, info = human_checkpoint_apply(df, target, drops, target_mapping, keep_cols)
    final = smart_profiler(clean, target)                     # re-profile after the fixes

    charts = generate_eda_charts(clean, target, chart_dir) if make_charts else []
    X_tr, X_te, y_tr, y_te, feats, pre = _split_and_transform(clean, target, test_size, random_state)

    return {
        "target_col": target,
        "leakage_cols": [c for c in prof["leakage_cols"] if c in info["dropped_cols"]],
        "imbalance_ratio": final["imbalance_ratio"],
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "feature_cols": feats,
        "eda_charts": charts,
        "profile_summary": prof,
        "dropped_cols": info["dropped_cols"],
        "target_mapping": info["target_mapping"],
        "rows_removed": info["rows_removed"],
        "finance_gate": gate,
        "preprocessor": pre,
    }
