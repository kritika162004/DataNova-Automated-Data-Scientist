"""
test_data_prep.py - runs the Member 1 stage on SYNTHETIC raw-style loan data (mock, not real).
Needs no CSV.  Run:  python test_data_prep.py
"""
import os
import warnings
import numpy as np
import pandas as pd

from data_prep import (finance_gate, smart_profiler, human_checkpoint_apply, preprocess_and_split,
                       prepare_for_modeling, encode_target)

warnings.filterwarnings("ignore")
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def raw_loans(n=6000, seed=7):
    """Looks like a raw Lending Club export: text percentages, text terms, ids, 'Current' loans, leakage."""
    rng = np.random.RandomState(seed)
    dti = rng.normal(18, 7, n).clip(0, 50)
    fico = rng.normal(690, 40, n).clip(540, 840)
    int_rate = (28 - (fico - 540) / 15 + rng.normal(0, 2, n)).clip(5, 30)
    inc = np.exp(rng.normal(11, 0.5, n))
    loan_amnt = rng.randint(1000, 35000, n)
    revol = rng.uniform(0, 100, n)
    delinq = rng.poisson(0.3, n)
    z = -5.0 + 0.05 * dti - 0.012 * (fico - 690) + 0.08 * int_rate + 0.6 * delinq - 0.3 * np.log(inc / 60000)
    bad = rng.rand(n) < 1 / (1 + np.exp(-z))
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    status[(~bad) & (rng.rand(n) < 0.15)] = "Current"             # outcome unknown -> must be removed
    status[bad & (rng.rand(n) < 0.05)] = "Late (31-120 days)"      # also counts as default
    revol_txt = np.array([f" {v:.1f}%" for v in revol], dtype=object)
    revol_txt[rng.rand(n) < 0.03] = None
    months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return pd.DataFrame({
        "id": np.arange(1000000, 1000000 + n),
        "member_id": np.arange(5000000, 5000000 + n),
        "loan_amnt": loan_amnt,
        "term": rng.choice([" 36 months", " 60 months"], n, p=[0.7, 0.3]),
        "int_rate": [f"{v:.2f}%" for v in int_rate],
        "installment": (loan_amnt / 30).round(2),
        "annual_inc": inc.round(0),
        "dti": dti.round(2),
        "fico_score": fico.round(0),
        "revol_util": revol_txt,
        "delinq_2yrs": delinq,
        "emp_title": [f"job_{i}" for i in rng.randint(0, 600, n)],
        "issue_d": [f"{months[rng.randint(12)]}-{rng.randint(2012, 2016)}" for _ in range(n)],
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car", "house"], n),
        "recoveries": np.where(bad, rng.uniform(100, 3000, n), 0).round(2),
        "total_rec_prncp": np.where(bad, rng.uniform(0, 5000, n), loan_amnt).round(2),
        "last_pymnt_amnt": np.where(bad, rng.uniform(0, 300, n), rng.uniform(200, 900, n)).round(2),
        "acct_health_score": np.where(bad, rng.normal(80, 3, n), rng.normal(20, 3, n)).round(2),  # hidden leak
        "loan_status": status,
    })


# ---------------------------------------------------------------------------------------
print("\n[1] Finance gate")
df = raw_loans()
g = finance_gate(df)
check(g["is_finance"], f"credit data recognised ({g['confidence']})")
check(not finance_gate(pd.DataFrame({"temp": [1, 2], "humidity": [3, 4], "wind": [5, 6], "city": ["a", "b"]}))["is_finance"],
      "weather data rejected")

# ---------------------------------------------------------------------------------------
print("\n[2] Profiler")
p = smart_profiler(df)
print("   target:", p["target_col"], "| mapping:", p["target_mapping"])
print("   imbalance:", p["imbalance_ratio"], "| unknown-outcome rows:", p["rows_with_unknown_outcome"])
print("   suggested drops:", p["suggested_drops"])
check(p["target_col"] == "loan_status", "target found: loan_status")
check(p["target_mapping"].get("Charged Off") == 1 and p["target_mapping"].get("Fully Paid") == 0,
      "Charged Off -> 1, Fully Paid -> 0")
check(p["target_mapping"].get("Current") is None, "'Current' loans have no label (will be removed)")
check(p["target_mapping"].get("Late (31-120 days)") == 1, "'Late' counts as default")
for c in ["recoveries", "total_rec_prncp", "last_pymnt_amnt"]:
    check(c in p["leakage_cols"], f"leakage by name: {c}")
check("acct_health_score" in p["leakage_cols"], "hidden leakage found by statistics: acct_health_score")
check("fico_score" not in p["leakage_cols"] and "int_rate" not in p["leakage_cols"], "legit strong features NOT flagged")
check({"id", "member_id"} <= set(p["id_cols"]), "id columns found")
check("emp_title" in p["suggested_drops"], "high-cardinality text column suggested for dropping")
check("issue_d" in p["suggested_drops"], "date column suggested for dropping")
check("int_rate" in p["numeric_cols"] and "term" in p["numeric_cols"] and "revol_util" in p["numeric_cols"],
      "'13.5%' and ' 36 months' text converted to numbers")
check(set(p["categorical_cols"]) == {"home_ownership", "purpose"}, f"categoricals: {p['categorical_cols']}")

# ---------------------------------------------------------------------------------------
print("\n[3] Checkpoint")
clean, info = human_checkpoint_apply(df, "loan_status", list(p["suggested_drops"]) + ["loan_status", "not_a_column"])
n_current = int((df.loan_status == "Current").sum())
check(info["rows_removed"] == n_current, f"{n_current} 'Current' rows removed")
check(set(clean.loan_status.unique()) == {0, 1}, "target is 0/1")
check("loan_status" in clean.columns, "target column cannot be dropped")
check("not_a_column" in info["ignored_drop_requests"], "unknown drop request ignored safely")

# ---------------------------------------------------------------------------------------
print("\n[4] Full stage: prepare_for_modeling")
out = prepare_for_modeling(df, chart_dir="charts_test")
Xtr, Xte, ytr, yte = out["X_train"], out["X_test"], out["y_train"], out["y_test"]
print("   X_train", Xtr.shape, "| X_test", Xte.shape, "| imbalance:", out["imbalance_ratio"])
print("   features:", out["feature_cols"])
check(isinstance(Xtr, pd.DataFrame) and isinstance(Xte, pd.DataFrame), "X_train / X_test are DataFrames with names")
check(Xtr.isna().sum().sum() == 0 and Xte.isna().sum().sum() == 0, "no missing values left")
check(all(pd.api.types.is_numeric_dtype(t) for t in Xtr.dtypes), "all features numeric")
check(set(np.unique(ytr)) == {0, 1} and set(np.unique(yte)) == {0, 1}, "y_train / y_test are 0/1")
check(abs(ytr.mean() - yte.mean()) < 0.005, f"stratified: default rate train {ytr.mean():.3f} vs test {yte.mean():.3f}")
check(not (set(out["dropped_cols"]) & set(out["feature_cols"])), "no dropped column survived into the features")
check(not any(c in out["feature_cols"] for c in ["id", "member_id", "recoveries", "acct_health_score"]), "ids and leakage not in features")
num_in = [c for c in ["int_rate", "dti", "fico_score"] if c in Xtr.columns]
check(np.allclose(Xtr[num_in].mean(), 0, atol=1e-8) and not np.allclose(Xte[num_in].mean(), 0, atol=1e-8),
      "scaler fitted on TRAIN only (train mean = 0 exactly, test mean is not)")
check(len(Xtr) + len(Xte) == len(df) - n_current, "row count = all rows minus unknown-outcome rows")
check(out["target_col"] == "loan_status" and "imbalance_ratio" in out and set(out["leakage_cols"]) >= {"recoveries"},
      "state keys target_col / leakage_cols / imbalance_ratio present")
check(len(out["eda_charts"]) >= 5 and all(os.path.exists(c) for c in out["eda_charts"]),
      f"{len(out['eda_charts'])} EDA charts written: {[os.path.basename(c) for c in out['eda_charts']]}")

# the old 5-value API still works
a, b, c, d, e = preprocess_and_split(clean, "loan_status")
check(len(e) == a.shape[1], "preprocess_and_split keeps its 5-value return")

# ---------------------------------------------------------------------------------------
print("\n[5] Other target formats")
rng = np.random.RandomState(1)
n = 1000
german = pd.DataFrame({"duration": rng.randint(6, 60, n), "credit_amount": rng.randint(300, 15000, n),
                       "age": rng.randint(19, 70, n), "checking_status": rng.choice(["<0", "0-200", ">200", "none"], n),
                       "purpose": rng.choice(["car", "radio", "education", "business"], n)})
german["class"] = np.where(rng.rand(n) < 0.3, 2, 1)               # 1 = good, 2 = bad, 30% bad
y, mapping, _ = encode_target(german["class"])
check(mapping == {2: 1, 1: 0}, f"numeric 1/2 target: minority class 2 -> default (mapping {mapping})")
o2 = prepare_for_modeling(german, chart_dir="charts_test_german", make_charts=False)
check(o2["target_col"] == "class" and abs(o2["y_train"].mean() - 0.3) < 0.03, "German-style data prepared")

try:
    encode_target(pd.Series(["A", "B", "C"] * 10, name="outcome"))
    check(False, "undecidable 3-value target should raise")
except ValueError as e:
    check("target_mapping" in str(e), "undecidable target raises a clear error that explains how to fix it")

y, m, _ = encode_target(pd.Series(["A", "B", "C"] * 10, name="outcome"), {"A": 1, "B": 0, "C": None})
check(y.isna().sum() == 10 and y.sum() == 10, "user-supplied target_mapping respected")

# ---------------------------------------------------------------------------------------
print("\n[6] Hand-off into the modeling stage (no glue code)")
try:
    from modeling import modeling_node
    sel = ["int_rate", "dti", "fico_score", "delinq_2yrs", "annual_inc", "revol_util"]
    state = dict(out)
    state.update({"selected_features": sel, "baseline_feature_sets": {"RFE": sel[:4], "Mutual Info": sel[:5]},
                  "pso_history": list(np.linspace(0.32, 0.2, 20))})
    res = modeling_node(state, out_dir="reports_test_dp", fast=True)
    check(res["metrics"]["Num_Features"] == 6, "modeling used exactly the 6 selected features")
    check([r["Method"] for r in res["comparison_table"]] == ["All Features", "RFE", "Mutual Info", "PSO (Ours)"],
          "all baselines present in the comparison table")
    print("   best model:", res["best_model_name"], "| AUC", res["metrics"]["AUC"], "(mock data)")
except ImportError:
    print("  (modeling.py not found next to this file, skipping the hand-off check)")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n - " + "\n - ".join(FAILS)))
