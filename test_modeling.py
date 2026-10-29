"""
test_modeling.py - run Member 3's stage on SYNTHETIC (mock) credit data.
All numbers it produces are for testing only, not real results.
"""
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from modeling import modeling_node

# --- mock Member 1 output: imbalanced credit-style data (~8% default) ---
X, y = make_classification(n_samples=6000, n_features=20, n_informative=6,
                           n_redundant=4, weights=[0.92, 0.08], flip_y=0.01,
                           random_state=1)
names = ["annual_income", "debt_to_income", "credit_score", "revol_util",
         "inq_last_6mths", "delinq_2yrs", "loan_amnt", "int_rate", "term",
         "emp_length", "open_acc", "total_acc", "pub_rec", "revol_bal",
         "installment", "home_owner", "verif_status", "purpose_code",
         "age_of_credit", "noise_1"]
X = pd.DataFrame(X, columns=names)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# --- mock Member 2 output ---
mi = pd.Series(mutual_info_classif(X_train, y_train, random_state=0), index=names)
selected = list(mi.sort_values(ascending=False).index[:8])
state = {
    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    "selected_features": selected,
    "baseline_feature_sets": {
        "Mutual Info": selected,
        "RFE": list(mi.sort_values(ascending=False).index[:10]),
        "GA": list(mi.sort_values(ascending=False).index[:9]),
    },
    "pso_history": list(np.round(np.linspace(0.32, 0.18, 30) + np.random.RandomState(0).rand(30) * 0.01, 4)),
}

state = modeling_node(state, out_dir="reports_test", fast=True)

print("Best model     :", state["best_model_name"])
print("Metrics        :", json.dumps(state["metrics"], indent=1))
print("\nCV results:")
for r in state["cv_results"]:
    print("  ", r["Model"], r["CV_AUC"], "+/-", r["CV_AUC_std"])
print("\nComparison table:")
print(pd.DataFrame(state["comparison_table"]).to_string(index=False))
print("\nSHAP summary:")
for s in state["shap_summary"]:
    print("  ", s)
print("\nCharts:", state["model_charts"])
print("Model :", state["model_path"])

# contract check against what reporter.py reads
for k in ["best_model_name", "metrics", "comparison_table", "shap_summary", "model_charts"]:
    assert k in state, k
for k in ["AUC", "F1", "Precision", "Recall"]:
    assert k in state["metrics"], k
for row in state["comparison_table"]:
    for k in ["Method", "Num_Features", "ROC_AUC", "F1"]:
        assert k in row, k
print("\nContract with reporter.py: OK")