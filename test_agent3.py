from reporter import reporter_node

fake_finance_state = {
    "report_style": "presentation",
    "target_col": "loan_status",
    "leakage_cols": ["recoveries", "total_rec_prncp", "last_payment_amnt"],
    "imbalance_ratio": "8.4% default (1:11 ratio)",
    "selected_features": [
        "annual_income", "debt_to_income", "credit_score",
        "revol_util", "inq_last_6mths", "delinq_2yrs"
    ],
    "pso_runtime": 14.8,
    "best_model_name": "XGBoost Classifier",
    "metrics": {
        "AUC": 0.884,
        "F1": 0.612,
        "Precision": 0.655,
        "Recall": 0.574
    },
    "comparison_table": [
        {"Method": "All Features", "Num_Features": 28, "ROC_AUC": 0.841, "F1": 0.542},
        {"Method": "Mutual Info", "Num_Features": 10, "ROC_AUC": 0.852, "F1": 0.561},
        {"Method": "RFE", "Num_Features": 8, "ROC_AUC": 0.860, "F1": 0.573},
        {"Method": "PSO (Ours)", "Num_Features": 6, "ROC_AUC": 0.884, "F1": 0.612},
    ],
    "shap_summary": [
        "debt_to_income (+0.42 to default probability)",
        "credit_score (-0.38 to default probability)",
        "inq_last_6mths (+0.25 to default probability)"
    ],
    "chart_paths": []
}

result = reporter_node(fake_finance_state)
print("File generated at:", result["pdf_path"])
