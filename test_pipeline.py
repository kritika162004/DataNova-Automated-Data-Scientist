"""
test_pipeline.py - end-to-end test of the whole backend on the REAL German Credit data.

  python test_pipeline.py          # quick settings, no API key needed (the LLM is replaced by a stub)

Checks: every stage runs, the state contract holds, the report/pptx files exist, the number
check passes, a made-up LLM number is caught, scoring and the follow-up chat work, and the
human-checkpoint (interactive) mode pauses and resumes.
"""
import os
import re
import shutil

import pandas as pd

import llm_client, reporter, data_understanding, chat_followup
import pipeline

OUT = "reports_test_e2e"
shutil.rmtree(OUT, ignore_errors=True)
CSV = "data/german_credit.csv"
CFG = {**pipeline.QUICK_CONFIG, "out_dir": OUT}


def stub_llm(prompt: str) -> str:
    """Stands in for Groq so the test needs no key. Returns the format each caller expects."""
    if "ONLY valid JSON" in prompt:                      # data understanding
        return '{"summary": "German credit applicants.", "column_notes": {"duration": "loan length"}, "concerns": []}'
    if "Question:" in prompt:                            # follow-up chat
        return "The model relies mainly on the checking account status."
    return ("## Dataset Overview\n- The data has 800 training rows.\n"
            "## Model Performance\n- AUC is 0.9999 on the test set.\n"     # deliberately WRONG number
            "## Feature Selection and Benchmark\n- PSO kept fewer features.\n"
            "## Risk Drivers and Recommendations\n- Checking account status matters most.\n")


for mod in (llm_client, reporter, data_understanding, chat_followup):
    if hasattr(mod, "call_llm"):
        mod.call_llm = stub_llm

# ---- 1. full run, executive PDF ---------------------------------------------------------
final = pipeline.run_pipeline(CSV, report_style="executive", config=CFG)
for k in ["X_train", "X_test", "y_train", "y_test", "feature_cols", "target_col", "leakage_cols",
          "imbalance_ratio", "eda_charts", "selected_features", "baseline_feature_sets", "pso_history",
          "pso_runtime", "best_model_name", "model_path", "metrics", "comparison_table", "shap_summary",
          "model_charts", "cv_results", "pdf_path", "report_warnings", "scoring_bundle", "results_json"]:
    assert k in final, f"missing state key: {k}"
assert set(final["y_train"].unique()) == {0, 1}
assert set(final["selected_features"]) <= set(final["feature_cols"])
assert os.path.getsize(final["pdf_path"]) > 10_000
assert len(final["eda_charts"]) >= 3 and len(final["model_charts"]) >= 4
assert any("0.9999" in w for w in final["report_warnings"]), "made-up number was not caught"
print("1. full run OK:", final["best_model_name"], final["metrics"]["AUC"])

# ---- 2. technical PDF and PowerPoint ----------------------------------------------------
for style in ("technical", "presentation"):
    st = dict(final); st["report_style"] = style
    st = reporter.reporter_node(st, out_dir=OUT)
    assert os.path.getsize(st["report_path"]) > 10_000, style
    print(f"2. {style} OK ->", st["report_path"])

# ---- 3. scoring new applicants ------------------------------------------------------------
from scoring import score_applicants
raw = pd.read_csv(CSV).drop(columns=["credit_risk"]).head(20)
scores = score_applicants(raw, final["scoring_bundle"])
assert len(scores) == 20 and scores["default_probability"].between(0, 1).all()
print("3. scoring OK:", scores["risk_band"].value_counts().to_dict())

# ---- 4. follow-up chat ----------------------------------------------------------------------
ans = chat_followup.answer_finance_question("Which feature matters most?", final, [])
assert isinstance(ans, str) and len(ans) > 10
print("4. chat OK:", ans)

# ---- 5. human checkpoint: analyze first, the user decides, then run with the decisions ----
props = pipeline.analyze_dataset(CSV)
assert props["target_col"] == "credit_risk" and props["target_mapping"] == {"1": 0, "0": 1}
assert any("inverted" in w for w in props["warnings"])           # the 1=good polarity was caught
done = pipeline.run_pipeline(CSV, decisions={"cols_to_drop": ["telephone"]}, config=CFG)
assert "telephone" in done["dropped_cols"] and not any(c.startswith("telephone") for c in done["feature_cols"])
print("5. human checkpoint OK: proposals shown, user dropped 'telephone', pipeline respected it")

print("\nALL END-TO-END CHECKS PASSED")
