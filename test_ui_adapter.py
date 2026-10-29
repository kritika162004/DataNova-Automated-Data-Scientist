"""
test_ui_adapter.py - the interface adapter on the REAL engine and the German Credit data. No API key needed.

    python test_ui_adapter.py

Checks: profile proposals, target polarity (1 = default after the fix), the run with quick PSO settings, the shapes the
UI reads, the report (PDF and PPTX) and the follow-up answer, all without an LLM key.
"""
import os

os.environ.pop("GROQ_API_KEY", None)
os.environ["ADS_USE_MOCK"] = "0"

import pipeline_adapter as ad

df = ad.sample_dataset()
assert df.shape == (1000, 21)

# ---- 1. profile ------------------------------------------------------------------------------------
p = ad.profile_data(df)
assert p["proposed_target"] == "credit_risk", p["proposed_target"]
assert p["domain"] == "credit / loan data" and p["finance_ok"]
assert any("inverted" in w for w in p["warnings"]), "polarity warning missing"
assert p["n_rows"] == 1000 and p["missing_total"] == 0 and p["suggested_fixes"] == []
assert ad.check_target(df, "credit_risk") is None
assert ad.check_target(df, "purpose") is not None, "a 10-level text column must not be accepted as a target"
print("1. profile OK:", p["proposed_target"], "|", p["imbalance_ratio"])

# ---- 2. run ----------------------------------------------------------------------------------------
stages, bars = [], []
r = ad.run_pipeline(df, p["proposed_target"], [c["col"] for c in p["leakage_candidates"]], [],
                    {"particles": 10, "iterations": 10, "lambda": 0.05},
                    progress_callback=lambda i, n, f=None: bars.append((i, n, f)), profile=p,
                    stage_callback=stages.append)
assert 0.28 < float(r["y_train"].mean()) < 0.32, "label not flipped: 1 must mean default (30%)"
assert set(r["metrics"]) >= {"auc", "f1", "precision", "recall", "pr_auc"}
assert 0.7 < r["metrics"]["auc"] < 0.9 and r["metrics"]["pr_auc"] < 0.8      # PR-AUC of the good class would be ~0.88
assert len(r["selected_features"]) < len(r["feature_cols"]) == 61
assert {"method", "auc"} <= set(r["comparison_table"][0])
assert r["shap_summary"] and r["shap_summary"][0]["direction"] in ("raise", "lower")
assert len(r["pso_history"]) == 10 and bars[-1][:2] == (10, 10)
assert len(r["eda_charts"]) >= 3 and len(r["model_charts"]) >= 4 and len(stages) >= 4
print("2. run OK:", r["best_model_name"], r["metrics"]["auc"], f"{len(r['selected_features'])}/61 features")

# ---- 3. report and chat -----------------------------------------------------------------------------
for style, ext in (("Executive", ".pdf"), ("Technical", ".pdf"), ("Presentation", ".pptx")):
    path = ad.generate_report(r, style)
    assert path.endswith(ext) and os.path.getsize(path) > 10_000, (style, path)
print("3. reports OK (pdf, pdf, pptx) with no API key")
ans = ad.answer_question(r, "Which feature matters most?", [])
assert isinstance(ans, str) and len(ans) > 20
print("4. follow-up OK:", ans[:90].replace("\n", " "), "...")
print("\nALL UI ADAPTER CHECKS PASSED")
