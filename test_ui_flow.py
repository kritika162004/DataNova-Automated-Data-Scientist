"""
test_ui_flow.py - walks the four Streamlit screens headlessly (Streamlit's AppTest harness).

    python test_ui_flow.py          # real engine, German Credit sample, quick PSO settings, no API key
    python test_ui_flow.py mock     # fake data, no engine
"""
import os
import sys

MOCK = len(sys.argv) > 1 and sys.argv[1] == "mock"
os.environ["ADS_USE_MOCK"] = "1" if MOCK else "0"
os.environ.pop("GROQ_API_KEY", None)

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=600).run()
assert not at.exception and at.session_state["stage"] == "upload"
[b for b in at.button if b.label == "Use sample dataset"][0].click().run()
assert not at.exception and at.session_state["stage"] == "checkpoint", at.exception
if not MOCK:
    assert at.session_state["target"] == "credit_risk"
    assert any("inverted" in w.value for w in at.warning), "polarity warning not shown on the review screen"
    at.slider[0].set_value(10)                         # particles
    at.slider[1].set_value(10)                         # iterations
    at.run()
[b for b in at.button if b.label.startswith("Approve and run")][0].click().run()
assert not at.exception and at.session_state["stage"] == "results", at.exception
vals = {m.label: m.value for m in at.metric}
assert {"auc", "f1", "model", "features kept", "pso time"} <= set(vals), vals
assert [t.label for t in at.tabs] == ["comparison", "features & shap", "charts", "report", "follow-up"]
[b for b in at.button if b.label == "Generate report"][0].click().run()
assert not at.exception and at.session_state["report_path"] and os.path.exists(at.session_state["report_path"])
print("metrics:", vals)
print("UI FLOW OK" + (" (mock)" if MOCK else " (real engine)"))
