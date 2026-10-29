# DataNova (merged project: engine + web interface)

Engine: `pipeline.py` (LangGraph: analyze -> prepare -> pso -> model -> report) and its modules (see README_BACKEND.md).
Interface: `app.py` (Streamlit) -> `pipeline_adapter.py` -> engine.  `mock_backend.py` gives fake data for UI work.

1. Create an environment (Python 3.10+, tested on 3.11):
       python -m venv .venv
       .venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
2. Install:                              pip install -r requirements.txt
3. Optional LLM key (new key from console.groq.com):
       copy .env.example .env            then edit .env and paste the key
   Without a key everything still runs: the dataset description, report text and chat answers fall back to
   rule-based or template output, and every number still comes from the engine.
4. Run the interface:                    streamlit run app.py
   Fake data, no engine:                 set ADS_USE_MOCK=1   (macOS/Linux: ADS_USE_MOCK=1 streamlit run app.py)
5. Open http://localhost:8501 and click "Use sample dataset" (German Credit, 1,000 loans).

Command line (no interface):             python pipeline.py data/german_credit.csv executive --quick
Tests (no key needed):                   python test_pipeline.py      python test_ui_adapter.py      python test_ui_flow.py
Never commit .env.
