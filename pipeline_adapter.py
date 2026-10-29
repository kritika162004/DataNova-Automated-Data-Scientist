"""
Seam between the Streamlit UI (app.py) and the DataNova engine (pipeline.py, data_prep.py, modeling.py ...).

app.py never imports engine code directly; it only calls the functions below, and this file translates the
engine's output formats into the shapes the UI expects.

    ADS_USE_MOCK=1  -> fake data, no engine, no key needed (UI development / offline demo)
    ADS_USE_MOCK=0  -> the real engine (default). GROQ_API_KEY is optional: without it the LLM-written parts
                       (dataset description, fix suggestions, report narrative, chat answers) fall back to
                       rule-based or template output, and every number still comes from the engine.
"""
import os
import re
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(HERE, "data", "german_credit.csv")
USE_MOCK = os.getenv("ADS_USE_MOCK", "0") == "1"

if USE_MOCK:
    import mock_backend as mock
else:
    try:                                   # pick up a local .env (GROQ_API_KEY)
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# ------------------------------------------------------------------ helpers
def _has_key():
    return bool(os.getenv("GROQ_API_KEY"))


def _fix_id(fix):
    return f"{fix['column']}|{fix['issue_type']}"


def _write_csv(df, name):
    path = os.path.join(tempfile.mkdtemp(prefix="ads_"), name)
    df.to_csv(path, index=False)
    return path


_SHAP = re.compile(r"^(.*?) \(mean \|SHAP\| ([\d.]+); higher values (raise|lower)")


def _parse_shap(items):
    """modeling.explain_with_shap returns strings like
    'income (mean |SHAP| 0.312; higher values raise default risk)'."""
    out = []
    for s in items or []:
        m = _SHAP.match(str(s))
        out.append({"feature": m.group(1), "mean_abs_shap": float(m.group(2)), "direction": m.group(3)} if m
                   else {"feature": str(s), "mean_abs_shap": 0.0, "direction": ""})
    return out


def _normalise_comparison(rows):
    return [{"method": r["Method"], "n_features": r["Num_Features"], "auc": r["ROC_AUC"], "f1": r["F1"],
             "precision": r.get("Precision"), "recall": r.get("Recall")} for r in rows]


def _engine_state(state):
    """The engine's own metric / comparison / SHAP formats, which reporter.py and chat_followup.py expect."""
    return dict(state, metrics=state["metrics_raw"], comparison_table=state["comparison_raw"],
                shap_summary=state["shap_raw"])


# ------------------------------------------------------------ sample + checks
def sample_dataset():
    """The dataset behind the 'Use sample dataset' button."""
    return mock.make_mock_df() if USE_MOCK else pd.read_csv(SAMPLE_CSV)


def check_target(df, col):
    """None if the column can be used as the target, otherwise a message for the user."""
    if USE_MOCK:
        s = df[col].dropna()
        ok = s.nunique() == 2 and pd.api.types.is_numeric_dtype(s)
        return None if ok else "The target should be a numeric 0/1 column (1 = defaulted)."
    from data_prep import coerce_numeric_strings, encode_target
    try:
        encode_target(coerce_numeric_strings(df[[col]])[col])
        return None
    except ValueError as e:
        return str(e)


# ------------------------------------------------------------ 1. profiling
def profile_data(df):
    """Fast analysis for the 'Review' screen (no model training)."""
    if USE_MOCK:
        return mock.mock_profile(df)

    import pipeline
    from profiler import profile_csv

    csv_path = _write_csv(df, "uploaded.csv")
    props = pipeline.analyze_dataset(csv_path)
    warnings = list(props["warnings"])
    gate = props["finance_gate"]
    if not gate["is_finance"]:
        warnings.append("This does not look like credit or loan data. The pipeline is built for "
                        "loan-default prediction, so results and the report may not make sense.")
    understanding = props["data_understanding"]
    for note in understanding.get("concerns", []):
        if note not in warnings:
            warnings.append(note)

    fixes = []
    if _has_key():                         # optional, LLM-written data-quality fixes
        try:
            from fix_proposer import propose_fixes
            fixes = propose_fixes(profile_csv(csv_path)).get("fixes", [])
        except Exception as e:
            warnings.append(f"AI fix suggestions unavailable ({e}). You can still continue without them.")

    target = props["target_col"]
    flipped = props["target_mapping"] == {"1": 0, "0": 1}
    if target:
        reason = f"'{target}' looks like the outcome column."
        reason += (" Its values will be recoded so that 1 means default (in this file 1 is the majority class)."
                   if flipped else " 1 is treated as default.")
    else:
        reason = "No obvious target column found - please choose one."

    miss = df.isna().sum()
    return {
        "n_rows": len(df), "n_cols": df.shape[1], "missing_total": int(miss.sum()),
        "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(miss[c]),
                     "unique": int(df[c].nunique())} for c in df.columns],
        "proposed_target": target, "target_reason": reason,
        "leakage_candidates": [{"col": c, "reason": r} for c, r in props["suggested_drops"].items()],
        "suggested_fixes": [{"id": _fix_id(f), "default": True, "raw": f,
                             "description": f"{f['column']}: {f['suggested_action']} - {f['reason']}"} for f in fixes],
        "imbalance_ratio": props["imbalance_ratio"],
        "domain": "credit / loan data" if gate["is_finance"] else "other data",
        "finance_ok": gate["is_finance"],
        "warnings": warnings,
        "summary_text": understanding.get("summary", ""),
    }


# ------------------------------------------------------------- 2. pipeline
def run_pipeline(df, target_col, leakage_cols, fixes, settings, progress_callback=None,
                 profile=None, stage_callback=None):
    """
    Runs everything after the human checkpoint with the engine's LangGraph pipeline:
    analyze -> prepare -> pso -> model -> report.

    fixes: fix ids ticked in the UI (profile_data -> suggested_fixes); they are applied to the data first.
    settings: {"particles", "iterations", "lambda"}.
    progress_callback(iteration, total, best_fitness): the engine reports PSO progress only when the search is
    finished, so the bar is replayed from the stored history and fills quickly.
    stage_callback(text): which engine stage is running.
    """
    if USE_MOCK:
        return mock.mock_run_pipeline(df, target_col, leakage_cols, fixes, settings, progress_callback)

    import pipeline
    say = stage_callback or (lambda _t: None)
    warnings, log = [], []

    by_id = {_fix_id(f["raw"]): f["raw"] for f in (profile or {}).get("suggested_fixes", [])}
    approved = [by_id[i] for i in fixes if i in by_id]
    clean = df.copy()
    if approved:
        from fix_proposer import apply_approved_fixes
        say("Applying approved data fixes...")
        clean, log = apply_approved_fixes(clean, approved)
    csv_path = _write_csv(clean, "cleaned.csv")

    config = {"out_dir": tempfile.mkdtemp(prefix="ads_out_"), "pso_particles": int(settings["particles"]),
              "pso_iters": int(settings["iterations"]), "pso_lambda": float(settings["lambda"])}
    decisions = {"target_col": target_col, "cols_to_drop": list(leakage_cols)}
    state = pipeline.run_pipeline(csv_path, decisions=decisions, report_style="executive", config=config,
                                  on_progress=lambda _stage, msg: say(msg))

    hist = state.get("pso_history") or []
    for i, f in enumerate(hist, 1):
        if progress_callback:
            progress_callback(i, len(hist), f)

    state["metrics_raw"] = state["metrics"]
    state["metrics"] = {k.lower(): v for k, v in state["metrics"].items()}
    state["comparison_raw"] = state["comparison_table"]
    state["comparison_table"] = _normalise_comparison(state["comparison_table"])
    state["shap_raw"] = state["shap_summary"]
    state["shap_summary"] = _parse_shap(state["shap_summary"])
    if state.get("report_warnings"):
        warnings.append("Numbers in the written report that were not found in the results: "
                        + ", ".join(state["report_warnings"]))
    state["profile"], state["fix_log"], state["warnings"] = profile, log, warnings
    state["leakage_cols"] = list(leakage_cols)
    return state


# ------------------------------------------------------------------ 3. report
def generate_report(state, style):
    """style: 'Executive' | 'Technical' | 'Presentation'. Returns the path of a .pdf (or .pptx)."""
    if USE_MOCK:
        return mock.mock_report(state, style)
    from reporter import reporter_node
    rs = dict(_engine_state(state), report_style=style.lower())
    return reporter_node(rs, out_dir=tempfile.mkdtemp(prefix="ads_report_"))["report_path"]


# ----------------------------------------------------------------------- Q&A
def answer_question(state, question, history=None):
    """history: UI chat messages [{'role','content'}]. Returns a plain string."""
    if USE_MOCK:
        return mock.mock_answer(state, question, history)
    from chat_followup import answer_finance_question
    pairs, pending = [], None
    for m in history or []:
        if m["role"] == "user":
            pending = m["content"]
        elif pending is not None:
            pairs.append({"question": pending, "answer": m["content"]})
            pending = None
    return answer_finance_question(question, _engine_state(state), pairs)
