"""
pipeline.py - the LangGraph backend for the credit-risk Auto DS system.

Flow:
    analyze  ->  [human checkpoint]  ->  prepare  ->  pso  ->  model  ->  report

    analyze   profile the CSV, finance gate, propose target / columns to drop, LLM data understanding
    prepare   apply the user's decisions, clean, split (train/test), preprocess, EDA charts   (data_prep.py)
    pso       binary PSO feature selection + baselines, on TRAIN data only                   (pso_feature_selection.py)
    model     model selection, training, single test evaluation, SHAP, model charts          (modeling.py)
    report    PDF / PPTX report                                                              (reporter.py)

After the run, chat_followup.answer_finance_question() answers questions from the final state.

--------------------------------------------------------------------------------------------
HOW THE UI TEAM USES THIS (three functions, nothing else needed)
--------------------------------------------------------------------------------------------
1.  proposals = analyze_dataset(csv_path)
        Fast (no model training). Show to the user:
            proposals["target_col"], proposals["target_mapping"], proposals["imbalance_ratio"],
            proposals["suggested_drops"]   {column: reason},   proposals["warnings"],
            proposals["data_understanding"]["summary"] / ["column_notes"] / ["concerns"],
            proposals["finance_gate"]["is_finance"]
2.  state = run_pipeline(csv_path, decisions=..., report_style="executive", config=..., on_progress=fn)
        decisions (all optional; leave None to accept every proposal):
            {"target_col": "loan_status",
             "cols_to_drop": ["id", "recoveries"],          # the user's final list
             "keep_cols": [],                               # columns the user wants kept despite suggestions
             "target_mapping": {"Fully Paid": 0, "Charged Off": 1, "Current": None}}
        on_progress(stage, message) is called as each stage starts/ends (for a progress bar).
        Returns the final state (see keys below).
3.  answer = chat_followup.answer_finance_question(question, state, chat_history)

Final state keys worth showing: pdf_path / report_path, metrics, best_model_name, comparison_table,
selected_features, shap_summary, eda_charts, model_charts, pso_history, pso_runtime, cv_results,
report_warnings, model_path.

Run from the terminal:   python pipeline.py data/german_credit.csv executive
"""
import json
import os
import sys
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END

from profiler import profile_csv
from data_prep import coerce_numeric_strings, finance_gate, smart_profiler, prepare_for_modeling
from data_understanding import understand_dataset
from pso_feature_selection import run_feature_selection
from modeling import modeling_node
from scoring import save_scoring_bundle
from reporter import reporter_node, build_facts

DEFAULT_CONFIG = {
    "out_dir": "reports",
    "test_size": 0.2,
    "random_state": 42,
    "pso_particles": 20,
    "pso_iters": 30,
    "pso_lambda": 0.05,
    "pso_cv": 3,
    "pso_max_rows": 20000,
    "run_ga": True,
    "fast_models": False,       # True = smaller search grids (quick demo)
}
QUICK_CONFIG = {"pso_particles": 10, "pso_iters": 10, "fast_models": True}


class PipelineState(TypedDict, total=False):
    # inputs
    csv_path: str
    original_csv_path: str
    report_style: str                 # "executive" | "technical" | "presentation"
    config: dict
    decisions: dict                   # the user's checkpoint answers (optional)
    progress_cb: object

    # analyze
    finance_gate: dict
    profile_report: dict              # data-quality report (missing values, outliers, dtypes)
    profile_summary: dict             # smart profiler output (target, leakage, drops, warnings)
    data_understanding: dict          # LLM (or rule-based) description of the dataset
    proposals: dict

    # prepare (data_prep.py)
    target_col: str
    target_mapping: dict
    leakage_cols: list
    dropped_cols: list
    rows_removed: int
    imbalance_ratio: str
    X_train: object
    X_test: object
    y_train: object
    y_test: object
    feature_cols: list
    eda_charts: list
    preprocessor: object

    # pso (pso_feature_selection.py)
    selected_features: list
    best_fitness: float
    pso_history: list
    pso_runtime: float
    baseline_feature_sets: dict

    # model (modeling.py)
    best_model_name: str
    model_path: str
    metrics: dict
    comparison_table: list
    shap_summary: list
    model_charts: list
    cv_results: list
    scoring_bundle: str

    # report
    insights: str
    pdf_path: str
    report_path: str
    report_warnings: list
    results_json: str

    # follow-up chat
    chat_history: list


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _cfg(state) -> dict:
    return {**DEFAULT_CONFIG, **(state.get("config") or {})}


def _say(state, stage: str, message: str):
    print(f"[{stage}] {message}")
    cb = state.get("progress_cb")
    if callable(cb):
        try:
            cb(stage, message)
        except Exception:
            pass


def _json_safe(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


# ----------------------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------------------
def analyze_node(state: PipelineState) -> dict:
    _say(state, "analyze", "Profiling the CSV...")
    csv_path = state["csv_path"]
    df = coerce_numeric_strings(pd.read_csv(csv_path))

    gate = finance_gate(df)
    prof = smart_profiler(df)
    quality = profile_csv(csv_path)                    # missing values / dtype problems / outliers
    understanding = understand_dataset(df, prof, gate)

    proposals = {
        "target_col": prof["target_col"],
        "target_candidates": prof["target_candidates"],
        "target_mapping": prof["target_mapping"],
        "imbalance_ratio": prof["imbalance_ratio"],
        "suggested_drops": prof["suggested_drops"],
        "leakage_cols": prof["leakage_cols"],
        "leakage_reasons": prof["leakage_reasons"],
        "warnings": prof["warnings"],
        "finance_gate": gate,
        "data_understanding": understanding,
        "total_rows": prof["total_rows"],
        "total_columns": prof["total_columns"],
    }
    if not gate["is_finance"]:
        _say(state, "analyze", f"WARNING: {gate['reason']}")
    _say(state, "analyze", f"Proposed target: {prof['target_col']} ({prof['imbalance_ratio']}); "
                           f"{len(prof['suggested_drops'])} column(s) proposed for removal.")
    return {"finance_gate": gate, "profile_report": quality, "profile_summary": prof,
            "data_understanding": understanding, "proposals": proposals,
            "original_csv_path": state.get("original_csv_path", csv_path)}


def prepare_node(state: PipelineState) -> dict:
    cfg = _cfg(state)
    dec = state.get("decisions") or {}
    prop = state.get("proposals") or {}
    _say(state, "prepare", "Applying decisions, cleaning, splitting and preprocessing...")
    df = pd.read_csv(state["csv_path"])

    target = dec.get("target_col") or prop.get("target_col")
    drops = dec["cols_to_drop"] if dec.get("cols_to_drop") is not None else list(prop.get("suggested_drops", {}))
    out = prepare_for_modeling(
        df, target_col=target, cols_to_drop=drops, keep_cols=dec.get("keep_cols") or (),
        target_mapping=dec.get("target_mapping"),
        test_size=cfg["test_size"], random_state=cfg["random_state"],
        chart_dir=os.path.join(cfg["out_dir"], "charts"))
    _say(state, "prepare", f"Train rows: {len(out['y_train'])}, test rows: {len(out['y_test'])}, "
                           f"features after encoding: {len(out['feature_cols'])}, "
                           f"dropped columns: {len(out['dropped_cols'])}.")
    out.pop("finance_gate", None)                      # already set by analyze
    return out


def pso_node(state: PipelineState) -> dict:
    cfg = _cfg(state)
    _say(state, "pso", f"PSO feature selection ({cfg['pso_particles']} particles x {cfg['pso_iters']} iterations)...")
    n_feat = len(state["feature_cols"])
    if n_feat < 3:                                     # nothing to select from
        _say(state, "pso", "Fewer than 3 features: skipping selection and keeping all.")
        return {"selected_features": list(state["feature_cols"]), "best_fitness": float("nan"),
                "pso_history": [], "pso_runtime": 0.0, "baseline_feature_sets": {}}
    res = run_feature_selection(
        X_train=state["X_train"], y_train=state["y_train"], feature_cols=state["feature_cols"],
        n_particles=cfg["pso_particles"], iters=cfg["pso_iters"], lambda_=cfg["pso_lambda"],
        cv=cfg["pso_cv"], seed=cfg["random_state"], max_rows=cfg["pso_max_rows"], run_ga=cfg["run_ga"])
    _say(state, "pso", f"Selected {len(res['selected_features'])} of {n_feat} features in {res['pso_runtime']} s.")
    return {k: res[k] for k in ("selected_features", "best_fitness", "pso_history", "pso_runtime",
                                "baseline_feature_sets")}


def model_node(state: PipelineState) -> dict:
    cfg = _cfg(state)
    _say(state, "model", "Selecting the model, training, testing, explaining (SHAP)...")
    work = dict(state)
    work = modeling_node(work, out_dir=cfg["out_dir"], fast=cfg["fast_models"])
    keys = ("best_model_name", "model_path", "metrics", "comparison_table", "shap_summary",
            "model_charts", "cv_results", "scoring_bundle")
    m = work["metrics"]
    work["scoring_bundle"] = save_scoring_bundle(work, cfg["out_dir"])
    _say(state, "model", f"{work['best_model_name']}: test ROC-AUC {m['AUC']}, F1 {m['F1']}, "
                         f"recall {m['Recall']}, precision {m['Precision']}.")
    return {k: work[k] for k in keys}


def report_node(state: PipelineState) -> dict:
    cfg = _cfg(state)
    _say(state, "report", f"Writing the {state.get('report_style', 'executive')} report...")
    work = dict(state)
    work = reporter_node(work, out_dir=cfg["out_dir"])

    # a plain-JSON copy of the results: for the paper, the UI and the follow-up chat
    facts = build_facts(work)
    facts["pso_history"] = work.get("pso_history", [])
    facts["cv_results"] = work.get("cv_results", [])
    res_path = os.path.join(cfg["out_dir"], "run_summary.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, default=_json_safe)
    _say(state, "report", f"Saved {work['pdf_path']}")
    return {"insights": work["insights"], "pdf_path": work["pdf_path"], "report_path": work["report_path"],
            "report_warnings": work["report_warnings"], "results_json": res_path}


# ----------------------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------------------
def build_pipeline():
    """The graph: analyze -> prepare -> pso -> model -> report.

    The human checkpoint is NOT a graph pause. The UI calls analyze_dataset() first (fast), shows the
    proposals, then calls run_pipeline(csv, decisions=...) with the user's answers. This avoids storing
    DataFrames / fitted models in a checkpointer and works the same in Streamlit, scripts and tests.
    """
    g = StateGraph(PipelineState)
    g.add_node("analyze", analyze_node)
    g.add_node("prepare", prepare_node)
    g.add_node("pso", pso_node)
    g.add_node("model", model_node)
    g.add_node("report", report_node)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "prepare")
    g.add_edge("prepare", "pso")
    g.add_edge("pso", "model")
    g.add_edge("model", "report")
    g.add_edge("report", END)
    return g.compile()


# ----------------------------------------------------------------------------------------
# simple API for the UI
# ----------------------------------------------------------------------------------------
def analyze_dataset(csv_path: str) -> dict:
    """Step 1 for the UI: fast analysis. Returns the proposals to show at the checkpoint."""
    return analyze_node({"csv_path": csv_path})["proposals"]


def run_pipeline(csv_path: str, decisions: dict | None = None, report_style: str = "executive",
                 config: dict | None = None, on_progress=None) -> dict:
    """Step 2 for the UI: run everything and return the final state."""
    state: PipelineState = {
        "csv_path": csv_path, "original_csv_path": csv_path, "report_style": report_style,
        "config": config or {}, "decisions": decisions or {}, "progress_cb": on_progress,
        "chat_history": [],
    }
    return build_pipeline().invoke(state)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/german_credit.csv"
    style = sys.argv[2] if len(sys.argv) > 2 else "executive"
    quick = "--quick" in sys.argv
    final = run_pipeline(path, report_style=style, config=QUICK_CONFIG if quick else None)

    print("\n" + "=" * 60 + "\nPIPELINE COMPLETE\n" + "=" * 60)
    print("Target          :", final["target_col"], "|", final["imbalance_ratio"])
    print("Dropped columns :", final["dropped_cols"])
    print("PSO selected    :", len(final["selected_features"]), "of", len(final["feature_cols"]), "features")
    print("Best model      :", final["best_model_name"])
    print("Test metrics    :", final["metrics"])
    print("Report          :", final["report_path"])
    print("Run summary     :", final["results_json"])
    if final.get("report_warnings"):
        print("Numbers not matched to results:", final["report_warnings"])
