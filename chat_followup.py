"""
chat_followup.py

Agent 4: Conversational Follow-Up.

Unlike Agents 1-3, this is NOT a node in the main LangGraph pipeline -
it's called on-demand, AFTER the pipeline finishes, whenever the user
types a question. It reuses the SAME shared LLM client as every other
agent, and reads from the already-cleaned data + profiling results
rather than re-running any analysis.
"""

import json

import pandas as pd
from llm_client import call_llm


def answer_followup_question(question: str, csv_path: str, profile_report: dict,
                               domain: dict, chat_history: list = None) -> str:
    """Answers a natural-language question about the dataset.

    Args:
        question: the user's typed question
        csv_path: path to the CLEANED csv (post Agent 1)
        profile_report: the data quality findings from Agent 1
        domain: the classified domain info from Agent 1
        chat_history: list of {"question": ..., "answer": ...} from earlier
                       turns in this session, so follow-ups can reference
                       previous answers (optional, defaults to none)

    Returns:
        The answer as plain text.
    """
    df = pd.read_csv(csv_path)

    # Give the model real, concrete grounding: actual column stats and a
    # small sample of real rows - NOT the whole dataset (could be huge).
    numeric_summary = df.describe(include="all").to_dict()
    sample_rows = df.head(5).to_dict(orient="records")

    history_text = ""
    if chat_history:
        history_text = "Previous conversation in this session:\n"
        for turn in chat_history[-3:]:  # only last 3 turns to keep prompt small
            history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n"

    prompt = f"""You are a data analysis assistant answering a user's question about
their dataset. Answer clearly and concisely, using only the information given below.
If the question cannot be answered from this data, say so honestly rather than guessing.

Dataset domain: {domain.get('domain', 'unknown')}

Data quality summary (from earlier profiling):
{profile_report}

Column statistics:
{numeric_summary}

Sample rows:
{sample_rows}

{history_text}
User's question: {question}

Answer:"""

    return call_llm(prompt)


if __name__ == "__main__":
    # Quick standalone test
    import json

    test_df = pd.DataFrame({
        "age": [25, 34, 29, 41, 38, 30, 27, 33],
        "income": [45000, 55500, 52000, 58000, 49000, 53000, 55500, 60000],
        "region": ["North", "South", "East", "North", "South", "West", "North", "South"]
    })
    test_df.to_csv("chat_test_data.csv", index=False)

    fake_profile_report = {
        "missing_values": {},
        "outliers": {},
        "schema": {"num_rows": 8, "num_columns": 3}
    }
    fake_domain = {"domain": "sales", "confidence": 0.9}

    print("Testing chat follow-up (mocked LLM response)...")


# ----------------------------------------------------------------------------------------
# Finance version: answers from the FINISHED run (metrics, PSO, SHAP, benchmark, data notes)
# ----------------------------------------------------------------------------------------
def _run_context(state: dict) -> dict:
    """Plain-JSON facts about the run. Everything the assistant may quote comes from here."""
    from reporter import build_facts
    facts = build_facts(state)
    facts["pso_history_first_last"] = ([state["pso_history"][0], state["pso_history"][-1]]
                                       if state.get("pso_history") else None)
    facts["data_quality"] = {k: v for k, v in (state.get("profile_report") or {}).items()
                             if k in ("missing_values", "dtype_inconsistencies")}
    facts["suggested_drops"] = (state.get("proposals") or {}).get("suggested_drops", {})
    facts["data_understanding"] = (state.get("data_understanding") or {}).get("summary", "")
    return facts


def answer_finance_question(question: str, state: dict, chat_history: list = None) -> str:
    """Answer a question about the credit-risk run (why a feature matters, how good the model is,
    what PSO did, limitations...). `state` is the final state returned by pipeline.run_pipeline."""
    ctx = _run_context(state)
    raw_sample = ""
    path = state.get("original_csv_path") or state.get("csv_path")
    try:
        raw_sample = pd.read_csv(path).head(3).to_dict(orient="records")
    except Exception:
        pass

    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n" + "".join(
            f"Q: {t['question']}\nA: {t['answer']}\n" for t in chat_history[-3:])

    prompt = f"""You are a credit-risk data scientist answering questions about ONE finished analysis.
Use ONLY the results below. Quote numbers exactly as given. If the answer is not in the results,
say so plainly instead of guessing. Do not claim causes: SHAP shows what the model relies on.
Keep the answer short (under 150 words) unless asked for detail.

RESULTS (JSON):
{json.dumps(ctx, indent=1, default=str)[:9000]}

First rows of the uploaded data: {json.dumps(raw_sample, default=str)[:1200]}

{history_text}
Question: {question}

Answer:"""
    try:
        return call_llm(prompt)
    except Exception as e:                     # no key / offline: still give something useful
        m = ctx.get("test_metrics", {})
        return (f"(The AI assistant is unavailable: {e}) Quick facts from the run: best model "
                f"{ctx.get('best_model')}, test ROC-AUC {m.get('AUC')}, F1 {m.get('F1')}; PSO kept "
                f"{ctx.get('n_selected')} of {ctx.get('features_after_encoding')} features; top drivers: "
                f"{'; '.join(ctx.get('top_risk_drivers', [])[:3])}.")
