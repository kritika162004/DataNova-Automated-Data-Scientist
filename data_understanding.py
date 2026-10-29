"""
data_understanding.py - "the LLM understands the data".

Asks the LLM to describe the dataset in plain words, explain what each kept column probably
means for credit risk, and comment on the target / leakage choices the profiler proposed.
It is advice only: if the LLM is unavailable the pipeline continues with a rule-based summary.
"""
import json
from llm_client import call_llm


def _fallback(profile: dict, gate: dict) -> dict:
    return {
        "summary": (f"{profile['total_rows']} rows and {profile['total_columns']} columns. "
                    f"Proposed target: {profile.get('target_col')}. "
                    f"Class balance: {profile.get('imbalance_ratio')}."),
        "column_notes": {},
        "concerns": list(profile.get("warnings", [])),
        "source": "rule-based (LLM not used)",
    }


def understand_dataset(df, profile: dict, gate: dict) -> dict:
    cols = {c: str(df[c].dtype) for c in df.columns}
    sample = df.head(3).to_dict(orient="records")
    prompt = f"""You are a credit-risk data analyst. A user uploaded a loan dataset.

Columns (name: dtype): {json.dumps(cols)}
Sample rows: {json.dumps(sample, default=str)[:3000]}
Proposed target column: {profile.get('target_col')} (mapping to 1 = default: {profile.get('target_mapping')})
Default rate: {profile.get('imbalance_ratio')}
Columns proposed for removal: {json.dumps(profile.get('suggested_drops', {}))}

Reply with ONLY valid JSON:
{{"summary": "<2 sentences on what this dataset is>",
  "column_notes": {{"<column>": "<5-10 words on what it means for credit risk>"}},
  "concerns": ["<possible data leakage, wrong target polarity, fairness-sensitive columns, etc.>"]}}
Include column_notes for at most 12 important columns. Do not invent numbers."""
    try:
        raw = call_llm(prompt)
        start, end = raw.find("{"), raw.rfind("}") + 1
        out = json.loads(raw[start:end])
        out.setdefault("column_notes", {})
        out.setdefault("concerns", [])
        out["source"] = "LLM"
        return out
    except Exception as e:  # LLM missing, offline, or bad JSON: never block the pipeline
        print(f"  [understanding] LLM skipped: {e}")
        return _fallback(profile, gate)
