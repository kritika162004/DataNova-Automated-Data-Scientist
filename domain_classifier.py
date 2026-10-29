import json
from llm_client import call_llm

DOMAIN_CATEGORIES = ["sales", "time-series", "survey", "healthcare", "generic"]


def build_classification_prompt(schema: dict, sample_rows: list[dict]) -> str:
    return f"""You are a data classification assistant. Classify the dataset below into
EXACTLY ONE of these categories: {DOMAIN_CATEGORIES}

Category definitions:
- "sales": transactional/purchase data, orders, revenue, products
- "time-series": data with a clear time/date progression meant to be analyzed over time (logs, sensor readings, stock prices)
- "survey": categorical/Likert-scale responses, opinions, ratings
- "healthcare": patient records, lab results, diagnoses, vitals, clinical trial data
- "generic": doesn't clearly fit any of the above

Dataset schema (column name -> data type):
{json.dumps(schema['columns'], indent=2)}

Sample rows:
{json.dumps(sample_rows, indent=2, default=str)}

Respond with ONLY valid JSON in this exact format, nothing else:
{{"domain": "<one of {DOMAIN_CATEGORIES}>", "confidence": <float 0-1>, "reason": "<one short sentence>"}}
"""


def classify_domain(schema: dict, sample_rows: list[dict]) -> dict:
    """Main function Agent 1 calls to get the domain classification."""
    prompt = build_classification_prompt(schema, sample_rows)
    raw_response = call_llm(prompt)

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        result = json.loads(raw_response[start:end])

    if result.get("domain") not in DOMAIN_CATEGORIES:
        result["domain"] = "generic"
        result["reason"] = "Model returned an invalid category; defaulted to generic."

    return result


if __name__ == "__main__":
    import pandas as pd
    from profiler import detect_schema

    df = pd.read_csv("data/sample.csv")
    schema = detect_schema(df)
    sample_rows = df.head(3).to_dict(orient="records")

    result = classify_domain(schema, sample_rows)
    print(json.dumps(result, indent=2))
