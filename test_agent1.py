"""
test_agent1.py

Run this to test your ENTIRE Data Profiler agent end-to-end,
without LangGraph. This is exactly what the LangGraph node will
call later - if this works, wiring it into the graph is just
plumbing.

Usage:
    export GROQ_API_KEY=your_real_key_here
    python test_agent1.py data/sample.csv
    python test_agent1.py data/healthcare_sample.csv
"""

import sys
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads XAI_API_KEY from your .env file

from profiler import profile_csv, load_csv
from domain_classifier import classify_domain
from fix_proposer import propose_fixes, apply_approved_fixes


def run_agent1(csv_path: str):
    print(f"\n{'='*60}")
    print(f"TESTING AGENT 1 ON: {csv_path}")
    print(f"{'='*60}\n")

    # ---- Step 1: Profile the raw CSV (pure Pandas, no AI) ----
    print("[1/4] Profiling CSV...")
    report = profile_csv(csv_path)
    print(json.dumps(report, indent=2, default=str))

    # ---- Step 2: Classify the domain (LLM call) ----
    print("\n[2/4] Classifying domain...")
    df = load_csv(csv_path)
    sample_rows = df.head(3).to_dict(orient="records")
    domain_result = classify_domain(report["schema"], sample_rows)
    print(json.dumps(domain_result, indent=2))

    # ---- Step 3: Propose fixes (LLM call) ----
    print("\n[3/4] Proposing fixes...")
    fixes_result = propose_fixes(report)
    print(json.dumps(fixes_result, indent=2))

    # ---- Step 4: Simulate user approving ALL fixes, then apply them ----
    print("\n[4/4] Applying approved fixes (simulating user approving all)...")
    approved_fixes = fixes_result.get("fixes", [])
    cleaned_df, log = apply_approved_fixes(df, approved_fixes)

    print("\nChange log:")
    for entry in log:
        print(" -", entry)

    print("\nCleaned data preview:")
    print(cleaned_df.head())

    print(f"\n{'='*60}")
    print("AGENT 1 TEST COMPLETE")
    print(f"{'='*60}\n")

    return {
        "profile_report": report,
        "domain": domain_result,
        "fixes": fixes_result,
        "cleaned_df": cleaned_df,
        "change_log": log
    }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    run_agent1(csv_path)