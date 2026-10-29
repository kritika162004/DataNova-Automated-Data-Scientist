"""
test_chat_followup.py

Standalone test for Agent 4 (Conversational Follow-Up), using your real
Groq API key. Lets you interactively ask questions about a real dataset
from the terminal, without needing Streamlit or the rest of the pipeline
running.

Usage:
    export GROQ_API_KEY=your_real_key_here
    python test_chat_followup.py data/sample_cleaned.csv
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from profiler import profile_csv
from domain_classifier import classify_domain
from chat_followup import answer_followup_question


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_cleaned.csv"

    print(f"Loading and profiling: {csv_path}")
    report = profile_csv(csv_path)

    import pandas as pd
    df = pd.read_csv(csv_path)
    sample_rows = df.head(3).to_dict(orient="records")
    domain_result = classify_domain(report["schema"], sample_rows)

    print(f"Domain detected: {domain_result['domain']} "
          f"(confidence: {domain_result.get('confidence')})")
    print()
    print("Ask questions about this dataset. Type 'quit' or 'exit' to stop.\n")

    chat_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit","bye"):
            print("Ending chat session.")
            break
        if not question:
            continue

        answer = answer_followup_question(
            question=question,
            csv_path=csv_path,
            profile_report=report,
            domain=domain_result,
            chat_history=chat_history
        )

        print(f"\nAssistant: {answer}\n")

        chat_history.append({"question": question, "answer": answer})


if __name__ == "__main__":
    main()


#python test_chat_followup.py data/sample_cleaned.csv