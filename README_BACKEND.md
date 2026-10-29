# Auto DS backend (credit risk + PSO)

## Run it
    pip install -r requirements.txt          # needs GROQ_API_KEY in .env for the LLM parts (everything else works without)
    python pipeline.py data/german_credit.csv executive            # or: technical | presentation   (add --quick for a fast demo)
    python test_pipeline.py                                        # end-to-end test, no API key needed
    python run_experiments.py data/german_credit.csv --seeds 10    # multi-split numbers for the paper

## Files
| File | Job |
|---|---|
| pipeline.py | LangGraph graph: analyze -> prepare -> pso -> model -> report. UI functions: analyze_dataset(), run_pipeline() |
| data_prep.py | Member 1: finance gate, target + polarity detection, leakage check, human-checkpoint cleaning, train-only preprocessing, split, EDA charts |
| data_understanding.py | LLM describes the dataset (rule-based fallback if no LLM) |
| profiler.py | data-quality report (missing values, bad dtypes, outliers); shown in the report and used by chat |
| pso_feature_selection.py | Member 2: binary PSO + baselines (Mutual Info, RFE, GA) + multi-seed experiments |
| modeling.py | Member 3: model selection, threshold, single test evaluation, benchmark table, SHAP, charts, model file |
| reporter.py | PDF / PPTX report; numbers come from results, LLM wording is checked against them |
| scoring.py | score new applicants with the trained model (saved bundle) |
| chat_followup.py | answer_finance_question(question, state): follow-up Q&A about the finished run |
| llm_client.py | the only LLM caller (Groq); lazy, retries, never crashes imports |
| run_experiments.py | repeated-split comparison PSO vs baselines (mean +/- std) |
| test_*.py | tests. test_pipeline.py is the main one |
| domain_classifier.py, fix_proposer.py, test_agent1.py | old EDA-only agent: no longer used by the finance graph |

## For the UI teammate
    props = pipeline.analyze_dataset(csv_path)                  # fast; show target, mapping, suggested_drops, warnings, data_understanding
    state = pipeline.run_pipeline(csv_path, decisions={"target_col": ..., "cols_to_drop": [...], "target_mapping": {...}},
                                  report_style="executive", on_progress=None)
    chat_followup.answer_finance_question(q, state, history)
    scoring.score_applicants(new_df, state["scoring_bundle"])
Show: state["pdf_path"], metrics, comparison_table, eda_charts, model_charts, shap_summary, report_warnings.
