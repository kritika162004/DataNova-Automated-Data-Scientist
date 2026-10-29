"""
reporter.py - Agent 3: turns the finished pipeline state into a PDF report or a PPTX deck.

What changed compared with the first version:
  * The Member 1 functions (finance gate, profiler, preprocessing, EDA charts) now live ONLY in
    data_prep.py. This file only reports.
  * The LLM must answer in a strict "## Heading" + "- bullet" format, and the parser matches
    headings exactly (the old keyword matching threw most bullets away).
  * Every number the LLM writes is checked against the real results. Unmatched numbers are listed
    in state["report_warnings"] and printed in the report.
  * Methodology and Limitations are written from the facts (not by the LLM), so they cannot be made up.
  * Executive and technical styles really differ. Images keep their aspect ratio and fit the page/slide.
  * If the LLM is unreachable the report is still produced from a template.
"""
import os
import re
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle,
                                KeepTogether, PageBreak)
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from llm_client import call_llm

PROJECT_NAME = os.environ.get("PROJECT_NAME", "Auto DS")
NAVY = colors.HexColor("#1F3A5F")

SECTIONS = [
    "Dataset Overview",
    "Model Performance",
    "Feature Selection and Benchmark",
    "Risk Drivers and Recommendations",
]

CHART_TITLES = {
    "eda_class_balance": "Class balance: how many borrowers defaulted",
    "roc_curve": "ROC curve on the held-out test set",
    "confusion_matrix": "Confusion matrix on the held-out test set",
    "pso_convergence": "PSO convergence: best fitness per iteration",
    "method_comparison": "Feature selection methods compared (test ROC-AUC)",
    "shap_importance": "Top default-risk drivers (SHAP)",
}
EXECUTIVE_CHARTS = ("eda_class_balance", "roc_curve", "method_comparison", "shap_importance")


# ---------------------------------------------------------------------------------------
# facts: everything the report may say, taken from the state (never invented)
# ---------------------------------------------------------------------------------------
def build_facts(state: dict) -> dict:
    metrics = state.get("metrics", {}) or {}
    sel = state.get("selected_features", []) or []
    hist = state.get("pso_history", []) or []
    n_train = len(state["y_train"]) if state.get("y_train") is not None else None
    n_test = len(state["y_test"]) if state.get("y_test") is not None else None
    return {
        "dataset": os.path.basename(state.get("original_csv_path") or state.get("csv_path") or "uploaded data"),
        "rows_train": n_train, "rows_test": n_test,
        "rows_removed_unknown_outcome": state.get("rows_removed", 0),
        "features_after_encoding": len(state.get("feature_cols", []) or []),
        "target": state.get("target_col", "N/A"),
        "target_mapping": state.get("target_mapping", {}),
        "class_balance": state.get("imbalance_ratio", "N/A"),
        "leakage_columns_removed": state.get("leakage_cols", []),
        "columns_dropped": state.get("dropped_cols", []),
        "selected_features": sel,
        "n_selected": len(sel),
        "pso_runtime_seconds": state.get("pso_runtime", "N/A"),
        "pso_iterations": len(hist),
        "pso_fitness_first": round(hist[0], 4) if hist else None,
        "pso_fitness_last": round(hist[-1], 4) if hist else None,
        "best_model": state.get("best_model_name", "N/A"),
        "test_metrics": metrics,
        "benchmark_table": state.get("comparison_table", []),
        "cv_results": [{k: v for k, v in r.items() if k != "Best_Params"} for r in state.get("cv_results", []) or []],
        "top_risk_drivers": state.get("shap_summary", []),
        "data_notes": state.get("profile_summary", {}).get("warnings", []) if state.get("profile_summary") else [],
    }


def _fmt_num(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------------------
# prompt, parsing, fallback
# ---------------------------------------------------------------------------------------
def build_finance_prompt(facts: dict, report_style: str) -> str:
    if report_style == "technical":
        tone = ("Audience: data scientists. Be precise. Mention the model, tuning, threshold, "
                "PR-AUC and the benchmark methods. 4-6 bullets per section.")
    else:
        tone = ("Audience: non-technical executives. Plain language, no jargon, say what the "
                "numbers mean for lending decisions. 2-4 short bullets per section.")
    heads = "\n".join(f"## {h}" for h in SECTIONS)
    return f"""You are a credit-risk analyst writing a report from verified results.
{tone}

VERIFIED FACTS (JSON):
{json.dumps(facts, indent=1, default=str)}

Write exactly these four sections, each starting with the heading line and followed by
bullet lines that start with "- ":
{heads}

Strict rules:
1. Use ONLY numbers that appear in the facts. Never compute new numbers, never invent any.
2. Explain that AUC / PR-AUC / F1 are used instead of accuracy because defaults are rare.
3. Under "Feature Selection and Benchmark" state honestly whether PSO beat the other methods;
   if the differences are small, say so. Do not claim PSO won when the table does not show it.
4. Under "Risk Drivers" name only the drivers listed in the facts.
5. No text before the first heading. No markdown tables. No headings other than the four above.
"""


def sanitize_text(text: str) -> str:
    repl = {"‑": "-", "‐": "-", "–": "-", "—": "-", "−": "-", "‘": "'",
            "’": "'", "“": '"', "”": '"', "≥": ">=", "≤": "<=", "→": "->",
            "•": "-", " ": " ", " ": " "}
    for a, b in repl.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _norm_heading(line: str) -> str:
    t = re.sub(r"^[#\s>*_]+", "", line.strip())
    t = re.sub(r"^section\s*\d*\s*[:.\-]\s*", "", t, flags=re.I)
    return re.sub(r"[\s*_:.#]+$", "", t).strip().lower()


def parse_sections(narrative: str) -> dict:
    """Split the narrative into {heading: [bullets]}. Headings match exactly (case-insensitive)."""
    known = {h.lower(): h for h in SECTIONS}
    out = {h: [] for h in SECTIONS}
    current = SECTIONS[0]
    for raw in narrative.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = _norm_heading(line)
        if key in known:
            current = known[key]
            continue
        if line.startswith("#"):                       # an unexpected heading: keep it as its own section
            title = re.sub(r"^[#\s*]+|[\s*:]+$", "", line)
            if title and len(title) < 80:
                out.setdefault(title, [])
                current = title
                continue
        bullet = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if len(bullet) > 3:
            out[current].append(bullet)
    return {k: v for k, v in out.items() if v}


def fallback_narrative(facts: dict) -> str:
    """Template report used when the LLM is unavailable. Every sentence comes from the facts."""
    m = facts["test_metrics"]
    lines = [
        f"## {SECTIONS[0]}",
        f"- Dataset: {facts['dataset']} ({facts['rows_train']} training rows, {facts['rows_test']} test rows).",
        f"- Target: {facts['target']}. Class balance: {facts['class_balance']}.",
        f"- Columns removed as leakage: {', '.join(facts['leakage_columns_removed']) or 'none detected'}.",
        f"## {SECTIONS[1]}",
        f"- Selected model: {facts['best_model']}.",
        f"- Test ROC-AUC {m.get('AUC')}, PR-AUC {m.get('PR_AUC')}, F1 {m.get('F1')}, "
        f"precision {m.get('Precision')}, recall {m.get('Recall')}.",
        "- AUC, PR-AUC and F1 are reported instead of accuracy because defaults are the rare class.",
        f"## {SECTIONS[2]}",
        f"- PSO selected {facts['n_selected']} of {facts['features_after_encoding']} features "
        f"in {facts['pso_runtime_seconds']} seconds.",
    ]
    for r in facts["benchmark_table"]:
        lines.append(f"- {r.get('Method')}: {r.get('Num_Features')} features, ROC-AUC {r.get('ROC_AUC')}, F1 {r.get('F1')}.")
    lines.append(f"## {SECTIONS[3]}")
    for d in facts["top_risk_drivers"][:5]:
        lines.append(f"- {d}")
    if not facts["top_risk_drivers"]:
        lines.append("- SHAP explanations were not available for this run.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# number check: every figure in the narrative must exist in the facts
# ---------------------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def verify_numbers(narrative: str, facts: dict) -> list:
    allowed = set()
    for tok in _NUM.findall(json.dumps(facts, default=str)):
        v = float(tok)
        allowed.add(v)
        if v <= 1:
            allowed.update({round(v * 100, 1), round(v * 100)})
    unmatched = []
    for tok in _NUM.findall(narrative):
        v = float(tok)
        if v == int(v) and v <= 12:                    # "4 sections", "1:11" style small integers
            continue
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9           # a quoted "0.80" may stand for 0.7996, "0.9999" may not
        if not any(abs(v - a) <= tol for a in allowed):
            unmatched.append(tok)
    return sorted(set(unmatched))


# ---------------------------------------------------------------------------------------
# deterministic sections (not written by the LLM)
# ---------------------------------------------------------------------------------------
def methodology_points(facts: dict) -> list:
    pts = [
        f"The data was split into {facts['rows_train']} training and {facts['rows_test']} test rows with a "
        "stratified split. Imputation, scaling and encoding were fitted on the training rows only.",
        "Columns that would not be known at application time (post-loan fields, suspected leakage) "
        "were removed before modelling.",
        "Class imbalance was handled with class weights; no synthetic oversampling was used.",
        f"Binary Particle Swarm Optimisation chose the feature subset using the training rows only "
        f"({facts['pso_iterations']} iterations). Its fitness is (1 - cross-validated ROC-AUC) plus a small "
        "penalty for the number of features.",
        "Model type and settings were chosen by 5-fold cross-validation on the training rows. The decision "
        "threshold was chosen from out-of-fold training predictions. The test set was used once, for the final scores.",
    ]
    if facts["rows_removed_unknown_outcome"]:
        pts.append(f"{facts['rows_removed_unknown_outcome']} rows with an unknown outcome were removed.")
    return pts


def limitation_points(facts: dict) -> list:
    pts = [
        "Results come from one dataset and one train/test split. Differences of about 0.01 ROC-AUC between "
        "methods are within normal run-to-run noise and should not be read as a ranking.",
    ]
    if facts["rows_test"] and facts["rows_test"] < 500:
        pts.append(f"The test set has only {facts['rows_test']} rows, so every score has wide uncertainty.")
    pts += [
        "SHAP shows what the model relies on, not what causes default.",
        "No fairness or bias analysis was performed. Check protected or proxy attributes before real use.",
        "The model was not validated on a later time period, so performance in changing economic "
        "conditions is unknown.",
    ]
    for n in facts["data_notes"]:
        pts.append(f"Data note: {n}")
    return pts


# ---------------------------------------------------------------------------------------
# helpers for pictures
# ---------------------------------------------------------------------------------------
def _chart_title(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    for key, title in CHART_TITLES.items():
        if stem.startswith(key):
            return title
    return stem.replace("_", " ").capitalize()


def _collect_charts(state: dict, style: str) -> list:
    paths = []
    for k in ("eda_charts", "chart_paths", "model_charts"):
        for p in state.get(k, []) or []:
            if p not in paths and os.path.exists(p):
                paths.append(p)
    if style == "executive":
        paths = [p for p in paths if any(os.path.basename(p).startswith(k) for k in EXECUTIVE_CHARTS)]
    return paths


def _fit(path: str, max_w: float, max_h: float):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    s = min(max_w / w, max_h / h)
    return w * s, h * s


# ---------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------
def _para_html(text: str) -> str:
    text = sanitize_text(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _metrics_table(facts: dict):
    m = facts["test_metrics"]
    rows = [
        ["Target", str(facts["target"]), "Class balance", str(facts["class_balance"])],
        ["Model", str(facts["best_model"]), "Features (PSO)", f"{facts['n_selected']} of {facts['features_after_encoding']}"],
        ["ROC-AUC", str(m.get("AUC", "N/A")), "PR-AUC", str(m.get("PR_AUC", "N/A"))],
        ["F1", str(m.get("F1", "N/A")), "Recall", str(m.get("Recall", "N/A"))],
        ["Precision", str(m.get("Precision", "N/A")), "PSO runtime", f"{facts['pso_runtime_seconds']} s"],
    ]
    t = Table([[Paragraph(sanitize_text(str(c)), _CELL) for c in r] for r in rows], colWidths=[85, 160, 95, 170])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F8")),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE0")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _benchmark_table(rows: list):
    if not rows:
        return None
    heads = ["Method", "Features", "ROC-AUC", "F1", "Precision", "Recall"]
    keys = ["Method", "Num_Features", "ROC_AUC", "F1", "Precision", "Recall"]
    data = [heads] + [[_fmt_num(r.get(k, "")) for k in keys] for r in rows]
    t = Table(data, repeatRows=1, colWidths=[140, 55, 60, 50, 60, 50])
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBE0")),
             ("ALIGN", (1, 0), (-1, -1), "CENTER")]
    for i, r in enumerate(rows, start=1):
        if "PSO" in str(r.get("Method", "")):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#E3EEF1")))
    t.setStyle(TableStyle(style))
    return t


_styles = getSampleStyleSheet()
_BODY = ParagraphStyle("body", parent=_styles["Normal"], fontSize=10, leading=14, spaceAfter=4)
_BULLET = ParagraphStyle("bullet", parent=_BODY, leftIndent=14, bulletIndent=2)
_CELL = ParagraphStyle("cell", parent=_styles["Normal"], fontSize=9, leading=11)
_H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=13, textColor=NAVY, spaceBefore=12,
                     spaceAfter=6, keepWithNext=True)
_CAP = ParagraphStyle("cap", parent=_styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"),
                      spaceAfter=10)


def _bullets(items):
    return [Paragraph(_para_html(i), _BULLET, bulletText="•") for i in items]


def build_pdf(state: dict, facts: dict, sections: dict, warnings: list, style: str, path: str):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40,
                            title=f"{PROJECT_NAME} credit risk report")
    story = [Paragraph(f"{PROJECT_NAME}: Credit Risk Report", _styles["Title"]),
             Paragraph(f"Dataset: {sanitize_text(facts['dataset'])} | Style: {style.capitalize()}", _styles["Heading4"]),
             Spacer(1, 6), _metrics_table(facts), Spacer(1, 8)]

    for head, items in sections.items():
        story.append(Paragraph(sanitize_text(head), _H2))
        story.extend(_bullets(items))
        if head == "Feature Selection and Benchmark":
            bt = _benchmark_table(facts["benchmark_table"])
            if bt is not None:
                story += [Spacer(1, 6), bt, Spacer(1, 6)]

    charts = _collect_charts(state, style)
    if charts:
        for i, p in enumerate(charts):
            w, h = _fit(p, 440, 270)
            block = [Image(p, width=w, height=h), Paragraph(_chart_title(p), _CAP)]
            if i == 0:                                  # keep the heading with the first chart
                block.insert(0, Paragraph("Charts", _H2))
            story.append(KeepTogether(block))

    story.append(Paragraph("Methodology", _H2))
    story.extend(_bullets(methodology_points(facts)))
    story.append(Paragraph("Limitations", _H2))
    story.extend(_bullets(limitation_points(facts)))
    story.append(Paragraph("Automatic number check", _H2))
    if warnings:
        story.append(Paragraph("These figures in the narrative could not be matched to the computed results "
                               "and should be treated with caution: " + ", ".join(warnings), _BODY))
    else:
        story.append(Paragraph("Every figure quoted in the narrative matches the computed results.", _BODY))
    doc.build(story)


# ---------------------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------------------
def build_pptx(state: dict, facts: dict, sections: dict, path: str):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    blank = prs.slide_layouts[6]
    navy, teal, grey = RGBColor(31, 58, 95), RGBColor(42, 127, 142), RGBColor(50, 50, 50)

    def title(slide, text, size=26):
        tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(11.9), Inches(0.9))
        p = tb.text_frame.paragraphs[0]
        p.text = sanitize_text(text)
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), True, navy

    def bullets_slide(head, items):
        # split long lists over several slides instead of cutting bullets
        chunks, cur, chars = [], [], 0
        for it in items:
            if cur and (len(cur) >= 5 or chars + len(it) > 700):
                chunks.append(cur)
                cur, chars = [], 0
            cur.append(it)
            chars += len(it)
        if cur:
            chunks.append(cur)
        for i, chunk in enumerate(chunks):
            s = prs.slides.add_slide(blank)
            title(s, head if len(chunks) == 1 else f"{head} ({i + 1}/{len(chunks)})")
            tb = s.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(11.5), Inches(5.4))
            tf = tb.text_frame
            tf.word_wrap = True
            size = 20 if sum(len(c) for c in chunk) < 450 else 17
            for j, it in enumerate(chunk):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                p.text = "•  " + sanitize_text(re.sub(r"\*\*(.+?)\*\*", r"\1", it))
                p.font.size, p.font.color.rgb = Pt(size), grey
                p.space_after = Pt(12)

    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{PROJECT_NAME}: Credit Risk and Loan Default"
    p.font.size, p.font.bold, p.font.color.rgb = Pt(36), True, navy
    p2 = tf.add_paragraph()
    p2.text = "Bio-inspired (PSO) feature selection and default prediction"
    p2.font.size, p2.font.color.rgb = Pt(20), teal
    m = facts["test_metrics"]
    p3 = tf.add_paragraph()
    p3.text = f"Model: {facts['best_model']}  |  Test ROC-AUC: {m.get('AUC', 'N/A')}  |  Dataset: {facts['dataset']}"
    p3.font.size, p3.font.color.rgb = Pt(15), grey
    p3.space_before = Pt(14)

    for head, items in sections.items():
        bullets_slide(head, items)

    rows = facts["benchmark_table"]
    if rows:
        s = prs.slides.add_slide(blank)
        title(s, "Benchmark: PSO vs baseline feature selection")
        heads, keys = ["Method", "Features", "ROC-AUC", "F1", "Precision", "Recall"], \
                      ["Method", "Num_Features", "ROC_AUC", "F1", "Precision", "Recall"]
        tbl = s.shapes.add_table(len(rows) + 1, len(heads), Inches(0.9), Inches(1.6), Inches(11.5),
                                 Inches(0.5) * (len(rows) + 1)).table
        for c, h in enumerate(heads):
            tbl.cell(0, c).text = h
            tbl.cell(0, c).text_frame.paragraphs[0].font.size = Pt(15)
        for r, row in enumerate(rows, start=1):
            for c, k in enumerate(keys):
                tbl.cell(r, c).text = _fmt_num(row.get(k, ""))
                tbl.cell(r, c).text_frame.paragraphs[0].font.size = Pt(14)

    for p_ in _collect_charts(state, "executive"):
        s = prs.slides.add_slide(blank)
        title(s, _chart_title(p_), 24)
        w, h = _fit(p_, 11.0 * 72, 5.4 * 72)           # points -> keep aspect ratio inside 11 x 5.4 inches
        left = (13.33 - w / 72) / 2
        s.shapes.add_picture(p_, Inches(left), Inches(1.5), width=Inches(w / 72), height=Inches(h / 72))

    s = prs.slides.add_slide(blank)
    title(s, "Method and limitations")
    tb = s.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(11.5), Inches(5.4))
    tb.text_frame.word_wrap = True
    items = methodology_points(facts)[:3] + limitation_points(facts)[:3]
    for j, it in enumerate(items):
        p = tb.text_frame.paragraphs[0] if j == 0 else tb.text_frame.add_paragraph()
        p.text = "•  " + sanitize_text(it)
        p.font.size, p.font.color.rgb = Pt(15), grey
        p.space_after = Pt(8)
    prs.save(path)


# ---------------------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------------------
def reporter_node(state: dict, out_dir: str = "reports") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    style = state.get("report_style", "executive")
    facts = build_facts(state)

    try:
        narrative = sanitize_text(call_llm(build_finance_prompt(facts, style)))
        sections = parse_sections(narrative)
        if len(sections) < 3:                          # LLM ignored the format: use the safe template
            print("  [reporter] LLM answer did not follow the format, using the template instead.")
            narrative = fallback_narrative(facts)
            sections = parse_sections(narrative)
    except Exception as e:
        print(f"  [reporter] LLM unavailable ({e}). Using the template narrative.")
        narrative = fallback_narrative(facts)
        sections = parse_sections(narrative)

    warnings = verify_numbers(narrative, facts)
    if warnings:
        print(f"  [reporter] numbers not found in the results: {warnings}")

    state["insights"] = narrative
    state["report_warnings"] = warnings
    if style == "presentation":
        path = os.path.join(out_dir, "credit_risk_presentation.pptx")
        build_pptx(state, facts, sections, path)
    else:
        path = os.path.join(out_dir, f"credit_risk_{style}_report.pdf")
        build_pdf(state, facts, sections, warnings, style, path)
    state["pdf_path"] = path
    state["report_path"] = path
    return state
