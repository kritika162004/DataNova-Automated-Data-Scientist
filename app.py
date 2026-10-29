"""DataNova - Streamlit UI ("Lab Notebook" design)."""
import hashlib
import html
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline_adapter as backend

st.set_page_config(page_title="DataNova", page_icon="📓", layout="wide", initial_sidebar_state="collapsed")

STAGES = ["upload", "review", "run", "results"]
STAGE_KEYS = ["upload", "checkpoint", "running", "results"]
DEFAULT_SETTINGS = {"particles": 20, "iterations": 30, "lambda": 0.05}
REPORT_STYLES = ["Executive", "Technical", "Presentation"]
STYLE_LABEL = {"Executive": "Executive summary", "Technical": "Technical deep-dive",
               "Presentation": "Presentation style"}
STYLE_DESC = {"Executive": "One page. Headline findings only, written for a decision-maker in a hurry.",
              "Technical": "Full methodology, every chart, statistical detail included.",
              "Presentation": "Slide-style output (.pptx), built to be projected or screen-shared."}

TEAL, AMBER, RED, INK, SOFT, LINE = "#1F6F63", "#B8722E", "#A8402F", "#1C2321", "#5B615C", "#D8D4C9"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--paper:#F7F6F2;--paper-alt:#EFEDE4;--ink:#1C2321;--ink-soft:#5B615C;--line:#D8D4C9;
      --teal:#1F6F63;--teal-dim:#DCE9E5;--amber:#B8722E;--amber-dim:#F1E3D1;--red:#A8402F;
      --serif:'Source Serif 4',Georgia,serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:var(--paper)!important;color:var(--ink);font-family:'Inter',system-ui,sans-serif;color-scheme:light;}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important;}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{max-width:1160px;padding:1.4rem 28px 5rem!important;}
h1,h2,h3,h4,h5{font-family:var(--serif)!important;font-weight:600!important;color:var(--ink)!important;letter-spacing:-0.01em;}
p,li,label,span{font-family:'Inter',system-ui,sans-serif;}
code{font-family:var(--mono)!important;background:var(--paper-alt)!important;color:var(--ink)!important;border-radius:3px;}

/* header + numbered nav */
.dn-top{display:flex;align-items:baseline;justify-content:space-between;}
.dn-brand{display:flex;align-items:baseline;gap:10px;}
.dn-brand b{font-family:var(--serif);font-size:24px;font-weight:600;letter-spacing:-0.01em;}
.dn-brand span,.dn-subject{font-family:var(--mono);font-size:12px;color:var(--ink-soft);}
.dn-nav{display:flex;gap:0;border-bottom:1px solid var(--line);margin:14px 0 30px;}
.dn-step{font-family:var(--mono);font-size:12.5px;color:var(--ink-soft);padding:13px 4px;margin-right:28px;
         border-bottom:2px solid transparent;white-space:nowrap;}
.dn-step .n{margin-right:8px;}
.dn-step.on{color:var(--ink);border-bottom-color:var(--teal);} .dn-step.on .n{color:var(--teal);}
.dn-step.done{color:var(--teal);}

/* headings + small pieces */
.dn-lede{margin-bottom:24px;}
.dn-lede h2{font-size:27px;margin:0;}
.dn-lede p{margin:6px 0 0;color:var(--ink-soft);font-size:14.5px;max-width:620px;}
.dn-h4{font-family:var(--serif);font-size:16px;font-weight:600;margin:0 0 2px;}
.dn-sub{font-family:var(--mono);font-size:11px;color:var(--ink-soft);margin-bottom:10px;}
.dn-mono{font-family:var(--mono);font-size:12.5px;color:var(--ink-soft);}
.dn-mono-h{font-family:var(--mono);font-size:11px;color:var(--ink-soft);text-transform:lowercase;}
.dn-tag{font-family:var(--mono);font-size:11px;padding:2px 7px;display:inline-block;border:1px solid var(--amber);color:var(--amber);}
.dn-tag.type{border-color:var(--red);color:var(--red);} .dn-tag.outlier{border-color:var(--ink-soft);color:var(--ink-soft);}
.dn-col{font-family:var(--mono);font-weight:500;font-size:13px;}
.dn-reason{color:var(--ink-soft);font-size:13.5px;margin-bottom:6px;}
.dn-domain{display:flex;align-items:center;gap:18px;border:1px solid var(--teal);background:var(--teal-dim);
           padding:14px 20px;margin:4px 0 24px;}
.dn-domain .tag{font-family:var(--mono);font-size:11.5px;color:var(--teal);border:1px solid var(--teal);padding:2px 7px;}
.dn-domain b{font-family:var(--serif);font-size:15px;} .dn-domain .conf{margin-left:auto;font-family:var(--mono);font-size:13px;color:var(--teal);}
.dn-steplog{font-family:var(--mono);font-size:12.5px;color:var(--ink-soft);line-height:2;margin-top:14px;}
.dn-steplog b{color:var(--teal);font-weight:500;}
.dn-foot{font-family:var(--mono);font-size:12.5px;color:var(--ink-soft);}

/* cards: any container created with key="card_*" */
[class*="st-key-card"]{background:#fff;border:1px solid var(--line);border-radius:3px;padding:16px 18px;}
[class*="st-key-ledger"]{background:#fff;border:1px solid var(--line);border-radius:3px;padding:6px 18px 14px;}

/* metrics */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:3px;padding:14px 16px;}
[data-testid="stMetricLabel"] p{font-family:var(--mono);font-size:11.5px;color:var(--ink-soft);text-transform:lowercase;}
[data-testid="stMetricValue"]{font-family:var(--serif);font-weight:600;font-size:1.4rem;color:var(--ink);}

/* tabs */
[data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{background:none;padding:13px 4px!important;margin-right:28px!important;}
button[data-baseweb="tab"] p,button[data-baseweb="tab"] div{font-family:var(--mono)!important;font-size:12.5px!important;color:var(--ink-soft);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink);}
[data-baseweb="tab-highlight"]{background:var(--teal)!important;height:2px!important;}
[data-baseweb="tab-border"]{display:none;}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:3px;font-weight:500;font-size:13.5px;border:1px solid var(--line);
  background:transparent;color:var(--ink);padding:.45rem 1.1rem;box-shadow:none;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--ink);color:var(--ink);background:transparent;}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--teal);border-color:var(--teal);color:#fff;}
.stButton>button[kind="primary"]:hover{background:#185a50;border-color:#185a50;color:#fff;}
.stButton>button:disabled{opacity:.45;}

/* inputs */
[data-testid="stFileUploaderDropzone"]{background:#fff;border:1px dashed var(--line);border-radius:0;padding:46px 24px;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]{background:#fff!important;border-radius:3px;border-color:var(--line)!important;}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:3px;}
[data-testid="stExpander"] summary p{font-family:var(--mono);font-size:12.5px;}
[data-testid="stAlert"]{border-radius:3px;border:1px solid var(--line);background:#fff;}
[data-testid="stProgress"] div[role="progressbar"]>div{background:var(--teal)!important;}
[data-testid="stProgress"] [data-baseweb="progress-bar"]>div{background:var(--line);}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:3px;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:3px;}
[data-testid="stCheckbox"] label span,[data-testid="stToggle"] label span{font-size:13.5px;}
</style>
"""


# ------------------------------------------------------------------ helpers
def esc(x):
    return html.escape(str(x))


def init_state():
    defaults = {"stage": "upload", "df": None, "profile": None, "target": None,
                "leakage": [], "fixes": [], "settings": dict(DEFAULT_SETTINGS),
                "result": None, "cache": {}, "chat_history": [], "report_path": None,
                "report_style": "Executive", "error": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go(stage):
    st.session_state.stage = stage
    st.rerun()


def reset_all():
    for k in list(st.session_state.keys()):
        if k != "cache":          # keep cache so re-running the same setup skips PSO
            del st.session_state[k]
    st.rerun()


def as_df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


def cache_key():
    s = st.session_state
    raw = (pd.util.hash_pandas_object(s.df, index=True).sum(), s.target,
           tuple(sorted(s.leakage)), tuple(sorted(s.fixes)), tuple(sorted(s.settings.items())))
    return hashlib.md5(str(raw).encode()).hexdigest()


def lede(title, text):
    st.markdown(f'<div class="dn-lede"><h2>{title}</h2><p>{text}</p></div>', unsafe_allow_html=True)


def card_title(title, sub=""):
    st.markdown(f'<div class="dn-h4">{esc(title)}</div><div class="dn-sub">{esc(sub)}</div>', unsafe_allow_html=True)


def style_fig(fig, title=None):
    fig.update_layout(font=dict(family="Inter, sans-serif", color=INK, size=13),
                      title=dict(text=title or "", font=dict(family="Source Serif 4, Georgia, serif", size=17), x=0),
                      paper_bgcolor="#fff", plot_bgcolor="#fff", colorway=[TEAL, AMBER, INK, SOFT, RED],
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), showlegend=fig.layout.showlegend)
    fig.update_xaxes(gridcolor="#EFEDE4", linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor="#EFEDE4", linecolor=LINE, zeroline=False)
    return fig


def header():
    s = st.session_state
    cur = STAGE_KEYS.index(s.stage)
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    mock = ' &nbsp;·&nbsp; <span style="color:#B8722E">mock data</span>' if backend.USE_MOCK else ""
    c1.markdown(f'<div class="dn-brand"><b>DataNova</b><span>automated data scientist agent{mock}</span></div>',
                unsafe_allow_html=True)
    if s.stage != "upload" and c2.button("Start over", key="start_over"):
        reset_all()
    steps = "".join(
        f'<span class="dn-step {"on" if i == cur else ("done" if i < cur else "")}">'
        f'<span class="n">{i + 1:02d}</span>{name}</span>' for i, name in enumerate(STAGES))
    st.markdown(f'<div class="dn-nav">{steps}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- screens
def screen_upload():
    lede("Bring a dataset, leave with a report.",
         "Upload a CSV of loan or credit data. The agents read its structure, check it for the problems that "
         "quietly ruin an analysis, and propose fixes for you to approve.")
    file = st.file_uploader("Drop a .csv file, or choose one", type=["csv"])
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    use_sample = c1.button("Use sample dataset")
    c2.markdown('<span class="dn-mono">%s</span>' % ("synthetic loan data, good for trying the app" if backend.USE_MOCK
                else "German Credit data, 1,000 loans, good for trying the app"), unsafe_allow_html=True)
    st.markdown('<div class="dn-foot" style="margin-top:18px">Your file stays in this session until you approve the '
                'pipeline running on it. The target column should say whether the borrower defaulted, as 0/1 or with words such as Fully Paid and Charged Off.</div>',
                unsafe_allow_html=True)
    if file or use_sample:
        try:
            df = pd.read_csv(file) if file else backend.sample_dataset()
            if df.shape[0] < 50 or df.shape[1] < 3:
                st.error("The file needs at least 50 rows and 3 columns.")
                return
            with st.spinner("Reading the schema and checking for problems..."):
                st.session_state.df = df
                st.session_state.profile = backend.profile_data(df)
            p = st.session_state.profile
            st.session_state.target = p["proposed_target"]
            st.session_state.leakage = [l["col"] for l in p["leakage_candidates"]]
            st.session_state.fixes = [f["id"] for f in p["suggested_fixes"] if f.get("default")]
            go("checkpoint")
        except Exception as e:
            st.error(f"Could not read or profile the file: {e}")


def _issue_class(issue_type):
    return {"missing_value": "", "dtype_inconsistency": "type", "outlier": "outlier"}.get(issue_type, "")


def _issue_label(issue_type):
    return {"missing_value": "missing", "dtype_inconsistency": "type", "outlier": "outlier"}.get(issue_type, "fix")


def _set_all_fixes(value):
    for f in st.session_state.profile["suggested_fixes"]:
        st.session_state[f"fix_{f['id']}"] = value


def screen_checkpoint():
    s, p = st.session_state, st.session_state.profile
    lede("Nothing gets fixed without your say-so.",
         "Confirm the target, the columns to drop and the data fixes. Nothing is applied or trained until you continue.")

    if p.get("domain"):
        st.markdown(f'<div class="dn-domain"><span class="tag">domain</span><b>{esc(p["domain"]).capitalize()}</b>'
                    f'<span class="dn-mono">{p["n_rows"]:,} rows · {p["n_cols"]} columns</span></div>',
                    unsafe_allow_html=True)
    for w in p.get("warnings", []):
        st.warning(w)

    ratio = str(p["imbalance_ratio"] or "n/a")
    short = re.search(r"\(1:(\d+) ratio\)", ratio)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("rows", f"{p['n_rows']:,}")
    m2.metric("columns", p["n_cols"])
    m3.metric("missing values", p["missing_total"])
    m4.metric("class ratio", f"1:{short.group(1)}" if short else ratio)
    extra = "" if "class balance" in p["summary_text"].lower() else f" &nbsp;·&nbsp; {esc(p['summary_text'])}"
    st.markdown(f'<div class="dn-foot" style="margin:10px 0 18px">class balance: {esc(ratio)}{extra}</div>',
                unsafe_allow_html=True)

    with st.expander("column overview"):
        st.dataframe(pd.DataFrame(p["columns"]), width="stretch", hide_index=True)
    with st.expander("data preview"):
        st.dataframe(s.df.head(20), width="stretch")

    cols = list(s.df.columns)
    st.markdown("#### Target column")
    st.markdown(f'<div class="dn-reason">{esc(p["target_reason"])}</div>', unsafe_allow_html=True)
    s.target = st.selectbox("Column to predict", cols, index=cols.index(s.target) if s.target in cols else None,
                            placeholder="Choose the column that says whether the loan defaulted")

    st.markdown("#### Columns to drop")
    st.markdown('<div class="dn-reason">Possible data leakage: information the model would not have in real life, '
                'which gives fake, inflated results.</div>', unsafe_allow_html=True)
    reasons = {l["col"]: l["reason"] for l in p["leakage_candidates"]}
    options = [c for c in cols if c != s.target]
    s.leakage = [c for c in s.leakage if c in options]
    s.leakage = st.multiselect("Drop these columns", options, default=s.leakage,
                               format_func=lambda c: f"{c} - {reasons[c]}" if c in reasons else c)

    st.markdown("#### Proposed fixes")
    fixes = p["suggested_fixes"]
    if not fixes:
        st.markdown('<div class="dn-reason">No data-quality fixes were proposed.</div>', unsafe_allow_html=True)
    else:
        widths = [0.9, 1.2, 1.7, 2.3, 0.7]
        with st.container(key="ledger_fixes"):
            head = st.columns(widths)
            for c, t in zip(head, ["issue", "column", "proposed fix", "reason", "approve"]):
                c.markdown(f'<span class="dn-mono-h">{t}</span>', unsafe_allow_html=True)
            for f in fixes:
                raw = f.get("raw") or {"column": "", "issue_type": "", "suggested_action": f["description"], "reason": ""}
                row = st.columns(widths, vertical_alignment="center")
                row[0].markdown(f'<span class="dn-tag {_issue_class(raw["issue_type"])}">'
                                f'{_issue_label(raw["issue_type"])}</span>', unsafe_allow_html=True)
                row[1].markdown(f'<span class="dn-col">{esc(raw["column"])}</span>', unsafe_allow_html=True)
                row[2].markdown(esc(raw["suggested_action"]).capitalize())
                row[3].markdown(f'<span class="dn-reason">{esc(raw["reason"])}</span>', unsafe_allow_html=True)
                key = f"fix_{f['id']}"
                if key not in st.session_state:
                    st.session_state[key] = f["id"] in s.fixes
                row[4].toggle("approve", key=key, label_visibility="collapsed")
            s.fixes = [f["id"] for f in fixes if st.session_state.get(f"fix_{f['id']}")]
            foot = st.columns([3, 1, 1], vertical_alignment="center")
            foot[0].markdown(f'<span class="dn-foot">{len(s.fixes)} of {len(fixes)} approved · '
                             f'{len(fixes) - len(s.fixes)} rejected</span>', unsafe_allow_html=True)
            foot[1].button("Approve all", key="appr_all", on_click=_set_all_fixes, args=(True,), width="stretch")
            foot[2].button("Reject all", key="rej_all", on_click=_set_all_fixes, args=(False,), width="stretch")

    st.markdown("#### Report style")
    s.report_style = st.radio("Report style", REPORT_STYLES, horizontal=True, label_visibility="collapsed",
                              index=REPORT_STYLES.index(s.report_style), format_func=lambda k: STYLE_LABEL[k])
    st.markdown(f'<div class="dn-reason">{STYLE_DESC[s.report_style]}</div>', unsafe_allow_html=True)

    with st.expander("advanced settings (the defaults work well)"):
        a, b, c = st.columns(3)
        s.settings["particles"] = a.slider("Particles", 5, 100, s.settings["particles"],
                                           help="How many candidate feature subsets PSO tries at once.")
        s.settings["iterations"] = b.slider("Iterations", 5, 200, s.settings["iterations"],
                                            help="How many rounds the swarm moves toward better subsets.")
        s.settings["lambda"] = c.slider("Lambda (feature penalty)", 0.0, 0.5, float(s.settings["lambda"]), 0.01,
                                        help="Higher = prefers fewer features. Fitness = (1 - AUC) + lambda x fraction kept.")
        if st.button("Reset to defaults"):
            s.settings = dict(DEFAULT_SETTINGS)
            st.rerun()

    target_ok = s.target is not None
    if target_ok:
        problem = backend.check_target(s.df, s.target)
        if problem:
            st.warning(f"{problem} Pick a different column or fix this one in your CSV first.")
            target_ok = False
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 2, 5])
    if c1.button("← Back"):
        go("upload")
    if c2.button("Approve and run analysis →", type="primary", disabled=not target_ok):
        go("running")


def screen_running():
    s = st.session_state
    lede("Running the pipeline.",
         "Applying your approved fixes, searching for the best features with PSO, then comparing models and "
         "explaining the winner.")
    key = cache_key()
    if key in s.cache:
        st.success("Same data and settings as a previous run - reusing saved results (PSO not repeated).")
        s.result = s.cache[key]
        go("results")

    bar = st.progress(0.0, text="Starting...")
    log_box = st.empty()
    best = st.empty()
    steps = []

    def render_log():
        lines = [f"<div><b>{'✓' if i < len(steps) - 1 else '…'}</b> {esc(t)}</div>" for i, t in enumerate(steps)]
        log_box.markdown(f'<div class="dn-steplog">{"".join(lines)}</div>', unsafe_allow_html=True)

    def stage_callback(text):
        steps.append(text)
        render_log()

    def progress_callback(iteration, total, best_fitness=None):
        bar.progress(iteration / total, text=f"PSO iteration {iteration}/{total}")
        if best_fitness is not None:
            best.metric("best fitness so far (lower is better)", f"{best_fitness:.4f}")

    try:
        result = backend.run_pipeline(s.df, s.target, s.leakage, s.fixes, s.settings, progress_callback,
                                      profile=s.profile, stage_callback=stage_callback)
    except Exception as e:
        st.error(f"The pipeline stopped: {e}")
        st.caption("Your choices are saved. Go back, adjust them and try again.")
        if st.button("← Back to review"):
            go("checkpoint")
        return
    s.cache[key] = result
    s.result = result
    s.chat_history, s.report_path = [], None
    go("results")


def screen_results():
    s, r = st.session_state, st.session_state.result
    lede("Charts first, report second.",
         "The model, how PSO-selected features compare with the baselines, and what drives the predictions. "
         "Pick how the report should be written before you download it.")
    m = r["metrics"]
    cols = st.columns(5)
    cols[0].metric("auc", m["auc"])
    cols[1].metric("f1", m["f1"])
    cols[2].metric("model", r["best_model_name"])
    n_all = len(r["feature_cols"]) if r.get("feature_cols") else \
        len([c for c in s.df.columns if c != s.target and c not in s.leakage])
    cols[3].metric("features kept", f"{len(r['selected_features'])} of {n_all}")
    cols[4].metric("pso time", f"{r['pso_runtime']}s")
    for w in r.get("warnings", []):
        st.warning(w)
    if m["auc"] >= 0.98:
        st.warning("An AUC this close to 1.0 on loan data is suspicious - it usually means a column that "
                   "reveals the outcome (data leakage) is still in the data. Go back and check the "
                   "'Columns to drop' list.")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["comparison", "features & shap", "charts", "report", "follow-up"])

    with t1:
        comp = as_df(r["comparison_table"])
        with st.container(key="card_comp"):
            card_title("PSO vs baselines", "test set · same model, same split")
            st.dataframe(comp, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1, st.container(key="card_auc"):
            card_title("Test AUC by method", "higher is better")
            colors = [TEAL if str(x).lower().startswith("pso") else "#C9C5B8" for x in comp["method"]]
            fig = px.bar(comp, x="method", y="auc", text="auc")
            fig.update_traces(marker_color=colors, textposition="outside")
            fig.update_layout(showlegend=False, yaxis_range=[max(0, comp["auc"].min() - 0.05), 1], xaxis_title=None)
            st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
        with c2, st.container(key="card_conv"):
            card_title("PSO convergence", "fitness per iteration · lower is better")
            hist = pd.DataFrame({"iteration": range(1, len(r["pso_history"]) + 1), "fitness": r["pso_history"]})
            fig = px.line(hist, x="iteration", y="fitness")
            fig.update_traces(line_color=TEAL)
            st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})

    with t2:
        c1, c2 = st.columns([1, 2])
        with c1, st.container(key="card_feats"):
            card_title("Selected features", f"{len(r['selected_features'])} kept by PSO")
            st.markdown("<br>".join(f'<span class="dn-col">{esc(f)}</span>' for f in r["selected_features"]),
                        unsafe_allow_html=True)
        with c2, st.container(key="card_shap"):
            card_title("What drives predictions", "mean |SHAP| · amber raises default risk, teal lowers it")
            sh = as_df(r["shap_summary"]).sort_values("mean_abs_shap")
            if "direction" in sh.columns and sh["direction"].astype(bool).any():
                sh["effect"] = sh["direction"].map({"raise": "raises risk", "lower": "lowers risk"}).fillna("")
                fig = px.bar(sh, x="mean_abs_shap", y="feature", orientation="h", color="effect",
                             color_discrete_map={"raises risk": AMBER, "lowers risk": TEAL, "": SOFT})
                fig.update_layout(legend_title_text="", showlegend=True, legend=dict(orientation="h", y=-0.15))
            else:
                fig = px.bar(sh, x="mean_abs_shap", y="feature", orientation="h")
                fig.update_traces(marker_color=TEAL)
            st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})

    with t3:
        for title, sub, paths in (("Data exploration", "exploration charts", r["eda_charts"]),
                                  ("Model results", "evaluation charts", r["model_charts"])):
            st.markdown(f"#### {title}")
            if not paths:
                st.markdown('<div class="dn-reason">No charts available.</div>', unsafe_allow_html=True)
                continue
            for i in range(0, len(paths), 3):
                for j, (col, p) in enumerate(zip(st.columns(3), paths[i:i + 3])):
                    with col, st.container(key=f"card_img_{title[:4]}_{i + j}"):
                        card_title(os.path.splitext(os.path.basename(p))[0].replace("_", " "), sub)
                        st.image(p, width="stretch")

    with t4:
        s.report_style = st.radio("Report style", REPORT_STYLES, horizontal=True, key="rs2",
                                  label_visibility="collapsed", index=REPORT_STYLES.index(s.report_style),
                                  format_func=lambda k: STYLE_LABEL[k])
        st.markdown(f'<div class="dn-reason">{STYLE_DESC[s.report_style]}</div>', unsafe_allow_html=True)
        if st.button("Generate report", type="primary"):
            try:
                with st.spinner("Writing the report..."):
                    s.report_path = backend.generate_report(r, s.report_style)
            except Exception as e:
                st.error(f"Could not build the report: {e}")
        if s.report_path:
            ext = os.path.splitext(s.report_path)[1]
            mime = ("application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    if ext == ".pptx" else "application/pdf")
            with st.container(key="card_report"):
                card_title(f"Credit risk report - {STYLE_LABEL[s.report_style]}",
                           f"generated by the reporter · {ext.lstrip('.')} · ready to download")
                with open(s.report_path, "rb") as f:
                    st.download_button(f"⬇ Download {ext.lstrip('.').upper()}", f, mime=mime, type="primary",
                                       file_name=f"credit_risk_report_{s.report_style.lower()}{ext}")

    with t5:
        st.markdown('<div class="dn-reason">Ask about the data or the results. Try: <i>Why was income kept?</i> or '
                    '<i>How much better is PSO than using all features?</i></div>', unsafe_allow_html=True)
        for msg in s.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        q = st.chat_input("Ask about the data or the results")
        if q:
            s.chat_history.append({"role": "user", "content": q})
            try:
                ans = backend.answer_question(r, q, s.chat_history[:-1])
            except Exception as e:
                ans = f"Sorry, I could not answer that ({e})."
            s.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()


# --------------------------------------------------------------------- main
init_state()
st.markdown(CSS, unsafe_allow_html=True)
header()
{"upload": screen_upload, "checkpoint": screen_checkpoint,
 "running": screen_running, "results": screen_results}[st.session_state.stage]()
