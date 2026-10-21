"""NewsLens AI - interactive dashboard for the multilingual Indian news classifier."""
import random
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

import inference

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="NewsLens AI", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- design tokens
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
INK, MUTED, GRID = "#E6EAF7", "#8E98B8", "rgba(142,152,184,0.16)"
ACCENT = "#8B7CFF"
MODEL_COLORS = {"IndicBERT-v2": "#8B7CFF", "XLM-RoBERTa": "#2DD4BF", "mBERT": "#F5A524"}
CAT_COLORS = {"crime": "#F87171", "entertainment": "#F472B6", "international": "#60A5FA",
              "sports": "#34D399", "technology": "#FBBF24"}
SCALE = [[0, "#111830"], [0.5, "#4B3FB5"], [1, "#B9AEFF"]]

NEW_ST = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
STRETCH = {"width": "stretch"} if NEW_ST else {"use_container_width": True}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(139,124,255,.14), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(45,212,191,.08), transparent 60%), #0A0F1E; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: #0D1327; border-right: 1px solid rgba(142,152,184,.14); }}
.brand {{ display:flex; align-items:center; gap:.7rem; margin: .2rem 0 1.2rem; }}
.brand-logo {{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem;
  background: linear-gradient(135deg,#8B7CFF,#2DD4BF); box-shadow:0 6px 20px rgba(139,124,255,.35); }}
.brand-name {{ font-weight:800; font-size:1.2rem; letter-spacing:-.01em; line-height:1.1; }}
.brand-sub {{ color:{MUTED}; font-size:.74rem; }}
.side-label {{ color:{MUTED}; font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.1rem 0 .35rem; }}
div[role="radiogroup"] {{ gap:.15rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding:.55rem .8rem; border-radius:10px; width:100%; cursor:pointer;
  border:1px solid transparent; transition: all .15s ease; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(139,124,255,.10); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(90deg, rgba(139,124,255,.25), rgba(139,124,255,.06)); border-color: rgba(139,124,255,.45); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-weight:600; font-size:.92rem; }}

/* hero */
.hero {{ position:relative; overflow:hidden; border-radius:20px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
  background: linear-gradient(120deg, rgba(139,124,255,.28), rgba(45,212,191,.12) 60%, rgba(17,24,48,.6));
  border: 1px solid rgba(139,124,255,.35); }}
.hero h1 {{ margin:0; font-size:1.9rem; font-weight:800; letter-spacing:-.02em; padding:0; }}
.hero p {{ margin:.35rem 0 0; color:#B9C1DE; font-size:.97rem; max-width: 760px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem; }}
.chip {{ font-size:.74rem; font-weight:600; padding:.22rem .65rem; border-radius:999px;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); color:#D5DBF2; }}
.chip.star {{ background:rgba(245,165,36,.16); border-color:rgba(245,165,36,.5); color:#FFCB6B; }}

/* KPI cards */
.kpi {{ background:#111830; border:1px solid rgba(142,152,184,.16); border-radius:16px; padding:1rem 1.1rem 1rem; position:relative; overflow:hidden; height:100%; }}
.kpi:before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:var(--c); }}
.kpi-label {{ color:{MUTED}; font-size:.74rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
.kpi-value {{ font-size:2rem; font-weight:800; letter-spacing:-.02em; margin:.2rem 0 .1rem; }}
.kpi-sub {{ font-size:.8rem; color:{MUTED}; }}
.up {{ color:#34D399; font-weight:700; }} .down {{ color:#F87171; font-weight:700; }}

/* cards & headings */
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px; border-color: rgba(142,152,184,.16); background: rgba(17,24,48,.55); }}
.sec-title {{ font-weight:700; font-size:1.02rem; margin:.1rem 0 0; }}
.sec-sub {{ color:{MUTED}; font-size:.82rem; margin:0 0 .4rem; }}
.page-gap {{ height:.6rem; }}
.pill {{ display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.78rem; font-weight:700; }}
.callout {{ background:rgba(139,124,255,.10); border:1px solid rgba(139,124,255,.35); border-radius:14px; padding:.8rem 1rem; color:#D5DBF2; font-size:.9rem; }}
.pred-card {{ border-radius:18px; padding:1.4rem 1.6rem; background:linear-gradient(120deg, rgba(52,211,153,.18), rgba(17,24,48,.7)); border:1px solid rgba(52,211,153,.4); }}
.pred-label {{ font-size:2.2rem; font-weight:800; text-transform:capitalize; letter-spacing:-.02em; }}

/* controls */
.stButton>button, .stDownloadButton>button {{ border-radius:10px; font-weight:600; border:1px solid rgba(139,124,255,.5); }}
.stButton>button[kind="primary"] {{ background:linear-gradient(135deg,#8B7CFF,#6D5BFF); border:none; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding:.5rem 1rem; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
@st.cache_data
def load():
    d = {n: pd.read_csv(DATA / f"{n}.csv") for n in
         ["overall_metrics", "language_wise_metrics", "predictions", "error_analysis"]}
    return d


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def style(fig, height=380, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13), margin=dict(l=10, r=10, t=30, b=10), title=None,
        showlegend=legend, legend=dict(orientation="h", y=1.12, x=0, title=None, font=dict(size=12)),
        hoverlabel=dict(bgcolor="#1B2447", font_size=13, font_family=FONT, bordercolor="#8B7CFF"),
        bargap=0.25)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=MUTED), linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(color=MUTED), linecolor=GRID)
    return fig


def plot(fig, **kw):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, **STRETCH, **kw)


def grid(df, **kw):
    st.dataframe(df, hide_index=True, **STRETCH, **kw)


def section(title, sub=""):
    st.markdown(f"<div class='sec-title'>{title}</div><div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)


def hero(title, text, chips=()):
    ch = "".join(f"<span class='chip{' star' if c.startswith('★') else ''}'>{c}</span>" for c in chips)
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{text}</p><div class='chips'>{ch}</div></div>",
                unsafe_allow_html=True)


def kpis(items):
    for col, (label, value, sub, color) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='kpi' style='--c:{color}'><div class='kpi-label'>{label}</div>"
                     f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>",
                     unsafe_allow_html=True)


def delta(a, b, unit=" pts vs next best"):
    d = (a - b) * 100
    cls = "up" if d >= 0 else "down"
    return f"<span class='{cls}'>{'▲' if d >= 0 else '▼'} {abs(d):.1f}{unit}</span>"


def gap():
    st.markdown("<div class='page-gap'></div>", unsafe_allow_html=True)


@st.cache_data
def model_scores(df, cats):
    rows = []
    for m, s in df.groupby("model"):
        rows.append(dict(model=m, accuracy=accuracy_score(s.true_label, s.predicted_label),
                         macro_f1=f1_score(s.true_label, s.predicted_label, labels=list(cats),
                                           average="macro", zero_division=0), n=len(s)))
    return pd.DataFrame(rows)


@st.cache_data
def category_f1(df, cats):
    rows = []
    for m, s in df.groupby("model"):
        f = f1_score(s.true_label, s.predicted_label, labels=list(cats), average=None, zero_division=0)
        rows += [dict(model=m, category=c, f1=v) for c, v in zip(cats, f)]
    return pd.DataFrame(rows)


def metrics(df, cats):
    return accuracy_score(df.true_label, df.predicted_label), f1_score(
        df.true_label, df.predicted_label, labels=cats, average="macro", zero_division=0)


def confusion(df, cats, key, height=460):
    norm = st.toggle("Show as % of true class", value=True, key=key)
    cm = confusion_matrix(df.true_label, df.predicted_label, labels=cats)
    if norm:
        z = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = px.imshow(z, x=cats, y=cats, text_auto=".0%", color_continuous_scale=SCALE, zmin=0, zmax=1,
                        labels=dict(x="Predicted", y="True", color="Share"), aspect="auto")
    else:
        fig = px.imshow(cm, x=cats, y=cats, text_auto=True, color_continuous_scale=SCALE,
                        labels=dict(x="Predicted", y="True", color="Count"), aspect="auto")
    fig.update_xaxes(side="bottom", showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_coloraxes(showscale=False)
    plot(style(fig, height, legend=False))


def pct_cfg(cols, label_map=None):
    label_map = label_map or {}
    return {c: st.column_config.ProgressColumn(label_map.get(c, c.replace("_", " ").title()),
                                               min_value=0, max_value=100, format="%.1f%%") for c in cols}


def table_cfg(q):
    cfg = {}
    if "text" in q: cfg["text"] = st.column_config.TextColumn("Headline", width="large")
    if "confidence" in q: cfg["confidence"] = st.column_config.ProgressColumn(
        "Confidence", min_value=0.0, max_value=1.0, format="%.3f")
    if "correct" in q: cfg["correct"] = st.column_config.CheckboxColumn("Correct")
    return cfg


def filters(df, key, outcome=True):
    """Filter panel. Returns the filtered frame."""
    with st.container(border=True):
        c = st.columns(5 if outcome else 4)
        m = c[0].multiselect("Model", sorted(df.model.unique()), key=key + "m", placeholder="All models") if "model" in df else []
        l = c[1].multiselect("Language", sorted(df.language.unique()), key=key + "l", placeholder="All languages")
        t = c[2].multiselect("True label", sorted(df.true_label.unique()), key=key + "t", placeholder="Any")
        p = c[3].multiselect("Predicted label", sorted(df.predicted_label.unique()), key=key + "p", placeholder="Any")
        o = c[4].selectbox("Outcome", ["All", "Correct", "Incorrect"], key=key + "o") if outcome else "All"
        r1, r2 = st.columns([2, 3])
        s = r1.text_input("Search headline text", key=key + "s", placeholder="Type a word…") if "text" in df else ""
        lo, hi = (0.0, 1.0)
        if "confidence" in df:
            lo, hi = r2.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01, key=key + "c")
    q = df
    if m: q = q[q.model.isin(m)]
    if l: q = q[q.language.isin(l)]
    if t: q = q[q.true_label.isin(t)]
    if p: q = q[q.predicted_label.isin(p)]
    if o != "All": q = q[q.correct == (o == "Correct")]
    if s: q = q[q.text.astype(str).str.contains(s, case=False, na=False, regex=False)]
    if "confidence" in q: q = q[q.confidence.between(lo, hi)]
    return q


def table_with_download(q, name):
    a, b = st.columns([4, 1])
    a.caption(f"{len(q):,} rows match the filters")
    b.download_button("⬇ Download CSV", csv_bytes(q), name, "text/csv", key="dl_" + name)
    grid(q, height=430, column_config=table_cfg(q))


# ----------------------------------------------------------------------------- data
D = load()
ov, lw, pr, er = D["overall_metrics"], D["language_wise_metrics"], D["predictions"], D["error_analysis"]
cats = sorted(pr.true_label.unique())
CATS = tuple(cats)
models = list(ov.model)
langs = list(lw.language.drop_duplicates())
best = ov.sort_values("validation_macro_f1", ascending=False).iloc[0]
BEST = best["model"]

# ----------------------------------------------------------------------------- sidebar
NAV = {"📊  Overview": "Overview", "⚖️  Model comparison": "Model comparison",
       "🌐  Language performance": "Language performance", "🔍  Prediction review": "Prediction review",
       "🧩  Error analysis": "Error analysis", "📤  Upload CSV": "Upload CSV", "⚡  Live classifier": "Live classifier"}
with st.sidebar:
    st.markdown("<div class='brand'><div class='brand-logo'>🔎</div><div><div class='brand-name'>NewsLens AI</div>"
                "<div class='brand-sub'>Indian multilingual news classifier</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='side-label'>Navigate</div>", unsafe_allow_html=True)
    page = NAV[st.radio("Navigate", list(NAV), label_visibility="collapsed")]
    st.markdown("<div class='side-label'>Global filter</div>", unsafe_allow_html=True)
    sel_langs = st.multiselect("Languages", langs, default=langs, placeholder="All languages",
                               help="Applies to every page except Upload CSV.")
    st.markdown(f"<div class='side-label'>Selected model</div><span class='pill' "
                f"style='background:rgba(245,165,36,.16);color:#FFCB6B'>★ {BEST}</span>"
                f"<div class='sec-sub' style='margin-top:.4rem'>Best validation macro-F1 "
                f"({best.validation_macro_f1:.2%})</div>", unsafe_allow_html=True)

sel_langs = sel_langs or langs
pr_f = pr[pr.language.isin(sel_langs)]
er_f = er[er.language.isin(sel_langs)]
lw_f = lw[lw.language.isin(sel_langs)]
scope = "All 5 languages" if len(sel_langs) == len(langs) else ", ".join(sel_langs)

# ----------------------------------------------------------------------------- pages
if page == "Overview":
    hero("Overview", "How well do multilingual transformers classify Indian news headlines? "
         "Compare models, languages and categories, then dig into the errors.",
         [f"★ Best: {BEST}", f"{len(pr_f) // len(models):,} test headlines", scope, f"{len(cats)} categories"])
    sel = st.selectbox("Model in focus", models, index=models.index(BEST))
    ms = model_scores(pr_f, CATS).set_index("model")
    others = ms.drop(index=sel)
    nb = others.sort_values("macro_f1", ascending=False).iloc[0]
    row = ov[ov.model == sel].iloc[0]
    kpis([
        ("Accuracy", f"{ms.loc[sel, 'accuracy']:.1%}", delta(ms.loc[sel, 'accuracy'], others.accuracy.max(), " pts vs best other"), MODEL_COLORS.get(sel, ACCENT)),
        ("Macro-F1", f"{ms.loc[sel, 'macro_f1']:.1%}", delta(ms.loc[sel, 'macro_f1'], nb.macro_f1, " pts vs best other"), "#2DD4BF"),
        ("Validation macro-F1", f"{row.validation_macro_f1:.1%}", "Model-selection metric (all languages)", "#F5A524"),
        ("Test headlines", f"{int(ms.loc[sel, 'n']):,}", scope, "#60A5FA"),
    ])
    gap()
    a, b = st.columns(2)
    with a.container(border=True):
        section("Accuracy and macro-F1 by model", "On the selected languages")
        d = ms.reset_index().melt("model", ["accuracy", "macro_f1"], var_name="metric", value_name="score")
        d["metric"] = d.metric.map({"accuracy": "Accuracy", "macro_f1": "Macro-F1"})
        fig = px.bar(d, x="model", y="score", color="metric", barmode="group", text_auto=".1%",
                     color_discrete_sequence=[ACCENT, "#2DD4BF"], labels={"score": "", "model": ""})
        fig.update_yaxes(range=[max(0, d.score.min() - 0.1), 1], tickformat=".0%")
        fig.update_traces(textposition="outside", cliponaxis=False)
        plot(style(fig, 360))
    with b.container(border=True):
        section("Category mix by language", f"True labels in the test set ({sel})")
        dist = pr_f[pr_f.model == sel].groupby(["language", "true_label"]).size().reset_index(name="n")
        fig = px.bar(dist, x="language", y="n", color="true_label", color_discrete_map=CAT_COLORS,
                     category_orders={"true_label": cats}, labels={"n": "Headlines", "language": "", "true_label": ""})
        plot(style(fig, 360))
    a, b = st.columns([3, 2])
    with a.container(border=True):
        section("Macro-F1 heatmap", "Model × language, from language_wise_metrics.csv")
        hm = lw_f.pivot(index="model", columns="language", values="macro_f1").reindex(models)[[l for l in langs if l in sel_langs]]
        fig = px.imshow(hm, text_auto=".1%", color_continuous_scale=SCALE, aspect="auto",
                        zmin=max(0, hm.values.min() - 0.08), zmax=1)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
        plot(style(fig, 280, legend=False))
    with b.container(border=True):
        section("Where predictions land", f"Predicted label share ({sel})")
        pc = pr_f[pr_f.model == sel].predicted_label.value_counts().reindex(cats).reset_index()
        pc.columns = ["category", "n"]
        fig = px.pie(pc, names="category", values="n", hole=0.62, color="category", color_discrete_map=CAT_COLORS)
        fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#0A0F1E", width=2)))
        plot(style(fig, 280))

elif page == "Model comparison":
    hero("Model comparison", "mBERT, XLM-RoBERTa and IndicBERT-v2 evaluated on the same held-out test set.",
         [f"★ Best: {BEST}", f"{len(models)} models"])
    with st.container(border=True):
        section("Leaderboard", "Test-set metrics (all languages) and validation macro-F1 used for model selection")
        t = ov.sort_values("validation_macro_f1", ascending=False).copy()
        cols = ["validation_macro_f1", "accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]
        t[cols] = t[cols] * 100
        cfg = pct_cfg(cols, {"validation_macro_f1": "Val macro-F1", "precision_macro": "Precision", "recall_macro": "Recall",
                             "macro_f1": "Macro-F1", "weighted_f1": "Weighted F1", "accuracy": "Accuracy"})
        cfg["model"] = st.column_config.TextColumn("Model")
        cfg["checkpoint"] = st.column_config.TextColumn("Checkpoint", width="medium")
        cfg["test_examples"] = st.column_config.NumberColumn("Test examples")
        grid(t, column_config=cfg)
        st.download_button("⬇ Download leaderboard CSV", csv_bytes(ov), "overall_metrics.csv", "text/csv")
    a, b = st.columns(2)
    with a.container(border=True):
        section("Metric profile", "Radar view of the test metrics")
        fig = go.Figure()
        keys = ["accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]
        names = ["Accuracy", "Precision", "Recall", "Macro-F1", "Weighted F1"]
        for _, r in ov.iterrows():
            v = [r[k] for k in keys]
            col = MODEL_COLORS.get(r["model"], ACCENT)
            fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=names + names[:1], name=r["model"], fill="toself",
                                          line=dict(color=col, width=2), opacity=.75))
        lo = max(0, ov[keys].values.min() - 0.05)
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[lo, 1], gridcolor=GRID, tickformat=".0%", linecolor=GRID),
                                     angularaxis=dict(gridcolor=GRID, linecolor=GRID)))
        plot(style(fig, 400))
    with b.container(border=True):
        section("Per-category F1", f"Derived from predictions.csv · {scope}")
        cf = category_f1(pr_f, CATS)
        fig = px.bar(cf, x="category", y="f1", color="model", barmode="group", color_discrete_map=MODEL_COLORS,
                     labels={"f1": "F1", "category": "", "model": ""}, text_auto=".0%")
        fig.update_yaxes(range=[max(0, cf.f1.min() - 0.1), 1], tickformat=".0%")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=10)
        plot(style(fig, 400))
    with st.container(border=True):
        section("Category × model heatmap", "Which categories are hardest for which model")
        hm = category_f1(pr_f, CATS).pivot(index="model", columns="category", values="f1").reindex(models)
        fig = px.imshow(hm, text_auto=".1%", color_continuous_scale=SCALE, aspect="auto", zmin=max(0, hm.values.min() - 0.08), zmax=1)
        fig.update_coloraxes(showscale=False); fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
        plot(style(fig, 270, legend=False))

elif page == "Language performance":
    hero("Language performance", "Per-language results for every model. Some scripts are clearly harder than others.",
         [scope])
    names = {"macro_f1": "Macro-F1", "accuracy": "Accuracy", "precision_macro": "Precision (macro)",
             "recall_macro": "Recall (macro)", "weighted_f1": "Weighted F1"}
    k = st.radio("Metric", list(names), format_func=names.get, horizontal=True)
    winners = lw_f.loc[lw_f.groupby("language")[k].idxmax()].set_index("language")
    lang_list = [l for l in langs if l in winners.index]
    for col, l in zip(st.columns(len(lang_list)), lang_list):
        w = winners.loc[l]
        col.markdown(f"<div class='kpi' style='--c:{MODEL_COLORS.get(w['model'], ACCENT)}'><div class='kpi-label'>{l}</div>"
                     f"<div class='kpi-value'>{w[k]:.1%}</div><div class='kpi-sub'>best: {w['model']}</div></div>",
                     unsafe_allow_html=True)
    gap()
    a, b = st.columns([3, 2])
    with a.container(border=True):
        section(f"{names[k]} by language", "Grouped by model")
        fig = px.bar(lw_f, x="language", y=k, color="model", barmode="group", color_discrete_map=MODEL_COLORS,
                     category_orders={"language": lang_list, "model": models}, text_auto=".0%",
                     labels={k: names[k], "language": "", "model": ""})
        fig.update_yaxes(range=[max(0, lw_f[k].min() - 0.1), 1], tickformat=".0%")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=10)
        plot(style(fig, 400))
    with b.container(border=True):
        section("Model lines across languages", "Spot the weak scripts")
        fig = px.line(lw_f, x="language", y=k, color="model", markers=True, color_discrete_map=MODEL_COLORS,
                      category_orders={"language": lang_list}, labels={k: names[k], "language": "", "model": ""})
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_yaxes(tickformat=".0%")
        plot(style(fig, 400))
    with st.container(border=True):
        section("Detail table", f"{names[k]} for each model and language")
        pv = lw_f.pivot(index="language", columns="model", values=k).reindex(lang_list)[models] * 100
        pv = pv.reset_index()
        grid(pv, column_config=pct_cfg(models, {m: m for m in models}))
        st.download_button("⬇ Download language metrics", csv_bytes(lw_f), "language_wise_metrics.csv", "text/csv")

elif page == "Prediction review":
    hero("Prediction review", "Browse every test prediction. Filter by model, language, label, outcome, confidence or headline text.",
         [f"{len(pr_f):,} predictions in scope", scope])
    q = filters(pr_f, "pr")
    n = len(q)
    kpis([("Matching rows", f"{n:,}", "after filters", ACCENT),
          ("Accuracy", f"{q.correct.mean():.1%}" if n else "–", "of matching rows", "#34D399"),
          ("Avg confidence", f"{q.confidence.mean():.1%}" if n else "–", "model certainty", "#60A5FA"),
          ("Misclassified", f"{(~q.correct).sum():,}", "in this view", "#F87171")])
    gap()
    table_with_download(q, "predictions_filtered.csv")

elif page == "Error analysis":
    hero("Error analysis", "Where and why models go wrong: confusions, confidence and the misclassified headlines.", [scope])
    m = st.radio("Model", models, index=models.index(BEST), horizontal=True)
    s = pr_f[pr_f.model == m]
    e = s[~s.correct]
    hi_conf = int((e.confidence > 0.9).sum())
    kpis([("Errors", f"{len(e):,}", f"of {len(s):,} headlines", "#F87171"),
          ("Error rate", f"{len(e) / max(len(s), 1):.1%}", m, "#F5A524"),
          ("Avg confidence on errors", f"{e.confidence.mean():.1%}" if len(e) else "–", "overconfident if high", "#60A5FA"),
          ("Confident mistakes", f"{hi_conf:,}", "wrong with >90% confidence", "#F472B6")])
    gap()
    a, b = st.columns(2)
    with a.container(border=True):
        section("Confusion matrix", f"{m} · rows are the true label")
        confusion(s, cats, "cm_" + m)
    with b.container(border=True):
        section("Most frequent confusions", "True → predicted label pairs")
        pairs = e.groupby(["true_label", "predicted_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
        pairs["pair"] = pairs.true_label + " → " + pairs.predicted_label
        fig = px.bar(pairs.head(10).iloc[::-1], x="count", y="pair", orientation="h", color="true_label",
                     color_discrete_map=CAT_COLORS, labels={"count": "Errors", "pair": "", "true_label": "True"})
        plot(style(fig, 500))
    a, b = st.columns(2)
    with a.container(border=True):
        section("Confidence: correct vs incorrect", "Well-calibrated models are less sure when wrong")
        fig = px.histogram(s, x="confidence", color="correct", nbins=20, barmode="overlay", opacity=.8,
                           color_discrete_map={True: "#34D399", False: "#F87171"},
                           labels={"confidence": "Confidence", "correct": "Correct"})
        plot(style(fig, 330))
    with b.container(border=True):
        section("Errors by language and true label", "Count of misclassified headlines")
        el = e.groupby(["language", "true_label"]).size().reset_index(name="n")
        fig = px.bar(el, x="language", y="n", color="true_label", color_discrete_map=CAT_COLORS,
                     category_orders={"true_label": cats}, labels={"n": "Errors", "language": "", "true_label": ""})
        plot(style(fig, 330))
    st.markdown("<div class='sec-title' style='margin-top:.8rem'>Misclassified examples</div>"
                "<div class='sec-sub'>From error_analysis.csv</div>", unsafe_allow_html=True)
    q = filters(er_f[er_f.model == m], "er", outcome=False)
    table_with_download(q, f"errors_{m}.csv")

elif page == "Upload CSV":
    hero("Upload predictions", "Bring your own results and get instant metrics, a confusion matrix and a filterable table.",
         ["Required: language, true_label, predicted_label", "Optional: model, text, confidence"])
    sample = pd.DataFrame({"model": ["IndicBERT-v2"] * 2, "language": ["Hindi", "Tamil"], "text": ["example headline"] * 2,
                           "true_label": ["sports", "crime"], "predicted_label": ["sports", "sports"], "confidence": [0.98, 0.61]})
    f = st.file_uploader("Drop a CSV file here", type="csv")
    st.download_button("⬇ Download a sample template", csv_bytes(sample), "template.csv", "text/csv")
    if f:
        u = pd.read_csv(f)
        miss = {"language", "true_label", "predicted_label"} - set(u.columns)
        if miss:
            st.error(f"Missing columns: {sorted(miss)}. Found: {list(u.columns)}")
        else:
            u["correct"] = u.true_label == u.predicted_label
            uc = sorted(set(u.true_label) | set(u.predicted_label))
            acc, f1 = metrics(u, uc)
            kpis([("Rows", f"{len(u):,}", f"{u.language.nunique()} languages", ACCENT), ("Accuracy", f"{acc:.1%}", "", "#34D399"),
                  ("Macro-F1", f"{f1:.1%}", f"{len(uc)} classes", "#2DD4BF"),
                  ("Misclassified", f"{(~u.correct).sum():,}", "", "#F87171")])
            gap()
            a, b = st.columns(2)
            with a.container(border=True):
                section("Confusion matrix", "Uploaded data")
                confusion(u, uc, "cm_up", 420)
            with b.container(border=True):
                section("Per-class report", "Precision, recall and F1")
                rep = pd.DataFrame(classification_report(u.true_label, u.predicted_label, labels=uc,
                                                         output_dict=True, zero_division=0)).T.loc[uc].reset_index()
                rep.columns = ["class", "precision", "recall", "f1", "support"]
                rep[["precision", "recall", "f1"]] *= 100
                grid(rep, column_config={**pct_cfg(["precision", "recall", "f1"]), "support": st.column_config.NumberColumn("Support")})
            q = filters(u, "up")
            table_with_download(q, "uploaded_filtered.csv")

else:
    hero("Live headline classifier", "Type or paste a headline in Hindi, Tamil, Telugu, Malayalam or Marathi and see the model's verdict.",
         ["Runs locally", "Model: IndicBERT-v2"])
    ok, msg = inference.model_ready()
    if not ok:
        with st.container(border=True):
            section("Live prediction is not set up yet", msg)
            st.markdown("Place your fine-tuned model in `models/best_model/` and restart the app:")
            for name in ["config.json", "model.safetensors (or pytorch_model.bin)", "tokenizer files"]:
                st.markdown(f"- `{name}`")
            st.code('model.save_pretrained("models/best_model")\ntokenizer.save_pretrained("models/best_model")', language="python")
            st.caption("The rest of the dashboard works without it.")
    else:
        @st.cache_resource
        def get():
            return inference.load_model()

        tok, model, id2label = get()

        def sample():
            r = pr[pr.model == BEST].sample(1).iloc[0]
            st.session_state["live_text"] = r["text"]
            st.session_state["live_meta"] = f"{r['language']} · true label: {r['true_label']}"

        a, b = st.columns([3, 2])
        with a.container(border=True):
            section("Headline", "Hindi · Tamil · Telugu · Malayalam · Marathi")
            text = st.text_area("Headline", key="live_text", height=140, label_visibility="collapsed",
                                placeholder="Paste a headline…")
            if st.session_state.get("live_meta"):
                st.caption("Sample: " + st.session_state["live_meta"])
            c1, c2 = st.columns([1, 1])
            go_click = c1.button("Classify", type="primary", **STRETCH)
            c2.button("🎲 Try a test headline", on_click=sample, **STRETCH)
        if go_click and text.strip():
            res = inference.predict(tok, model, id2label, text)
            df = pd.DataFrame(res, columns=["category", "probability"])
            col = CAT_COLORS.get(res[0][0], ACCENT)
            with b:
                st.markdown(f"<div class='pred-card' style='border-color:{col}'><div class='kpi-label'>Predicted category</div>"
                            f"<div class='pred-label' style='color:{col}'>{res[0][0]}</div>"
                            f"<div class='kpi-sub'>{res[0][1]:.1%} confidence</div></div>", unsafe_allow_html=True)
            with st.container(border=True):
                section("Class probabilities")
                fig = px.bar(df.iloc[::-1], x="probability", y="category", orientation="h", color="category",
                             color_discrete_map=CAT_COLORS, text_auto=".1%", labels={"probability": "", "category": ""})
                fig.update_xaxes(range=[0, 1], tickformat=".0%")
                fig.update_traces(textposition="outside", cliponaxis=False)
                plot(style(fig, 300, legend=False))
