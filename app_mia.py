import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# MedShield (VLM MIA) - Streamlit Dashboard
# - Runs out-of-the-box with mock data
# - Swap `load_or_generate_records()` with your real CSV/JSONL pipeline later
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MedShield: VLM Privacy Auditing & MIA Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Aero-glass / modern telemetry styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
  background-color: #0b0f14;
  color: #e2e8f0;
}

.main-title {
  background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 45%, #8b5cf6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
  font-size: 2.65rem;
  margin-bottom: 0.15rem;
}

.subtitle {
  color: #94a3b8;
  font-size: 1.05rem;
  font-weight: 300;
  margin-bottom: 1.4rem;
}

.glass-card, div[data-testid="stVerticalBlockBorderDiv"] {
  background: rgba(30, 41, 59, 0.42) !important;
  backdrop-filter: blur(14px) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  padding: 1.25rem !important;
  margin-bottom: 1.0rem !important;
  box-shadow: 0 10px 36px rgba(0, 0, 0, 0.45) !important;
}

.metric-label {
  font-size: 0.86rem;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.metric-value {
  font-size: 2.25rem;
  font-weight: 800;
  color: #38bdf8;
  line-height: 1.05;
  margin-top: 0.35rem;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid rgba(255,255,255,0.10);
}

.badge-high { background: rgba(239, 68, 68, 0.18); color: #ef4444; border-color: rgba(239, 68, 68, 0.35); }
.badge-med  { background: rgba(245, 158, 11, 0.16); color: #f59e0b; border-color: rgba(245, 158, 11, 0.35); }
.badge-low  { background: rgba(34, 197, 94, 0.14); color: #22c55e; border-color: rgba(34, 197, 94, 0.35); }

hr {
  border-color: rgba(255, 255, 255, 0.10);
  margin: 1.7rem 0;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  background-color: rgba(15, 23, 42, 0.55) !important;
}


/* SIDEBAR */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  background: rgba(7, 12, 18, 0.94) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] * {
  color: #e2e8f0;
}
section[data-testid="stSidebar"] hr {
  border-color: rgba(255, 255, 255, 0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1) Mocked model metrics (swap with real eval outputs later)
# -----------------------------------------------------------------------------

# These are *attack* models (higher AUC => stronger MIA model (higher benchmark performance)).
MODELS_METRICS: Dict[str, Dict[str, float | str]] = {
    # VLM MIA only (milestone4–6).
    # Naming aligns with the scripts/pipelines in milestone4–6 `src/` rather than generic labels.

    # milestone6: MV‑CLG-Rerank merged (base stack + cauldron rerank override)
    "method3_MV-CLG-Rerank (merged)": {
        "auc": 0.856481,
        "tpr_at_01": 0.386667,
        "color": "#38bdf8",
    },

    # milestone5: Step3 stacking classifier (default `--model_type logreg` in `milestone5/src/step3_stack.py`)
    "m5_step3_stack (logreg)": {
        "auc": 0.837374,
        "tpr_at_01": 0.290000,
        "color": "#a855f7",
    },

    # milestone5: simplified ratio+CPU LogReg (`milestone5/src/step3_not_ensembled.py`)
    "m5_step3_not_ensembled (cpu+ratio logreg)": {
        "auc": 0.830256,
        "tpr_at_01": 0.283333,
        "color": "#fb923c",
    },

    # milestone6: cauldron-only rerank model (ablation/reference; not merged back into full dev here)
    "method3_MV-CLG-Rerank (cauldron-only)": {
        "auc": 0.569478,
        "tpr_at_01": 0.126667,
        "color": "#94a3b8",
    },
}


MODEL_NOTES: Dict[str, str] = {
    "method3_MV-CLG-Rerank (merged)": "Full-dev score after replacing only the cauldron subset with a multi-variant reranker.",
    "m5_step3_stack (logreg)": "Validation stacking model trained on CPU text stats + dual-model loss-gap features.",
    "m5_step3_not_ensembled (cpu+ratio logreg)": "Lightweight baseline using CPU stats plus a single loss ratio feature.",
    "method3_MV-CLG-Rerank (cauldron-only)": "Reranker evaluated on the cauldron subset only (shown as an ablation/reference).",
}


def _performance_tier(auc: float) -> Tuple[str, str]:
    if auc >= 0.92:
        return "🏁 TOP PERFORMANCE", "badge badge-high"
    if auc >= 0.85:
        return "📈 STRONG PERFORMANCE", "badge badge-med"
    return "🧪 BASELINE PERFORMANCE", "badge badge-low"


# -----------------------------------------------------------------------------
# 2) Mock data generator (replace with real data loading later)
# -----------------------------------------------------------------------------

VLM_SNIPPETS: List[str] = [
    "User: Describe the image in one sentence.\nAssistant: A person is standing outdoors near a vehicle under a cloudy sky.",
    "User: What objects are visible?\nAssistant: A table, plates, cups, and utensils are arranged for a meal.",
    "User: Answer the question: what is the animal doing?\nAssistant: The animal is running across a grassy field.",
    "User: Provide a brief caption.\nAssistant: A city street at night with bright storefront lights and pedestrians.",
    "User: Summarize the scene.\nAssistant: A kitchen counter with vegetables and a cutting board prepared for cooking.",
    "User: Identify the activity.\nAssistant: Two people are playing a sport on a court with a net.",
    "User: What is the main subject?\nAssistant: A close-up of a handheld device with a lit screen.",
    "User: Give a concise description.\nAssistant: A group of people sitting around a conference table in a meeting.",
    "User: What is happening in the image?\nAssistant: A vehicle is parked beside a road while someone takes a photo.",
]

@st.cache_data(show_spinner=False)
def load_or_generate_records(dataset_name: str, n_records: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42 if "v1" in dataset_name else 99)

    ids = [f"PT-{int(rng.integers(10000, 99999))}" for _ in range(n_records)]
    texts = [str(rng.choice(VLM_SNIPPETS)) for _ in range(n_records)]

    # Standard 50/50 member distribution for benchmarking visuals
    is_member = rng.binomial(1, 0.5, size=n_records).astype(int)

    df = pd.DataFrame({
        "Identifier": ids,
        "Raw Text": texts,
        "is_member": is_member,
    })

    # Simulate per-model membership probabilities with separability roughly tied to AUC.
    # This is only for UI; swap with real model predictions later.
    for model_name, meta in MODELS_METRICS.items():
        auc = float(meta["auc"])
        sep = max(0.05, (auc - 0.5) * 4.5)  # higher AUC => stronger separation
        noise = rng.normal(0.0, 1.0, size=n_records)
        logit = (is_member * sep) - (sep / 2.0) + noise
        prob = 1.0 / (1.0 + np.exp(-logit))
        df[f"score_{model_name}"] = np.clip(prob, 0.0, 1.0).round(4)

    return df


# -----------------------------------------------------------------------------
# 3) Header
# -----------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">VLM Privacy Auditing & MIA Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Executive Summary: This dashboard audits <b>Membership Inference Attacks (MIA)</b> against <b>vision-language models (VLMs)</b>. '
    'A higher <b>AUC</b> indicates a <b>stronger MIA classifier</b> (higher membership-identification success on this benchmark).</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4) Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🎛️ Audit Configuration")
selected_dataset = st.sidebar.selectbox(
    "Dataset",
    ["VLM-MIA v1 (Mixed prompts)", "VLM-MIA v2 (Synthetic variants)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Attack Models")
st.sidebar.caption("Select attack models to overlay ROC curves and compare benchmark performance metrics.")

selected_models: List[str] = []
for model_name in MODELS_METRICS.keys():
    default_on = model_name in [
        "VLM Dual-Loss Stack (XGBoost)",
        "VLM Dual-Loss Stack (LogReg)",
        "Single-Model Loss (Baseline)",
    ]
    if st.sidebar.checkbox(model_name, value=default_on):
        selected_models.append(model_name)

st.sidebar.markdown("")
with st.sidebar.expander("🧠 Model notes", expanded=False):
    for m in selected_models or list(MODELS_METRICS.keys()):
        st.markdown(f"- **{m}**: {MODEL_NOTES.get(m, '—')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Performance Key")
st.sidebar.markdown(
    "- <span class='badge badge-high'>🚨 HIGH</span> (AUC ≥ 0.92)\n"
    "- <span class='badge badge-med'>⚠️ MEDIUM</span> (0.85 ≤ AUC < 0.92)\n"
    "- <span class='badge badge-low'>🛡️ LOW</span> (AUC < 0.85)",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5) Load data + compute headline metrics
# -----------------------------------------------------------------------------

raw_db = load_or_generate_records(selected_dataset, n_records=120)

if selected_models:
    highest_auc = float(max(float(MODELS_METRICS[m]["auc"]) for m in selected_models))
    avg_tpr = float(np.mean([float(MODELS_METRICS[m]["tpr_at_01"]) for m in selected_models]))
else:
    highest_auc = 0.5
    avg_tpr = 0.0

n_audited = int(len(raw_db))

# -----------------------------------------------------------------------------
# 6) KPI row
# -----------------------------------------------------------------------------

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"""
<div class="glass-card">
  <div class="metric-label">📈 Highest MIA AUC Detected</div>
  <div class="metric-value">{highest_auc:.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
<div class="glass-card">
  <div class="metric-label">🎯 Avg Member Recall (TPR @ FPR=10%)</div>
  <div class="metric-value">{avg_tpr * 100.0:.2f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
<div class="glass-card">
  <div class="metric-label">📋 Total Audited Patient Records</div>
  <div class="metric-value">{n_audited}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 7) Main dashboard split: ROC + leaderboard
# -----------------------------------------------------------------------------

col_left, col_right = st.columns([3, 2])

with col_left:
    with st.container(border=True):
        st.markdown("### 📊 Multi-Model ROC Curves")
        st.caption("Attack ROC curves: closer to top-left implies stronger membership inference performance on this benchmark.")

        fig = go.Figure()

        # random guess reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#64748b", width=2, dash="dash"),
                name="Random Baseline (AUC=0.50)",
            )
        )

        fpr_grid = np.linspace(0, 1, 160)
        for model_name in selected_models:
            auc_val = float(MODELS_METRICS[model_name]["auc"])
            color = str(MODELS_METRICS[model_name]["color"])

            # Parametric ROC with exact AUC: AUC = a/(a+1) => a = AUC/(1-AUC)
            if auc_val >= 0.999:
                tpr = np.ones_like(fpr_grid)
                tpr[0] = 0.0
            else:
                a = auc_val / max(1e-8, (1.0 - auc_val))
                tpr = 1.0 - (1.0 - fpr_grid) ** a

            fig.add_trace(
                go.Scatter(
                    x=fpr_grid,
                    y=tpr,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"{model_name} (AUC={auc_val:.4f})",
                )
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=20, t=10, b=35),
            xaxis=dict(
                title="False Positive Rate (FPR)",
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.10)",
                tickfont=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                title="True Positive Rate (TPR)",
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.10)",
                tickfont=dict(color="#94a3b8"),
            ),
            legend=dict(
                x=0.52,
                y=0.05,
                bgcolor="rgba(15,23,42,0.86)",
                bordercolor="rgba(255,255,255,0.10)",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=11),
            ),
            height=390,
        )

        st.plotly_chart(fig, use_container_width=True)

with col_right:
    with st.container(border=True):
        st.markdown("### 🏆 MIA Performance Leaderboard")
        st.caption("Sorted by AUC (higher = better membership identification on this benchmark).")

        rows = []
        for model_name, meta in MODELS_METRICS.items():
            auc = float(meta["auc"])
            status_text, badge_class = _performance_tier(auc)
            rows.append(
                {
                    "Attack Model": model_name,
                    "MIA AUC": auc,
                    "TPR @ FPR=10%": float(meta["tpr_at_01"]),
                    "Performance Tier": status_text,
                    "Method Note": MODEL_NOTES.get(model_name, "—"),
                    "_badge_class": badge_class,
                }
            )

        ld_df = pd.DataFrame(rows).sort_values(by="MIA AUC", ascending=False).reset_index(drop=True)

        styled = (
            ld_df.drop(columns=["_badge_class"]).style.background_gradient(
                cmap="Blues", subset=["MIA AUC"], vmin=0.50, vmax=0.95
            )
            .format({"MIA AUC": "{:.4f}", "TPR @ FPR=10%": "{:.2%}"})
        )

        st.dataframe(styled, use_container_width=True, height=385)

# -----------------------------------------------------------------------------
# 8) Probe workspace
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🔍 Sample Privacy Probe Workspace")
st.caption("Browse individual multimodal prompt/response texts and inspect membership probabilities produced by the selected attack model.")

if selected_models:
    probe_model = st.selectbox("Classifier to probe", selected_models)
else:
    probe_model = st.selectbox("Classifier to probe", list(MODELS_METRICS.keys()), index=0)

score_col = f"score_{probe_model}"

display_df = raw_db[["Identifier", "Raw Text", "is_member", score_col]].copy()

# Filters
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    query = st.text_input("✍️ Filter text (e.g., 'kitchen', 'scene', 'street')", "")
with f2:
    min_prob = st.slider("Min membership score", 0.0, 1.0, 0.0, 0.01)
with f3:
    show_only = st.selectbox("Show", ["All", "Members only", "Non-members only"], index=0)

if query:
    display_df = display_df[display_df["Raw Text"].str.contains(query, case=False, na=False)]

display_df = display_df[display_df[score_col] >= float(min_prob)]

if show_only == "Members only":
    display_df = display_df[display_df["is_member"] == 1]
elif show_only == "Non-members only":
    display_df = display_df[display_df["is_member"] == 0]

# Status tags
status = []
insight = []
for _, r in display_df.iterrows():
    gt = int(r["is_member"])
    p = float(r[score_col])

    if gt == 1 and p >= 0.70:
        status.append("🔴 HIGH-CONFIDENCE MEMBER")
        insight.append("✅ HIGH-CONFIDENCE HIT")
    elif gt == 1 and p >= 0.55:
        status.append("🟠 LIKELY MEMBER")
        insight.append("📈 MODERATE SIGNAL")
    elif gt == 0 and p >= 0.70:
        status.append("🟠 FALSE POSITIVE")
        insight.append("⚠️ OVERLAP / GENERIC PATTERN")
    else:
        status.append("🟢 SECURE")
        insight.append("🧪 NO STRONG SIGNAL")

out_df = display_df.copy()
out_df["Audit Result"] = status
out_df["Notes"] = insight
out_df["Text Sample"] = out_df["Raw Text"].apply(lambda x: (x[:90] + "…") if len(x) > 90 else x)

# Table display
final_cols = ["Identifier", "Text Sample", "is_member", score_col, "Audit Result", "Notes"]
table = out_df[final_cols].rename(columns={"is_member": "In Training Set", score_col: "Membership Score"})

st.dataframe(
    table.style.map(
        lambda x: "color:#ef4444;font-weight:800;" if x == "🔴 HIGH-CONFIDENCE MEMBER" else (
            "color:#f59e0b;font-weight:800;" if x in ["🟠 LIKELY MEMBER", "🟠 FALSE POSITIVE"] else (
                "color:#22c55e;font-weight:800;" if x == "🟢 SECURE" else ""
            )
        ),
        subset=["Audit Result"],
    ).format({"Membership Score": "{:.2%}"}),
    use_container_width=True,
    height=420,
)

st.markdown("#### 🕵️ Narrative Deep-Dive")
st.caption("Expand a few samples to inspect ground truth vs. predicted membership score.")

sample_view = out_df.head(3)
if sample_view.empty:
    st.info("No records match the current filters.")
else:
    for _, r in sample_view.iterrows():
        tag = r["Audit Result"]
        color = "#ef4444" if tag == "🔴 HIGH-CONFIDENCE MEMBER" else ("#f59e0b" if "🟠" in tag else "#22c55e")
        with st.expander(f"📋 {r['Identifier']}  |  {tag}"):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown("**Full Text**")
                st.text_area("Raw Text", value=str(r["Raw Text"]), height=120, disabled=True, key=f"txt_{r['Identifier']}")
            with c2:
                st.markdown("**Audit Stats**")
                st.markdown(f"- **Ground Truth**: {'Member' if int(r['is_member']) == 1 else 'Non-member'}")
                st.markdown(
                    f"- **Membership Score**: <span style='font-weight:900;color:{color};font-size:1.1rem'>{float(r[score_col])*100:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"- **Signal Note**: {r['Notes']}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.85rem;margin-top:1.6rem;'>"
    "MedShield v1 • VLM MIA Auditing Dashboard • Replace mock generators with your real inference outputs when ready."
    "</div>",
    unsafe_allow_html=True,
)
