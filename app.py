import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedShield: Clinical Data Privacy & MIA Auditing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Aero-Glass Dark Tech Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main layout fonts and background styling */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Header styling with gradient */
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 1.8rem;
    }
    
    /* Aero-glass panel styling for cards and container blocks */
    .glass-card, div[data-testid="stVerticalBlockBorderDiv"] {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1;
        margin-top: 0.4rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom status tags */
    .badge-breached {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .badge-secure {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }

    /* Style dataframe tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
    }
    
    /* Divider customization */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MOCK DATA & MODEL METRICS
# -----------------------------------------------------------------------------
# Standardised models and metrics matching our actual Text-only MIA findings
MODELS_METRICS = {
    "WEL-Weighted-XGBoost": {
        "auc": 0.9370,
        "tpr_at_01": 0.8020,
        "color": "#38bdf8", # Neon Sky Blue
        "status": "🚨 HIGH LEAKAGE RISK"
    },
    "WBC-Ensemble-XGBoost": {
        "auc": 0.9228,
        "tpr_at_01": 0.7464,
        "color": "#a855f7", # Neon Purple
        "status": "🚨 HIGH LEAKAGE RISK"
    },
    "WBC-Ensemble-LR": {
        "auc": 0.9060,
        "tpr_at_01": 0.7288,
        "color": "#f43f5e", # Rose Red
        "status": "⚠️ MEDIUM LEAKAGE RISK"
    },
    "Heuristic-Ensemble-LR": {
        "auc": 0.8616,
        "tpr_at_01": 0.5732,
        "color": "#fb923c", # Vibrant Orange
        "status": "⚠️ MEDIUM LEAKAGE RISK"
    },
    "SmolLM2-360M-Mean (Baseline)": {
        "auc": 0.6372,
        "tpr_at_01": 0.1964,
        "color": "#94a3b8", # Steel Slate Gray
        "status": "🛡️ LOW LEAKAGE RISK"
    }
}

# Clinical/medical text snippets for testing/probing
CLINICAL_SNIPPETS = [
    "Patient presented with acute dyspnea, history of severe COPD. Oxygen saturation at 88% on room air. Initiated non-invasive positive pressure ventilation (NIPPV) with immediate stabilizing effect.",
    "A 45-year-old female with recurrent migraines underwent routine MRI scanning showing microvascular white matter ischemic lesions. Initiated prophylaxis regimen using Propranolol 40mg daily.",
    "Discharge Summary: Patient admitted for elective laparoscopic cholecystectomy due to symptomatic cholelithiasis. Post-operative course uneventful. Discharged home in stable condition.",
    "Cardiology Report: Echocardiogram reveals LVEF estimated at 35%. Severe global hypokinesis of the left ventricle. Recommendation: initiate Metoprolol Succinate and Lisinopril titration.",
    "Oncology Update: Stage IIIA invasive ductal carcinoma, ER/PR positive, HER2 neu negative. Completed adjuvant cycle 4 of AC chemotherapy. Patient reports mild peripheral neuropathy.",
    "Routine physical: 62-year-old male with history of Type 2 Diabetes Mellitus. HbA1c remains elevated at 8.2% despite Metformin max dosage. Advised lifestyle changes and initiated basal Insulin.",
    "Neurology consult: 78-year-old male presenting with progressive memory decline over 18 months. MMSE score 19/30. MRI brain displays mild generalized hippocampal atrophy.",
    "ICU Note: Admitted following motor vehicle collision with multiple rib fractures and moderate left-sided hemothorax. Chest tube placed, draining serosanguinous fluid.",
    "Patient referred for persistent epigastric pain. Upper endoscopy revealed a deep 2cm duodenal ulcer positive for Helicobacter pylori. Initiated triple antibiotic and PPI eradication therapy.",
    "Discharge Note: Pediatric patient admitted for acute asthmatic exacerbation. Responded well to scheduled Albuterol nebulizers and oral Dexamethasone. Discharged with spacer education."
]

# Synthetic clinical-like snippets for Synthetic-v2
SYNTHETIC_SNIPPETS = [
    "SIMULATED RECORD: 45-year-old virtual subject displays normal cardiovascular sinus rhythm. No significant clinical anomalies noted in the mock assessment.",
    "SYNTHETIC PROFILE: Subject is a simulated diabetic profile. HbA1c modeled at 7.5% under synthetic glucose titration algorithms. Advised simulated lifestyle updates.",
    "GENERATED NARRATIVE: Artificial patient admitted for virtual appendectomy simulation. Virtual post-op recovery modeled as fully stable within 48-hour synthetic window.",
    "MOCK ECG REPORT: Left ventricular ejection fraction parameterized at 55%. Simulated cardiac output registers within normal bounds on artificial telemetry.",
    "VIRTUAL ONCOLOGY CASE: Simulated breast cancer profile, HER2 negative, ER positive. Generated mock record for privacy auditing and differential validation.",
    "SYNTHETIC SUMMARY: Virtual subject presents with modeled seasonal allergic rhinitis. Prescribed synthetic Loratadine 10mg daily as virtual baseline test.",
    "SIMULATED NEUROLOGY: 67-year-old mock profile presents with minor artificial cognitive fluctuation. Mock test scores register at stable virtual baseline."
]

@st.cache_data
def generate_patient_records(dataset_name: str, n_records: int = 100) -> pd.DataFrame:
    """Generates mock patient records with realistic clinical or synthetic markers."""
    is_synthetic = "Synthetic" in dataset_name
    seed = 99 if is_synthetic else 42
    np.random.seed(seed)
    
    prefix = "SYN" if is_synthetic else "PT"
    ids = [f"{prefix}-{np.random.randint(10000, 99999)}" for _ in range(n_records)]
    
    # Choose snippets based on dataset selection
    snippets = SYNTHETIC_SNIPPETS if is_synthetic else CLINICAL_SNIPPETS
    texts = [np.random.choice(snippets) for _ in range(n_records)]
    
    # 50% member distribution (standard for MIA benchmarking)
    is_member = np.random.binomial(1, 0.5, n_records)
    
    df = pd.DataFrame({
        "Identifier": ids,
        "Raw Text": texts,
        "is_member": is_member
    })
    
    # Generate scores based on actual ground truth with noise scaled by model quality
    for model_name, info in MODELS_METRICS.items():
        auc = info["auc"]
        # Shift scores slightly differently on synthetic data for visual validation
        if is_synthetic:
            auc = min(0.9999, auc + 0.015)
            
        sep = (auc - 0.5) * 5.0
        
        # Draw probabilities from a beta or logit distribution styled based on member status
        base_noise = np.random.normal(0, 1.2, n_records)
        raw_signal = is_member * sep - (sep / 2) + base_noise
        
        # Map to [0, 1] probability range via sigmoid
        probs = 1 / (1 + np.exp(-raw_signal))
        df[f"score_{model_name}"] = np.round(probs, 4)
        
    return df


# -----------------------------------------------------------------------------
# 3. HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ MedShield: Clinical Data Privacy & MIA Auditing Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced telemetry and auditing dashboard measuring Membership Inference Attack (MIA) vulnerability on fine-tuned LLMs trained on medical record systems.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. SIDEBAR & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/shield.png", width=70)
st.sidebar.markdown("### 🎛️ Audit Configuration")

selected_dataset = st.sidebar.selectbox(
    "Medical Dataset",
    ["MIA-Clinical-v1 (Clinical Summaries)", "MIA-Synthetic-v2 (Fictional Patients)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Audited Models")
st.sidebar.caption("Select models to overlay on the ROC curve and compare privacy risk metrics.")

selected_models = []
for model_name in MODELS_METRICS.keys():
    default_val = model_name in ["WEL-Weighted-XGBoost", "WBC-Ensemble-XGBoost", "SmolLM2-360M-Mean (Baseline)"]
    checked = st.sidebar.checkbox(model_name, value=default_val)
    if checked:
        selected_models.append(model_name)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Risk Status Color Key")
st.sidebar.markdown("""
- <span class="badge-breached">🚨 HIGH LEAKAGE RISK</span> (AUC > 0.92)
- <span class="badge-warning">⚠️ MEDIUM LEAKAGE RISK</span> (AUC 0.85 - 0.92)
- <span class="badge-secure">🛡️ LOW LEAKAGE RISK</span> (AUC < 0.85)
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. DATA INGESTION & METRICS COMPILATION
# -----------------------------------------------------------------------------
# Load standard patient records
raw_db = generate_patient_records(selected_dataset, 100)
n_audited = len(raw_db)

# Compute metrics based on selected models
if len(selected_models) > 0:
    highest_auc = max([MODELS_METRICS[m]["auc"] for m in selected_models])
    avg_tpr = np.mean([MODELS_METRICS[m]["tpr_at_01"] for m in selected_models])
else:
    highest_auc = 0.5000
    avg_tpr = 0.1000

# -----------------------------------------------------------------------------
# 6. TOP METRICS ROW
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">📈 Highest MIA AUC Detected</div>
            <div class="metric-value">{highest_auc:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">🎯 Avg Member Recall (TPR @ FPR=10%)</div>
            <div class="metric-value">{avg_tpr * 100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">📋 Total Audited Patient Records</div>
            <div class="metric-value">{n_audited}</div>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. MAIN DASHBOARD SPLIT (ROC CURVE VS LEADERBOARD)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

# Left Column: Multi-Model ROC Curves (Plotly)
with col_left:
    with st.container(border=True):
        st.markdown("### 📊 Multi-Model ROC Curves")
        st.caption("Visual representation of attack models converging towards the top-left (higher area indicates greater privacy leakage risk).")
        
        fig = go.Figure()
        
        # Add Diagonal Random Guess Reference Line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#475569', width=2, dash='dash'),
            name='Random Baseline (AUC = 0.50)'
        ))
        
        # Generate smooth mathematical ROC points for selected models
        fpr_grid = np.linspace(0, 1, 150)
        for model_name in selected_models:
            auc_val = MODELS_METRICS[model_name]["auc"]
            color = MODELS_METRICS[model_name]["color"]
            
            # Perfect mathematical ROC curves parameterized to match AUC exactly
            # TPR = 1 - (1 - FPR)^a, where area under curve = a/(a+1) = AUC -> a = AUC/(1-AUC)
            if auc_val >= 0.999:
                tpr = np.ones_like(fpr_grid)
                tpr[0] = 0
            else:
                a = auc_val / (1.0 - auc_val)
                tpr = 1.0 - (1.0 - fpr_grid)**a
                
            fig.add_trace(go.Scatter(
                x=fpr_grid, y=tpr,
                mode='lines',
                line=dict(color=color, width=3),
                name=f"{model_name} (AUC={auc_val:.4f})"
            ))
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(
                title='False Positive Rate (FPR)',
                gridcolor='rgba(255, 255, 255, 0.05)',
                zerolinecolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='#94a3b8')
            ),
            yaxis=dict(
                title='True Positive Rate (TPR)',
                gridcolor='rgba(255, 255, 255, 0.05)',
                zerolinecolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='#94a3b8')
            ),
            legend=dict(
                x=0.55, y=0.05,
                bgcolor='rgba(15, 23, 42, 0.85)',
                bordercolor='rgba(255, 255, 255, 0.08)',
                borderwidth=1,
                font=dict(color='#e2e8f0', size=11)
            ),
            height=380,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Right Column: MIA Leaderboard & Vulnerability Audit
with col_right:
    with st.container(border=True):
        st.markdown("### 🏆 MIA Vulnerability Leaderboard")
        st.caption("Global assessment ranking target architectures based on security leakage scores. Red zones signify compromised parameters.")
        
        # Compile leaderboard data
        leaderboard_data = []
        for model_name, info in MODELS_METRICS.items():
            leaderboard_data.append({
                "Target Architecture": model_name,
                "MIA AUC": info["auc"],
                "TPR @ FPR=10%": info["tpr_at_01"],
                "Audit Assessment": info["status"]
            })
            
        ld_df = pd.DataFrame(leaderboard_data)
        ld_df = ld_df.sort_values(by="MIA AUC", ascending=False).reset_index(drop=True)
        
        # Beautiful Pandas Gradient Styling
        # Highlight high AUC values representing weak privacy
        styled_table = ld_df.style.background_gradient(
            cmap="OrRd", 
            subset=["MIA AUC"],
            vmin=0.50, 
            vmax=0.95
        ).format(
            {"MIA AUC": "{:.4f}", "TPR @ FPR=10%": "{:.2%}"}
        )
        
        st.dataframe(styled_table, use_container_width=True, height=360)


# -----------------------------------------------------------------------------
# 8. BOTTOM SECTION: INTERACTIVE DATA PROBE WORKSPACE
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Patient Record Privacy Probe Workspace")
st.markdown("Audit individual sequence instances. This portal lets you explore specific clinical profiles to determine how strongly their unique structures leak out-of-fold membership indicators.")

# Filter/Probe Model selection
if len(selected_models) > 0:
    probe_model = st.selectbox(
        "Evaluate Membership Probabilities Using Classifier Engine:",
        selected_models
    )
else:
    probe_model = list(MODELS_METRICS.keys())[0]

# Display interactive probe table
score_col = f"score_{probe_model}"

# Format display table
display_df = raw_db[["Identifier", "Raw Text", "is_member", score_col]].copy()

# Add interactive Search / Filter
search_query = st.text_input("✍️ Filter Clinical Text (e.g. 'COPD', 'chemotherapy', 'MRI')", "")
if search_query:
    display_df = display_df[display_df["Raw Text"].str.contains(search_query, case=False)]

# Create visual markers and status flags
status_list = []
badges = []

for _, row in display_df.iterrows():
    is_mem = row["is_member"]
    prob = row[score_col]
    
    # Conditional logic to tag status based on actual membership and prediction confidence
    if is_mem == 1 and prob >= 0.70:
        status_list.append("🔴 BREACHED")
        badges.append("🚨 HIGH CONFIDENCE LEAKAGE")
    elif is_mem == 1 and prob < 0.70:
        status_list.append("🟡 VULNERABLE")
        badges.append("⚠️ PARTIAL MEMORY SIGNAL")
    elif is_mem == 0 and prob >= 0.70:
        status_list.append("🟡 FALSE POSITIVE RISK")
        badges.append("⚠️ BOILERPLATE OVERLAP")
    else:
        status_list.append("🟢 SECURE")
        badges.append("🛡️ CONTAINS NO SIGNALS")

display_df["Audit Result"] = status_list
display_df["Clinical Insights"] = badges

# Truncate raw text for beautiful view
display_df["Clinical Text Sample"] = display_df["Raw Text"].apply(lambda x: x[:75] + "..." if len(x) > 75 else x)

# Rearrange columns for display
final_cols = ["Identifier", "Clinical Text Sample", "is_member", score_col, "Audit Result", "Clinical Insights"]
table_out = display_df[final_cols].rename(columns={
    "is_member": "In Training Set",
    score_col: "Breach Prob."
})

# Streamlit Dataframe display with formatting
st.dataframe(
    table_out.style.map(
        lambda x: "color: #ef4444; font-weight: bold;" if x == "🔴 BREACHED" else (
            "color: #fb923c; font-weight: bold;" if x in ["🟡 VULNERABLE", "🟡 FALSE POSITIVE RISK"] else (
                "color: #22c55e; font-weight: bold;" if x == "🟢 SECURE" else ""
            )
        ),
        subset=["Audit Result"]
    ).format(
        {"Breach Prob.": "{:.2%}"}
    ),
    use_container_width=True,
    height=400
)

# Patient Detail Inspector Expanders
st.markdown("#### 🕵️ Patient Narrative Deep-Dive Inspector")
st.caption("Click to expand and view the full clinical record and comparison metrics for targeted audits.")

# Display top 3 sample expanders from search results
inspected_samples = display_df.head(3)
if not inspected_samples.empty:
    for idx, row in inspected_samples.iterrows():
        status_color = "#ef4444" if row["Audit Result"] == "🔴 BREACHED" else (
            "#fb923c" if "🟡" in row["Audit Result"] else "#22c55e"
        )
        
        with st.expander(f"📋 Record ID: {row['Identifier']} | Result: {row['Audit Result']}"):
            col_exp_1, col_exp_2 = st.columns([3, 1])
            with col_exp_1:
                st.markdown(f"**Full Narrative Structure:**")
                st.text_area("Clinical Text", value=row["Raw Text"], height=100, disabled=True, key=f"txt_{row['Identifier']}")
            with col_exp_2:
                st.markdown("**Privacy Audit Stats:**")
                st.markdown(f"- **Ground Truth**: {'In Training Set (Member)' if row['is_member'] == 1 else 'Not in Training Set (Non-Member)'}")
                st.markdown(f"- **Leakage Score**: <span style='font-size: 1.1rem; font-weight: bold; color: {status_color};'>{row[score_col]:.2%}</span>", unsafe_allow_html=True)
                st.markdown(f"- **Vector Reliability**: {row['Clinical Insights']}")
else:
    st.write("No matching patient records found.")

# -----------------------------------------------------------------------------
# 9. FOOTER SECTION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;'>"
    "MedShield Platform v1.2.0 • UBC MDS-CL Privacy Auditing Engine • Secured via Out-of-Fold Cross-Validation Framework."
    "</div>", 
    unsafe_allow_html=True
)
