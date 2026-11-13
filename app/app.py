"""
app.py
------
Streamlit application for the Loan Default Prediction project.

Loads the saved end-to-end pipeline (preprocessing + model) and lets a user
enter borrower/loan information to get a default risk prediction. No
preprocessing logic is duplicated here — the same feature-engineering
functions used during training are reused, and the saved sklearn Pipeline
handles imputation/encoding/scaling internally.

Run with:
    streamlit run app/app.py
"""

import json
import os
import sys
from datetime import date

import joblib
import pandas as pd
import streamlit as st

# Make `src` importable when running from the app/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import full_feature_pipeline  # noqa: E402

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

st.set_page_config(page_title="Loan Default Prediction", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------
# Design tokens & global styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg:        #0A0E16;
            --bg-alt:    #0C111C;
            --surface:   #131926;
            --surface-2: #1B2333;
            --border:    #232B3E;
            --text:      #FFFFFF;
            --text-muted:#FFFFFF;
            --accent:    #3B82F6;
            --accent-2:  #60A5FA;
            --success:   #22C55E;
            --warning:   #F0A93A;
            --danger:    #EF4444;
        }

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }

        .stApp { background-color: var(--bg); color: var(--text); }
        section[data-testid="stSidebar"] {
            background-color: var(--bg-alt);
            border-right: 1px solid var(--border);
        }
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Inputs */
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div > div,
        .stNumberInput input,
        .stTextInput input {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"],
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
        div[data-testid="stNumberInput"] button {
            background: var(--surface-2) !important;
            color: var(--text) !important;
        }
        .react-aria-ComboBox [role="group"],
        .react-aria-ComboBox input,
        .react-aria-ComboBox button,
        div[data-testid="stNumberInputContainer"],
        div[data-testid="stNumberInputContainer"] > div,
        div[data-testid="stNumberInputContainer"] button {
            background: var(--surface-2) !important;
            color: var(--text) !important;
        }
        .react-aria-ComboBox [role="group"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stNumberInput"] button:hover {
            background: var(--border) !important;
        }
        div[data-baseweb="select"] *,
        div[data-baseweb="select"] [aria-selected="true"],
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] > div,
        div[role="listbox"],
        div[role="option"] {
            background-color: var(--surface-2) !important;
            color: var(--text) !important;
        }
        div[role="option"]:hover,
        div[role="option"][aria-selected="true"] {
            background-color: var(--border) !important;
            color: var(--text) !important;
        }
        /* Slider track + thumb only (not the value labels) */
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background-color: var(--accent) !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        /* Kill the blue/red highlight boxes on min, max and current slider values */
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"],
        div[data-testid="stSlider"] div[data-baseweb="slider"] span,
        div[data-testid="stSlider"] div[data-baseweb="tooltip"],
        div[data-testid="stThumbValue"] {
            background: transparent !important;
            box-shadow: none !important;
            color: #FFFFFF !important;
        }

        /* Force every bit of text white, everywhere */
        * { color: #FFFFFF !important; }
        ::selection { background: rgba(59,130,246,0.35); color: #FFFFFF; }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid var(--border);
        }
        .stButton > button:not([kind="primary"]) {
            background: var(--surface-2) !important;
            color: var(--text) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            border: none;
            color: white;
        }

        /* Generic card */
        .card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }
        .card-title {
            font-family: 'Sora', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.15rem;
        }
        .card-subtitle { color: var(--text-muted); font-size: 0.82rem; margin-bottom: 0; }

        /* Hero card */
        .hero-card {
            background: linear-gradient(160deg, #1D3A63 0%, #142642 55%, #0C111C 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.6rem 1.6rem;
            color: white;
            margin-bottom: 1rem;
        }
        .hero-card h4 { color: white !important; margin-bottom: 0.5rem; }
        .hero-card p { color: #C7D3E8 !important; font-size: 0.88rem; line-height: 1.5; }

        /* Metric rows */
        .metric-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.55rem 0; border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-row .val { font-weight: 600; color: var(--text); }
        .metric-row .lbl { color: var(--text-muted); }

        /* Risk result cards */
        .risk-card {
            padding: 1.4rem 1.6rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            background-color: var(--surface);
            margin-top: 0.5rem;
        }
        .risk-low { border-left: 5px solid var(--success); }
        .risk-medium { border-left: 5px solid var(--warning); }
        .risk-high { border-left: 5px solid var(--danger); }
        .risk-card h3 { margin-bottom: 0.4rem; }
        .risk-card p { margin: 0.15rem 0; color: var(--text-muted); }
        .risk-card b { color: var(--text); }

        /* Status pill */
        .status-pill {
            display: inline-flex; align-items: center; gap: 6px;
            background-color: rgba(34,197,94,0.12); color: var(--success);
            border: 1px solid rgba(34,197,94,0.35);
            padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background-color: var(--success); }

        /* Sidebar nav buttons */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%; text-align: left; background: transparent;
            border: 1px solid transparent; color: var(--text-muted);
            font-weight: 500; padding: 0.55rem 0.9rem; margin-bottom: 2px;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: var(--surface); color: var(--text); border-color: var(--border);
        }

        hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    return "High"


def card(title, subtitle=None, body_html=""):
    sub = f'<p class="card-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div>{sub}{body_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; padding:4px 4px 18px 4px;">
            <div style="font-size:1.6rem;">🛡️</div>
            <div>
                <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:1.05rem; line-height:1.15;">
                    Loan Default<br/>Prediction
                </div>
            </div>
        </div>
        <p style="color:#8B93A8; font-size:0.8rem; margin-top:-10px;">AI-powered credit risk assessment</p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("Home", "🏠"),
        ("Predict Loan Risk", "🎯"),
        ("Model Insights", "📊"),
        ("About", "ℹ️"),
    ]
    for label, icon in nav_items:
        if st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="background-color:#131926;">
            <div style="font-size:1.3rem; margin-bottom:6px;">📈</div>
            <div style="font-weight:600; font-size:0.9rem;">Smarter predictions.<br/>Safer lending.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ---------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------
header_l, header_r = st.columns([3, 1])
with header_l:
    if page == "Home":
        st.markdown("## Welcome to <span style='color:var(--accent);'>Loan Default Prediction</span>", unsafe_allow_html=True)
        st.caption("Enter the borrower details below to predict the likelihood of loan default.")
    elif page == "Predict Loan Risk":
        st.markdown("## Predict Loan Risk")
        st.caption("Fill in the borrower and loan details, then run the model.")
    elif page == "Model Insights":
        st.markdown("## Model Insights")
        st.caption("Performance and configuration of the deployed model.")
    else:
        st.markdown("## About this Project")
        st.caption("How the prediction pipeline works, end to end.")
with header_r:
    model_exists = os.path.exists(MODEL_PATH)
    dot_color = "var(--success)" if model_exists else "var(--danger)"
    pill_text = "Model Ready" if model_exists else "Model Missing"
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:10px;">
            <span class="status-pill" style="color:{dot_color}; border-color:{dot_color};
                background-color: transparent;">
                <span class="status-dot" style="background-color:{dot_color};"></span> {pill_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not model_exists:
    st.error(
        "No trained model found. Please run `python main.py` from the "
        "project root first to train and save the model."
    )
    st.stop()

model, metadata = load_model_and_metadata()
val_metrics = metadata.get("validation_metrics_default_threshold", {})

# =======================================================================
# HOME
# =======================================================================
if page == "Home":
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        card(
            "What this tool does",
            body_html="""
            <p style="color:#8B93A8; font-size:0.9rem; line-height:1.6; margin-top:8px;">
                This app scores a loan application against a trained machine-learning model
                and returns a probability of default, a plain-language risk level, and the
                decision behind it. Head to <b style="color:#E8EBF1;">Predict Loan Risk</b>
                in the sidebar to run a scenario, or check <b style="color:#E8EBF1;">Model
                Insights</b> to see how the model performs.
            </p>
            """,
        )
        if st.button("Go to Predict Loan Risk →", type="primary"):
            st.session_state.page = "Predict Loan Risk"
            st.rerun()

    with side_col:
        st.markdown(
            """
            <div class="hero-card">
                <h4>Better Data. Lower Risk.</h4>
                <p>Our AI model analyzes key financial and credit factors to predict the
                likelihood of loan default, helping you make smarter lending decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card(
            "Model Information",
            body_html=f"""
            <div class="metric-row"><span class="lbl">Model Used</span><span class="val">{metadata.get('model_name', 'N/A')}</span></div>
            <div class="metric-row"><span class="lbl">ROC-AUC</span><span class="val">{val_metrics.get('ROC-AUC', 0):.3f}</span></div>
            <div class="metric-row"><span class="lbl">F1 Score</span><span class="val">{val_metrics.get('F1', 0):.3f}</span></div>
            """,
        )
        card(
            "Your data is secure",
            body_html='<p style="color:#8B93A8; font-size:0.82rem; margin-top:4px;">We never store your personal information.</p>',
        )

# =======================================================================
# PREDICT LOAN RISK
# =======================================================================
elif page == "Predict Loan Risk":
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        card("Borrower & Loan Information", "Fill in the details below. All information is used for risk analysis.")

        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
            term = st.selectbox("Loan Term", options=[36, 60], format_func=lambda x: f"{x} months")
            int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
            installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=1500.0, value=300.0, step=10.0)
            grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"], index=2)
            sub_grade_num = st.slider("Sub-grade (1=best, 5=worst within grade)", 1, 5, 1)
            annual_inc = st.number_input("Annual Income ($)", min_value=4000, max_value=1000000, value=60000, step=1000)
            emp_length_label = st.selectbox(
                "Employment Length",
                options=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                         "6 years", "7 years", "8 years", "9 years", "10+ years"],
                index=5,
            )
            home_ownership = st.selectbox("Home Ownership", options=["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"])
            verification_status = st.selectbox(
                "Income Verification", options=["Not Verified", "Source Verified", "Verified"]
            )

        with col2:
            purpose = st.selectbox(
                "Loan Purpose",
                options=["debt_consolidation", "credit_card", "home_improvement", "car", "other",
                         "small_business", "wedding", "medical", "renewable_energy", "house",
                         "major_purchase", "vacation", "moving", "educational"],
            )
            addr_state = st.selectbox(
                "State",
                options=["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "MI",
                         "VA", "NC", "AZ", "WA", "MA", "CO", "MD", "OTHER"],
            )
            dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 15.0, 0.1)
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=8000, step=500)
            revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, 0.5)
            open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=9, step=1)
            total_acc = st.number_input("Total Credit Lines", min_value=0, max_value=100, value=20, step=1)
            delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", min_value=0, max_value=20, value=0, step=1)
            inq_last_6mths = st.number_input("Credit Inquiries (last 6 mths)", min_value=0, max_value=15, value=1, step=1)
            pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0, step=1)
            pub_rec_bankruptcies = st.number_input("Public Bankruptcies", min_value=0, max_value=5, value=0, step=1)

            earliest_cr_year = st.slider("Year Credit History Began", 1960, 2015, 2002)
            issue_year_input = st.slider("Loan Issue Year", 2007, 2016, 2012)
            has_desc = st.checkbox("Borrower provided a loan description", value=True)

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            reset_clicked = st.button("↺ Reset", use_container_width=True)
        with btn_col2:
            predict_clicked = st.button("✨ Predict Loan Risk →", type="primary", use_container_width=True)

        if reset_clicked:
            st.rerun()

        if predict_clicked:
            raw_row = pd.DataFrame([{
                "loan_amnt": loan_amnt,
                "funded_amnt": loan_amnt,
                "funded_amnt_inv": loan_amnt,
                "term": f" {term} months",
                "int_rate": f"{int_rate}%",
                "installment": installment,
                "grade": grade,
                "sub_grade": f"{grade}{sub_grade_num}",
                "emp_length": emp_length_label,
                "home_ownership": home_ownership,
                "annual_inc": annual_inc,
                "verification_status": verification_status,
                "issue_d": date(issue_year_input, 6, 1).strftime("%b-%y"),
                "desc": ("Borrower provided additional context." if has_desc else None),
                "purpose": purpose,
                "title": purpose,
                "zip_code": "000xx",
                "addr_state": addr_state,
                "dti": dti,
                "delinq_2yrs": delinq_2yrs,
                "earliest_cr_line": date(earliest_cr_year, 1, 1).strftime("%b-%y"),
                "inq_last_6mths": inq_last_6mths,
                "mths_since_last_delinq": None,
                "mths_since_last_record": None,
                "open_acc": open_acc,
                "pub_rec": pub_rec,
                "revol_bal": revol_bal,
                "revol_util": f"{revol_util}%",
                "total_acc": total_acc,
                "pub_rec_bankruptcies": pub_rec_bankruptcies,
            }])

            engineered_row = full_feature_pipeline(raw_row, id_col="id", target_col=None)
            feature_cols = metadata["features_used"]
            for col in feature_cols:
                if col not in engineered_row.columns:
                    engineered_row[col] = None
            X_input = engineered_row[feature_cols]

            probability = model.predict_proba(X_input)[0, 1]
            threshold = metadata.get("decision_threshold", 0.5)
            prediction = "Likely to Default" if probability >= threshold else "Unlikely to Default"
            level = risk_level(probability)

            st.markdown("#### Prediction Result")
            risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[level]
            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <h3>{prediction}</h3>
                    <p><b>Default Probability:</b> {probability * 100:.2f}%</p>
                    <p><b>Risk Level:</b> {level}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with side_col:
        st.markdown(
            """
            <div class="hero-card">
                <h4>Better Data. Lower Risk.</h4>
                <p>Our AI model analyzes key financial and credit factors to predict the
                likelihood of loan default, helping you make smarter lending decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card(
            "Model Information",
            body_html=f"""
            <div class="metric-row"><span class="lbl">Model Used</span><span class="val">{metadata.get('model_name', 'N/A')}</span></div>
            <div class="metric-row"><span class="lbl">ROC-AUC</span><span class="val">{val_metrics.get('ROC-AUC', 0):.3f}</span></div>
            <div class="metric-row"><span class="lbl">F1 Score</span><span class="val">{val_metrics.get('F1', 0):.3f}</span></div>
            """,
        )
        card(
            "Example Input",
            "Here's a sample of how to fill the form:",
            body_html="""
            <div class="metric-row"><span class="lbl">Monthly Installment</span><span class="val">300.00</span></div>
            <div class="metric-row"><span class="lbl">Revolving Balance</span><span class="val">8000</span></div>
            <div class="metric-row"><span class="lbl">Loan Grade</span><span class="val">C</span></div>
            <div class="metric-row"><span class="lbl">Revolving Utilization</span><span class="val">45%</span></div>
            """,
        )
        card(
            "Your data is secure",
            body_html='<p style="color:#8B93A8; font-size:0.82rem; margin-top:4px;">We never store your personal information.</p>',
        )

# =======================================================================
# MODEL INSIGHTS
# =======================================================================
elif page == "Model Insights":
    m1, m2, m3 = st.columns(3)
    with m1:
        card("Model Used", body_html=f'<div style="font-size:1.4rem; font-weight:700;">{metadata.get("model_name", "N/A")}</div>')
    with m2:
        card("ROC-AUC", body_html=f'<div style="font-size:1.4rem; font-weight:700;">{val_metrics.get("ROC-AUC", 0):.3f}</div>')
    with m3:
        card("F1 Score", body_html=f'<div style="font-size:1.4rem; font-weight:700;">{val_metrics.get("F1", 0):.3f}</div>')

    remaining_metrics = {k: v for k, v in val_metrics.items() if k not in ("ROC-AUC", "F1")}
    if remaining_metrics:
        rows = "".join(
            f'<div class="metric-row"><span class="lbl">{k}</span><span class="val">{v:.3f}'
            f'</span></div>' if isinstance(v, (int, float)) else
            f'<div class="metric-row"><span class="lbl">{k}</span><span class="val">{v}</span></div>'
            for k, v in remaining_metrics.items()
        )
        card("Additional Validation Metrics", body_html=rows)

    decision_threshold = metadata.get("decision_threshold", 0.5)
    features_used = metadata.get("features_used", [])
    card(
        "Configuration",
        body_html=f"""
        <div class="metric-row"><span class="lbl">Decision Threshold</span><span class="val">{decision_threshold}</span></div>
        <div class="metric-row"><span class="lbl">Number of Features</span><span class="val">{len(features_used)}</span></div>
        """,
    )
    if features_used:
        with st.expander("View feature list used by the model"):
            st.write(", ".join(features_used))

# =======================================================================
# ABOUT
# =======================================================================
else:
    main_col, side_col = st.columns([2, 1], gap="large")
    with main_col:
        card(
            "How it works",
            body_html="""
            <p style="color:#8B93A8; font-size:0.9rem; line-height:1.7; margin-top:8px;">
                1. You enter borrower and loan details on the <b style="color:#E8EBF1;">Predict Loan Risk</b> page.<br/>
                2. The same feature-engineering functions used during training transform the raw
                inputs into model-ready features.<br/>
                3. A saved scikit-learn pipeline handles imputation, encoding, and scaling internally,
                then the trained classifier estimates a default probability.<br/>
                4. That probability is compared against the model's decision threshold to produce a
                clear, plain-language risk level.
            </p>
            """,
        )
        card(
            "Data & privacy",
            body_html='<p style="color:#8B93A8; font-size:0.9rem; line-height:1.6; margin-top:8px;">Inputs are used only to compute a prediction for this session — nothing you enter is stored or sent anywhere else.</p>',
        )
    with side_col:
        st.markdown(
            """
            <div class="hero-card">
                <h4>Built for transparency</h4>
                <p>Every prediction ships with the probability, the threshold used, and the model
                behind it — no black boxes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
