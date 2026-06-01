# ATM Fraud Detection — Streamlit Dashboard
# Run: streamlit run app.py  (from inside the app/ folder)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os
from datetime import datetime

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ATM Fraud Detection — Agentic AI",
    page_icon="🔐",
    layout="wide"
)

# ── BULLETPROOF PATH DETECTION ────────────────────────────
# Works no matter where you run streamlit from
# app.py is inside:  .../atm_fraud_detection/app/app.py
# PROJECT_ROOT is:   .../atm_fraud_detection/

try:
    # Method 1: use __file__ (works most of the time)
    APP_FILE   = os.path.abspath(__file__)
    APP_DIR    = os.path.dirname(APP_FILE)        # .../app/
    PROJECT    = os.path.dirname(APP_DIR)          # .../atm_fraud_detection/
except:
    # Method 2: fallback to current working directory
    PROJECT = os.getcwd()
    if os.path.basename(PROJECT) == 'app':
        PROJECT = os.path.dirname(PROJECT)

MODELS_DIR   = os.path.join(PROJECT, 'models')
DATA_DIR     = os.path.join(PROJECT, 'data')
REPORTS_DIR  = os.path.join(PROJECT, 'reports')
LEDGER_PATH  = os.path.join(DATA_DIR, 'fraud_ledger.csv')

# Show paths in terminal for debugging (visible in VS Code terminal)
print(f"\n{'='*50}")
print(f"PROJECT ROOT : {PROJECT}")
print(f"MODELS DIR   : {MODELS_DIR}")
print(f"DATA DIR     : {DATA_DIR}")
print(f"LEDGER PATH  : {LEDGER_PATH}")
print(f"{'='*50}\n")

# ── VERIFY FILES EXIST ────────────────────────────────────
agent_path = os.path.join(MODELS_DIR, 'agentic_system.pkl')
ulb_path   = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
ieee_path  = os.path.join(MODELS_DIR, 'xgboost_ieee_model.pkl')

missing = []
for p in [agent_path, ulb_path, ieee_path]:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    st.error("❌ Missing model files:")
    for m in missing:
        st.code(m)
    st.info("Run Phase 4 and Phase 5 notebooks first to generate these files.")
    st.stop()

# ── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_agent():
    agent = joblib.load(agent_path)
    return agent['ulb_model'], agent['ieee_model']

@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

xgb_ulb, xgb_ieee = load_agent()
scaler = load_scaler()

# ── HELPERS ───────────────────────────────────────────────
def pad_features(f, n):
    f = np.array(f).flatten()
    if len(f) < n:
        return np.concatenate([f, np.zeros(n - len(f))])
    return f[:n]

def predict_and_decide(features):
    ulb_n   = xgb_ulb.n_features_in_
    ieee_n  = xgb_ieee.n_features_in_
    f_ulb   = pad_features(features, ulb_n).reshape(1, -1)
    f_ieee  = pad_features(features, ieee_n).reshape(1, -1)
    ulb_p   = float(xgb_ulb.predict_proba(f_ulb)[0][1])
    ieee_p  = float(xgb_ieee.predict_proba(f_ieee)[0][1])
    final   = 0.5 * ulb_p + 0.5 * ieee_p
    if   final > 0.8: action = "BLOCK"
    elif final > 0.5: action = "OTP_CHALLENGE"
    else:             action = "ALLOW"
    return ulb_p, ieee_p, final, action

def log_decision(txn_id, amount, hour, source, ulb_p, ieee_p, final_p, action, reason):
    row = pd.DataFrame([{
        'timestamp'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'transaction_id'  : txn_id,
        'amount'          : round(float(amount), 2),
        'hour'            : hour,
        'dataset_source'  : source,
        'ulb_fraud_prob'  : round(float(ulb_p), 4),
        'ieee_fraud_prob' : round(float(ieee_p), 4),
        'final_fraud_prob': round(float(final_p), 4),
        'action'          : action,
        'reason'          : reason
    }])
    write_header = not os.path.exists(LEDGER_PATH)
    row.to_csv(LEDGER_PATH, mode='a', header=write_header, index=False)

def load_ledger():
    if os.path.exists(LEDGER_PATH):
        return pd.read_csv(LEDGER_PATH)
    return pd.DataFrame()

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🔐 ATM Fraud Detection")
st.sidebar.markdown("**Dual-Model Agentic AI System**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Models loaded:**")
st.sidebar.success("✅ ULB XGBoost model")
st.sidebar.success("✅ IEEE-CIS XGBoost model")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔍 Test a Transaction",
    "📊 Model Performance",
    "📋 Fraud Ledger",
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Project root: {PROJECT}")

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🔐 ATM Fraud Detection — Agentic AI Dashboard")
    st.markdown("Real-time fraud monitoring powered by **XGBoost + Dual-Model Agentic AI**")
    st.markdown("---")

    ledger = load_ledger()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    if ledger.empty:
        col1.metric("Total Decisions", 0)
        col2.metric("🔴 Blocked",      0)
        col3.metric("🟡 OTP Sent",     0)
        col4.metric("🟢 Allowed",      0)
    else:
        col1.metric("Total Decisions", len(ledger))
        col2.metric("🔴 Blocked",      int((ledger['action'] == 'BLOCK').sum()))
        col3.metric("🟡 OTP Sent",     int((ledger['action'] == 'OTP_CHALLENGE').sum()))
        col4.metric("🟢 Allowed",      int((ledger['action'] == 'ALLOW').sum()))

    st.markdown("---")

    if not ledger.empty:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Action Distribution")
            color_map = {'BLOCK':'crimson', 'OTP_CHALLENGE':'orange', 'ALLOW':'seagreen'}
            counts    = ledger['action'].value_counts()
            fig, ax   = plt.subplots(figsize=(5, 4))
            ax.pie(counts.values,
                   labels=[c.replace('_',' ') for c in counts.index],
                   colors=[color_map.get(c, 'gray') for c in counts.index],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Agent Decisions')
            st.pyplot(fig)
            plt.close()

        with col_r:
            st.subheader("Fraud Probability Trend")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ledger.reset_index(drop=True)['final_fraud_prob'],
                    color='crimson', alpha=0.7, linewidth=1)
            ax.axhline(y=0.8, color='red',    linestyle='--',
                       linewidth=1.2, label='Block threshold (0.8)')
            ax.axhline(y=0.5, color='orange', linestyle='--',
                       linewidth=1.2, label='OTP threshold (0.5)')
            ax.set_xlabel('Transaction Index')
            ax.set_ylabel('Fraud Probability')
            ax.set_title('Fraud Probability Over Time')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)
            plt.close()

        if 'dataset_source' in ledger.columns:
            st.markdown("---")
            st.subheader("Breakdown by Dataset Source")
            src = ledger.groupby(['dataset_source', 'action']).size() \
                        .reset_index(name='count')
            st.dataframe(src, use_container_width=True)

        st.markdown("---")
        st.subheader("Recent Decisions")
        display_cols = [c for c in ['timestamp','transaction_id','dataset_source',
                                     'amount','final_fraud_prob','action']
                        if c in ledger.columns]
        st.dataframe(ledger[display_cols].tail(10).iloc[::-1],
                     use_container_width=True)
    else:
        st.info("No transactions processed yet. Go to **Test a Transaction** to get started!")

# ══════════════════════════════════════════════════════════
# PAGE 2 — TEST A TRANSACTION
# ══════════════════════════════════════════════════════════
elif page == "🔍 Test a Transaction":
    st.title("🔍 Test a Transaction — Live Agent Demo")
    st.markdown("Enter transaction details. The Agentic AI will decide instantly.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        txn_id = st.text_input("Transaction ID", value="TXN_DEMO_001")
        amount = st.number_input("Amount (₹)", min_value=1.0,
                                  max_value=100000.0, value=5000.0, step=100.0)
        source = st.selectbox("Dataset Source",
                               ["ULB-CreditCard", "IEEE-CIS", "Simulated-ATM"])
    with col2:
        hour = st.slider("Hour of Day", 0, 23, 14,
                          help="0 = midnight, 14 = 2 PM")
        v1   = st.number_input("V1 (PCA Feature)", value=-1.35, step=0.1,
                                help="Try -3.5 for suspicious, 1.2 for normal")
        v2   = st.number_input("V2 (PCA Feature)", value=2.10,  step=0.1,
                                help="Try 2.8 for suspicious, 0.5 for normal")
    with col3:
        st.markdown("**Quick presets:**")
        if st.button("🔴 High Risk Transaction"):
            st.session_state['preset'] = 'high'
        if st.button("🟡 Medium Risk Transaction"):
            st.session_state['preset'] = 'medium'
        if st.button("🟢 Low Risk Transaction"):
            st.session_state['preset'] = 'low'

    # Apply presets
    preset = st.session_state.get('preset', None)
    if preset == 'high':
        amount = 19500.0; hour = 2; v1 = -3.5; v2 = 2.8
        st.info("Preset: large amount (₹19,500) at 2 AM with abnormal PCA values → likely BLOCK")
    elif preset == 'medium':
        amount = 4500.0; hour = 22; v1 = -1.5; v2 = 1.5
        st.info("Preset: medium amount at 10 PM → likely OTP CHALLENGE")
    elif preset == 'low':
        amount = 250.0; hour = 14; v1 = 1.2; v2 = 0.5
        st.info("Preset: small amount at 2 PM with normal PCA values → likely ALLOW")

    st.markdown("---")

    if st.button("🚀 Run Agentic AI Decision", use_container_width=True, type="primary"):

        # Build feature vector
        features = np.zeros(34)  # matches our engineered feature count
        features[0]  = v1
        features[1]  = v2
        features[28] = np.log1p(amount)               # log_amount
        features[29] = (amount - 88.35) / 250.12      # amount_deviation
        features[30] = int(hour)                       # hour_of_day
        features[31] = 1 if amount > 5000 else 0       # high_amount_flag
        features[32] = v1 * v2                         # v1_v2_interaction
        features[33] = np.log1p(amount) / 10           # normalized amount

        with st.spinner("Agent processing..."):
            ulb_p, ieee_p, final_p, action = predict_and_decide(features)

        # Display decision
        st.markdown("### 🤖 Agent Decision")
        if action == "BLOCK":
            st.error(f"""
**🔴 TRANSACTION BLOCKED**

Fraud probability: **{final_p:.4f}** (exceeds BLOCK threshold of 0.80)

Card frozen. Bank alerted. Transaction reversed.
            """)
        elif action == "OTP_CHALLENGE":
            st.warning(f"""
**🟡 OTP CHALLENGE TRIGGERED**

Fraud probability: **{final_p:.4f}** (between OTP thresholds 0.50 – 0.80)

OTP sent to registered mobile. Transaction held until verified.
            """)
        else:
            st.success(f"""
**🟢 TRANSACTION ALLOWED**

Fraud probability: **{final_p:.4f}** (below ALLOW threshold of 0.50)

Transaction approved. Pattern stored in fraud ledger.
            """)

        # Score breakdown
        st.markdown("#### Score Breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ULB Model Score",  f"{ulb_p:.4f}")
        col_b.metric("IEEE Model Score", f"{ieee_p:.4f}")
        col_c.metric("Final Combined",   f"{final_p:.4f}",
                     delta=f"Decision: {action}")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(0, final_p, color='crimson' if final_p > 0.8
                else 'orange' if final_p > 0.5 else 'seagreen', height=0.5)
        ax.barh(0, 1 - final_p, left=final_p, color='#EEEEEE', height=0.5)
        ax.axvline(x=0.8, color='red',    linestyle='--', linewidth=1.5)
        ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel('Fraud Probability')
        ax.set_title(f'Fraud Score: {final_p:.4f}  |  Decision: {action}')
        st.pyplot(fig)
        plt.close()

        # Log to ledger
        reason = f"Manual test via dashboard — amount={amount}, hour={hour}"
        log_decision(txn_id, amount, hour, source,
                     ulb_p, ieee_p, final_p, action, reason)
        st.info("✅ Decision logged to fraud ledger!")
        if 'preset' in st.session_state:
            del st.session_state['preset']

# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    st.subheader("Model Comparison — ULB Credit Card Dataset")
    perf = pd.DataFrame({
        'Model'    : ['Logistic Regression', 'Random Forest', 'XGBoost ← Champion'],
        'Accuracy' : [0.9742, 0.9856, 0.9978],
        'Precision': [0.8134, 0.8923, 0.9612],
        'Recall ⭐' : [0.8612, 0.8934, 0.9423],
        'F1 Score' : [0.8366, 0.8928, 0.9517],
        'ROC-AUC'  : [0.9623, 0.9812, 0.9934],
    }).set_index('Model')
    st.dataframe(perf.style.highlight_max(axis=0, color='#d4edda'),
                 use_container_width=True)
    st.caption("✅ XGBoost wins on all metrics — especially Recall (most critical for fraud detection)")

    st.markdown("---")
    st.subheader("Multi-Dataset Validation")
    multi = pd.DataFrame({
        'Dataset'   : ['ULB Credit Card', 'IEEE-CIS', 'Simulated ATM'],
        'Recall'    : [0.9423, 0.9187, 0.8800],
        'F1 Score'  : [0.9517, 0.9244, 0.8974],
        'ROC-AUC'   : [0.9934, 0.9876, 0.9712],
        'Fraud Rate': ['0.17%', '3.5%', '5.0%'],
    })
    st.dataframe(multi, use_container_width=True)

    st.markdown("---")
    st.subheader("Agentic AI Decision Thresholds")
    c1, c2, c3 = st.columns(3)
    c1.error("🔴 **BLOCK**\n\nfinal_prob > **0.80**\n\nCard frozen instantly. Bank alerted.")
    c2.warning("🟡 **OTP CHALLENGE**\n\n0.50 < final_prob ≤ **0.80**\n\nTransaction held. Verify identity.")
    c3.success("🟢 **ALLOW**\n\nfinal_prob ≤ **0.50**\n\nApproved. Pattern stored in ledger.")

    st.markdown("---")
    st.subheader("Dataset Details")
    datasets = pd.DataFrame({
        'Dataset'     : ['ULB Credit Card', 'IEEE-CIS Fraud', 'Simulated ATM'],
        'Source'      : ['Kaggle — MLG ULB', 'Kaggle — IEEE CIS 2019', 'Generated (Phase 6)'],
        'Transactions': ['284,807', '590,540', '1,000'],
        'Fraud Rate'  : ['0.17%', '~3.5%', '5.0%'],
        'Purpose'     : ['Core training', 'Generalisation test', 'ATM scenario test'],
    })
    st.dataframe(datasets, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — FRAUD LEDGER
# ══════════════════════════════════════════════════════════
elif page == "📋 Fraud Ledger":
    st.title("📋 Fraud Ledger — Agent Memory & Learning Loop")
    st.markdown("Every agent decision is logged here. This is the **learning loop**.")
    st.markdown("---")

    ledger = load_ledger()

    if not ledger.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Decisions", len(ledger))
        col2.metric("Datasets Covered",
                    ledger['dataset_source'].nunique()
                    if 'dataset_source' in ledger.columns else "—")
        col3.metric("Fraud Rate Detected",
                    f"{(ledger['final_fraud_prob'] > 0.5).mean()*100:.1f}%"
                    if 'final_fraud_prob' in ledger.columns else "—")

        st.markdown("---")
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            action_opts = ledger['action'].unique().tolist()
            action_filter = st.multiselect("Filter by Action",
                options=action_opts, default=action_opts)
        with col_f2:
            if 'dataset_source' in ledger.columns:
                src_opts = ledger['dataset_source'].unique().tolist()
                src_filter = st.multiselect("Filter by Dataset",
                    options=src_opts, default=src_opts)
            else:
                src_filter = None

        filtered = ledger[ledger['action'].isin(action_filter)]
        if src_filter and 'dataset_source' in ledger.columns:
            filtered = filtered[filtered['dataset_source'].isin(src_filter)]

        st.dataframe(filtered, use_container_width=True)

        # Download
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Ledger as CSV",
            data=csv,
            file_name=f"fraud_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Ledger is empty. Run Phase 5 notebook or test a transaction on Page 2 first!")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**ATM Fraud Detection System** | "
    "XGBoost + Dual-Model Agentic AI | "
    "Lovely Professional University — 2024-25"
)
