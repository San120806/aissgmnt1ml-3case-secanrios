"""
Machine Learning Fundamentals - Interactive Exploration App
Run with: streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

st.set_page_config(
    page_title="ML Fundamentals Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-supervised { background-color: #DBEAFE; color: #1E40AF; }
    .badge-unsupervised { background-color: #DCFCE7; color: #166534; }
    .badge-regression { background-color: #FEF3C7; color: #92400E; }
    .badge-classification { background-color: #FCE7F3; color: #9D174D; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a Section:",
    [
        "📘 Overview & Core Concepts",
        "1️⃣ Scenario 1: Customer Spend (Regression)",
        "2️⃣ Scenario 2: Fraud Detection (Classification)",
        "3️⃣ Scenario 3: Viewing Habits (Clustering)",
        "🎓 Part B: Student Scenario (Pass/Fail)",
        "⚠️ Part C: Risk Analysis (Misapplying ML)",
        "📄 View / Download PDF Report"
    ]
)

# -------------------------------------------------------------
# PAGE: OVERVIEW
# -------------------------------------------------------------
if page == "📘 Overview & Core Concepts":
    st.markdown('<div class="main-title">Machine Learning Fundamentals — Interactive Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A beginner-friendly demonstration of core ML principles, scenarios, and error trade-offs.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🎓 What is Supervised Learning?\n"
                "Learning with a 'teacher' or 'answer key'. The model is trained on past data that already contains the right answer (ground-truth labels).\n\n"
                "**Two Main Types:**\n"
                "- **Regression:** The answer is a continuous number (e.g. Dollar spend, house price, temperature).\n"
                "- **Classification:** The answer is a discrete category (e.g. Fraud vs Normal, Pass vs Fail, Cat vs Dog).")

    with col2:
        st.success("### 🧩 What is Unsupervised Learning?\n"
                   "Learning without labels or predefined answers. The model analyzes patterns, groupings, or distributions in data organically.\n\n"
                   "**Main Application:**\n"
                   "- **Clustering:** Partitioning data into groups where items in the same cluster are similar to each other, but different from other clusters (e.g. Viewer habits, customer personas).")

    st.markdown("---")
    st.subheader("📋 Assignment Matrix Summary")
    df_matrix = pd.DataFrame([
        {"Scenario": "1. Customer Spend Next Month", "Paradigm": "Supervised", "Task Type": "Regression", "Key Features": "Avg spend, Frequency, Recency", "Target Variable": "next_month_spend ($)"},
        {"Scenario": "2. Fraud Detection", "Paradigm": "Supervised", "Task Type": "Binary Classification", "Key Features": "Amount, Distance, Velocity", "Target Variable": "is_fraud (0 or 1)"},
        {"Scenario": "3. Viewing Habit Segmentation", "Paradigm": "Unsupervised", "Task Type": "Clustering", "Key Features": "Watch hours, Weekend ratio, Genres", "Target Variable": "None (No labels)"},
        {"Scenario": "4. Student Exam Outcome (Part B)", "Paradigm": "Supervised", "Task Type": "Binary Classification", "Key Features": "Attendance %, Study hrs, Assignments", "Target Variable": "course_passed (0 or 1)"}
    ])
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# PAGE: SCENARIO 1
# -------------------------------------------------------------
elif page == "1️⃣ Scenario 1: Customer Spend (Regression)":
    st.markdown('<div class="main-title">Scenario 1: Customer Spend Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-supervised">Supervised Learning</span>
    <span class="badge badge-regression">Regression Task</span>
    """, unsafe_allow_html=True)
    st.write("")

    st.write("""
    **Brief:** *Predict how much a customer will spend next month from their past purchases.*
    - **(a) Supervised or Unsupervised:** **Supervised Learning** (we have past records with known historical monthly spend).
    - **(b) Classification or Regression:** **Regression** (predicting a continuous dollar value).
    - **(c) 2–3 Likely Features:** Past average spend, purchase frequency (orders/mo), recency (days since last purchase).
    - **(d) The Target:** `next_month_spend` (continuous numeric value in $).
    """)

    st.markdown("---")
    col_input, col_chart = st.columns([1, 1.2])

    with col_input:
        st.subheader("Interactive Predictor")
        in_past_spend = st.slider("Past 3-Month Average Spend ($)", min_value=10.0, max_value=800.0, value=150.0, step=5.0)
        in_freq = st.slider("Monthly Purchase Frequency (orders)", min_value=1, max_value=25, value=4, step=1)
        in_recency = st.slider("Days Since Last Purchase", min_value=1, max_value=60, value=7, step=1)

        # Baseline linear model formula: y = 0.85*spend + 12*freq - 1.2*recency
        predicted_spend = max(5.0, (0.85 * in_past_spend) + (12 * in_freq) - (1.2 * in_recency))

        st.metric(label="Predicted Next Month Spend", value=f"${predicted_spend:.2f}")
        st.caption("Notice: Higher past spend & frequent visits boost predicted spend, while inactive recency lowers it.")

    with col_chart:
        st.subheader("Model Validation Plot")
        img_path = os.path.join(FIGURES_DIR, "scenario_1_spend_regression.png")
        st.image(img_path, caption="Actual vs. Predicted Spend (R² = 0.974, RMSE = $20.05)")


# -------------------------------------------------------------
# PAGE: SCENARIO 2
# -------------------------------------------------------------
elif page == "2️⃣ Scenario 2: Fraud Detection (Classification)":
    st.markdown('<div class="main-title">Scenario 2: Fraudulent Transaction Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-supervised">Supervised Learning</span>
    <span class="badge badge-classification">Binary Classification</span>
    """, unsafe_allow_html=True)
    st.write("")

    st.write("""
    **Brief:** *Flag whether a transaction is fraud or not.*
    - **(a) Supervised or Unsupervised:** **Supervised Learning** (trained on confirmed fraudulent and legitimate transaction logs).
    - **(b) Classification or Regression:** **Classification (Binary)** (predicting Fraud = 1 vs Legitimate = 0).
    - **(c) 2–3 Likely Features:** Transaction amount ($), distance from home (km), transaction timestamp / velocity.
    - **(d) The Target:** `is_fraud` (binary label: 1 or 0).
    """)

    st.markdown("---")
    col_input, col_chart = st.columns([1, 1.2])

    with col_input:
        st.subheader("Test a Transaction")
        in_amount = st.slider("Transaction Amount ($)", min_value=5.0, max_value=2000.0, value=45.0, step=10.0)
        in_dist = st.slider("Distance from Home Location (km)", min_value=0.0, max_value=500.0, value=12.0, step=5.0)

        # Simple synthetic risk scoring
        risk_score = (in_amount / 250.0) * 0.5 + (in_dist / 100.0) * 0.5
        risk_prob = 1 / (1 + np.exp(-(risk_score - 2.0)))

        st.progress(float(min(1.0, risk_prob)))
        if risk_prob > 0.5:
            st.error(f"🚨 Flagged as **SUSPICIOUS / FRAUD** (Risk Probability: {risk_prob:.1%})")
        else:
            st.success(f"✅ Flagged as **LEGITIMATE** (Risk Probability: {risk_prob:.1%})")

    with col_chart:
        st.subheader("Model Confusion Matrix")
        st.image(os.path.join(FIGURES_DIR, "scenario_2_fraud_confusion_matrix.png"), caption="Confusion Matrix on Test Dataset (98% Accuracy)")


# -------------------------------------------------------------
# PAGE: SCENARIO 3
# -------------------------------------------------------------
elif page == "3️⃣ Scenario 3: Viewing Habits (Clustering)":
    st.markdown('<div class="main-title">Scenario 3: Viewer Habit Segmentation</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-unsupervised">Unsupervised Learning</span>
    <span class="badge badge-regression" style="background:#E0E7FF;color:#3730A3;">Clustering Task</span>
    """, unsafe_allow_html=True)
    st.write("")

    st.write("""
    **Brief:** *Group users into segments based on viewing habits (no preset labels).*
    - **(a) Supervised or Unsupervised:** **Unsupervised Learning** (no preset labels or correct answers provided).
    - **(b) Classification or Regression:** **Neither (Clustering)** (groups items based on feature similarities).
    - **(c) 2–3 Likely Features:** Weekly watch hours, weekend vs weekday ratio, genre diversity ratio.
    - **(d) The Target:** **None** (unsupervised algorithms do not use target labels).
    """)

    st.markdown("---")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Discovered Viewer Personas")
        st.markdown("""
        **Segment 1: Casual Weekend Viewers (Blue)**
        - Low total weekly hours (~5 hrs/week).
        - High weekend ratio (>80% on Sat/Sun).
        
        **Segment 2: Heavy Binge Watchers (Green)**
        - Very high weekly consumption (~28 hrs/week).
        - Balanced viewing across all days.
        
        **Segment 3: Daily Evening Habitual Viewers (Orange)**
        - Moderate weekly hours (~14 hrs/week).
        - Consistent weekday evening viewings.
        """)
    with col2:
        st.subheader("Cluster Distribution Map")
        st.image(os.path.join(FIGURES_DIR, "scenario_3_user_clusters.png"), caption="K-Means (K=3) Discovered Cohorts")


# -------------------------------------------------------------
# PAGE: PART B
# -------------------------------------------------------------
elif page == "🎓 Part B: Student Scenario (Pass/Fail)":
    st.markdown('<div class="main-title">Part B: Student-Created Scenario</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-supervised">Supervised Learning</span>
    <span class="badge badge-classification">Binary Classification</span>
    """, unsafe_allow_html=True)
    st.write("")

    st.write("""
    **Proposed Scenario:** *Predict whether a university student will pass or fail a course based on semester study habits.*
    - **(a) Supervised or Unsupervised:** **Supervised Learning** (historical students have recorded study metrics and final pass/fail marks).
    - **(b) Classification or Regression:** **Classification (Binary)** (predicting Pass = 1 vs Fail = 0).
    - **(c) 2–3 Likely Features:** Class attendance %, weekly self-study hours, on-time homework submission rate %.
    - **(d) The Target:** `course_passed` (binary categorical label: 1 = Pass, 0 = Fail).
    """)

    st.markdown("---")
    col_input, col_chart = st.columns([1, 1.2])
    with col_input:
        st.subheader("Test Student Outcome")
        in_att = st.slider("Class Attendance Rate (%)", min_value=30, max_value=100, value=75, step=5)
        in_hrs = st.slider("Weekly Self-Study Hours", min_value=1, max_value=30, value=10, step=1)
        in_hw = st.slider("Homework Completion Rate (%)", min_value=20, max_value=100, value=80, step=5)

        # Logit equation
        logit = -6.0 + (0.05 * in_att) + (0.18 * in_hrs) + (3.0 * (in_hw / 100.0))
        prob_pass = 1 / (1 + np.exp(-logit))

        st.metric("Predicted Pass Probability", f"{prob_pass:.1%}")
        if prob_pass >= 0.5:
            st.success("🎓 Predicted Outcome: **PASS**")
        else:
            st.error("⚠️ Predicted Outcome: **AT RISK / FAIL**")

    with col_chart:
        st.subheader("Decision Boundary")
        st.image(os.path.join(FIGURES_DIR, "part_b_student_pass_fail.png"), caption="Study Hours vs Attendance with Outcome (84% Model Accuracy)")


# -------------------------------------------------------------
# PAGE: PART C
# -------------------------------------------------------------
elif page == "⚠️ Part C: Risk Analysis (Misapplying ML)":
    st.markdown('<div class="main-title">Part C: The Risks of Misapplying Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("### Selected Scenario: Scenario 2 — Fraudulent Transaction Detection (Word Count: ~255 Words)")

    st.warning("""
    A major risk of misapplying machine learning to fraud detection is failing to account for **severe class imbalance** and the resulting **asymmetric cost of prediction errors**. In real-world payment networks, legitimate transactions make up over 99% of all activity, while actual fraud represents less than 1%. If a practitioner naively optimizes for overall accuracy, a completely trivial model that predicts 'legitimate' 100% of the time achieves 99% accuracy while catching zero fraudulent transactions, rendering the system practically useless.

    Furthermore, improper thresholding produces severe business consequences. If decision boundaries are tuned too aggressively to catch every potential threat, the rate of **False Positives** spikes. When honest cardholders traveling abroad or making atypical anniversary purchases have their cards blocked, it causes acute customer frustration. Industry studies indicate that insulted customers frequently abandon their payment cards and switch to competitors, inflicting long-term loss of customer lifetime value that often exceeds the cost of fraud itself. Conversely, excessive **False Negatives** cause direct financial liability, chargeback fees, and regulatory penalties.

    Lastly, deploying supervised models as static black boxes overlooks **adversarial concept drift**. Organized fraud syndicates rapidly adapt patterns to bypass fixed thresholds. Without continuous retraining, cost-sensitive loss calibration, precision-recall curve monitoring, and human-in-the-loop dispute workflows, institutions acquire a false sense of automation while remaining highly exposed.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("#### 1. Severe Imbalance Trap\n99% accuracy is useless if you miss 100% of the 1% actual fraud.")
    with col2:
        st.warning("#### 2. Cost Asymmetry\nDeclining an innocent customer (False Positive) destroys trust & causes churn.")
    with col3:
        st.info("#### 3. Adversarial Drift\nFraudsters continuously evolve methods; static models quickly decay.")


# -------------------------------------------------------------
# PAGE: PDF REPORT
# -------------------------------------------------------------
elif page == "📄 View / Download PDF Report":
    st.markdown('<div class="main-title">Assignment PDF Report</div>', unsafe_allow_html=True)
    st.write("A 3-page submission report has been compiled and saved locally.")

    pdf_file_path = os.path.join(BASE_DIR, "ML_Fundamentals_Assignment_Report.pdf")
    with open(pdf_file_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📥 Download ML_Fundamentals_Assignment_Report.pdf",
        data=pdf_bytes,
        file_name="ML_Fundamentals_Assignment_Report.pdf",
        mime="application/pdf"
    )
    st.success("The report contains the complete question answers, plain-English conceptual explanations, Part B, Part C risk analysis (255 words), and all 4 embedded diagrams!")
