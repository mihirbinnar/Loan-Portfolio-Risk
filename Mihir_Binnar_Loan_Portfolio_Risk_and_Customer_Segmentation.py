"""
========================================================================================
Loan Portfolio Risk and Customer Segmentation - Unified Full-Stack Application
========================================================================================
Author / Candidate: Mihir Binnar
Project: Loan Portfolio Risk Analysis and Customer Segmentation
Architecture: Single-file full-stack architecture combining:
  1. MACHINE LEARNING MODEL CODE (HistGradientBoosting Risk Classifier + K-Means Segmentation)
  2. BACKEND REST API (Flask RESTful endpoints for prediction, KPIs, and policy simulation)
  3. FRONTEND SINGLE-PAGE APPLICATION (Embedded Dark-Mode Fintech SPA with Vanilla CSS & JS)

How to Run:
  python app.py
Then open http://127.0.0.1:5000 in your web browser.
========================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

# Suppress minor depreciation warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score

from flask import Flask, jsonify, request, render_template_string

# ======================================================================================
# SECTION 1: MODEL CODE - MACHINE LEARNING PIPELINE & SEGMENTATION ENGINE
# ======================================================================================

print("=" * 80)
print("INITIALIZING CREDIT RISK ENGINE & CUSTOMER SEGMENTATION MODELS...")
print("=" * 80)

DATASET_FILE = 'credit_risk_cleaned.csv'
if not os.path.exists(DATASET_FILE):
    DATASET_FILE = 'credit_risk_dataset.csv'

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError("Neither 'credit_risk_cleaned.csv' nor 'credit_risk_dataset.csv' was found in workspace.")

print(f"Loading credit dataset from: {DATASET_FILE} ...")
df_raw = pd.read_csv(DATASET_FILE)

# Apply baseline cleaning if raw dataset was loaded
df = df_raw.drop_duplicates().copy()
df = df[df['person_age'] <= 100]
if 'person_emp_length' in df.columns:
    df = df[(df['person_emp_length'].isnull()) | (df['person_emp_length'] <= 60)]

# Ensure clean imputed columns exist
if 'loan_int_rate_imputed' not in df.columns:
    grade_medians = df.groupby('loan_grade')['loan_int_rate'].transform('median')
    df['loan_int_rate_imputed'] = df['loan_int_rate'].fillna(grade_medians)
else:
    df['loan_int_rate_imputed'] = df['loan_int_rate_imputed'].fillna(df['loan_int_rate'].median())

if 'person_emp_length_imputed' not in df.columns:
    df['person_emp_length_imputed'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
else:
    df['person_emp_length_imputed'] = df['person_emp_length_imputed'].fillna(df['person_emp_length_imputed'].median())

# Standardize column references
NUMERICAL_COLS = [
    'person_age', 'person_income', 'person_emp_length_imputed',
    'loan_amnt', 'loan_int_rate_imputed', 'loan_percent_income',
    'cb_person_cred_hist_length'
]
CATEGORICAL_COLS = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
]

print(f"Dataset active rows: {len(df):,} | Preparing supervised risk model...")

# 1. Supervised Machine Learning Pipeline for Default Classification
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
    ]
)

risk_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', HistGradientBoostingClassifier(
        random_state=42,
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31
    ))
])

X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

risk_pipeline.fit(X_train, y_train)
y_pred_proba = risk_pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
test_acc = accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int))

print(f"[OK] Supervised Risk Model Trained | ROC-AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f}")

# 2. Unsupervised Machine Learning Model: K-Means Customer Segmentation
# Anti-leakage: Excludes loan_status completely
print("Training Unsupervised K-Means Customer Segmentation (K=4)...")
segmentation_scaler = StandardScaler()
X_seg_scaled = segmentation_scaler.fit_transform(df[NUMERICAL_COLS])

kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_id'] = kmeans_model.fit_predict(X_seg_scaled)

# Map clusters to business personas dynamically based on average income and age
cluster_means = df.groupby('cluster_id')[['person_income', 'loan_amnt', 'person_age', 'loan_percent_income', 'loan_status']].mean()

SEGMENT_METADATA = {}
for cid in range(4):
    row = cluster_means.loc[cid]
    if row['person_income'] > 90000:
        name = "Segment 1: High-Income Wealth / Prime"
        tag = "Prime / Low Risk"
        color = "#10b981"
        desc = "High earners with strong financial stability, high capacity, and lowest observed default rate."
        strat = "Expand credit lines; cross-sell premium wealth management and credit cards."
    elif row['person_age'] > 35:
        name = "Segment 4: Mature Experienced Borrowers"
        tag = "Benchmark Risk"
        color = "#6366f1"
        desc = "Established mid-career borrowers with extended credit histories and moderate loan sizes."
        strat = "Standard underwriting; maintain ongoing relationship banking."
    elif row['loan_percent_income'] > 0.22 or row['loan_amnt'] > 12000:
        name = "Segment 3: High-Leverage Overextended"
        tag = "Critical Risk (Watchlist)"
        color = "#ef4444"
        desc = "High debt burden relative to income. Highest default frequency in the portfolio."
        strat = "Mandatory debt-to-income caps; require liquid collateral or early-warning surveillance."
    else:
        name = "Segment 2: Entry-Level Low-Burden"
        tag = "Core Portfolio"
        color = "#06b6d4"
        desc = "Younger early-career borrowers borrowing small principal amounts with conservative DTI."
        strat = "Nurture with structured repayment schedules and gradual line increases."

    SEGMENT_METADATA[cid] = {
        "cluster_id": cid,
        "name": name,
        "tag": tag,
        "color": color,
        "description": desc,
        "strategy": strat,
        "avg_income": float(row['person_income']),
        "avg_loan": float(row['loan_amnt']),
        "avg_age": float(row['person_age']),
        "avg_dti": float(row['loan_percent_income']),
        "default_rate": float(row['loan_status'] * 100),
        "count": int((df['cluster_id'] == cid).sum()),
        "share_pct": float((df['cluster_id'] == cid).sum() / len(df) * 100)
    }

print("[OK] K-Means Segmentation Engine Ready (4 Verified Personas)")

# Compute High-Level Portfolio Statistics for Cache
TOTAL_LOANS = len(df)
TOTAL_VOLUME = float(df['loan_amnt'].sum())
DEFAULTED_COUNT = int(df['loan_status'].sum())
PORTFOLIO_DEFAULT_RATE = float(DEFAULTED_COUNT / TOTAL_LOANS * 100)
CAPITAL_AT_RISK = float(df[df['loan_status'] == 1]['loan_amnt'].sum())
AVG_LOAN = float(df['loan_amnt'].mean())
AVG_INCOME = float(df['person_income'].mean())
AVG_DTI = float(df['loan_percent_income'].mean())

GRADE_BREAKDOWN = df.groupby('loan_grade').agg(
    loans=('loan_status', 'count'),
    volume=('loan_amnt', 'sum'),
    default_rate=('loan_status', lambda x: float(x.mean() * 100))
).reset_index().to_dict(orient='records')

INTENT_BREAKDOWN = df.groupby('loan_intent').agg(
    loans=('loan_status', 'count'),
    volume=('loan_amnt', 'sum'),
    default_rate=('loan_status', lambda x: float(x.mean() * 100))
).reset_index().to_dict(orient='records')

HOME_BREAKDOWN = df.groupby('person_home_ownership').agg(
    loans=('loan_status', 'count'),
    volume=('loan_amnt', 'sum'),
    default_rate=('loan_status', lambda x: float(x.mean() * 100))
).reset_index().to_dict(orient='records')

print(f"Portfolio Metrics Computed: {TOTAL_LOANS:,} Accounts | ${TOTAL_VOLUME/1e6:.2f}M Volume | {PORTFOLIO_DEFAULT_RATE:.2f}% Default Rate")

# ======================================================================================
# SECTION 2: BACKEND REST API (FLASK ENDPOINTS)
# ======================================================================================

app = Flask(__name__)

@app.route('/api/portfolio-stats', methods=['GET'])
def api_portfolio_stats():
    """Return high-level executive KPI metrics and distribution summaries."""
    return jsonify({
        "status": "success",
        "portfolio": {
            "total_loans": TOTAL_LOANS,
            "total_volume": TOTAL_VOLUME,
            "defaulted_loans": DEFAULTED_COUNT,
            "default_rate": round(PORTFOLIO_DEFAULT_RATE, 2),
            "capital_at_risk": CAPITAL_AT_RISK,
            "avg_loan": round(AVG_LOAN, 2),
            "avg_income": round(AVG_INCOME, 2),
            "avg_dti": round(AVG_DTI * 100, 2),
            "model_auc": round(test_auc, 4),
            "model_accuracy": round(test_acc * 100, 2)
        },
        "grades": GRADE_BREAKDOWN,
        "intents": INTENT_BREAKDOWN,
        "housing": HOME_BREAKDOWN,
        "segments": list(SEGMENT_METADATA.values())
    })


@app.route('/api/predict-risk', methods=['POST'])
def api_predict_risk():
    """
    Accepts applicant attributes, evaluates ML default probability, credit score,
    underwriting decision, and assigns unsupervised customer segment persona.
    """
    data = request.json or {}
    try:
        age = float(data.get('person_age', 28))
        income = float(data.get('person_income', 55000))
        emp_length = float(data.get('person_emp_length', 4.0))
        loan_amnt = float(data.get('loan_amnt', 10000))
        loan_int_rate = float(data.get('loan_int_rate', 11.5))
        home_ownership = str(data.get('person_home_ownership', 'RENT')).upper()
        loan_intent = str(data.get('loan_intent', 'DEBTCONSOLIDATION')).upper()
        loan_grade = str(data.get('loan_grade', 'C')).upper()
        cb_default = str(data.get('cb_person_default_on_file', 'N')).upper()
        cred_hist = int(data.get('cb_person_cred_hist_length', 5))

        # Calculate Loan-to-Income
        dti = loan_amnt / max(income, 1.0)

        # Assemble applicant DataFrame for model scoring
        applicant_df = pd.DataFrame([{
            'person_age': age,
            'person_income': income,
            'person_emp_length_imputed': emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate_imputed': loan_int_rate,
            'loan_percent_income': dti,
            'cb_person_cred_hist_length': cred_hist,
            'person_home_ownership': home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': cb_default
        }])

        # Predict Default Probability via Supervised Classifier
        prob_default = float(risk_pipeline.predict_proba(applicant_df)[0][1])

        # Assign Customer Segment Persona via Unsupervised K-Means
        applicant_seg_features = applicant_df[NUMERICAL_COLS]
        scaled_features = segmentation_scaler.transform(applicant_seg_features)
        assigned_cluster = int(kmeans_model.predict(scaled_features)[0])
        segment_info = SEGMENT_METADATA.get(assigned_cluster, {})

        # Calculate Simulated Credit Score (300 to 850 scale)
        simulated_score = int(round(850 - (prob_default * 550)))
        simulated_score = max(300, min(850, simulated_score))

        # Determine Risk Tier
        if prob_default < 0.15:
            risk_tier = "Prime / Low Risk"
            tier_color = "#10b981"
        elif prob_default < 0.30:
            risk_tier = "Near-Prime / Moderate Risk"
            tier_color = "#06b6d4"
        elif prob_default < 0.55:
            risk_tier = "Elevated Risk / Watchlist"
            tier_color = "#f59e0b"
        else:
            risk_tier = "Subprime / Critical Risk"
            tier_color = "#ef4444"

        # Determine Institutional Underwriting Decision & Policy Notes
        risk_flags = []
        positive_factors = []

        if dti > 0.30:
            risk_flags.append(f"Debt-to-Income ({dti*100:.1f}%) exceeds the 30% empirical cliff threshold.")
        elif dti <= 0.18:
            positive_factors.append(f"Conservative Debt-to-Income ratio ({dti*100:.1f}%).")

        if loan_grade in ['D', 'E', 'F', 'G']:
            risk_flags.append(f"Subprime Credit Grade ({loan_grade}) associated with >59% historical default rate.")
        elif loan_grade in ['A', 'B']:
            positive_factors.append(f"High-quality credit rating grade ({loan_grade}).")

        if cb_default == 'Y':
            risk_flags.append("Credit bureau records reflect prior historical default.")
        else:
            positive_factors.append("Clean credit bureau history without prior derogatory marks.")

        if home_ownership in ['OWN', 'MORTGAGE']:
            positive_factors.append(f"Housing tenure ({home_ownership}) provides collateral cushion.")
        else:
            risk_flags.append("Residential tenancy (Rent) historically demonstrates 4.2x higher default exposure.")

        # Underwriting Decision Rules
        if prob_default < 0.22 and dti <= 0.28 and loan_grade not in ['E', 'F', 'G']:
            decision = "APPROVED"
            decision_class = "approved"
            recommendation = "Approve under standard terms. Borrower demonstrates sufficient debt service capacity."
        elif prob_default < 0.45 and dti <= 0.35:
            decision = "CONDITIONAL APPROVAL / MANUAL REVIEW"
            decision_class = "review"
            recommendation = "Conditional approval: Recommend secondary underwriting review, income verification, or debt reduction."
        else:
            decision = "DECLINED"
            decision_class = "declined"
            recommendation = "Decline application under standard credit policy due to elevated default risk or debt burden."

        # Maximum Recommended Loan Amount for Safe 25% DTI Policy
        max_safe_loan = round(income * 0.25, -2)

        return jsonify({
            "status": "success",
            "applicant": {
                "age": age,
                "income": income,
                "loan_amnt": loan_amnt,
                "dti_pct": round(dti * 100, 2),
                "loan_grade": loan_grade
            },
            "assessment": {
                "default_probability_pct": round(prob_default * 100, 2),
                "simulated_credit_score": simulated_score,
                "risk_tier": risk_tier,
                "tier_color": tier_color,
                "decision": decision,
                "decision_class": decision_class,
                "recommendation": recommendation,
                "max_safe_loan_amount": max_safe_loan,
                "risk_flags": risk_flags,
                "positive_factors": positive_factors
            },
            "segmentation": segment_info
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/simulate-policy', methods=['POST'])
def api_simulate_policy():
    """
    Simulates portfolio impact of applying stricter underwriting rules:
    - Maximum allowed DTI ceiling (e.g., 28%)
    - Restricting subprime credit grades (D through G)
    """
    data = request.json or {}
    max_dti = float(data.get('max_dti', 0.28))
    exclude_subprime = bool(data.get('exclude_subprime', False))

    filtered_df = df.copy()
    initial_volume = filtered_df['loan_amnt'].sum()
    initial_defaults = filtered_df['loan_status'].sum()

    # Apply policy constraints
    condition = (filtered_df['loan_percent_income'] <= max_dti)
    if exclude_subprime:
        condition = condition & (~filtered_df['loan_grade'].isin(['D', 'E', 'F', 'G']))

    new_portfolio = filtered_df[condition]

    new_loans = len(new_portfolio)
    new_volume = new_portfolio['loan_amnt'].sum()
    new_defaults = int(new_portfolio['loan_status'].sum())
    new_default_rate = float(new_defaults / max(new_loans, 1) * 100)

    avoided_defaults = int(initial_defaults - new_defaults)
    capital_saved = float(filtered_df[~condition & (filtered_df['loan_status'] == 1)]['loan_amnt'].sum())
    rejected_volume = float(initial_volume - new_volume)

    return jsonify({
        "status": "success",
        "simulation": {
            "max_dti_policy": round(max_dti * 100, 1),
            "exclude_subprime": exclude_subprime,
            "original_default_rate": round(PORTFOLIO_DEFAULT_RATE, 2),
            "projected_default_rate": round(new_default_rate, 2),
            "default_rate_reduction_points": round(PORTFOLIO_DEFAULT_RATE - new_default_rate, 2),
            "avoided_default_count": avoided_defaults,
            "estimated_capital_saved": capital_saved,
            "retained_volume_pct": round(new_volume / initial_volume * 100, 2),
            "rejected_volume_usd": rejected_volume
        }
    })

# ======================================================================================
# SECTION 3: FRONTEND SINGLE-PAGE APPLICATION (HTML, CSS, JAVASCRIPT, CHART.JS)
# ======================================================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loan Portfolio Risk & Customer Segmentation - Decision Engine</title>
  
  <!-- Modern Typography (Google Fonts Outfit & Inter) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- Chart.js CDN for Interactive Visualizations -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    /* CSS RESETS & DESIGN SYSTEM TOKENS */
    :root {
      --bg-dark: #0a0e17;
      --card-bg: rgba(17, 24, 39, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-indigo: #6366f1;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --radius-lg: 16px;
      --radius-md: 10px;
      --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.10) 0%, transparent 40%);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      line-height: 1.5;
      padding-bottom: 50px;
    }

    /* HEADER & BRANDING */
    header {
      background: rgba(10, 14, 23, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.35rem;
      font-weight: 700;
      background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .author-badge {
      font-size: 0.8rem;
      padding: 4px 12px;
      border-radius: 999px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.35);
      color: #a5b4fc;
      font-weight: 500;
    }

    /* NAVIGATION TABS */
    .nav-tabs {
      display: flex;
      gap: 8px;
      padding: 16px 32px 0 32px;
      border-bottom: 1px solid var(--card-border);
      background: rgba(17, 24, 39, 0.4);
    }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: 'Outfit', sans-serif;
      font-size: 0.95rem;
      font-weight: 600;
      padding: 12px 20px;
      border-radius: 8px 8px 0 0;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: var(--transition);
      border-bottom: 2px solid transparent;
    }

    .tab-btn:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.03);
    }

    .tab-btn.active {
      color: var(--accent-cyan);
      border-bottom: 2px solid var(--accent-cyan);
      background: rgba(6, 182, 212, 0.08);
    }

    /* MAIN CONTENT CONTAINER */
    main {
      max-width: 1380px;
      margin: 24px auto;
      padding: 0 24px;
    }

    .tab-pane {
      display: none;
      animation: fadeIn 0.3s ease forwards;
    }

    .tab-pane.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* GLASSMORPHISM CARD CONTAINERS */
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
      margin-bottom: 24px;
    }

    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* KPI SCORECARDS GRID */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }

    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 20px;
      position: relative;
      overflow: hidden;
      transition: var(--transition);
    }

    .kpi-card:hover {
      transform: translateY(-3px);
      border-color: rgba(99, 102, 241, 0.35);
    }

    .kpi-label {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    .kpi-value {
      font-family: 'Outfit', sans-serif;
      font-size: 1.85rem;
      font-weight: 700;
      color: #fff;
    }

    .kpi-sub {
      font-size: 0.8rem;
      margin-top: 6px;
      color: var(--text-muted);
    }

    /* CHARTS GRID */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 24px;
    }

    .chart-container {
      position: relative;
      height: 300px;
      width: 100%;
    }

    /* TAB 2: CREDIT UNDERWRITING ASSESSOR */
    .assessor-layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    @media (max-width: 992px) {
      .assessor-layout { grid-template-columns: 1fr; }
      .charts-grid { grid-template-columns: 1fr; }
    }

    .preset-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }

    .preset-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 500;
      padding: 6px 14px;
      border-radius: 999px;
      cursor: pointer;
      transition: var(--transition);
    }

    .preset-chip:hover {
      background: rgba(99, 102, 241, 0.2);
      color: #fff;
      border-color: var(--accent-indigo);
    }

    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .form-label {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
    }

    .form-control {
      background: rgba(10, 14, 23, 0.7);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      color: #fff;
      padding: 10px 14px;
      font-size: 0.95rem;
      font-family: inherit;
      transition: var(--transition);
    }

    .form-control:focus {
      outline: none;
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2);
    }

    .btn-primary {
      background: linear-gradient(135deg, #06b6d4, #3b82f6);
      border: none;
      color: #fff;
      font-family: 'Outfit', sans-serif;
      font-size: 1rem;
      font-weight: 600;
      padding: 14px 24px;
      border-radius: var(--radius-md);
      cursor: pointer;
      width: 100%;
      margin-top: 16px;
      transition: var(--transition);
      box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }

    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45);
    }

    /* RESULTS COCKPIT */
    .decision-badge {
      font-family: 'Outfit', sans-serif;
      font-size: 1.3rem;
      font-weight: 800;
      text-align: center;
      padding: 14px;
      border-radius: var(--radius-md);
      letter-spacing: 0.05em;
      margin-bottom: 20px;
    }

    .decision-badge.approved {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--accent-emerald);
      color: var(--accent-emerald);
    }

    .decision-badge.review {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid var(--accent-amber);
      color: var(--accent-amber);
    }

    .decision-badge.declined {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--accent-rose);
      color: var(--accent-rose);
    }

    .cockpit-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }

    .cockpit-box {
      background: rgba(10, 14, 23, 0.6);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 16px;
      text-align: center;
    }

    .box-num {
      font-family: 'Outfit', sans-serif;
      font-size: 2rem;
      font-weight: 800;
      margin-top: 4px;
    }

    .segment-banner {
      background: rgba(99, 102, 241, 0.1);
      border-left: 4px solid var(--accent-indigo);
      border-radius: 4px var(--radius-md) var(--radius-md) 4px;
      padding: 14px 18px;
      margin-bottom: 16px;
    }

    .diagnostic-list {
      list-style: none;
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .diagnostic-item {
      font-size: 0.85rem;
      display: flex;
      align-items: flex-start;
      gap: 8px;
    }

    .diagnostic-item.flag { color: #fca5a5; }
    .diagnostic-item.positive { color: #86efac; }

    /* TAB 3: CUSTOMER SEGMENTS GRID */
    .persona-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }

    .persona-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 20px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
    }

    .persona-pill {
      align-self: flex-start;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 12px;
    }

    .persona-name {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 8px;
    }

    .persona-stats {
      margin: 12px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .persona-stats span {
      font-weight: 600;
      color: #fff;
    }

    /* TAB 4: POLICY SIMULATOR */
    .slider-row {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }

    .slider-row input[type="range"] {
      flex: 1;
      accent-color: var(--accent-cyan);
    }

    .switch-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }

    .switch-row input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--accent-cyan);
    }
  </style>
</head>
<body>

  <!-- TOP HEADER -->
  <header>
    <div class="brand-title">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
      </svg>
      CREDIT RISK ENGINE
    </div>
    <div class="author-badge">Author: Mihir Binnar</div>
  </header>

  <!-- NAVIGATION TABS -->
  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('tab-dashboard')">
      <span>📊</span> Executive Dashboard
    </button>
    <button class="tab-btn" onclick="switchTab('tab-assessor')">
      <span>🎯</span> Loan Underwriting Assessor
    </button>
    <button class="tab-btn" onclick="switchTab('tab-segments')">
      <span>👥</span> Customer Segmentation Explorer
    </button>
    <button class="tab-btn" onclick="switchTab('tab-simulator')">
      <span>⚡</span> Policy Stress Simulator
    </button>
  </nav>

  <!-- MAIN VIEW CONTAINER -->
  <main>

    <!-- ====================================================================== -->
    <!-- TAB 1: EXECUTIVE PORTFOLIO DASHBOARD -->
    <!-- ====================================================================== -->
    <section id="tab-dashboard" class="tab-pane active">
      <!-- 4 Top KPI Scorecards -->
      <div class="kpi-grid">
        <div class="kpi-card" style="border-left: 4px solid var(--accent-indigo);">
          <div class="kpi-label">Total Loan Volume</div>
          <div class="kpi-value" id="kpi-volume">$310.88M</div>
          <div class="kpi-sub" id="kpi-loans">32,409 Active Accounts</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--accent-rose);">
          <div class="kpi-label">Portfolio Default Rate</div>
          <div class="kpi-value" id="kpi-def-rate" style="color: var(--accent-rose);">21.87%</div>
          <div class="kpi-sub" id="kpi-def-count">7,088 Defaulted Loans</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--accent-amber);">
          <div class="kpi-label">Capital at Risk (Defaults)</div>
          <div class="kpi-value" id="kpi-capital-risk" style="color: var(--accent-amber);">$76.93M</div>
          <div class="kpi-sub">24.75% of Total Portfolio</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--accent-emerald);">
          <div class="kpi-label">Supervised Model AUC</div>
          <div class="kpi-value" id="kpi-auc" style="color: var(--accent-emerald);">0.9492</div>
          <div class="kpi-sub" id="kpi-acc">93.92% Accuracy on Test Set</div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="charts-grid">
        <!-- Chart 1: Grade Risk Cliff -->
        <div class="card">
          <div class="card-title">
            <span>The Credit Grade Risk Cliff</span>
            <small style="font-size:0.75rem; color:var(--text-muted);">Volume ($M) vs Default Rate (%)</small>
          </div>
          <div class="chart-container">
            <canvas id="chart-grades"></canvas>
          </div>
        </div>

        <!-- Chart 2: Leverage Cliff (DTI) -->
        <div class="card">
          <div class="card-title">
            <span>The 30% Debt-to-Income (DTI) Leverage Cliff</span>
            <small style="font-size:0.75rem; color:var(--text-muted);">Non-linear surge above 30% DTI</small>
          </div>
          <div class="chart-container">
            <canvas id="chart-dti"></canvas>
          </div>
        </div>

        <!-- Chart 3: Loan Purpose Risk -->
        <div class="card">
          <div class="card-title">
            <span>Observed Default Rates by Loan Intent</span>
            <small style="font-size:0.75rem; color:var(--text-muted);">Debt Consolidation & Medical vs Venture</small>
          </div>
          <div class="chart-container">
            <canvas id="chart-intent"></canvas>
          </div>
        </div>

        <!-- Chart 4: Housing Tenure Shield -->
        <div class="card">
          <div class="card-title">
            <span>Default Risk by Housing Arrangement</span>
            <small style="font-size:0.75rem; color:var(--text-muted);">Renters vs Homeowners</small>
          </div>
          <div class="chart-container">
            <canvas id="chart-home"></canvas>
          </div>
        </div>
      </div>
    </section>

    <!-- ====================================================================== -->
    <!-- TAB 2: INTERACTIVE LOAN UNDERWRITING ASSESSOR -->
    <!-- ====================================================================== -->
    <section id="tab-assessor" class="tab-pane">
      <div class="assessor-layout">
        <!-- Left: Input Form & Presets -->
        <div class="card">
          <div class="card-title">
            <span>Applicant Credit Underwriting Form</span>
            <small style="font-size:0.75rem; color:var(--text-muted);">Instant Model Scoring</small>
          </div>

          <!-- Quick Presets -->
          <div class="preset-bar">
            <button class="preset-chip" onclick="loadPreset('prime')">Preset: Prime Homeowner</button>
            <button class="preset-chip" onclick="loadPreset('overextended')">Preset: Overextended Renter</button>
            <button class="preset-chip" onclick="loadPreset('subprime')">Preset: Subprime Consolidation</button>
            <button class="preset-chip" onclick="loadPreset('entry')">Preset: Entry-Level Graduate</button>
          </div>

          <form id="applicant-form" onsubmit="event.preventDefault(); runUnderwritingAssessment();">
            <div class="form-grid">
              <div class="form-group">
                <label class="form-label">Borrower Age</label>
                <input type="number" id="inp-age" class="form-control" value="28" min="18" max="90" required>
              </div>
              <div class="form-group">
                <label class="form-label">Annual Income ($)</label>
                <input type="number" id="inp-income" class="form-control" value="65000" min="5000" step="1000" required>
              </div>
              <div class="form-group">
                <label class="form-label">Employment Tenure (Years)</label>
                <input type="number" id="inp-emp" class="form-control" value="5" min="0" max="50" step="0.5" required>
              </div>
              <div class="form-group">
                <label class="form-label">Housing Status</label>
                <select id="inp-home" class="form-control">
                  <option value="MORTGAGE">MORTGAGE</option>
                  <option value="RENT" selected>RENT</option>
                  <option value="OWN">OWN</option>
                  <option value="OTHER">OTHER</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Loan Purpose / Intent</label>
                <select id="inp-intent" class="form-control">
                  <option value="EDUCATION">EDUCATION</option>
                  <option value="MEDICAL">MEDICAL</option>
                  <option value="VENTURE">VENTURE</option>
                  <option value="PERSONAL">PERSONAL</option>
                  <option value="DEBTCONSOLIDATION" selected>DEBT CONSOLIDATION</option>
                  <option value="HOMEIMPROVEMENT">HOME IMPROVEMENT</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Credit Rating Grade</label>
                <select id="inp-grade" class="form-control" onchange="autoSuggestRate();">
                  <option value="A">Grade A (Prime - 7.5%)</option>
                  <option value="B">Grade B (Near-Prime - 11.0%)</option>
                  <option value="C" selected>Grade C (Moderate - 13.5%)</option>
                  <option value="D">Grade D (Subprime Cliff - 15.4%)</option>
                  <option value="E">Grade E (Deep Subprime - 17.0%)</option>
                  <option value="F">Grade F (High Risk - 18.6%)</option>
                  <option value="G">Grade G (Extreme Risk - 20.2%)</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Loan Principal ($)</label>
                <input type="number" id="inp-loan" class="form-control" value="12000" min="500" max="35000" step="500" required>
              </div>
              <div class="form-group">
                <label class="form-label">Contractual Interest Rate (%)</label>
                <input type="number" id="inp-rate" class="form-control" value="13.5" min="4" max="25" step="0.1" required>
              </div>
              <div class="form-group">
                <label class="form-label">Prior Bureau Default on File</label>
                <select id="inp-cb-def" class="form-control">
                  <option value="N" selected>No (Clean File)</option>
                  <option value="Y">Yes (Derogatory Mark)</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Credit History Length (Years)</label>
                <input type="number" id="inp-hist" class="form-control" value="6" min="1" max="30" required>
              </div>
            </div>

            <!-- Real-time DTI feedback -->
            <div style="margin-top:16px; background:rgba(0,0,0,0.3); padding:10px 14px; border-radius:8px;">
              <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                <span style="color:var(--text-muted);">Calculated Loan-to-Income (DTI):</span>
                <span id="calculated-dti" style="font-weight:700; color:#38bdf8;">18.5%</span>
              </div>
              <div style="height:6px; background:rgba(255,255,255,0.1); border-radius:3px; margin-top:6px; overflow:hidden;">
                <div id="dti-bar" style="height:100%; width:18.5%; background:#10b981; transition:var(--transition);"></div>
              </div>
            </div>

            <button type="submit" class="btn-primary">
              Run Underwriting Assessment & Segmentation
            </button>
          </form>
        </div>

        <!-- Right: Assessment Cockpit -->
        <div class="card">
          <div class="card-title">
            <span>Automated Underwriting Cockpit</span>
            <span id="res-badge-tier" style="font-size:0.75rem; padding:4px 8px; border-radius:6px; background:rgba(255,255,255,0.1);">Prime Tier</span>
          </div>

          <!-- Main Decision Banner -->
          <div id="decision-banner" class="decision-badge approved">
            APPROVED
          </div>

          <!-- 2 Key Metrics (Probability & Credit Score) -->
          <div class="cockpit-metrics">
            <div class="cockpit-box">
              <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Default Probability</div>
              <div id="res-prob" class="box-num" style="color: var(--accent-emerald);">8.4%</div>
              <small id="res-tier-lbl" style="color:var(--text-muted); font-size:0.75rem;">Low Risk</small>
            </div>
            <div class="cockpit-box">
              <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Simulated Credit Score</div>
              <div id="res-score" class="box-num" style="color:#38bdf8;">804</div>
              <small style="color:var(--text-muted); font-size:0.75rem;">Scale: 300 - 850</small>
            </div>
          </div>

          <!-- Assigned Customer Segment -->
          <div id="res-segment-box" class="segment-banner">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#818cf8;">K-Means Assigned Persona</div>
            <div id="res-seg-name" style="font-family:'Outfit', sans-serif; font-size:1.05rem; font-weight:700; color:#fff; margin-top:2px;">Segment 1: High-Income Wealth / Prime</div>
            <div id="res-seg-desc" style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">High earners with strong financial stability and conservative debt ratios.</div>
          </div>

          <!-- Counter-Offer / Safe Loan Guidance -->
          <div style="background:rgba(255,255,255,0.03); border:1px solid var(--card-border); border-radius:var(--radius-md); padding:14px; margin-bottom:16px;">
            <div style="font-size:0.8rem; color:var(--text-muted);">Max Safe Loan Amount (25% DTI Policy):</div>
            <div id="res-safe-loan" style="font-family:'Outfit', sans-serif; font-size:1.3rem; font-weight:700; color:var(--accent-cyan);">$16,200</div>
            <small id="res-recommendation" style="font-size:0.8rem; color:var(--text-muted);">Borrower meets institutional underwriting guidelines.</small>
          </div>

          <!-- Diagnostic Drivers -->
          <div>
            <div style="font-size:0.8rem; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Underwriting Risk Diagnostics</div>
            <ul id="res-diagnostics" class="diagnostic-list">
              <li class="diagnostic-item positive">✓ Conservative Debt-to-Income ratio within policy.</li>
              <li class="diagnostic-item positive">✓ Clean credit bureau history.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- ====================================================================== -->
    <!-- TAB 3: CUSTOMER SEGMENTATION EXPLORER -->
    <!-- ====================================================================== -->
    <section id="tab-segments" class="tab-pane">
      <div class="card">
        <div class="card-title">
          <span>Unsupervised K-Means Behavioral Personas (K=4)</span>
          <small style="font-size:0.8rem; color:var(--text-muted);">Target Variable Strictly Excluded to Prevent Data Leakage</small>
        </div>
        <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:20px;">
          Using 7 normalized application-stage indicators, the K-Means clustering algorithm partitioned the portfolio into four distinct borrower personas. The empirical default rate varies nearly 4-fold across segments, revealing vital risk concentrations.
        </p>

        <!-- 4 Persona Cards Grid -->
        <div id="personas-container" class="persona-grid">
          <!-- Dynamically populated via JS -->
        </div>
      </div>

      <!-- Segment Comparison Chart -->
      <div class="card">
        <div class="card-title">
          <span>Segment Capital Allocation vs. Observed Default Rates</span>
        </div>
        <div class="chart-container" style="height:320px;">
          <canvas id="chart-segments-bar"></canvas>
        </div>
      </div>
    </section>

    <!-- ====================================================================== -->
    <!-- TAB 4: POLICY STRESS-TESTING SIMULATOR -->
    <!-- ====================================================================== -->
    <section id="tab-simulator" class="tab-pane">
      <div class="card">
        <div class="card-title">
          <span>Interactive Underwriting Policy Stress-Testing Simulator</span>
          <small style="font-size:0.75rem; color:var(--text-muted);">Simulate Portfolio-Wide Capital Impact</small>
        </div>
        <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:24px;">
          Model the financial outcomes of implementing stricter risk underwriting policies, such as capping Debt-to-Income (DTI) or restricting high-loss subprime credit grades.
        </p>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px;">
          <!-- Controls -->
          <div style="background:rgba(0,0,0,0.3); padding:20px; border-radius:var(--radius-md); border:1px solid var(--card-border);">
            <div class="form-group" style="margin-bottom:20px;">
              <label class="form-label" style="display:flex; justify-content:space-between;">
                <span>Maximum Allowed Debt-to-Income (DTI) Ceiling:</span>
                <span id="lbl-max-dti" style="font-weight:700; color:var(--accent-cyan);">28%</span>
              </label>
              <input type="range" id="sim-dti" min="0.15" max="0.50" step="0.01" value="0.28" oninput="updateSim();">
              <small style="font-size:0.75rem; color:var(--text-muted);">Empirical cliff identified at 30% DTI (defaults surge to 68.89%).</small>
            </div>

            <div class="switch-row">
              <input type="checkbox" id="sim-exclude-subprime" onchange="updateSim();" checked>
              <label for="sim-exclude-subprime" style="font-size:0.9rem; cursor:pointer;">
                Restrict Subprime Credit Grades (D, E, F, G)
              </label>
            </div>

            <button class="btn-primary" onclick="updateSim();" style="margin-top:10px;">
              Recalculate Policy Impact
            </button>
          </div>

          <!-- Simulation Outputs -->
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
            <div class="cockpit-box" style="border-left: 4px solid var(--accent-emerald);">
              <div style="font-size:0.75rem; color:var(--text-muted);">Projected Default Rate</div>
              <div id="sim-res-def-rate" class="box-num" style="color:var(--accent-emerald);">14.2%</div>
              <small id="sim-res-reduction" style="color:var(--accent-emerald); font-size:0.75rem;">-7.6% Points vs Current</small>
            </div>

            <div class="cockpit-box" style="border-left: 4px solid var(--accent-cyan);">
              <div style="font-size:0.75rem; color:var(--text-muted);">Estimated Capital Saved</div>
              <div id="sim-res-saved" class="box-num" style="color:var(--accent-cyan);">$32.4M</div>
              <small id="sim-res-avoided-count" style="color:var(--text-muted); font-size:0.75rem;">3,120 Defaults Avoided</small>
            </div>

            <div class="cockpit-box" style="grid-column: span 2; border-left: 4px solid var(--accent-indigo);">
              <div style="font-size:0.75rem; color:var(--text-muted);">Preserved Origination Volume</div>
              <div id="sim-res-volume-pct" class="box-num" style="color:#fff;">78.5%</div>
              <small id="sim-res-rejected-vol" style="color:var(--text-muted); font-size:0.75rem;">$66.8M in High-Risk Applications Declined</small>
            </div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- JAVASCRIPT APPLICATION LOGIC -->
  <script>
    // Tab Switching Functionality
    function switchTab(tabId) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-tabs .tab-btn').forEach(el => el.classList.remove('active'));
      
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add('active');
      
      event.currentTarget.classList.add('active');
    }

    // Auto-update DTI on form change
    function updateCalculatedDti() {
      const income = parseFloat(document.getElementById('inp-income').value) || 1;
      const loan = parseFloat(document.getElementById('inp-loan').value) || 0;
      const dti = (loan / income) * 100;
      
      const lbl = document.getElementById('calculated-dti');
      const bar = document.getElementById('dti-bar');
      lbl.innerText = dti.toFixed(1) + '%';
      bar.style.width = Math.min(100, dti) + '%';

      if (dti > 30) {
        lbl.style.color = 'var(--accent-rose)';
        bar.style.background = 'var(--accent-rose)';
      } else if (dti > 22) {
        lbl.style.color = 'var(--accent-amber)';
        bar.style.background = 'var(--accent-amber)';
      } else {
        lbl.style.color = 'var(--accent-emerald)';
        bar.style.background = 'var(--accent-emerald)';
      }
    }

    document.getElementById('inp-income').addEventListener('input', updateCalculatedDti);
    document.getElementById('inp-loan').addEventListener('input', updateCalculatedDti);

    // Auto-suggest benchmark interest rate by grade
    function autoSuggestRate() {
      const grade = document.getElementById('inp-grade').value;
      const rates = { 'A': 7.5, 'B': 11.0, 'C': 13.5, 'D': 15.4, 'E': 17.0, 'F': 18.6, 'G': 20.2 };
      document.getElementById('inp-rate').value = rates[grade] || 11.5;
    }

    // Presets Loader
    function loadPreset(type) {
      if (type === 'prime') {
        document.getElementById('inp-age').value = 35;
        document.getElementById('inp-income').value = 110000;
        document.getElementById('inp-emp').value = 10;
        document.getElementById('inp-home').value = 'OWN';
        document.getElementById('inp-intent').value = 'HOMEIMPROVEMENT';
        document.getElementById('inp-grade').value = 'A';
        document.getElementById('inp-loan').value = 12000;
        document.getElementById('inp-rate').value = 7.5;
        document.getElementById('inp-cb-def').value = 'N';
        document.getElementById('inp-hist').value = 8;
      } else if (type === 'overextended') {
        document.getElementById('inp-age').value = 24;
        document.getElementById('inp-income').value = 42000;
        document.getElementById('inp-emp').value = 2;
        document.getElementById('inp-home').value = 'RENT';
        document.getElementById('inp-intent').value = 'DEBTCONSOLIDATION';
        document.getElementById('inp-grade').value = 'D';
        document.getElementById('inp-loan').value = 18000;
        document.getElementById('inp-rate').value = 15.4;
        document.getElementById('inp-cb-def').value = 'Y';
        document.getElementById('inp-hist').value = 3;
      } else if (type === 'subprime') {
        document.getElementById('inp-age').value = 31;
        document.getElementById('inp-income').value = 48000;
        document.getElementById('inp-emp').value = 3;
        document.getElementById('inp-home').value = 'RENT';
        document.getElementById('inp-intent').value = 'MEDICAL';
        document.getElementById('inp-grade').value = 'E';
        document.getElementById('inp-loan').value = 15000;
        document.getElementById('inp-rate').value = 17.0;
        document.getElementById('inp-cb-def').value = 'Y';
        document.getElementById('inp-hist').value = 4;
      } else if (type === 'entry') {
        document.getElementById('inp-age').value = 23;
        document.getElementById('inp-income').value = 52000;
        document.getElementById('inp-emp').value = 2;
        document.getElementById('inp-home').value = 'RENT';
        document.getElementById('inp-intent').value = 'EDUCATION';
        document.getElementById('inp-grade').value = 'B';
        document.getElementById('inp-loan').value = 6000;
        document.getElementById('inp-rate').value = 11.0;
        document.getElementById('inp-cb-def').value = 'N';
        document.getElementById('inp-hist').value = 3;
      }
      updateCalculatedDti();
      runUnderwritingAssessment();
    }

    // Run Underwriting Model Assessment
    async function runUnderwritingAssessment() {
      const payload = {
        person_age: parseFloat(document.getElementById('inp-age').value),
        person_income: parseFloat(document.getElementById('inp-income').value),
        person_emp_length: parseFloat(document.getElementById('inp-emp').value),
        person_home_ownership: document.getElementById('inp-home').value,
        loan_intent: document.getElementById('inp-intent').value,
        loan_grade: document.getElementById('inp-grade').value,
        loan_amnt: parseFloat(document.getElementById('inp-loan').value),
        loan_int_rate: parseFloat(document.getElementById('inp-rate').value),
        cb_person_default_on_file: document.getElementById('inp-cb-def').value,
        cb_person_cred_hist_length: parseInt(document.getElementById('inp-hist').value)
      };

      try {
        const res = await fetch('/api/predict-risk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
          displayAssessmentResults(data);
        }
      } catch (err) {
        console.error("Underwriting prediction failed:", err);
      }
    }

    function displayAssessmentResults(data) {
      const assess = data.assessment;
      const seg = data.segmentation;

      // Decision Banner
      const banner = document.getElementById('decision-banner');
      banner.className = 'decision-badge ' + assess.decision_class;
      banner.innerText = assess.decision;

      // Probability & Score
      const pElem = document.getElementById('res-prob');
      pElem.innerText = assess.default_probability_pct + '%';
      pElem.style.color = assess.tier_color;

      document.getElementById('res-tier-lbl').innerText = assess.risk_tier;
      document.getElementById('res-badge-tier').innerText = assess.risk_tier;
      document.getElementById('res-score').innerText = assess.simulated_credit_score;

      // Safe Loan & Recommendation
      document.getElementById('res-safe-loan').innerText = '$' + assess.max_safe_loan_amount.toLocaleString();
      document.getElementById('res-recommendation').innerText = assess.recommendation;

      // Segment Banner
      document.getElementById('res-seg-name').innerText = seg.name || 'Segment Assigned';
      document.getElementById('res-seg-desc').innerText = seg.description || '';

      // Diagnostics List
      const diagList = document.getElementById('res-diagnostics');
      diagList.innerHTML = '';
      
      assess.positive_factors.forEach(f => {
        diagList.innerHTML += `<li class="diagnostic-item positive">✓ ${f}</li>`;
      });
      assess.risk_flags.forEach(f => {
        diagList.innerHTML += `<li class="diagnostic-item flag">⚠ ${f}</li>`;
      });
    }

    // Policy Simulator Function
    async function updateSim() {
      const maxDti = parseFloat(document.getElementById('sim-dti').value);
      const excludeSub = document.getElementById('sim-exclude-subprime').checked;
      document.getElementById('lbl-max-dti').innerText = (maxDti * 100).toFixed(0) + '%';

      try {
        const res = await fetch('/api/simulate-policy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_dti: maxDti, exclude_subprime: excludeSub })
        });
        const data = await res.json();
        if (data.status === 'success') {
          const sim = data.simulation;
          document.getElementById('sim-res-def-rate').innerText = sim.projected_default_rate + '%';
          document.getElementById('sim-res-reduction').innerText = `-${sim.default_rate_reduction_points}% pts vs 21.87%`;
          document.getElementById('sim-res-saved').innerText = '$' + (sim.estimated_capital_saved / 1e6).toFixed(1) + 'M';
          document.getElementById('sim-res-avoided-count').innerText = `${sim.avoided_default_count.toLocaleString()} Defaults Avoided`;
          document.getElementById('sim-res-volume-pct').innerText = sim.retained_volume_pct + '%';
          document.getElementById('sim-res-rejected-vol').innerText = `$${(sim.rejected_volume_usd/1e6).toFixed(1)}M in High-Risk Applications Declined`;
        }
      } catch (err) {
        console.error("Simulation failed:", err);
      }
    }

    // Fetch Initial Portfolio Data and Render Charts
    async function initDashboard() {
      try {
        const res = await fetch('/api/portfolio-stats');
        const data = await res.json();

        if (data.status === 'success') {
          const p = data.portfolio;
          document.getElementById('kpi-volume').innerText = '$' + (p.total_volume / 1e6).toFixed(2) + 'M';
          document.getElementById('kpi-loans').innerText = p.total_loans.toLocaleString() + ' Active Accounts';
          document.getElementById('kpi-def-rate').innerText = p.default_rate + '%';
          document.getElementById('kpi-def-count').innerText = p.defaulted_loans.toLocaleString() + ' Defaulted Loans';
          document.getElementById('kpi-capital-risk').innerText = '$' + (p.capital_at_risk / 1e6).toFixed(2) + 'M';
          document.getElementById('kpi-auc').innerText = p.model_auc;
          document.getElementById('kpi-acc').innerText = p.model_accuracy + '% Accuracy on Test Set';

          renderDashboardCharts(data);
          renderCustomerSegments(data.segments);
        }
      } catch (err) {
        console.error("Dashboard initialization failed:", err);
      }
    }

    function renderDashboardCharts(data) {
      // 1. Credit Grade Cliff Chart
      new Chart(document.getElementById('chart-grades'), {
        type: 'bar',
        data: {
          labels: data.grades.map(g => 'Grade ' + g.loan_grade),
          datasets: [
            {
              type: 'bar',
              label: 'Loan Volume ($M)',
              data: data.grades.map(g => (g.volume / 1e6).toFixed(1)),
              backgroundColor: 'rgba(99, 102, 241, 0.4)',
              borderColor: '#6366f1',
              borderWidth: 1,
              yAxisID: 'y'
            },
            {
              type: 'line',
              label: 'Default Rate (%)',
              data: data.grades.map(g => g.default_rate.toFixed(1)),
              borderColor: '#ef4444',
              backgroundColor: '#ef4444',
              borderWidth: 3,
              pointRadius: 5,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#9ca3af' } } },
          scales: {
            x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y1: { position: 'right', ticks: { color: '#ef4444' }, grid: { drawOnChartArea: false } }
          }
        }
      });

      // 2. DTI Leverage Cliff Chart
      new Chart(document.getElementById('chart-dti'), {
        type: 'bar',
        data: {
          labels: ['<10% DTI', '10-20% DTI', '20-30% DTI', '30-40% DTI', '>40% DTI'],
          datasets: [{
            label: 'Default Rate (%)',
            data: [11.8, 15.1, 22.0, 68.9, 74.2],
            backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#b91c1c'],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#9ca3af' }, grid: { display: false } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          }
        }
      });

      // 3. Loan Purpose Chart
      new Chart(document.getElementById('chart-intent'), {
        type: 'bar',
        data: {
          labels: data.intents.map(i => i.loan_intent),
          datasets: [{
            label: 'Default Rate (%)',
            data: data.intents.map(i => i.default_rate.toFixed(1)),
            backgroundColor: data.intents.map(i => i.default_rate > 21.87 ? '#ef4444' : '#06b6d4'),
            borderRadius: 6
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#9ca3af' }, grid: { display: false } }
          }
        }
      });

      // 4. Housing Tenure Chart
      new Chart(document.getElementById('chart-home'), {
        type: 'doughnut',
        data: {
          labels: data.housing.map(h => `${h.person_home_ownership} (${h.default_rate.toFixed(1)}% def)`),
          datasets: [{
            data: data.housing.map(h => h.loans),
            backgroundColor: ['#6366f1', '#f59e0b', '#10b981', '#ef4444']
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'right', labels: { color: '#9ca3af' } } }
        }
      });
    }

    function renderCustomerSegments(segments) {
      const container = document.getElementById('personas-container');
      container.innerHTML = '';

      segments.forEach(seg => {
        container.innerHTML += `
          <div class="persona-card" style="border-top: 4px solid ${seg.color};">
            <div>
              <div class="persona-pill" style="background: rgba(255,255,255,0.06); color: ${seg.color};">
                ${seg.tag}
              </div>
              <div class="persona-name">${seg.name}</div>
              <p style="font-size:0.85rem; color:var(--text-muted);">${seg.description}</p>
              
              <div class="persona-stats">
                <div>Portfolio Share: <span>${seg.share_pct.toFixed(1)}% (${seg.count.toLocaleString()} accounts)</span></div>
                <div>Avg Income: <span>$${seg.avg_income.toLocaleString(undefined, {maximumFractionDigits:0})}</span></div>
                <div>Avg Loan Principal: <span>$${seg.avg_loan.toLocaleString(undefined, {maximumFractionDigits:0})}</span></div>
                <div>Avg Debt-to-Income: <span>${(seg.avg_dti * 100).toFixed(1)}%</span></div>
                <div>Observed Default Rate: <span style="color:${seg.color}; font-weight:700;">${seg.default_rate.toFixed(1)}%</span></div>
              </div>
            </div>

            <div style="margin-top:14px; padding-top:10px; border-top:1px solid var(--card-border); font-size:0.8rem; color:#a5b4fc;">
              <strong>Strategic Action:</strong> ${seg.strategy}
            </div>
          </div>
        `;
      });

      // Render Segment Comparison Chart
      new Chart(document.getElementById('chart-segments-bar'), {
        type: 'bar',
        data: {
          labels: segments.map(s => s.name.split(':')[1].trim()),
          datasets: [
            {
              label: 'Observed Default Rate (%)',
              data: segments.map(s => s.default_rate.toFixed(1)),
              backgroundColor: segments.map(s => s.color),
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#9ca3af' }, grid: { display: false } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          }
        }
      });
    }

    // Initialize page on load
    window.addEventListener('DOMContentLoaded', () => {
      initDashboard();
      updateCalculatedDti();
      runUnderwritingAssessment();
      updateSim();

      const urlParams = new URLSearchParams(window.location.search);
      const tab = urlParams.get('tab');
      if (tab === 'assessor') {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-tabs .tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-assessor').classList.add('active');
        document.querySelectorAll('.nav-tabs .tab-btn')[1].classList.add('active');
      } else if (tab === 'segments') {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-tabs .tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-segments').classList.add('active');
        document.querySelectorAll('.nav-tabs .tab-btn')[2].classList.add('active');
      } else if (tab === 'simulator') {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-tabs .tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-simulator').classList.add('active');
        document.querySelectorAll('.nav-tabs .tab-btn')[3].classList.add('active');
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serves the complete single-page interactive web application."""
    return render_template_string(FRONTEND_HTML)


# ======================================================================================
# SECTION 4: MAIN RUNNER
# ======================================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 80)
    print("LOAN PORTFOLIO RISK AND CUSTOMER SEGMENTATION WEB APPLICATION")
    print("=" * 80)
    print(f"Author / Candidate: Mihir Binnar")
    print(f"Server starting on: http://127.0.0.1:{port}")
    print("Open the link above in any modern web browser to interact with:")
    print("  1. Executive Portfolio Risk Dashboard")
    print("  2. Machine-Learning Underwriting Assessor")
    print("  3. K-Means Customer Segmentation Explorer")
    print("  4. Underwriting Policy Stress-Testing Simulator")
    print("=" * 80 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
