# Loan Portfolio Risk and Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project%20Status-Submission%20Ready-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Credit%20Risk-orange.svg)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

> **End-to-End Data Analytics Internship Capstone Project**  
> Empirical Portfolio Performance Analysis, Credit Grade Risk Cliffs, Debt Leverage Thresholds, and Behavioral Borrower Segmentation using Machine Learning.

---

## 1. Project Overview & Problem Statement
In retail credit operations, financial institutions must maintain a disciplined balance between asset growth and credit default exposure. Inadequate risk grading, unmonitored subprime exposures, and aggressive borrower leverage frequently culminate in non-performing loan spikes during macroeconomic shocks.

This project investigates a commercial retail credit portfolio of **32,581 loan records** representing **$310.88M in disbursed capital**. With a baseline portfolio default rate of **21.87%** ($76.93M in non-performing assets), the institution requires a systematic portfolio diagnostic to:
1. Identify empirical drivers of loan default across borrower demographics and loan structures.
2. Uncover non-linear risk thresholds that trigger disproportionate credit defaults.
3. Segment the borrower population using both policy rules and unsupervised machine learning (without data leakage).
4. Deliver actionable underwriting policies and Business Intelligence (BI) datasets for executive monitoring.

---

## 2. Core Objectives
- **Data Quality & Hygiene:** Audit schema integrity, handle anomalies without arbitrary deletions, and apply domain-aligned statistical imputation.
- **Exploratory Data Analysis (EDA):** Uncover distributions of borrower age, income, employment tenure, loan size, and contractual interest rates.
- **Credit Risk Measurement:** Quantify portfolio performance by credit grade, loan purpose, housing tenure, income quartile, and debt-to-income (DTI) ratio.
- **Customer Segmentation:** Formulate four distinct borrower personas using K-Means clustering (strictly excluding the target variable `loan_status` to prevent data leakage) and policy tiering.
- **Strategic Policy & BI Deployment:** Formulate risk mitigation guidelines for credit committees and structure exportable data models for Excel and Power BI dashboards.

---

## 3. Dataset Information & Placement Instructions
- **Dataset Title:** Credit Risk Dataset
- **Provider / Host:** Kaggle (`laotse/credit-risk-dataset`)
- **Direct Kaggle URL:** [https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Raw Dimensions:** 32,581 rows × 12 columns
- **Target Variable:** `loan_status` (0 = Non-Default / Fully Paid, 1 = Defaulted)

### Download & Placement Instructions:
1. Download the dataset from the Kaggle URL above.
2. Extract the file as `credit_risk_dataset.csv`.
3. Place `credit_risk_dataset.csv` directly in the project root directory alongside the Jupyter Notebook:
   ```
   d:/Loan_credit_risk/credit_risk_dataset.csv
   ```
   *(Note: The notebook uses a configurable path `DATASET_PATH = 'credit_risk_dataset.csv'` and validates file presence upon execution).*

---

## 4. Key Empirical Findings (From Actual Data Analysis)

| Portfolio Metric / Dimension | Analyzed Value | Business / Operational Interpretation |
| :--- | :---: | :--- |
| **Total Analyzed Accounts** | **32,409** | Preserved **99.47%** of data after deduplication (165 rows) and filtering 7 physical anomalies. |
| **Total Disbursed Volume** | **$310.88M** | Total principal committed across all loan facilities. |
| **Average Loan Principal** | **$9,592.49** | Median loan size is $8,000.00; maximum portfolio loan ceiling is $35,000.00. |
| **Portfolio Default Rate** | **21.87%** | 7,088 defaulted accounts representing **$76.93M in capital at risk** (24.75% of volume). |
| **The Credit Grade Risk Cliff** | **C (20.8%) vs D (59.0%)** | Default rate triples from Grade C to D, reaching **98.44% in Grade G**. Grades D–G hold **$56.35M in exposure**. |
| **The 30% Leverage Cliff** | **DTI > 30% → 68.89%** | For DTI < 20%, default rate is 11–15%. Above 30% DTI, default surges to **68.89%** (and 74.15% above 40%). |
| **Housing Tenure Risk** | **Rent (31.6%) vs Own (7.5%)** | Residential tenants exhibit a **4.2x higher default rate** compared to outright homeowners. |
| **Loan Purpose Divergence** | **Debt Consolidation (28.7%)** | Consolidation and Medical (26.8%) carry the highest risk; Venture (14.9%) and Education (17.3%) are safest. |
| **Credit Bureau History** | **Prior Default: 37.86%** | Borrowers with prior defaults on bureau files default at more than double the rate of clean accounts (18.44%). |

---

## 5. Unsupervised Customer Segmentation (K-Means, K=4)

*Note: The default target variable was strictly excluded during clustering to avoid data leakage.*

| Segment Identifier | Portfolio Share | Disbursed Volume | Avg Income | Avg Loan | Avg DTI | Observed Default Rate | Risk Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Segment 1: High-Income Wealth / Prime** | 12.44% | $46.85M | $118,500 | $11,624 | 12.3% | **10.57%** | **Prime / Low Risk** |
| **Segment 2: Entry-Level Low-Burden** | 52.02% | $103.06M | $57,365 | $6,113 | 12.0% | **15.12%** | **Core / Low Risk** |
| **Segment 3: High-Leverage Overextended** | 25.98% | $132.06M | $58,746 | $15,684 | 29.5% | **40.84%** | **CRITICAL RISK (Watchlist)** |
| **Segment 4: Mature Experienced Borrowers** | 9.57% | $28.91M | $63,290 | $9,324 | 16.7% | **21.74%** | **Benchmark Risk** |

---

## 6. Strategic Business Recommendations
1. **Institutional 28% DTI Ceiling:** Implement an automated underwriting rule capping unsecured loan-to-income at 28%. Applications exceeding 30% DTI must be declined or require liquid collateral.
2. **Subprime Restructuring (Grades D through G):** Terminate unsecured lending for Grades D–G. Require co-signers, reduce credit lines to <$5,000, and shift to Risk-Adjusted Return on Capital (RAROC) pricing.
3. **Surveillance of Segment 3 Accounts:** Segment 3 represents $132.06M in loan commitments and 40.84% default risk. Implement early-warning contact upon 15-day late payment signals.
4. **Targeted Growth in Resilient Categories:** Increase marketing and origination incentives for Venture (14.86% default rate) and Education loans (17.26%) to safely expand interest margin.

---

## 7. Technologies & Dependencies Used
- **Core Analytics & Manipulation:** `pandas>=2.0.0`, `numpy>=1.24.0`
- **Machine Learning & Preprocessing:** `scikit-learn>=1.3.0`, `scipy>=1.10.0`
- **Statistical Visualization:** `matplotlib>=3.7.0`, `seaborn>=0.12.0`
- **Reporting & Business Intelligence:** `openpyxl>=3.1.0`, `python-docx>=1.1.0`
- **Interactive Computation:** `jupyter>=1.0.0`, `nbconvert>=7.10.0`

---

## 8. Repository Directory Structure
```
d:/Loan_credit_risk/
│
├── credit_risk_dataset.csv                  # Raw dataset (Primary input)
├── credit_risk_cleaned.csv                  # Cleaned, imputed, and audited dataset
├── requirements.txt                         # External Python dependencies
├── README.md                                # Project documentation (This file)
│
├── YourName_ProjectName.ipynb               # Fully executed Jupyter Notebook
├── YourName_Loan_Portfolio_Risk_... .ipynb  # Canonical named copy of executed notebook
│
├── YourName_ProjectReport.docx              # Complete executive Word report (with charts)
├── YourName_Loan_Portfolio_Risk_... .docx   # Canonical named copy of Word report
│
├── build_visualizations_and_kpis.py         # Automation script for charts & KPI exports
├── build_word_report.py                     # Script to construct the formatted .docx
├── generate_notebook.py                     # Script to build and format the .ipynb
│
├── figures/                                 # High-resolution (300 DPI) visualization figures
│   ├── fig1_distributions.png               # Age, income, loan amount, and rate distributions
│   ├── fig2_categoricals.png                # Intent, housing, grade, and prior default breakdown
│   ├── fig3_grade_risk_cliff.png            # Credit grade loan exposure & default risk cliff
│   ├── fig4_intent_and_home_risk.png        # Default rates across loan purposes and housing
│   ├── fig5_leverage_and_income_risk.png    # DTI ratio risk cliff and income quartiles
│   ├── fig6_correlation_matrix.png          # Correlation heatmap of financial attributes
│   ├── fig7_clustering_optimization.png     # Elbow method and silhouette analysis
│   ├── fig8_segment_risk_comparison.png     # Customer segment comparison and default disparity
│   └── fig9_executive_dashboard_kpi.png     # Executive KPI scorecard summary
│
└── data_outputs/                            # Dashboard-ready datasets
    ├── loan_portfolio_summary_kpis.xlsx     # Multi-tab executive Excel workbook
    ├── loan_portfolio_dashboard_bi_ready.csv# Flat BI table for Power BI / Tableau
    └── portfolio_statistics.json            # Machine-readable JSON summary metrics
```

---

## 9. Installation & Execution Guide

### Step 1: Clone or Open the Workspace
```bash
cd d:/Loan_credit_risk
```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Interactive Jupyter Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook YourName_ProjectName.ipynb
```
Select **Kernel -> Restart & Run All** to execute all analytical cells from beginning to end.

---

## 10. Analytical Limitations & Governance Notice
- **Static Cross-Sectional Data:** The dataset captures a cross-sectional point in time. It lacks time-series loan vintage performance, macro trends (interest rate changes, inflation), and repayment histories.
- **Not an Automated Underwriting System:** The clustering models and risk tiers provide descriptive behavioral grouping. They are **not** automated credit granting/declining models and must not be used as sole determinants for credit denial.
- **Regulatory Model Governance:** Real-world credit risk deployment requires adherence to supervisory guidelines (e.g., Federal Reserve SR 11-7, Equal Credit Opportunity Act, Fair Credit Reporting Act), including continuous Population Stability Index (PSI) tracking and disparate impact testing.

---

## 11. Author & Submission Information
- **Author / Candidate:** `[Your Name / Candidate Name]`
- **Role:** Data Analytics Intern / Credit Risk Analyst
- **Course / Program:** Data Analytics Internship Program
- **Submission Date:** October 2026
- **Contact:** `[your.email@example.com]`
- **LinkedIn / Portfolio:** `[https://www.linkedin.com/in/yourprofile]`
