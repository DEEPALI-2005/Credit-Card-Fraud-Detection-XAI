# Credit Card Fraud Detection with Explainable AI

> **Author:** Deepali  
> **Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
> **Primary Metric:** PR-AUC (Precision-Recall AUC)

---

## Overview

A complete, end-to-end machine learning project that detects fraudulent credit card transactions using three classifiers — Logistic Regression, Random Forest, and XGBoost — with SMOTE oversampling for class imbalance and SHAP-based Explainable AI for model interpretability. The best model is served via an interactive Streamlit web application.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Records | 284,807 transactions |
| Fraud | 492 (0.173%) — severely imbalanced |
| Features | V1–V28 (PCA-transformed), Time, Amount, Class |
| Target | `Class` — 0: Legitimate, 1: Fraud |

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.11 |
| Data | pandas 2.3.1, numpy 1.26.4 |
| Visualisation | matplotlib 3.10.3, seaborn 0.13.2 |
| ML | scikit-learn 1.7.2, xgboost 3.2.0 |
| Imbalance | imbalanced-learn 0.14.2 (SMOTE) |
| XAI | shap 0.51.0 |
| Web App | streamlit 1.53.1 |
| Serialisation | joblib 1.5.2 |
| Notebook | JupyterLab 4.6.3 |

---

## Folder Structure

```
Deepali_FraudDetection/
├── data/
│   └── creditcard.csv              ← raw dataset
├── outputs/
│   ├── charts/                     ← all saved PNG visualisations
│   └── models/
│       └── fraud_model.pkl         ← best saved model (XGBoost)
├── Deepali_FraudDetection.ipynb    ← main analysis notebook (11 sections)
├── app.py                          ← Streamlit web application
├── requirements.txt                ← pinned dependencies
├── README.md                       ← this file
└── Deepali_ProjectReport.docx      ← formal project report
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook (trains models, saves charts + model)
```bash
jupyter notebook Deepali_FraudDetection.ipynb
```
Run all cells top-to-bottom. The notebook will:
- Perform EDA and generate 14+ charts to `outputs/charts/`
- Train 3 models and save the best to `outputs/models/fraud_model.pkl`
- Generate SHAP explanations

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Key Results

| Model | Accuracy | ROC-AUC | **PR-AUC ↑** | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9708 | 0.7231 | 0.9184 | 0.0569 | 0.1072 |
| Random Forest | 0.9983 | 0.9793 | 0.8323 | 0.8469 | 0.5092 | 0.6360 |
| **XGBoost** | **0.9994** | **0.9997** | **0.9791 ★** | **0.9796** | **0.7619** | **0.8571** |

> XGBoost achieves PR-AUC = **0.9791**, well above the KPI target of 0.80.
> PR-AUC is the primary metric — ROC-AUC is misleading under 0.173% fraud rate.

---

## KPI Targets

| Metric | Target |
|---|---|
| Recall (Fraud) | > 90% |
| Precision (Fraud) | > 85% |
| F1-Score (Fraud) | > 88% |
| PR-AUC | > 0.80 |

---

## EDA Charts Generated

| # | Chart |
|---|---|
| 01 | Class imbalance bar chart (log scale) |
| 02 | Amount distribution — Fraud vs Legitimate |
| 03 | Transaction count by hour of day |
| 04 | Correlation heatmap — top 15 features vs Class |
| 05 | Boxplots of top 6 V-features vs Class |
| 06 | Fraud rate across Amount quantile bins |
| 07 | PCA 2D scatter coloured by Class |
| 08 | KDE feature distribution comparison |
| 09 | Combined ROC curves |
| 10 | Combined Precision-Recall curves |
| 11 | Model comparison bar chart |
| 12 | SHAP global feature importance |
| 13 | SHAP beeswarm summary plot |
| 14 | SHAP waterfall (single fraud explanation) |

---

## Why PR-AUC (not ROC-AUC)?

With only 0.173% fraud, ROC-AUC is inflated and misleading — a model predicting all-0 achieves high ROC-AUC. PR-AUC focuses only on the minority class (fraud) and provides an honest assessment of model performance.

---

## Author

**Deepali**  
B.Tech / Data Science  
Project: Credit Card Fraud Detection with Explainable AI (XAI)
