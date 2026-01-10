
# Heart Disease Classification — End-to-End ML & Streamlit App

## Problem Statement
Predict the presence of **heart disease** (binary: 0/1) using clinical and demographic features, and deploy an interactive web app for evaluation.

## Dataset Description
We use the **Heart Disease / Heart Failure Prediction** dataset widely derived from UCI heart disease databases and consolidated on Kaggle (918 rows, 12 features + target). Download `heart.csv` and place it under `data/`.

- Source: Kaggle — `fedesoriano/heart-failure-prediction`
- Target column: `HeartDisease` (1 = disease, 0 = healthy)
- Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

> **Why not the raw UCI Cleveland (303 rows)?** The assignment requires at least **500 instances**; the Kaggle-consolidated dataset satisfies this requirement while keeping the classic UCI feature schema.

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table
<!--METRICS_TABLE_START-->
| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|---|---|---|---|---|---|---|
| _metrics will be injected here by `train.py`_ |  |  |  |  |  |  |
<!--METRICS_TABLE_END-->

### Observations
(Add comments after training, e.g., Random Forest/XGBoost typically achieve higher AUC and MCC; Logistic Regression is interpretable; kNN may underperform with many one-hot features; Naive Bayes can be fast but assumes feature independence.)

## Repository Structure
StreamlitApp/
├── app.py
├── train.py
├── requirements.txt
├── utils.py
├── data/
│   └── heart.csv
└── models/
├── *.pkl
└── metrics.csv
