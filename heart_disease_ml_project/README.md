
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
| Logistic Regression | 0.8859 | 0.9299 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| Decision Tree | 0.7935 | 0.7910 | 0.8137 | 0.8137 | 0.8137 | 0.5820 |
| kNN | 0.9130 | 0.9503 | 0.9135 | 0.9314 | 0.9223 | 0.8238 |
| Naive Bayes (Gaussian) | 0.8859 | 0.9118 | 0.8932 | 0.9020 | 0.8976 | 0.7688 |
| Random Forest | 0.9022 | 0.9331 | 0.8962 | 0.9314 | 0.9135 | 0.8018 |
| XGBoost | 0.8696 | 0.9284 | 0.8980 | 0.8627 | 0.8800 | 0.7380 |

<!--METRICS_TABLE_END-->

### Observations
ML Model Name	Observation about Model Performance
Logistic Regression	Achieved 88.59% accuracy with a strong AUC of 0.93. It shows good generalization and high recall (93.14%), making it effective at identifying heart disease cases, though slightly weaker than top-performing models.
Decision Tree	Produced the lowest performance with 79.35% accuracy and MCC of 0.58, indicating overfitting and weaker generalization compared to ensemble and distance-based methods.
kNN	Delivered the best overall performance with 91.30% accuracy, highest AUC (0.95), F1-score 0.92, and highest MCC (0.82). This indicates excellent predictive power and balanced classification.
Naive Bayes	Achieved 88.59% accuracy with solid balance between precision and recall. Performs well despite strong independence assumptions, making it a reliable and computationally efficient model.
Random Forest (Ensemble)	Reached 90.21% accuracy with high AUC (0.93) and MCC (0.80). The ensemble approach significantly improves stability and robustness over a single decision tree.
XGBoost (Ensemble)	Obtained 86.96% accuracy with strong precision (0.90). While powerful, it slightly underperformed compared to kNN and Random Forest on this dataset, likely due to hyperparameter sensitivity.

Overall Conclusion for Report

kNN emerged as the top-performing model on this dataset, followed closely by Random Forest, while Decision Tree showed the weakest performance. Ensemble methods improved stability and accuracy, but the distance-based kNN classifier achieved the most balanced and reliable results across all evaluation metrics.


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
