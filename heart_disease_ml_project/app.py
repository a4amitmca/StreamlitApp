import streamlit as st
import os
import requests
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
from joblib import dump
from utils import TARGET_COL, build_preprocessor
st.set_page_config(page_title="Bits ML Classification and Models & Metrics")
st.title("Classification Model and Evaluation matrix")
st.markdown("""
This app train six classification model on a single data set and reports the following  metrices:
- Accurecy
- AUC Score
- Precesion
- Recall
- F1 Score
- Mathhews corelation coefficent (MCC)

 you can upload your owndataset(CSV) min 12 features and 1000 instances).           
""")
with st.sidebar:
     st.header("Data & Setting")
     data_choice=st.selectbox("choose dataset",["Upload CSV","Uploaad Excel CSV supported"])
     #test_size=st.slider("Test Size (Validation Split)",0.1,0.4,0.2,0.02)
     #scale_numeric=st.checkbox("Scale Numeric features(StandradScalar)",value=True)
     model_option = st.selectbox("Select Model",["Logistic Regression", "Random Forest", "KNN", "Decision Tree", "SVM", "XGBoost"])
     #random_state=st.number_input("Random seed",min_value=0,max_value=10000, value=42,step=1)
if data_choice=="Upload CSV":
     uploaded=st.file_uploader("Upload the CSV file (last colum target recomended)",type=["csv"])
     df=None
     if uploaded is not None:
        df=pd.read_csv(uploaded)
        st.write("preview:",df.head())
        st.write("shape",df.shape)
        save_name = "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/data/heart.csv"
        df.to_csv(save_name, index=False)
        st.success(f"File saved as {save_name}")
        subprocess.run(["python3", "train.py"])
        st.success("Model Training Successfull for train.py")
        #DATA_PATH = os.path.join('data', 'heart.csv')
#assert os.path.exists(save_name), f"Dataset not found at {DATA_PATH}. Please download heart.csv from Kaggle and place it here."

# Load dataset

#assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}"

# Train/test split
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessor
preprocessor, feature_cols, num_cols, cat_cols = build_preprocessor(df)

# Helper to build pipelines
def make_pipeline(model):
    return Pipeline(steps=[('pre', preprocessor), ('clf', model)])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    ),
}

records = []
report_dir = 'models'
os.makedirs(report_dir, exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe.named_steps['clf'], 'decision_function'):
        y_prob = pipe.decision_function(X_test)
    else:
        y_prob = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    records.append({
        'ML Model Name': name,
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc,
    })

    # Save model pipeline (preprocessor inside)
    dump(pipe, os.path.join(report_dir, f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}.pkl"))

    # Save confusion matrix & classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    with open(os.path.join(report_dir, f"{name}_report.txt"), 'w') as f:
        f.write(f"Classification Report for {name}\n\n")
        f.write(cr)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

# Also save the standalone preprocessor fitted on training data
preprocessor.fit(X_train)
dump(preprocessor, os.path.join(report_dir, 'preprocessor.pkl'))

# Save metrics
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(os.path.join(report_dir, 'metrics.csv'), index=False)
print("\nSaved models and metrics to 'models/'")

# Auto-update README.md comparison table
readme_path = 'README.md'
if os.path.exists(readme_path):
    md_table_header = '| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |\n|---|---|---|---|---|---|---|\n'
    rows = []
    for r in records:
        rows.append(f"| {r['ML Model Name']} | {r['Accuracy']:.4f} | {r['AUC']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['MCC']:.4f} |\n")
    md_table = md_table_header + ''.join(rows)

    with open(readme_path, 'r', encoding='utf-8') as f:
        md = f.read()
    start_tag = '<!--METRICS_TABLE_START-->'
    end_tag = '<!--METRICS_TABLE_END-->'
    if start_tag in md and end_tag in md:
        before = md.split(start_tag)[0]
        after = md.split(end_tag)[-1]
        new_md = before + start_tag + '\n\n' + md_table + '\n' + end_tag + after
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_md)
        print("README.md updated with metrics table.")
        st.write("README.md updated with metrics table.")
    else:
        print("Placeholder tags not found in README.md; skipping auto-update.")
      
else:
        st.markdown("--------------Default File Heart.csv---------------------------")    
        csv_url = "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/data/heart.csv"
        df = pd.read_csv(csv_url)
        st.dataframe(df.head())
 
st.markdown("--------------Matrix for all 6 Model of Heart.csv---------------------------") 
csv_url="https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/metrics.csv"
df = pd.read_csv(csv_url)
st.dataframe(df.head())
st.markdown("--------------This is Matrix for all model selection of Heart.csv---------------------------") 

metrics_urls = {
    "Logistic Regression": "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/LogisticRegression_report.txt",
    "Random Forest":       "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/RandomForest_report.txt",
    "KNN":                 "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/kNN_report.txt",
    "Decision Tree":       "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/Decision_Tree_report.txt",
    "SVM":                 "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/NaiveBayes(Gaussian)_report.txt",
    "XGBoost":             "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/XGBoost_report.txt",
}

# Fetch and display metrics
url = metrics_urls.get(model_option)  # Get URL for selected model
if url:
    response = requests.get(url)
    if response.status_code == 200:
        metrics_text = response.text
        st.subheader(f"Evaluation Metrics for {model_option}")
        st.markdown(f"```\n{metrics_text}\n```")  # preserves formatting
    else:
        st.error(f"Could not fetch the metrics file for {model_option}")
else:
    st.warning("No metrics available for the selected model")
         
   
