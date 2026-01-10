import streamlit as st
import pandas as pd
import os
import requests
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
     test_size=st.slider("Test Size (Validation Split)",0.1,0.4,0.2,0.02)
     scale_numeric=st.checkbox("Scale Numeric features(StandradScalar)",value=True)
     model_option = st.selectbox("Select Model",("Logistic Regression", "Decision Tree Classifier","K-Nearest Neighbor Classifier","Naive Bayes Classifier - Gaussian or Multinomial","Ensemble Model - Random Forest","Ensemble Model - XGBoost"))
     random_state=st.number_input("Random seed",min_value=0,max_value=10000, value=42,step=1)
if data_choice=="Upload CSV":
     uploaded=st.file_uploader("Upload the CSV file (last colum target recomended)",type=["csv"])
     df=None
     if uploaded is not None:
        df=pd.read_csv(uploaded)
        st.write("preview:",df.head())
        st.write("shape",df.shape)
     
     else:
        csv_url = "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/data/heart.csv"
        df = pd.read_csv(csv_url)
        st.dataframe(df.head())
st.markdown("--------------Evaluation metrics---------------------------")     


if model_option == "Logistic Regression":
    url = "https://raw.githubusercontent.com/a4amitmca/StreamlitApp/master/heart_disease_ml_project/models/Logistic Regression_report.txt"
    response = requests.get(url)
    if response.status_code == 200:
       metrics_text = response.text
       st.subheader("Evaluation Metrics")
       st.text(metrics_text)  # or st.markdown(f"```\n{metrics_text}\n```")
    else:
       st.error("Could not fetch the metrics file from GitHub"
         
   
