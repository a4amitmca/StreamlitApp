import streamlit as st

st.set_page_config(page_title="Bits ML Classification and Models & Metrics")
st.title("Classification Model and Evaluation matrix")
st.markdown("""This app train six classification model on a single data set and reports the following  metrices:
-Accurecy
-AUC Score
-Precesion
-Recall
-F1 Score
-Mathhews corelation coefficent (MCC)
 you can upload your owndata set(CSV) or use the simple ** Breast cancer Wisconsin ** dataset (30 features , 550 instances).           
""")
