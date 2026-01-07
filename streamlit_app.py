import streamlit as st

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
     data_choice=st.radio("choose dataset",["Upload CSV","upload your owndataset"])
     test_size=st.slider("Test Size (Validation Split)",0.1,0.2,0.02)
     scale_numeric=st.checkbox("Scale Numeric features(StandradScalar)",value=True)
     nb_variant=st.selectbox("Naive Bayes varient",["GaussianNB","MultinomialNB"])
     random_state=st.number_input("Random seed",min_value=0,max_value=10000, value=42,step=1)
