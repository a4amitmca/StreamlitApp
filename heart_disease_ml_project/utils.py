
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEDESORIANO_EXPECTED_COLUMNS = [
    'Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
    'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'
]

TARGET_COL = 'HeartDisease'

def infer_columns(df: pd.DataFrame):
    """Infer numeric and categorical columns; drop target if present."""
    cols = list(df.columns)
    if TARGET_COL in cols:
        feature_cols = [c for c in cols if c != TARGET_COL]
    else:
        feature_cols = cols
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols

def build_preprocessor(df: pd.DataFrame):
    feature_cols, num_cols, cat_cols = infer_columns(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ],
        remainder='drop'
    )
    return preprocessor, feature_cols, num_cols, cat_cols
