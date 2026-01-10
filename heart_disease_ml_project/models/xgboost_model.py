from xgboost import XGBClassifier

def load_model():
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42)
