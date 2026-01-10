from sklearn.ensemble import RandomForestClassifier

def load_model():
    return RandomForestClassifier(n_estimators=200, random_state=42)
