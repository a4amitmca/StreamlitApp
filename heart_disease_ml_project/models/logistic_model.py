from sklearn.linear_model import LogisticRegression

def load_model():
    return LogisticRegression(max_iter=1000)
