from sklearn.neighbors import KNeighborsClassifier

def load_model():
    return KNeighborsClassifier(n_neighbors=7)
