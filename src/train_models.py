from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def train_and_evaluate(X, y, output_path):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        results[name] = accuracy_score(y, y_pred)
    pd.DataFrame.from_dict(results, orient='index').to_csv(output_path)
