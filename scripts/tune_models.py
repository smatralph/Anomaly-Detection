from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.model_selection import ParameterGrid
import pandas as pd

def tune_isolation_forest(X):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "contamination": [0.05, 0.1, 0.15]
    }
    grid = ParameterGrid(param_grid)
    scores = []

    for params in grid:
        model = IForest(**params)
        model.fit(X)
        scores.append({
            "params": params,
            "n_anomalies": sum(model.labels_)
        })
    pd.DataFrame(scores).to_csv("results/iforest_tuning.csv", index=False)
    print("Isolation Forest tuning complete. Results saved to results/iforest_tuning.csv")

def tune_knn(X):
    param_grid = {
        "n_neighbors": [3, 5, 10],
        "contamination": [0.05, 0.1]
    }
    grid = ParameterGrid(param_grid)
    scores = []

    for params in grid:
        model = KNN(**params)
        model.fit(X)
        scores.append({
            "params": params,
            "n_anomalies": sum(model.labels_)
        })
    pd.DataFrame(scores).to_csv("results/knn_tuning.csv", index=False)
    print("KNN tuning complete. Results saved to results/knn_tuning.csv")

def main():
    df = pd.read_excel("data/processed/cleaned_data.xlsx")
    X = df.select_dtypes(include="number")

    tune_isolation_forest(X)
    tune_knn(X)

if __name__ == "__main__":
    main()
