from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ee import EllipticEnvelope
import pandas as pd

def run_models(df):
    X = df.select_dtypes(include='number')
    models = {
        "IsolationForest": IForest(),
        "LOF": LOF(),
        "KNN": KNN(),
        "HBOS": HBOS(),
        "EllipticEnvelope": EllipticEnvelope()
    }

    results = {}
    for name, model in models.items():
        model.fit(X)
        results[name] = model.labels_
    return pd.DataFrame(results)
