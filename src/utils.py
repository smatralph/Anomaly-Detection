import pandas as pd
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    return pd.read_excel(path)
