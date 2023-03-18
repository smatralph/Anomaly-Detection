import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')
    df = df.fillna(df.mean(numeric_only=True))
    return df
