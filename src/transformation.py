import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def transform_features(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = np.log1p(df[numeric_cols])
    return df

def create_gdp_index(df):
    df['GDP_Index'] = df['GDP'] / df['Population']
    return df

def apply_pca(df, n_components=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    return components, pca.explained_variance_ratio_
