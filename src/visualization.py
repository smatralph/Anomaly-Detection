import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_heatmap(df, save_path="results/correlation_heatmap.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Economic Variables")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def time_series_plot(df, column, save_path="results/timeseries_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[column], marker='o')
    plt.title(f"Time Series of {column}")
    plt.xlabel("Year")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def anomaly_scatter(df, x_col, y_col, anomalies, save_path="results/anomaly_scatter.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], label="Normal", alpha=0.6)
    plt.scatter(df.loc[anomalies.index, x_col], df.loc[anomalies.index, y_col], 
                color="red", label="Anomalies", marker="x")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Anomalies Highlighted in Scatter Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
