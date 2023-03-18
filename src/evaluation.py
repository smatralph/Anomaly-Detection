import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def consensus_results(results_df, df_original, threshold=2):
    results_df["Consensus"] = results_df.sum(axis=1)
    anomalies = results_df[results_df["Consensus"] >= threshold]

    anomalies_report = df_original.loc[anomalies.index]
    anomalies_report["ConsensusScore"] = anomalies["Consensus"]
    anomalies_report.to_csv("results/anomaly_reports.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df_original.index, y=results_df["Consensus"])
    plt.title("Consensus Anomaly Score Over Time")
    plt.xlabel("Index or Year")
    plt.ylabel("Consensus Score")
    plt.tight_layout()
    plt.savefig("results/consensus_score.png")
    plt.close()

    print(f"{len(anomalies_report)} anomalies detected and saved to results/anomaly_reports.csv")
    return anomalies_report
