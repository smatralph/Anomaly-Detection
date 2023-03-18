from src.data_preprocessing import load_and_clean_data
from src.transformation import transform_features, create_gdp_index
from src.anomaly_models import run_models
from src.evaluation import consensus_results
from src.utils import ensure_dir

def main():
    ensure_dir("results")
    ensure_dir("data/processed")

    print("Loading data...")
    df = load_and_clean_data("data/raw/FIRS_Taxes.xlsx")

    print("Transforming features...")
    df = transform_features(df)
    df = create_gdp_index(df)

    print("Running anomaly detection...")
    results = run_models(df)

    print("Building consensus and saving results...")
    consensus_results(results, df)

    print("Pipeline complete! Check results/ folder.")

if __name__ == "__main__":
    main()
