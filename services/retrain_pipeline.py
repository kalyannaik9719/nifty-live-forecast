from services.data_ingest import run_once as ingest_once
from services.feature_engineering import save_features, add_features, load_data
from services.baseline_model import load_data as load_feat_data, train_model, save_model
import pandas as pd

def run_retrain():
    print("Starting retrain pipeline...")

    ingest_once()

    df = pd.read_parquet("data/raw/nifty_5m.parquet")
    feat = add_features(df)
    save_features(feat)

    train_df = load_feat_data()
    model = train_model(train_df)
    save_model(model)

    print("Retrain complete.")

if __name__ == "__main__":
    run_retrain()