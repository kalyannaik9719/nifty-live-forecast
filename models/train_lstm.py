from pathlib import Path
import pandas as pd

from models.lstm_model import train_lstm

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
ARTIFACT_DIR = Path("artifacts/lstm")


def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["target_return"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna().reset_index(drop=True)

    art = train_lstm(df=df, artifact_dir=ARTIFACT_DIR, epochs=25)
    print("Saved LSTM model:", art.model_path)
    print("Saved LSTM meta:", art.meta_path)


if __name__ == "__main__":
    main()