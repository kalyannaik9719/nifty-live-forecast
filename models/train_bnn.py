from pathlib import Path
import pandas as pd

from models.bnn_model import train_bnn

DATA_PATH = Path("data/processed/nifty_regimes.parquet")
ARTIFACT_DIR = Path("artifacts/bnn")


def main():
    df = pd.read_parquet(DATA_PATH).copy()
    df["target_return"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna().reset_index(drop=True)

    train_bnn(df=df, artifact_dir=ARTIFACT_DIR, steps=1200)
    print("Saved BNN artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()