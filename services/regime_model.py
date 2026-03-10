import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/nifty_features.parquet")
OUT_PATH = Path("data/processed/nifty_regimes.parquet")

def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

def classify_regime(df):
    df = df.copy()

    vol_q75 = df["volatility"].quantile(0.75)

    def get_regime(row):
        if row["volatility"] > vol_q75:
            return "high_vol"
        elif row["close_sma20_gap"] > 0.002:
            return "bull"
        elif row["close_sma20_gap"] < -0.002:
            return "bear"
        else:
            return "sideways"

    df["regime"] = df.apply(get_regime, axis=1)
    return df

def save_data(df):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print("Regime data saved →", OUT_PATH)
    print(df["regime"].value_counts())

if __name__ == "__main__":
    df = load_data()
    df = classify_regime(df)
    save_data(df)