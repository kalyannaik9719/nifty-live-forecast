import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/nifty_5m.parquet")
OUT_PATH = Path("data/processed/nifty_features.parquet")

def load_data():
    df = pd.read_parquet(RAW_PATH)
    df = df.sort_values("ts")
    return df

def add_features(df):

    # returns
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)

    # moving averages
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()

    # exponential moving avg
    df["ema12"] = df["close"].ewm(span=12).mean()

    # volatility
    df["volatility"] = df["ret1"].rolling(12).std()

    # price distance from averages
    df["close_sma5_gap"] = (df["close"] - df["sma5"]) / df["sma5"]
    df["close_sma20_gap"] = (df["close"] - df["sma20"]) / df["sma20"]

    # candle range
    df["range"] = df["high"] - df["low"]

    # time features
    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["minute"] = pd.to_datetime(df["ts"]).dt.minute

    # prediction target (next candle direction)
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df

def save_features(df):

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.dropna().to_parquet(OUT_PATH,index=False)
    print("Features saved →", OUT_PATH)

if __name__ == "__main__":

    df = load_data()
    df = add_features(df)
    save_features(df)