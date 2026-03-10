import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/nifty_5m.parquet")

def fetch_data():
    df = yf.download("^NSEI", interval="5m", period="5d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    df.rename(columns={"datetime":"ts","date":"ts"}, inplace=True)

    return df

def save_data(df):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PATH,index=False)
    print("Saved data →", DATA_PATH)

if __name__ == "__main__":
    df = fetch_data()
    save_data(df)