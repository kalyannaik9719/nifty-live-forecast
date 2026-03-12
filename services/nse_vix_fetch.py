import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

OUT_PATH = Path("data/raw/nse_vix_quote.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

def fetch_vix():
    session = requests.Session()
    session.headers.update(HEADERS)

    session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)

    url = "https://www.nseindia.com/api/allIndices"
    r = session.get(url, timeout=15)
    r.raise_for_status()

    data = r.json()["data"]

    vix_row = None
    for row in data:
        if str(row.get("index", "")).upper() == "INDIA VIX":
            vix_row = row
            break

    if vix_row is None:
        raise ValueError("India VIX not found")

    df = pd.DataFrame([{
        "ts": datetime.now(),
        "vix": float(vix_row["last"]),
        "change": float(vix_row.get("variation", 0)),
        "pChange": float(vix_row.get("percentChange", 0)),
    }])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print("Saved VIX →", OUT_PATH)
    print(df)

if __name__ == "__main__":
    fetch_vix()