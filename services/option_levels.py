import pandas as pd
from pathlib import Path
import json

CHAIN_PATH = Path("data/raw/nifty_option_chain.parquet")
OUT_PATH = Path("data/processed/option_levels.json")

def build_option_levels():
    df = pd.read_parquet(CHAIN_PATH).copy()

    call_wall = int(df.loc[df["ce_oi"].idxmax(), "strike"])
    put_wall = int(df.loc[df["pe_oi"].idxmax(), "strike"])

    top_calls = df.nlargest(5, "ce_oi")[["strike", "ce_oi"]].to_dict(orient="records")
    top_puts = df.nlargest(5, "pe_oi")[["strike", "pe_oi"]].to_dict(orient="records")

    out = {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "top_calls": top_calls,
        "top_puts": top_puts,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("Saved option levels →", OUT_PATH)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    build_option_levels()