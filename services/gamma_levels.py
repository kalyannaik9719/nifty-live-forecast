import json
from pathlib import Path
import pandas as pd

CHAIN_PATH = Path("data/raw/nifty_option_chain.parquet")
OUT_PATH = Path("data/processed/gamma_levels.json")


def compute_gamma_levels():
    if not CHAIN_PATH.exists():
        out = {
            "gamma_support": None,
            "gamma_resistance": None,
            "top_call_walls": [],
            "top_put_walls": [],
            "status": "option_chain_missing"
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    df = pd.read_parquet(CHAIN_PATH).copy()

    for col in ["strike", "ce_oi", "pe_oi", "ce_ltp", "pe_ltp"]:
        if col not in df.columns:
            df[col] = 0

    df["call_gamma_proxy"] = df["ce_oi"].fillna(0) * df["ce_ltp"].fillna(0)
    df["put_gamma_proxy"] = df["pe_oi"].fillna(0) * df["pe_ltp"].fillna(0)

    call_sorted = df.sort_values("call_gamma_proxy", ascending=False).head(5)
    put_sorted = df.sort_values("put_gamma_proxy", ascending=False).head(5)

    gamma_resistance = None
    gamma_support = None

    if len(call_sorted) > 0:
        gamma_resistance = float(call_sorted.iloc[0]["strike"])

    if len(put_sorted) > 0:
        gamma_support = float(put_sorted.iloc[0]["strike"])

    out = {
        "gamma_support": gamma_support,
        "gamma_resistance": gamma_resistance,
        "top_call_walls": call_sorted[["strike", "call_gamma_proxy"]].to_dict(orient="records"),
        "top_put_walls": put_sorted[["strike", "put_gamma_proxy"]].to_dict(orient="records"),
        "status": "ok"
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print("Saved gamma levels →", OUT_PATH)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    compute_gamma_levels()