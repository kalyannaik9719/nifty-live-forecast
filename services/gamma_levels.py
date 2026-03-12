import pandas as pd
from pathlib import Path

OPTION_CHAIN = Path("data/raw/nifty_option_chain.parquet")
OUTPUT = Path("data/processed/gamma_levels.json")

def compute_gamma_levels():

    df = pd.read_parquet(OPTION_CHAIN)

    df["call_gamma"] = df["CE_openInterest"] * df["CE_lastPrice"]
    df["put_gamma"] = df["PE_openInterest"] * df["PE_lastPrice"]

    gamma_call = df.groupby("strike")["call_gamma"].sum()
    gamma_put = df.groupby("strike")["put_gamma"].sum()

    gamma_resistance = gamma_call.idxmax()
    gamma_support = gamma_put.idxmax()

    result = {
        "gamma_resistance": float(gamma_resistance),
        "gamma_support": float(gamma_support)
    }

    OUTPUT.write_text(str(result))

    print(result)


if __name__ == "__main__":
    compute_gamma_levels()