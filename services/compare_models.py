import pandas as pd
from pathlib import Path

BASELINE_PATH = Path("data/processed/paper_trades.parquet")
RL_PATH = Path("data/processed/rl_trades.parquet")
OUT_PATH = Path("data/processed/model_comparison.json")


def summarize_equity(df: pd.DataFrame, equity_col: str = "equity_curve") -> dict:
    equity = df[equity_col].dropna().reset_index(drop=True)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    total_return = float(equity.iloc[-1] - 1.0)
    max_drawdown = float(drawdown.min())

    rets = equity.pct_change().dropna()
    sharpe_like = float(rets.mean() / rets.std()) if len(rets) > 1 and rets.std() != 0 else 0.0

    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
        "num_rows": int(len(df)),
    }


def run():
    baseline_df = pd.read_parquet(BASELINE_PATH)
    rl_df = pd.read_parquet(RL_PATH)

    baseline_summary = summarize_equity(baseline_df, "equity_curve")
    rl_summary = summarize_equity(rl_df, "equity_curve")

    comparison = {
        "baseline": baseline_summary,
        "rl": rl_summary,
        "winner": "rl" if rl_summary["final_equity"] > baseline_summary["final_equity"] else "baseline"
    }

    pd.Series(comparison).to_json(OUT_PATH, indent=2)
    print("Comparison saved →", OUT_PATH)
    print("Baseline Final Equity:", round(baseline_summary["final_equity"], 4))
    print("RL Final Equity:", round(rl_summary["final_equity"], 4))
    print("Winner:", comparison["winner"])


if __name__ == "__main__":
    run()