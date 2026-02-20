"""Run final backtest from existing predictions.pkl with tuned params and regenerate outputs/charts."""
from __future__ import annotations

import json
from pathlib import Path

from backtest_engine import run_backtest
from charts import generate_all_charts
from config import ML_PARAMS, OUTPUT_DIR, PREDICTION_FILE
from main import save_pyfolio_outputs
from ml_strategy import load_prediction_dict


def main() -> None:
    signal_dict = load_prediction_dict(PREDICTION_FILE)

    result, pyfolio_data = run_backtest(
        signal_dict=signal_dict,
        top_k=ML_PARAMS.top_k,
        prob_threshold=ML_PARAMS.prob_threshold,
        trailing_stop_pct=ML_PARAMS.trailing_stop_pct,
    )

    save_pyfolio_outputs(pyfolio_data=pyfolio_data, output_dir=OUTPUT_DIR)
    generate_all_charts(pyfolio_data=pyfolio_data, result=result, output_dir=OUTPUT_DIR)

    returns = pyfolio_data["returns"]
    if hasattr(returns, "dropna"):
        returns = returns.dropna()
    if getattr(returns, "ndim", 1) > 1:
        returns = returns.iloc[:, 0]

    n_days = len(returns)
    ann = None
    calmar = None
    if n_days > 0:
        eq_final = float((1.0 + returns).cumprod().iloc[-1])
        if eq_final > 0:
            ann = float(eq_final ** (252 / n_days) - 1)
        else:
            ann = -1.0
        mdd = float(result.max_drawdown or 0.0)
        if mdd > 0:
            calmar = ann / (mdd / 100.0)

    # Holding ratio
    active_days = int((returns.abs() > 1e-8).sum())
    holding_pct = active_days / n_days * 100 if n_days > 0 else 0.0

    summary = {
        "top_k": ML_PARAMS.top_k,
        "prob_threshold": ML_PARAMS.prob_threshold,
        "trailing_stop_pct": ML_PARAMS.trailing_stop_pct,
        "start_value": float(result.start_value),
        "final_value": float(result.final_value),
        "total_return": float(result.total_return),
        "annual_return": ann,
        "sharpe": None if result.sharpe is None else float(result.sharpe),
        "max_drawdown_pct": float(result.max_drawdown or 0.0),
        "calmar": calmar,
        "holding_pct": round(holding_pct, 1),
        "trading_days": n_days,
        "active_days": active_days,
    }

    out_path = Path(OUTPUT_DIR) / "final_backtest_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Final summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
