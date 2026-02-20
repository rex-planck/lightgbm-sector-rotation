# -*- coding: utf-8 -*-
"""Efficient grid search over top_k and prob_threshold."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from backtest_engine import run_backtest
from ml_strategy import load_prediction_dict

PREDICTION_FILE = Path("output/predictions.pkl")
signal_dict = load_prediction_dict(PREDICTION_FILE)

# Wide grid
TOP_K_VALUES = [1, 2, 3, 5]
THRESHOLD_VALUES = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.91, 0.93, 0.95, 0.97]

results = []
total = len(TOP_K_VALUES) * len(THRESHOLD_VALUES)
idx = 0

for tk in TOP_K_VALUES:
    for th in THRESHOLD_VALUES:
        idx += 1
        sys.stdout.write(f"\r[{idx}/{total}] top_k={tk} thresh={th:.2f} ...")
        sys.stdout.flush()
        try:
            res, pf = run_backtest(
                signal_dict=signal_dict,
                top_k=tk,
                prob_threshold=th,
                trailing_stop_pct=0.0,
            )
            returns = pf["returns"]
            if hasattr(returns, "dropna"):
                returns = returns.dropna()
            if getattr(returns, "ndim", 1) > 1:
                returns = returns.iloc[:, 0]

            n_days = len(returns)
            eq_final = float((1.0 + returns).cumprod().iloc[-1]) if n_days > 0 else 1.0
            ann = 0.0
            if eq_final > 0 and n_days > 0:
                ann = float(eq_final ** (252 / n_days) - 1)
            active = int((returns.abs() > 1e-8).sum())
            hold_pct = active / max(n_days, 1) * 100

            row = {
                "top_k": tk,
                "threshold": th,
                "total_return": round(res.total_return * 100, 2),
                "ann_return": round(ann * 100, 2),
                "sharpe": round(res.sharpe, 3) if res.sharpe else None,
                "max_dd": round(res.max_drawdown, 2),
                "hold_pct": round(hold_pct, 1),
            }
            results.append(row)
        except Exception as e:
            print(f"\n  ERROR: {e}")

# Sort by total return descending
results.sort(key=lambda r: r["total_return"], reverse=True)

print("\n\n" + "=" * 85)
print("GRID SEARCH RESULTS (sorted by total return)")
print("=" * 85)
print(f"{'top_k':>5} {'thresh':>7} {'return%':>10} {'ann%':>10} {'sharpe':>8} {'maxDD%':>8} {'hold%':>7}")
print("-" * 65)
for r in results:
    sh = f"{r['sharpe']:.3f}" if r['sharpe'] else "N/A"
    print(
        f"{r['top_k']:>5} {r['threshold']:>7.2f} "
        f"{r['total_return']:>+10.2f} {r['ann_return']:>+10.2f} "
        f"{sh:>8} {r['max_dd']:>7.2f}% {r['hold_pct']:>6.1f}%"
    )

out = Path("output/grid_search_results.json")
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out}")
