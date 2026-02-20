"""Refine search on shortlisted params with trailing stop, using existing predictions.pkl."""
import io
import contextlib
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from backtest_engine import run_backtest

sig = joblib.load("output/predictions.pkl")

# 聚焦搜索：围绕第一轮最优年化附近 (top_k≈10, threshold≈0.58~0.70)
topks = [9, 10, 11]
thresholds = [0.55, 0.58, 0.60, 0.65, 0.70, 0.75]
trailing_list = [0.00, 0.03, 0.05, 0.08, 0.10]

rows = []
total = len(topks) * len(thresholds) * len(trailing_list)
done = 0

for k in topks:
    for th in thresholds:
        for ts in trailing_list:
            done += 1
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                r, pf = run_backtest(
                    sig,
                    top_k=k,
                    prob_threshold=th,
                    trailing_stop_pct=ts,
                )

            ret = pf["returns"]
            if isinstance(ret, pd.DataFrame):
                ret = ret.iloc[:, 0]
            ret = ret.dropna()

            if len(ret) == 0:
                continue

            eq = (1 + ret).cumprod()
            ann = float(eq.iloc[-1] ** (252 / len(ret)) - 1)
            dd = float((eq / eq.cummax() - 1).min())
            calmar = float(ann / abs(dd)) if dd < 0 else np.nan
            active = int((ret != 0).sum())
            sharpe = float(r.sharpe) if r.sharpe is not None else np.nan

            rows.append(
                {
                    "top_k": k,
                    "threshold": th,
                    "trailing": ts,
                    "total_ret": float(r.total_return),
                    "ann_ret": ann,
                    "sharpe": sharpe,
                    "max_dd_pct": float(r.max_drawdown or 0.0),
                    "calmar": calmar,
                    "active_days": active,
                    "n_days": int(len(ret)),
                }
            )

            if done % 10 == 0:
                print(
                    f"[{done}/{total}] k={k} th={th:.2f} ts={ts:.2f} => "
                    f"ann={ann:+.2%} ret={r.total_return:+.2%} sharpe={sharpe:+.3f}",
                    flush=True,
                )

df = pd.DataFrame(rows)
# 过滤：避免极端低交易活跃度
valid = df[df["active_days"] > 200].copy()

# 排序：优先年化，再看Sharpe
best_ann = valid.sort_values(["ann_ret", "sharpe"], ascending=False).head(20)
best_sharpe = valid.sort_values(["sharpe", "ann_ret"], ascending=False).head(20)

print("\n=== Top 20 by Annual Return ===")
for _, row in best_ann.iterrows():
    print(
        f"k={int(row['top_k']):2d} th={row['threshold']:.2f} ts={row['trailing']:.2f} "
        f"ret={row['total_ret']:+7.2%} ann={row['ann_ret']:+7.2%} "
        f"sharpe={row['sharpe']:+.3f} maxDD={row['max_dd_pct']:5.1f}% active={int(row['active_days'])}"
    )

print("\n=== Top 20 by Sharpe ===")
for _, row in best_sharpe.iterrows():
    print(
        f"k={int(row['top_k']):2d} th={row['threshold']:.2f} ts={row['trailing']:.2f} "
        f"ret={row['total_ret']:+7.2%} ann={row['ann_ret']:+7.2%} "
        f"sharpe={row['sharpe']:+.3f} maxDD={row['max_dd_pct']:5.1f}% active={int(row['active_days'])}"
    )

valid.to_csv("output/refine_grid.csv", index=False)
print("\nSaved output/refine_grid.csv")
