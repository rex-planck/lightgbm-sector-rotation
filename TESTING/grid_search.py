"""Grid search: top_k x threshold, using existing predictions.pkl."""
import io, contextlib, warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import joblib, numpy as np, pandas as pd
from backtest_engine import run_backtest

sig = joblib.load("output/predictions.pkl")

topks = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
thresholds = [0.0, 0.30, 0.40, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.91]
rows = []
total = len(topks) * len(thresholds)
done = 0

for k in topks:
    for th in thresholds:
        done += 1
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            r, pf = run_backtest(sig, top_k=k, prob_threshold=th, trailing_stop_pct=0.0)
        ret = pf["returns"]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = ret.dropna()
        eq = (1 + ret).cumprod()
        n = len(ret)
        if n == 0:
            continue
        ann = eq.iloc[-1] ** (252 / n) - 1
        dd = (eq / eq.cummax() - 1).min()
        calmar = (ann / abs(dd)) if dd < 0 else np.nan
        active = int((ret != 0).sum())
        sh = r.sharpe if r.sharpe is not None else np.nan
        rows.append({
            "top_k": k, "threshold": th,
            "total_ret": r.total_return,
            "ann_ret": ann,
            "sharpe": sh,
            "max_dd_pct": r.max_drawdown,
            "calmar": calmar,
            "active_days": active,
            "n_days": n,
        })
        if done % 10 == 0:
            print(f"  [{done}/{total}] k={k} th={th} => ret={r.total_return:+.2%} sharpe={sh:.3f} dd={r.max_drawdown:.1f}%", flush=True)

df = pd.DataFrame(rows)
valid = df[(df["active_days"] > 50)].copy()
print(f"\nGrid: {len(df)} combos, {len(valid)} with >50 active days")

print("\n=== Top 20 by Sharpe ===")
top = valid.sort_values("sharpe", ascending=False).head(20)
for _, row in top.iterrows():
    print(f"  k={int(row['top_k']):2d}  th={row['threshold']:.2f}  ret={row['total_ret']:+7.2%}  ann={row['ann_ret']:+7.2%}  "
          f"sharpe={row['sharpe']:+.3f}  maxDD={row['max_dd_pct']:5.1f}%  calmar={row['calmar']:+.2f}  active={int(row['active_days'])}")

print("\n=== Top 20 by Annual Return ===")
top2 = valid.sort_values("ann_ret", ascending=False).head(20)
for _, row in top2.iterrows():
    print(f"  k={int(row['top_k']):2d}  th={row['threshold']:.2f}  ret={row['total_ret']:+7.2%}  ann={row['ann_ret']:+7.2%}  "
          f"sharpe={row['sharpe']:+.3f}  maxDD={row['max_dd_pct']:5.1f}%  calmar={row['calmar']:+.2f}  active={int(row['active_days'])}")

valid.to_csv("output/grid_full.csv", index=False)
print("\nSaved output/grid_full.csv")
