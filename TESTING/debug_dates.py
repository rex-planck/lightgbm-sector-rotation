"""Quick diagnostic: why only 144 active days in backtest?"""
import joblib, pandas as pd, numpy as np

sig = joblib.load("output/predictions.pkl")
bt_start = pd.Timestamp("20200101")
bt_end = pd.Timestamp("20260218")

rows = []
for tk, df in sig.items():
    mask = (df.index >= bt_start) & (df.index <= bt_end)
    sub = df.loc[mask]
    nz = int((sub["pred_prob"] > 0).sum()) if len(sub) > 0 else 0
    first_nz = None
    if nz > 0:
        first_nz = sub.index[sub["pred_prob"] > 0][0]
    rows.append({
        "ticker": tk,
        "bt_rows": len(sub),
        "nz_pred": nz,
        "data_start": sub.index.min() if len(sub) > 0 else None,
        "first_nonzero": first_nz,
    })

ls = pd.DataFrame(rows)
has_data = ls["bt_rows"] > 0
has_pred = ls["nz_pred"] > 0
print(f"Tickers with bt_rows>0: {has_data.sum()}/{len(ls)}")
print(f"Tickers with nz_pred>0: {has_pred.sum()}/{len(ls)}")

valid = ls[ls["data_start"].notna()].sort_values("data_start")
print("\n--- First 10 (earliest start) ---")
print(valid[["ticker", "data_start", "bt_rows", "nz_pred", "first_nonzero"]].head(10).to_string(index=False))
print("\n--- Last 10 (latest start) ---")
print(valid[["ticker", "data_start", "bt_rows", "nz_pred", "first_nonzero"]].tail(10).to_string(index=False))

starts = valid["data_start"]
print(f"\nStart date: min={starts.min()}, median={starts.median()}, max={starts.max()}")

fns = valid["first_nonzero"].dropna()
print(f"First nonzero pred: min={fns.min()}, median={fns.median()}, max={fns.max()}")
print(f"Nonzero starts after 2024: {(fns >= pd.Timestamp('20240101')).sum()}")
print(f"Nonzero starts after 2025: {(fns >= pd.Timestamp('20250101')).sum()}")

# How many predictions > threshold on each date in 2020-2025?
print("\n--- Monthly count of stocks with pred_prob > 0.5 ---")
all_dfs = []
for tk, df in sig.items():
    sub = df.loc[(df.index >= bt_start) & (df.index <= bt_end), ["pred_prob"]].copy()
    sub["ticker"] = tk
    all_dfs.append(sub)
big = pd.concat(all_dfs)
big["above50"] = big["pred_prob"] > 0.5
monthly = big.resample("ME")["above50"].sum()
print(monthly.to_string())
