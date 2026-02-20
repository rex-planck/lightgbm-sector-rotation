# -*- coding: utf-8 -*-
"""Phase 5 pipeline: (1) retrain ML model, (2) grid search for best params."""

import json
import sys
import time
import traceback

def main():
    t0 = time.time()

    # ---- Step 1: Retrain model ----
    print("=" * 60)
    print("STEP 1: Retraining ML model (Phase 5 enhanced features)")
    print("=" * 60)

    from config import DATA_DIR, PREDICTION_FILE, OUTPUT_DIR, ensure_directories
    ensure_directories()

    from ml_strategy import train_and_predict_all
    signal_dict = train_and_predict_all(DATA_DIR, PREDICTION_FILE, n_jobs=-1)

    t1 = time.time()
    print(f"\nTraining complete in {t1 - t0:.0f}s")
    print(f"Tickers: {len(signal_dict)}")

    # Quick prob distribution check
    import numpy as np
    all_probs = []
    for _, df in signal_dict.items():
        arr = df["pred_prob"].values
        all_probs.append(arr[arr > 0])
    combined = np.concatenate(all_probs)
    print(f"Prob dist: count={len(combined)}, mean={combined.mean():.4f}, "
          f"median={np.median(combined):.4f}, p75={np.percentile(combined, 75):.4f}, "
          f"p90={np.percentile(combined, 90):.4f}, p95={np.percentile(combined, 95):.4f}")

    # ---- Step 2: Grid Search ----
    print("\n" + "=" * 60)
    print("STEP 2: Grid search for optimal parameters")
    print("=" * 60)

    from backtest_engine import run_backtest

    top_k_list = [1, 2, 3, 5]
    # Adaptive thresholds based on actual probability distribution
    pcts = [50, 60, 70, 75, 80, 85, 90, 93, 95, 97]
    thresh_list = sorted(set([round(np.percentile(combined, p), 2) for p in pcts] + [0.55, 0.60, 0.65, 0.70]))

    print(f"Thresholds to test: {thresh_list}")
    print(f"top_k values: {top_k_list}")
    total = len(top_k_list) * len(thresh_list)

    results = []
    idx = 0
    for tk in top_k_list:
        for thr in thresh_list:
            idx += 1
            print(f"\n[{idx}/{total}] top_k={tk} thresh={thr:.2f} ...", end="", flush=True)
            try:
                res, pyf = run_backtest(signal_dict, top_k=tk, prob_threshold=thr)
                total_ret = res.total_return * 100
                years = 5.14  # approx 2020-01 to 2025-02
                eq_final = 1 + res.total_return
                ann_ret = ((eq_final ** (1.0 / years)) - 1.0) * 100 if eq_final > 0 else -100.0
                sharpe = res.sharpe if res.sharpe is not None else 0.0

                # Calculate holding %
                pf_positions = pyf.get("positions")
                if pf_positions is not None and len(pf_positions) > 0:
                    pos_df = pf_positions
                    if "cash" in pos_df.columns:
                        pos_df = pos_df.drop(columns=["cash"], errors="ignore")
                    holding = (pos_df.abs().sum(axis=1) > 0).mean() * 100
                else:
                    holding = 0.0

                results.append({
                    "top_k": tk,
                    "threshold": round(thr, 3),
                    "total_return": round(total_ret, 2),
                    "ann_return": round(ann_ret, 2),
                    "sharpe": round(sharpe, 3),
                    "max_dd": round(res.max_drawdown, 2),
                    "hold_pct": round(holding, 1),
                })
                print(f" ret={total_ret:+.2f}%, sharpe={sharpe:.3f}, dd={res.max_drawdown:.1f}%", flush=True)
            except Exception as e:
                print(f" ERROR: {e}")
                traceback.print_exc()

    # Sort by total return desc
    results.sort(key=lambda x: x["total_return"], reverse=True)

    outfile = OUTPUT_DIR / "grid_search_v2.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 RESULTS")
    print("=" * 60)
    print(f"{'Rank':>4} {'top_k':>5} {'thresh':>7} {'Return':>8} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>6} {'Hold%':>6}")
    print("-" * 55)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} {r['top_k']:>5} {r['threshold']:>7.3f} "
              f"{r['total_return']:>+7.1f}% {r['ann_return']:>+6.1f}% "
              f"{r['sharpe']:>7.3f} {r['max_dd']:>5.1f}% {r['hold_pct']:>5.1f}%")

    t2 = time.time()
    print(f"\nTotal time: {t2 - t0:.0f}s")
    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()
