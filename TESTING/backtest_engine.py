# -*- coding: utf-8 -*-
"""Backtrader backtest engine (Phase 4 - robust version).

Key design:
  - Uses cheat-on-close (COC) so orders execute at the SAME bar's close price.
    This eliminates pending-order conflicts that caused accidental short positions.
  - Slight realism trade-off (we assume we can trade at the close) but avoids
    the cross-bar order overlap bug entirely.
  - A-share commission: buy 0.03%, sell 0.03% + 0.1% stamp duty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict

import backtrader as bt
import pandas as pd

from config import BACKTEST_START_DATE, END_DATE, ML_PARAMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom data feed with pred_prob line
# ---------------------------------------------------------------------------

class MLData(bt.feeds.PandasData):
    lines = ("pred_prob",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "vol"),
        ("openinterest", -1),
        ("pred_prob", "pred_prob"),
    )


# ---------------------------------------------------------------------------
# A-share commission model
# ---------------------------------------------------------------------------

class AShareCommission(bt.CommInfoBase):
    params = (
        ("commission", 0.0003),
        ("stamp_duty", 0.001),
    )

    def _getcommission(self, size, price, pseudoexec):
        trade_value = abs(size) * price
        comm = trade_value * self.p.commission
        if size < 0:
            comm += trade_value * self.p.stamp_duty
        return comm


# ---------------------------------------------------------------------------
# Phase 4 Strategy
# ---------------------------------------------------------------------------

class Phase4Strategy(bt.Strategy):
    """
    Phase 5 core logic:
      1) Absolute prob threshold: all pred_prob < threshold -> cash
      2) Top-K ranking: concentrate on highest confidence picks
      3) Dynamic weight: per_stock = 0.95 / n_targets
      4) Fixed stop-loss at 8%
      5) Minimum holding period (min_hold_days)

    With COC enabled, orders execute at current bar's close.
    """

    params = (
        ("top_k", ML_PARAMS.top_k),
        ("prob_threshold", ML_PARAMS.prob_threshold),
        ("stop_loss", ML_PARAMS.stop_loss_pct),
        ("trailing_stop", ML_PARAMS.trailing_stop_pct),
        ("min_hold_days", getattr(ML_PARAMS, "min_hold_days", 3)),
    )

    def __init__(self):
        self.trade_days = 0
        self.entry_bar = {}  # ticker -> bar_number when entered

    def prenext(self):
        self.next()

    def next(self):
        self.trade_days += 1

        # === 1. Stop-loss scan (ignores min hold) ===
        stopped_names = set()
        for d in self.datas:
            if len(d) == 0:
                continue
            pos = self.getposition(d)
            if pos.size <= 0 or pos.price <= 0:
                continue
            price = float(d.close[0])
            cost = pos.price

            if price < cost * (1.0 - self.p.stop_loss):
                self.close(d)
                stopped_names.add(d._name)
                self.entry_bar.pop(d._name, None)
                continue

        # === 2. Collect today's candidates ===
        candidates = []
        for d in self.datas:
            if len(d) == 0:
                continue
            if d._name in stopped_names:
                continue
            prob = float(d.pred_prob[0])
            if math.isnan(prob):
                prob = 0.0
            if prob >= self.p.prob_threshold:
                candidates.append((d, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # === 3. Top K ===
        top_list = candidates[: self.p.top_k]
        target_set = {d._name for d, _ in top_list}
        n_targets = len(top_list)
        weight = 0.95 / n_targets if n_targets > 0 else 0.0

        # === 4. Debug log (first 5 days) ===
        if self.trade_days <= 5:
            dt_str = str(self.datas[0].datetime.date(0))
            if n_targets > 0:
                names = [d._name for d, _ in top_list]
                probs = [f"{p:.3f}" for _, p in top_list]
                print(f"[{dt_str}] Top-{n_targets}: {names} prob:{probs} w={weight:.2%}")
            else:
                print(f"[{dt_str}] Cash - no candidates above {self.p.prob_threshold}")

        # === 5. Close positions not in top-K (respect min_hold) ===
        for d in self.datas:
            if d._name in stopped_names:
                continue
            pos = self.getposition(d)
            if pos.size > 0 and d._name not in target_set:
                # Check minimum holding period
                entry = self.entry_bar.get(d._name, 0)
                held = self.trade_days - entry
                if held >= self.p.min_hold_days:
                    self.close(d)
                    self.entry_bar.pop(d._name, None)

        # === 6. Open / rebalance ===
        for d, _ in top_list:
            if d._name in stopped_names:
                continue
            pos = self.getposition(d)
            if pos.size <= 0:
                # New position - record entry bar
                self.entry_bar[d._name] = self.trade_days
            self.order_target_percent(d, target=weight)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    start_value: float
    final_value: float
    total_return: float
    sharpe: float | None
    max_drawdown: float | None


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_backtest(
    signal_dict: Dict[str, pd.DataFrame],
    initial_cash: float = 10_000_000.0,
    top_k: int | None = None,
    prob_threshold: float | None = None,
    trailing_stop_pct: float | None = None,
) -> tuple[BacktestResult, dict]:
    if not signal_dict:
        raise ValueError("signal_dict is empty")

    strategy_top_k = ML_PARAMS.top_k if top_k is None else int(top_k)
    strategy_threshold = (
        ML_PARAMS.prob_threshold if prob_threshold is None else float(prob_threshold)
    )
    strategy_trailing = (
        ML_PARAMS.trailing_stop_pct
        if trailing_stop_pct is None
        else float(trailing_stop_pct)
    )

    # ---- Probability distribution stats ----
    import numpy as _np
    all_probs = []
    for _, df in signal_dict.items():
        if "pred_prob" in df.columns:
            arr = _np.asarray(df["pred_prob"].values, dtype=float)
            all_probs.append(arr[arr > 0])
    if all_probs:
        combined = _np.concatenate(all_probs)
        print(
            f"DEBUG prob dist: count={len(combined)}, "
            f"mean={combined.mean():.4f}, "
            f"median={float(_np.median(combined)):.4f}, "
            f"p75={float(_np.percentile(combined, 75)):.4f}, "
            f"p90={float(_np.percentile(combined, 90)):.4f}, "
            f">={strategy_threshold}: "
            f"{int((combined >= strategy_threshold).sum())}"
        )

    # ---- Cerebro setup ----
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.addcommissioninfo(AShareCommission())
    cerebro.broker.set_coc(True)  # Cheat-on-close: execute at same bar close

    bt_start = pd.Timestamp(BACKTEST_START_DATE)
    bt_end = pd.Timestamp(END_DATE)
    print(f"Loading data... (backtest range: {BACKTEST_START_DATE} ~ {END_DATE})")

    added_feeds = 0
    for ticker, df in signal_dict.items():
        if df.empty:
            continue
        bt_df = df.copy()
        if not isinstance(bt_df.index, pd.DatetimeIndex):
            bt_df.index = pd.to_datetime(bt_df.index)
        bt_df = bt_df.sort_index()
        bt_df = bt_df.loc[(bt_df.index >= bt_start) & (bt_df.index <= bt_end)]
        if bt_df.empty:
            continue
        required = ["open", "high", "low", "close", "vol", "pred_prob"]
        if any(col not in bt_df.columns for col in required):
            logger.warning("%s missing required columns, skipping", ticker)
            continue
        feed = MLData(dataname=bt_df, name=ticker)
        cerebro.adddata(feed)
        added_feeds += 1

    if added_feeds == 0:
        raise ValueError("No usable data feeds")

    cerebro.addstrategy(
        Phase4Strategy,
        top_k=strategy_top_k,
        prob_threshold=strategy_threshold,
        trailing_stop=strategy_trailing,
    )
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    print(
        f"Running backtest... ({added_feeds} stocks, "
        f"Top-K={strategy_top_k}, prob_threshold={strategy_threshold})"
    )
    start_value = cerebro.broker.getvalue()
    strat = cerebro.run(maxcpus=1)[0]
    final_value = cerebro.broker.getvalue()

    sharpe_res = strat.analyzers.sharpe.get_analysis()
    drawdown_res = strat.analyzers.drawdown.get_analysis()

    pf_returns, pf_positions, pf_transactions, pf_gross_lev = (
        strat.analyzers.pyfolio.get_pf_items()
    )
    pyfolio_res = {
        "returns": pf_returns,
        "positions": pf_positions,
        "transactions": pf_transactions,
        "gross_lev": pf_gross_lev,
    }

    sharpe = sharpe_res.get("sharperatio") if isinstance(sharpe_res, dict) else None
    max_drawdown = 0.0
    if isinstance(drawdown_res, dict):
        max_info = drawdown_res.get("max", {})
        max_drawdown = float(max_info.get("drawdown", 0.0))

    result = BacktestResult(
        start_value=float(start_value),
        final_value=float(final_value),
        total_return=float(final_value / start_value - 1.0),
        sharpe=None if sharpe is None else float(sharpe),
        max_drawdown=float(max_drawdown),
    )
    return result, pyfolio_res
