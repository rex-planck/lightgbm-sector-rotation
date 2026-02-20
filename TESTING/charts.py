# -*- coding: utf-8 -*-
"""Phase 4 -- Backtest Chart Generator.

Generates 4 charts:
  1. phase4_report.png   (combined: NAV + drawdown + monthly heatmap)
  2. equity_curve.png    (NAV curve)
  3. drawdown.png        (underwater curve)
  4. monthly_heatmap.png (monthly returns heatmap)

All print / title / annotations avoid non-GBK characters (e.g. emoji)
to ensure correct output on Windows Chinese consoles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

COLORS = {
    "equity": "#1a5fb4",
    "benchmark": "#aaaaaa",
    "drawdown": "#c01c28",
    "positive": "#26a269",
    "negative": "#c01c28",
    "grid": "#e8e8e8",
    "bg": "#fafafa",
    "fill_up": "#d4edda",
    "fill_down": "#f8d7da",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns_series(pyfolio_data: dict) -> pd.Series:
    returns = pyfolio_data.get("returns")
    if returns is None:
        raise ValueError("pyfolio_data missing 'returns'")
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns.index = pd.to_datetime(returns.index)
    return returns


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def _calc_calmar(ann_return_pct: float, max_dd_pct: float) -> float | None:
    if max_dd_pct is None or max_dd_pct <= 0:
        return None
    return ann_return_pct / max_dd_pct


def _holding_ratio(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    active = (returns.abs() > 1e-8).sum()
    return float(active / len(returns))


# ---------------------------------------------------------------------------
# Chart 1 -- NAV Curve (enhanced)
# ---------------------------------------------------------------------------

def plot_equity_curve(
    returns: pd.Series,
    result,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    equity = _equity_curve(returns)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    ax.set_facecolor(COLORS["bg"])

    # Fill profit / loss zones
    ax.fill_between(
        equity.index, 1.0, equity.values,
        where=equity.values >= 1.0,
        color=COLORS["fill_up"], alpha=0.5, interpolate=True,
    )
    ax.fill_between(
        equity.index, 1.0, equity.values,
        where=equity.values < 1.0,
        color=COLORS["fill_down"], alpha=0.5, interpolate=True,
    )

    ax.plot(equity.index, equity.values, color=COLORS["equity"], linewidth=1.6, label="Strategy NAV")

    # Key metrics panel
    ret_pct = (equity.iloc[-1] - 1.0) * 100
    n_days = len(equity)
    eq_final = float(equity.iloc[-1])
    if eq_final > 0 and n_days > 0:
        ann_ret = (eq_final ** (252 / n_days) - 1) * 100
    else:
        ann_ret = -100.0
    sharpe_str = f"{result.sharpe:.3f}" if result.sharpe else "N/A"
    mdd_val = result.max_drawdown if result.max_drawdown else 0.0
    mdd_str = f"{mdd_val:.2f}%"
    calmar = _calc_calmar(ann_ret, mdd_val)
    calmar_str = f"{calmar:.2f}" if calmar else "N/A"
    hold_pct = _holding_ratio(returns) * 100

    info_text = (
        f"Total Return: {ret_pct:+.2f}%\n"
        f"Annualized:   {ann_ret:+.2f}%\n"
        f"Sharpe:       {sharpe_str}\n"
        f"Max DD:       {mdd_str}\n"
        f"Calmar:       {calmar_str}\n"
        f"Holding:      {hold_pct:.1f}%"
    )
    ax.text(
        0.02, 0.97, info_text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.90,
                  edgecolor="#bbbbbb", linewidth=0.8),
    )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Phase 4 | Strategy NAV (top_k=3, prob>=0.91, no trailing stop)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return ax


# ---------------------------------------------------------------------------
# Chart 2 -- Underwater Curve
# ---------------------------------------------------------------------------

def plot_drawdown(
    returns: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    equity = _equity_curve(returns)
    dd = _drawdown_series(equity) * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.set_facecolor(COLORS["bg"])
    ax.fill_between(dd.index, dd.values, 0, color=COLORS["drawdown"], alpha=0.30)
    ax.plot(dd.index, dd.values, color=COLORS["drawdown"], linewidth=0.9)

    # Annotate max drawdown
    min_idx = dd.idxmin()
    min_val = dd.min()
    ax.annotate(
        f"{min_val:.1f}%",
        xy=(min_idx, min_val),
        xytext=(min_idx, min_val - 2),
        fontsize=8, color=COLORS["drawdown"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["drawdown"], lw=0.8),
        ha="center",
    )

    ax.set_title("Underwater Curve (Drawdown)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    return ax


# ---------------------------------------------------------------------------
# Chart 3 -- Monthly Returns Heatmap
# ---------------------------------------------------------------------------

def plot_monthly_heatmap(
    returns: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
    # Use Chinese month labels with safe chars
    month_names = {1: "1\u6708", 2: "2\u6708", 3: "3\u6708", 4: "4\u6708",
                   5: "5\u6708", 6: "6\u6708", 7: "7\u6708", 8: "8\u6708",
                   9: "9\u6708", 10: "10\u6708", 11: "11\u6708", 12: "12\u6708"}
    pivot.columns = [month_names.get(m, str(m)) for m in pivot.columns]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Monthly Return (%)"},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax


# ---------------------------------------------------------------------------
# Generate all
# ---------------------------------------------------------------------------

def generate_all_charts(
    pyfolio_data: dict,
    result,
    output_dir: str | Path = "output",
) -> None:
    """Generate all charts and save as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = _returns_series(pyfolio_data)

    # --- Combined figure (3-in-1) ---
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1.3], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    plot_equity_curve(returns, result, ax=ax1)
    plot_drawdown(returns, ax=ax2)
    plot_monthly_heatmap(returns, ax=ax3)

    fig.suptitle(
        "Phase 4 Backtest Report | LightGBM Top-K Rotation Strategy",
        fontsize=15, fontweight="bold", y=0.98,
    )
    combo_path = output_dir / "phase4_report.png"
    fig.savefig(combo_path)
    plt.close(fig)
    logger.info("Combined chart saved: %s", combo_path)

    # --- Individual NAV ---
    fig2, ax = plt.subplots(figsize=(14, 5))
    plot_equity_curve(returns, result, ax=ax)
    eq_path = output_dir / "equity_curve.png"
    fig2.savefig(eq_path)
    plt.close(fig2)

    # --- Individual drawdown ---
    fig3, ax = plt.subplots(figsize=(14, 3))
    plot_drawdown(returns, ax=ax)
    dd_path = output_dir / "drawdown.png"
    fig3.savefig(dd_path)
    plt.close(fig3)

    # --- Individual heatmap ---
    fig4, ax = plt.subplots(figsize=(12, 4))
    plot_monthly_heatmap(returns, ax=ax)
    hm_path = output_dir / "monthly_heatmap.png"
    fig4.savefig(hm_path)
    plt.close(fig4)

    # GBK-safe output (no emoji, no special unicode)
    print("")
    print("[Charts] Saved to %s:" % output_dir.resolve())
    print("   - %s  (combined report)" % combo_path.name)
    print("   - %s" % eq_path.name)
    print("   - %s" % dd_path.name)
    print("   - %s" % hm_path.name)
