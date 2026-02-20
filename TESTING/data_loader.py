"""Tushare 数据下载与清洗。"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import tushare as ts

try:
    tqdm = import_module("tqdm").tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from config import DOWNLOAD_START_DATE, END_DATE

logger = logging.getLogger(__name__)


def _merge_and_adjust(daily_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    """合并并计算前复权价格序列（qfq）。"""
    if daily_df.empty or adj_df.empty:
        return pd.DataFrame()

    daily_df = daily_df.copy()
    adj_df = adj_df.copy()

    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"], format="%Y%m%d")
    adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d")

    merged = pd.merge(
        daily_df,
        adj_df[["trade_date", "adj_factor"]],
        on="trade_date",
        how="left",
        validate="one_to_one",
    ).sort_values("trade_date")

    merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
    if merged["adj_factor"].isna().all():
        return pd.DataFrame()

    last_factor = float(merged["adj_factor"].iloc[-1])
    if last_factor == 0:
        return pd.DataFrame()

    qfq_factor = merged["adj_factor"] / last_factor

    for col in ["open", "high", "low", "close"]:
        merged[col] = merged[col] * qfq_factor

    out = merged[["trade_date", "open", "high", "low", "close", "vol", "amount"]].copy()
    out = out.rename(columns={"trade_date": "date"})
    out = out.sort_values("date").set_index("date")
    out = out[~out.index.duplicated(keep="first")]
    out = out.dropna(how="any")
    return out


def fetch_single_ticker(
    pro: Any,
    ticker: str,
    data_dir: Path,
    start_date: str = DOWNLOAD_START_DATE,
    end_date: str = END_DATE,
) -> Optional[Path]:
    """下载单只股票并保存 CSV。"""
    try:
        daily_df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
        adj_df = pro.adj_factor(ts_code=ticker, start_date=start_date, end_date=end_date)

        clean_df = _merge_and_adjust(daily_df=daily_df, adj_df=adj_df)
        if clean_df.empty:
            logger.warning("%s 数据为空或调整失败，已跳过。", ticker)
            return None

        file_path = data_dir / f"{ticker}.csv"
        clean_df.to_csv(file_path, encoding="utf-8")
        return file_path
    except Exception as exc:
        logger.exception("下载 %s 失败: %s", ticker, exc)
        return None


def fetch_and_save_all(
    token: str,
    tickers: Iterable[str],
    data_dir: Path,
    start_date: str = DOWNLOAD_START_DATE,
    end_date: str = END_DATE,
) -> list[Path]:
    """批量下载全市场标的并保存为本地 CSV。"""
    ts.set_token(token)
    pro = ts.pro_api(token)

    data_dir.mkdir(parents=True, exist_ok=True)

    ticker_list = list(tickers)
    success_files: list[Path] = []
    for ticker in tqdm(ticker_list, desc="Downloading A-share data", ncols=100):
        file_path = fetch_single_ticker(
            pro=pro,
            ticker=ticker,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
        )
        if file_path is not None:
            success_files.append(file_path)

    logger.info("数据下载完成: 成功 %d / 总计 %d", len(success_files), len(ticker_list))
    return success_files
