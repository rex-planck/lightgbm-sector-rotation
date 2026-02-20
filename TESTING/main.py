"""Full-Market Sector Leader Rolling LightGBM Strategy 入口。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtest_engine import run_backtest
from config import (
    BACKTEST_START_DATE,
    DATA_DIR,
    DOWNLOAD_START_DATE,
    END_DATE,
    OUTPUT_DIR,
    PREDICTION_FILE,
    TARGET_ASSETS,
    TS_TOKEN,
    ensure_directories,
)
from data_loader import fetch_and_save_all
from ml_strategy import train_and_predict_all


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def save_pyfolio_outputs(pyfolio_data: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = pyfolio_data.get("returns")
    positions = pyfolio_data.get("positions")
    transactions = pyfolio_data.get("transactions")
    gross_lev = pyfolio_data.get("gross_lev")

    if isinstance(returns, pd.Series) and not returns.empty:
        returns.to_csv(output_dir / "pyfolio_returns.csv", header=["returns"])
    if isinstance(positions, pd.DataFrame) and not positions.empty:
        positions.to_csv(output_dir / "pyfolio_positions.csv")
    if isinstance(transactions, pd.DataFrame) and not transactions.empty:
        transactions.to_csv(output_dir / "pyfolio_transactions.csv")
    if isinstance(gross_lev, pd.Series) and not gross_lev.empty:
        gross_lev.to_csv(output_dir / "pyfolio_gross_lev.csv", header=["gross_lev"])


def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    ensure_directories()

    existing_csvs = list(DATA_DIR.glob("*.csv"))
    if existing_csvs:
        logger.info("步骤1/3: 检测到 %d 份本地 CSV，跳过下载", len(existing_csvs))
    else:
        logger.info("步骤1/3: 下载并清洗数据")
        success_files = fetch_and_save_all(
            token=TS_TOKEN,
            tickers=TARGET_ASSETS,
            data_dir=DATA_DIR,
            start_date=DOWNLOAD_START_DATE,
            end_date=END_DATE,
        )
        if not success_files:
            raise RuntimeError("未下载到任何可用数据，流程终止")

    logger.info("步骤2/3: 训练滚动 LightGBM 并生成信号")
    signal_dict = train_and_predict_all(
        data_dir=DATA_DIR,
        output_file=PREDICTION_FILE,
        n_jobs=-1,
    )
    if not signal_dict:
        raise RuntimeError("模型未生成任何信号，流程终止")

    logger.info("步骤3/3: 运行回测")
    bt_result, pyfolio_data = run_backtest(signal_dict=signal_dict)

    print("=" * 72)
    print("Full-Market Sector Leader Rolling LightGBM Strategy")
    print("=" * 72)
    print(f"回测区间: {BACKTEST_START_DATE} ~ {END_DATE}")
    print(f"初始资金: {bt_result.start_value:,.2f}")
    print(f"期末资金: {bt_result.final_value:,.2f}")
    print(f"总收益率: {bt_result.total_return:.2%}")
    print(f"Sharpe: {bt_result.sharpe}")
    print(f"最大回撤(%): {bt_result.max_drawdown}")

    save_pyfolio_outputs(pyfolio_data=pyfolio_data, output_dir=OUTPUT_DIR)

    # --- 生成图表 ---
    try:
        from charts import generate_all_charts
        generate_all_charts(pyfolio_data=pyfolio_data, result=bt_result, output_dir=OUTPUT_DIR)
    except Exception as e:
        logger.warning("图表生成失败: %s", e)

    logger.info("结果文件输出目录: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
