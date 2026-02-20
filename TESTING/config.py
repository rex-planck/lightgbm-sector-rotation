"""全局配置模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


TS_TOKEN = "02a9f2204bcbed25321fed5f92d4e708a2294ce68afe61f1f4034e20"

DOWNLOAD_START_DATE = "20160101"
BACKTEST_START_DATE = "20200101"
END_DATE = "20260218"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PREDICTION_FILE = OUTPUT_DIR / "predictions.pkl"

TARGET_ASSETS = [
    "300442.SZ", "601360.SH",  # 互联网
    "002292.SZ", "002489.SZ",  # 休闲用品
    "601628.SH", "601318.SH",  # 保险
    "001965.SZ", "600377.SH",  # 公路
    "002311.SZ", "000876.SZ",  # 农业
    "600276.SH", "688235.SH",  # 制药
    "600030.SH", "601211.SH",  # 券商
    "603899.SH", "002103.SZ",  # 办公用品
    "002831.SZ", "002701.SZ",  # 包装
    "600309.SH", "600989.SH",  # 化工原料
    "002064.SZ", "300699.SZ",  # 化纤
    "000792.SZ", "000408.SZ",  # 化肥农药
    "300760.SZ", "300015.SZ",  # 医疗保健
    "688981.SH", "688256.SH",  # 半导体
    "601727.SH", "600875.SH",  # 发电设备
    "600415.SH", "603300.SH",  # 商业服务
    "603993.SH", "600111.SH",  # 基本金属
    "000617.SZ", "000987.SZ",  # 多元金融
    "603833.SH", "603816.SH",  # 家居用品
    "000333.SZ", "000651.SZ",  # 家用电器
    "002050.SZ", "601100.SH",  # 工业机械
    "600031.SH", "000425.SZ",  # 工程机械
    "600585.SH", "002080.SZ",  # 建材
    "601668.SH", "601800.SH",  # 建筑
    "600048.SH", "001979.SZ",  # 房地产
    "603129.SH", "603529.SH",  # 摩托车
    "002607.SZ", "300010.SZ",  # 教育及其它
    "002027.SZ", "300251.SZ",  # 文化传媒
    "603605.SH", "300957.SZ",  # 日用品
    "600009.SH", "600004.SH",  # 机场
    "601118.SH", "002043.SZ",  # 林木
    "600008.SH", "601158.SH",  # 水务
    "002594.SZ", "600104.SH",  # 汽车
    "600660.SH", "601689.SH",  # 汽车零部件
    "601919.SH", "601872.SH",  # 海运
    "600018.SH", "601018.SH",  # 港口
    "600925.SH", "000937.SZ",  # 煤炭
    "600803.SH", "600956.SH",  # 燃气
    "300779.SZ", "603568.SH",  # 环保
    "603259.SH", "603392.SH",  # 生物科技
    "600941.SH", "601728.SH",  # 电信
    "600900.SH", "600930.SH",  # 电力
    "601138.SH", "300308.SZ",  # 电子元件及设备
    "300750.SZ", "300274.SZ",  # 电工电网
    "603019.SH", "000938.SZ",  # 电脑硬件
    "600346.SH", "002648.SZ",  # 石油化工
    "601857.SH", "600938.SH",  # 石油天然气
    "002709.SZ", "002340.SZ",  # 精细化工
    "300979.SZ", "600177.SH",  # 纺织服装
    "600295.SH", "000009.SZ",  # 综合类
    "601808.SH",  # 中海油服
    "002353.SZ",  # 杰瑞股份
    "302132.SZ", "600760.SH",  # 航天军工
    "002352.SZ", "601111.SH",  # 航空与物流
    "601899.SH", "600547.SH",  # 贵金属
    "601888.SH", "000963.SZ",  # 贸易
    "688111.SH", "002230.SZ",  # 软件
    "605499.SH", "603156.SH",  # 软饮料
    "300502.SZ", "000063.SZ",  # 通信设备
    "002078.SZ", "603733.SH",  # 造纸
    "600519.SH", "000858.SZ",  # 酒类
    "601766.SH", "600150.SH",  # 重型机械
    "600019.SH", "600010.SH",  # 钢铁
    "601398.SH", "601288.SH",  # 银行
    "601816.SH", "601006.SH",  # 陆路运输
    "601933.SH", "000564.SZ",  # 零售
    "002714.SZ", "603288.SH",  # 食品
    "600754.SH", "300144.SZ",  # 餐饮旅游
]


@dataclass(frozen=True)
class MLConfig:
    # --- 训练窗口 ---
    lookback_window: int = 400
    min_train_size: int = 200
    retrain_frequency: int = 20

    # --- LightGBM 参数 (Phase 5 增强版) ---
    n_estimators: int = 300         # more trees, early stopping controls actual count
    learning_rate: float = 0.03     # slower learning for better generalization
    num_leaves: int = 31            # deeper trees for complex patterns
    min_child_samples: int = 20     # more regularization
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1          # L1 regularization
    reg_lambda: float = 1.0         # L2 regularization
    random_state: int = 42
    n_jobs_model: int = 1

    # --- Phase 5 风控 (grid search optimal) ---
    top_k: int = 2
    prob_threshold: float = 0.70    # optimal from grid search (Sharpe=1.756)
    stop_loss_pct: float = 0.08     # wider stop for multi-day holding
    trailing_stop_pct: float = 0.0  # disabled
    min_hold_days: int = 3          # minimum holding period

    feature_columns: List[str] = field(
        default_factory=lambda: [
            # --- Original 10 ---
            "rsi14",
            "roc5",
            "roc20",
            "macd",
            "macd_hist",
            "sma_ratio_5_60",
            "atr14_norm",
            "vol_ratio_20",
            "ret_1",
            "ret_5",
            # --- NEW: Momentum & Trend ---
            "rsi5",
            "roc10",
            "adx14",
            "cci14",
            "willr14",
            "mfi14",
            # --- NEW: Bollinger & MA ---
            "bb_pos",
            "sma_ratio_5_20",
            "price_vs_sma20",
            # --- NEW: Volatility ---
            "atr5_norm",
            "ret_10",
            "ret_20",
            # --- NEW: Volume ---
            "vol_ratio_5",
            "pv_corr_10",
            # --- NEW: Candlestick ---
            "body_ratio",
        ]
    )


ML_PARAMS = MLConfig()

# 兼容旧调用（逐步迁移）
START_DATE = BACKTEST_START_DATE


def ensure_directories() -> None:
    """创建数据与输出目录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
