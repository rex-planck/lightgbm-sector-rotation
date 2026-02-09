"""
终极优化版配置 V3.0
整合所有优化方向：数据、模型、策略、风控
"""
import os

# ==================== Tushare 配置 ====================
TUSHARE_TOKEN = "02a9f2204bcbed25321fed5f92d4e708a2294ce68afe61f1f4034e20"

# ==================== 数据配置 (3年+) ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH_V3 = os.path.join(DATA_DIR, "stock_data.db")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_v3")
for subdir in ['factors', 'models', 'backtest', 'reports']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# 扩展股票池：沪深300 + 中证500 + 中证1000
INDEX_CODES_V3 = {
    'CSI300': '000300.SH',
    'CSI500': '000905.SH',
    'CSI1000': '000852.SH',
}

# 3年历史数据
START_DATE_V3 = "20220101"  # 2022年开始
END_DATE_V3 = "20260209"    # 到最新

# 训练/验证/测试划分
def split_dates():
    # 2022-2024年训练，2025年验证，2026年测试
    # Adjusting based on actual data range 20240621-20260206
    # Train: 20240621 - 20250630
    # Valid: 20250701 - 20251101
    # Test: 20251102 - 20260206
    return {
        'train_end': '20250630',
        'valid_end': '20251101',
    }

# ==================== GP挖掘配置 (保持优化) ====================
GP_CONFIG_V3 = {
    "population_size": 2000,      # 进一步提升
    "generations": 30,
    "hall_of_fame": 50,
    "n_components": 50,
    "function_set": ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan'],
    "parsimony_coefficient": 0.0005,
    "max_samples": 0.9,
    "random_state": 42,
}

# ==================== Transformer + 对抗训练配置 ====================
TRANSFORMER_CONFIG_V3 = {
    "d_model": 256,           # 提升维度
    "nhead": 8,
    "num_layers": 6,          # 更深网络
    "dim_feedforward": 1024,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "batch_size": 256,
    "epochs": 100,
    "seq_len": 30,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15,
}

# 对抗训练配置
ADVERSARIAL_CONFIG = {
    "enabled": True,
    "epsilon": 0.01,          # 扰动大小
    "alpha": 0.5,             # 对抗损失权重
    "pgd_steps": 3,           # PGD步数
}

# ==================== 市场状态判断配置 ====================
MARKET_REGIME_CONFIG = {
    "method": "hmm",          # HMM隐马尔可夫模型
    "n_regimes": 3,           # 牛/熊/震荡
    "features": ['index_return', 'volatility', 'volume'],
    "retrain_freq": 60,       # 每60天重新训练
}

# ==================== 行业中性配置 ====================
SECTOR_NEUTRAL_CONFIG = {
    "enabled": True,
    "max_sector_deviation": 0.05,  # 行业偏离基准最多5%
    "sector_col": "industry",       # 行业列名
}

# ==================== 动态风控配置 ====================
RISK_CONFIG_V3 = {
    "initial_capital": 10000000,
    "commission": 0.001,
    "slippage": 0.002,
    
    # 动态仓位管理
    "position_sizing": "kelly",     # Kelly公式 / "equal" / "risk_parity"
    "max_position_weight": 0.08,    # 单票最大8%
    "min_position_weight": 0.01,    # 单票最小1%
    
    # 自适应止损
    "stop_loss_method": "atr",      # ATR动态止损 / "fixed"
    "stop_loss_atr_mult": 2.0,      # 2倍ATR止损
    "trailing_stop": True,          # 追踪止损
    "trailing_stop_pct": 0.10,      # 10%回撤止盈
    
    # 风险预算
    "max_portfolio_var": 0.02,      # 组合日波动不超过2%
    "max_drawdown_limit": 0.15,     # 15%最大回撤限制
    "var_lookback": 20,             # VaR计算回看窗口
}

# ==================== 特征配置 (扩展) ====================
FEATURE_CONFIG_V3 = {
    "price_volume": ["open", "high", "low", "close", "vol", "amount"],
    "returns": ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d"],
    "volatility": ["volatility_20d", "volatility_60d", "atr_14"],
    "technical": ["rsi_6", "rsi_14", "rsi_28", "macd", "macd_signal", 
                  "bb_upper", "bb_lower", "bb_position", 
                  "kdj_k", "kdj_d", "kdj_j"],
    "volume_ind": ["volume_ratio", "volume_ma5", "volume_ma20", "obv"],
    "fundamental": ["pe", "pb", "ps", "turnover_rate", "free_float"],
    "market": ["index_return", "index_volatility", "market_sentiment"],
}

print(f"[Config V3] Loaded. Output dir: {OUTPUT_DIR}")
