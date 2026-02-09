"""
优化版项目配置文件
基于用户反馈的全面优化
"""
import os

# ==================== Tushare 配置 ====================
TUSHARE_TOKEN = "02a9f2204bcbed25321fed5f92d4e708a2294ce68afe61f1f4034e20"

# ==================== 数据配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_optimized")
FACTOR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "factors")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")

for dir_path in [DATA_DIR, OUTPUT_DIR, FACTOR_OUTPUT_DIR, MODEL_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== 扩展股票池配置 ====================
INDEX_CODES = {
    'CSI300': '000300.SH',    # 沪深300
    'CSI500': '000905.SH',    # 中证500
    'CSI1000': '000852.SH',   # 中证1000
}

# 时间范围：3年+
START_DATE = "20220101"
END_DATE = "20260209"

# 训练/验证/测试集划分 (6:2:2)
TRAIN_END = "20241231"
VALID_END = "20250131"

# ==================== 优化版 GP 配置 ====================
GP_CONFIG_OPTIMIZED = {
    "population_size": 1500,      # 提升至1500
    "generations": 30,            # 提升至30代
    "tournament_size": 30,        # 锦标赛大小
    "stopping_criteria": 0.005,   # 更严格的停止条件
    "p_crossover": 0.8,           # 提高交叉概率
    "p_subtree_mutation": 0.15,   # 提高变异概率
    "p_hoist_mutation": 0.05,
    "p_point_mutation": 0.1,
    "max_samples": 0.9,
    "parsimony_coefficient": 0.0005,  # 降低惩罚，允许更复杂因子
    "function_set": [
        "add", "sub", "mul", "div", 
        "sqrt", "log", "abs", "neg", 
        "inv", "max", "min", "sin", "cos"
    ],
    "n_factors": 50,              # 目标50个高质量因子
    "min_ic_threshold": 0.025,    # 最低IC阈值
    "max_correlation": 0.6,       # 最大相关性
}

# ==================== 优化版 Transformer 配置 ====================
TRANSFORMER_CONFIG = {
    "input_size": None,           # 动态设置
    "d_model": 128,               # 模型维度
    "nhead": 8,                   # 注意力头数
    "num_layers": 4,              # 层数
    "dim_feedforward": 512,       # 前馈网络维度
    "dropout": 0.3,               # 增加dropout防止过拟合
    "learning_rate": 0.0001,      # 降低学习率
    "batch_size": 256,
    "epochs": 50,                 # 增加epoch
    "step_len": 30,               # 增加序列长度
    "weight_decay": 1e-4,         # L2正则化
    "early_stopping_patience": 10,
    "lr_scheduler": "cosine",     # 余弦退火
}

# ==================== 集成学习配置 ====================
ENSEMBLE_CONFIG = {
    "n_models": 5,                # 集成5个模型
    "bagging_ratio": 0.8,         # 每次采样80%数据
    "model_types": ["transformer", "gru", "lstm"],
}

# ==================== 回测优化配置 ====================
BACKTEST_CONFIG = {
    "initial_capital": 10000000,
    "commission": 0.001,          # 千分之一手续费
    "slippage": 0.002,            # 千分之二滑点
    "top_n": 30,                  # 持仓30只
    "rebalance_freq": 5,          # 每5天调仓
    "max_position_weight": 0.05,  # 单票最大5%权重
    "stop_loss": 0.08,            # 8%止损线
    "take_profit": 0.15,          # 15%止盈线
    "risk_free_rate": 0.03,       # 无风险利率3%
}

# ==================== 增强特征配置 ====================
FEATURE_GROUPS = {
    "price_volume": ["open", "high", "low", "close", "vol", "amount"],
    "returns": ["ret_1d", "ret_5d", "ret_10d", "ret_20d"],
    "volatility": ["volatility_20d", "volatility_60d"],
    "volume_indicators": ["volume_ratio", "volume_ma5", "volume_ma20"],
    "technical": ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"],
    "fundamental": ["pe", "pb", "turnover_rate"],
}
