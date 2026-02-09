"""
项目配置文件
"""
import os

# ==================== Tushare 配置 ====================
TUSHARE_TOKEN = "02a9f2204bcbed25321fed5f92d4e708a2294ce68afe61f1f4034e20"  # 2000积分账户

# ==================== 数据配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据存储路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FACTOR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "factors")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")

# 自动创建目录
for dir_path in [DATA_DIR, OUTPUT_DIR, FACTOR_OUTPUT_DIR, MODEL_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== 股票池配置 ====================
# 指数成分股（避免幸存者偏差）
INDEX_CODE = "000300.SH"  # 沪深300
# INDEX_CODE = "000905.SH"  # 中证500
# INDEX_CODE = "000852.SH"  # 中证1000

# 时间范围
START_DATE = "20230101"
END_DATE = "20260209"

# 训练/验证/测试集划分
TRAIN_END = "20211231"
VALID_END = "20221231"

# ==================== 因子挖掘配置 ====================
GP_CONFIG = {
    "population_size": 500,       # 种群大小（减小以加速）
    "generations": 15,            # 进化代数（减小以加速）
    "tournament_size": 20,        # 锦标赛大小
    "stopping_criteria": 0.01,    # 停止条件
    "p_crossover": 0.7,           # 交叉概率
    "p_subtree_mutation": 0.1,    # 子树突变概率
    "p_hoist_mutation": 0.05,     # 提升突变概率
    "p_point_mutation": 0.1,      # 点突变概率
    "max_samples": 0.8,           # 每次训练样本比例
    "parsimony_coefficient": 0.001,  # 复杂度惩罚系数
    "function_set": [             # 函数集
        "add", "sub", "mul", "div", 
        "sqrt", "log", "abs", "neg", 
        "inv", "max", "min"
    ],
    "n_factors": 30,              # 目标因子数量（减小以加速）
    "min_ic_threshold": 0.02,     # 最小 IC 阈值（降低以获得更多因子）
    "max_correlation": 0.7,       # 因子间最大相关性（放宽）
}

# ==================== GRU 模型配置 ====================
GRU_CONFIG = {
    "input_size": None,  # 动态设置（GP挖掘出的因子数）
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.0005,
    "batch_size": 1024,
    "epochs": 10,
    "step_len": 20,  # 滑动窗口长度
    "early_stopping_patience": 3,
}

# ==================== 特征字段定义 ====================
# 基础量价特征（输入 GP）
BASE_FEATURES = {
    "ohlcv": ["open", "high", "low", "close", "vol"],
    "indicators": ["turnover_rate", "pe", "pb"],  # 需要 daily_basic 接口
}

# 标签字段
LABEL_COL = "label"
# 预测未来 N 天的收益率
FORECAST_HORIZON = 5
