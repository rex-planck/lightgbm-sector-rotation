"""
Tushare API 配置
"""

# API Token
TUSHARE_TOKEN = "02a9f2204bcbed25321fed5f92d4e708a2294ce68afe61f1f4034e20"

# 数据时间范围
START_DATE = "20230101"
END_DATE = "20260206"

# 低估值蓝筹策略筛选条件
BLUE_CHIP_FILTER = {
    "max_pe": 20,              # 市盈率上限
    "max_pb": 2,               # 市净率上限
    "min_market_cap": 50,      # 最小市值(亿)
    "min_profit_growth": 10,   # 最小净利润增长率(%)
    "min_roe": 10,             # 最小ROE(%)
}

# 数据存储路径
DATA_DIR = "multi_factor_stock_selection/data"
RESULTS_DIR = "multi_factor_stock_selection/results"
