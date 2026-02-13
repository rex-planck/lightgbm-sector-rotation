import akshare as ak
import pandas as pd

def find_api(keywords):
    """在本地 AKShare 库中搜索包含指定关键词的函数"""
    all_apis = dir(ak)
    matches = []
    for api in all_apis:
        # 排除私有函数
        if api.startswith("_"):
            continue
        # 检查是否包含任一关键词
        if any(k in api.lower() for k in keywords):
            matches.append(api)
    return matches

print(f"当前 AKShare 版本: {ak.__version__}")
print("-" * 30)

# 1. 核心：宏观数据 (用于宏观状态识别)
macro_apis = find_api(['macro', 'cpi', 'ppi', 'gdp'])
print(f"【宏观数据】候选接口 ({len(macro_apis)}):")
# 重点关注: macro_china_cpi_yearly, macro_china_ppi_yearly
print(macro_apis[:5])  # 展示前5个

# 2. 核心：北向资金 (用于另类数据 Alpha)
north_apis = find_api(['hsgt', 'north', 'money', 'flow'])
print(f"\n【北向/资金】候选接口 ({len(north_apis)}):")
# 重点关注: stock_hsgt_north_net_flow_in_em
print(north_apis[:5])

# 3. 核心：行业/指数 (用于行业轮动)
index_apis = find_api(['industry', 'sector', 'plate', 'board'])
print(f"\n【行业/板块】候选接口 ({len(index_apis)}):")
# 重点关注: stock_board_industry_name_em (行业列表), stock_board_industry_hist_em (行业行情)
print(index_apis[:5])

# 4. 核心：A股指数 (用于 Benchmark)
# 重点关注: index_zh_a_hist (替代了旧版的不稳定接口)
benchmark_apis = find_api(['index_zh_a_hist'])
print(f"\n【指数行情】候选接口:")
print(benchmark_apis)