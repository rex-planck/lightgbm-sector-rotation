"""
多因子选股模型 - 主程序入口

项目一：低估值蓝筹策略
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import TushareDataFetcher
from data.storage import DataStorage
from strategies.blue_chip_strategy import BlueChipStrategy
from config.tushare_config import START_DATE, END_DATE


def main():
    """主程序"""
    print("\n" + "=" * 80)
    print("🚀 多因子选股模型 - 项目一：低估值蓝筹策略")
    print("=" * 80)
    print(f"数据时间范围: {START_DATE} - {END_DATE}")
    print("=" * 80)
    
    # 初始化组件
    fetcher = TushareDataFetcher()
    storage = DataStorage()
    strategy = BlueChipStrategy()
    
    # ========== 第一步：获取股票基础信息 ==========
    print("\n" + "-" * 80)
    print("📋 步骤 1/5: 获取股票基础信息")
    print("-" * 80)
    
    stock_basic_file = "stock_basic.csv"
    if storage.check_data_exists(stock_basic_file):
        stock_basic = storage.load_dataframe(stock_basic_file)
    else:
        stock_basic = fetcher.get_stock_basic()
        if stock_basic is not None:
            storage.save_dataframe(stock_basic, stock_basic_file)
    
    # ========== 第二步：获取每日指标数据 ==========
    print("\n" + "-" * 80)
    print("📊 步骤 2/5: 获取每日指标数据 (PE, PB, 市值等)")
    print("-" * 80)
    
    daily_file = "daily_basic_monthly.csv"
    if storage.check_data_exists(daily_file):
        print(f"发现已保存的数据文件: {daily_file}")
        choice = input("是否使用已有数据? (y/n, 默认y): ").strip().lower()
        if choice != 'n':
            df_daily = storage.load_dataframe(daily_file)
        else:
            df_daily = None
    else:
        df_daily = None
    
    if df_daily is None:
        print("\n🔄 开始从 Tushare 获取数据...")
        df_daily = fetcher.get_daily_basic_monthly(START_DATE, END_DATE)
        if df_daily is not None:
            storage.save_dataframe(df_daily, daily_file)
    
    if df_daily is None or df_daily.empty:
        print("❌ 获取每日指标数据失败")
        return
    
    print(f"\n✅ 每日指标数据: {len(df_daily)} 条记录")
    print(f"  日期范围: {df_daily['trade_date'].min()} - {df_daily['trade_date'].max()}")
    print(f"  股票数: {df_daily['ts_code'].nunique()}")
    
    # ========== 第三步：使用最新数据进行初步筛选 ==========
    print("\n" + "-" * 80)
    print("🔍 步骤 3/5: 初步筛选 (PE, PB, 市值)")
    print("-" * 80)
    
    latest_date = df_daily['trade_date'].max()
    print(f"\n使用最新交易日数据: {latest_date}")
    
    latest_daily = df_daily[df_daily['trade_date'] == latest_date].copy()
    print(f"当日股票数量: {len(latest_daily)}")
    
    # 初步筛选出低估值、大市值股票
    df_prep = latest_daily.copy()
    df_prep['total_mv'] = df_prep['total_mv'] / 10000  # 转为亿
    
    # 基础筛选
    df_prep = df_prep[(df_prep['pe'] > 0) & (df_prep['pb'] > 0)]
    df_prep = df_prep[(df_prep['pe'] < strategy.filters['max_pe']) & 
                      (df_prep['pb'] < strategy.filters['max_pb'])]
    df_prep = df_prep[df_prep['total_mv'] > strategy.filters['min_market_cap']]
    
    print(f"\n初步筛选后: {len(df_prep)} 只股票")
    print("(PE < 20, PB < 2, 市值 > 50亿)")
    
    # ========== 第四步：获取财务数据 ==========
    print("\n" + "-" * 80)
    print("📈 步骤 4/5: 获取财务数据 (ROE、净利润增长等)")
    print("-" * 80)
    
    # 获取初步筛选后股票的财务数据
    candidate_codes = df_prep['ts_code'].tolist()
    
    fina_data_file = "fina_data_latest.csv"
    income_data_file = "income_data_latest.csv"
    
    if storage.check_data_exists(fina_data_file) and storage.check_data_exists(income_data_file):
        print(f"发现已保存的财务数据")
        choice = input("是否使用已有数据? (y/n, 默认y): ").strip().lower()
        if choice != 'n':
            df_fina = storage.load_dataframe(fina_data_file)
            df_income = storage.load_dataframe(income_data_file)
        else:
            df_fina = None
            df_income = None
    else:
        df_fina = None
        df_income = None
    
    if df_fina is None or df_income is None:
        print(f"\n将为 {len(candidate_codes)} 只候选股票获取财务数据")
        print("(非VIP用户限制，最多获取前200只股票的数据)")
        
        fina_results = fetcher.get_latest_fina_data_for_stocks(candidate_codes, max_stocks=200)
        
        df_income = fina_results.get('income')
        df_fina = fina_results.get('fina')
        
        if df_fina is not None:
            storage.save_dataframe(df_fina, fina_data_file)
        if df_income is not None:
            storage.save_dataframe(df_income, income_data_file)
    
    if df_fina is None or df_fina.empty:
        print("⚠️ 财务指标数据获取失败，将跳过 ROE 筛选")
        df_fina = None
    else:
        print(f"\n✅ 财务指标数据: {len(df_fina)} 条记录")
        
    if df_income is None or df_income.empty:
        print("⚠️ 利润表数据获取失败，将跳过净利润增长率筛选")
        df_income = None
    else:
        print(f"✅ 利润表数据: {len(df_income)} 条记录")
    
    # ========== 第五步：执行完整筛选策略 ==========
    print("\n" + "-" * 80)
    print("🔍 步骤 5/5: 执行完整低估值蓝筹筛选")
    print("-" * 80)
    
    # 执行完整筛选
    result = strategy.filter_stocks(latest_daily, df_income, df_fina, stock_basic)
    
    # 生成报告
    report = strategy.generate_report(result, top_n=30)
    print(report)
    
    # 保存结果
    if result is not None and not result.empty:
        strategy.save_results(result, f"blue_chip_results_{latest_date}.csv")
        
        # 保存完整报告
        report_file = os.path.join(storage.results_dir, f"blue_chip_report_{latest_date}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ 程序执行完成!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
