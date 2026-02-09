"""
快速运行 - 低估值蓝筹策略
使用已有的 daily_basic 数据进行筛选，限制获取少量财务数据
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from data.fetcher import TushareDataFetcher
from data.storage import DataStorage
from strategies.blue_chip_strategy import BlueChipStrategy


def quick_filter():
    """快速筛选 - 基于已有数据"""
    print("\n" + "=" * 80)
    print("🚀 低估值蓝筹策略 - 快速筛选")
    print("=" * 80)
    
    storage = DataStorage()
    strategy = BlueChipStrategy()
    
    # 加载已有数据
    print("\n📂 加载已保存的数据...")
    df_daily = storage.load_dataframe("daily_basic_monthly.csv")
    stock_basic = storage.load_dataframe("stock_basic.csv")
    
    if df_daily is None:
        print("❌ 没有找到 daily_basic 数据，请先运行 main.py 获取数据")
        return
    
    # 使用最新日期数据
    latest_date = df_daily['trade_date'].max()
    print(f"\n📅 使用最新交易日数据: {latest_date}")
    
    latest_daily = df_daily[df_daily['trade_date'] == latest_date].copy()
    print(f"当日股票数量: {len(latest_daily)}")
    
    # 执行筛选（不获取额外财务数据）
    result = strategy.filter_stocks(latest_daily, None, None, stock_basic)
    
    # 生成报告
    report = strategy.generate_report(result, top_n=50)
    print(report)
    
    # 保存结果
    if result is not None and not result.empty:
        strategy.save_results(result, f"blue_chip_quick_{latest_date}.csv")
        
        # 保存报告
        report_file = os.path.join(storage.results_dir, f"blue_chip_quick_report_{latest_date}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ 快速筛选完成!")
    print("=" * 80)
    
    return result


def enhanced_filter(max_fina_stocks=50):
    """
    增强筛选 - 获取部分股票的财务数据
    
    Parameters:
    -----------
    max_fina_stocks : int
        最多获取财务数据的股票数量
    """
    print("\n" + "=" * 80)
    print("🚀 低估值蓝筹策略 - 增强筛选 (含部分财务数据)")
    print("=" * 80)
    
    fetcher = TushareDataFetcher()
    storage = DataStorage()
    strategy = BlueChipStrategy()
    
    # 加载已有数据
    print("\n📂 加载已保存的数据...")
    df_daily = storage.load_dataframe("daily_basic_monthly.csv")
    stock_basic = storage.load_dataframe("stock_basic.csv")
    
    if df_daily is None:
        print("❌ 没有找到 daily_basic 数据")
        return
    
    # 使用最新日期数据
    latest_date = df_daily['trade_date'].max()
    print(f"\n📅 使用最新交易日数据: {latest_date}")
    
    latest_daily = df_daily[df_daily['trade_date'] == latest_date].copy()
    
    # 先进行基础筛选
    df_prep = latest_daily.copy()
    df_prep['total_mv'] = df_prep['total_mv'] / 10000
    df_prep = df_prep[(df_prep['pe'] > 0) & (df_prep['pb'] > 0)]
    df_prep = df_prep[(df_prep['pe'] < strategy.filters['max_pe']) & 
                      (df_prep['pb'] < strategy.filters['max_pb'])]
    df_prep = df_prep[df_prep['total_mv'] > strategy.filters['min_market_cap']]
    
    print(f"\n初步筛选后: {len(df_prep)} 只股票")
    
    # 获取前N只股票的财务数据
    candidate_codes = df_prep.sort_values('pe')['ts_code'].head(max_fina_stocks).tolist()
    
    print(f"\n📈 获取前 {len(candidate_codes)} 只股票的财务数据...")
    fina_results = fetcher.get_latest_fina_data_for_stocks(candidate_codes, max_stocks=max_fina_stocks)
    
    # 执行完整筛选
    result = strategy.filter_stocks(
        latest_daily, 
        fina_results.get('income'), 
        fina_results.get('fina'), 
        stock_basic
    )
    
    # 生成报告
    report = strategy.generate_report(result, top_n=30)
    print(report)
    
    # 保存结果
    if result is not None and not result.empty:
        strategy.save_results(result, f"blue_chip_enhanced_{latest_date}.csv")
        
        # 保存报告
        report_file = os.path.join(storage.results_dir, f"blue_chip_enhanced_report_{latest_date}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ 增强筛选完成!")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='低估值蓝筹策略快速筛选')
    parser.add_argument('--mode', choices=['quick', 'enhanced'], default='quick',
                       help='运行模式: quick=快速筛选, enhanced=增强筛选(含财务数据)')
    parser.add_argument('--fina-stocks', type=int, default=50,
                       help='增强模式下获取财务数据的股票数量')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            quick_filter()
        else:
            enhanced_filter(max_fina_stocks=args.fina_stocks)
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
