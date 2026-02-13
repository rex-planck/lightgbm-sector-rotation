import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 配置区 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 自动定位到数据文件夹
RAW_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'raw')
PROCESSED_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'processed')
IMG_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'plots')


def load_and_merge_data(sector_name="半导体"):
    """
    加载数据并缝合：[行业行情] + [HMM宏观信号]
    """
    print(f"📥 正在加载【{sector_name}】数据与 HMM 信号...")

    # 1. 加载行业数据 (Raw)
    sector_file = os.path.join(RAW_PATH, f'sector_{sector_name}.csv')
    if not os.path.exists(sector_file):
        raise FileNotFoundError(f"找不到行业数据: {sector_file}，请先运行 data_loader.py")

    df_sector = pd.read_csv(sector_file)
    # 兼容东财/新浪列名
    date_col = [c for c in df_sector.columns if '日期' in c][0]
    close_col = [c for c in df_sector.columns if '收盘' in c][0]

    df_sector = df_sector[[date_col, close_col]].rename(columns={date_col: 'date', close_col: 'price'})
    df_sector['date'] = pd.to_datetime(df_sector['date'])

    # 2. 加载 HMM 信号 (Processed)
    hmm_file = os.path.join(PROCESSED_PATH, 'hmm_signals.csv')
    if not os.path.exists(hmm_file):
        raise FileNotFoundError(f"找不到HMM信号: {hmm_file}，请先运行 models/hmm_regime.py")

    df_hmm = pd.read_csv(hmm_file)
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])
    # 只取需要的列
    df_hmm = df_hmm[['date', 'hidden_state']]

    # 3. 合并 (Inner Join)
    df_merge = pd.merge(df_sector, df_hmm, on='date', how='inner').sort_values('date')

    return df_merge


def calculate_strategy(df):
    """
    核心策略逻辑：宏观状态 + 行业动量
    """
    print("🧠 正在计算策略信号...")

    # 1. 计算行业本身的动量因子 (Momentum)
    # 定义: 当前价格 / 20日均线 - 1
    # 实务: 这是最简单的趋势因子，也可以用 RSI 或 MACD 替代
    df['MA20'] = df['price'].rolling(window=20).mean()
    df['Momentum'] = df['price'] / df['MA20'] - 1

    # 2. 计算下期收益率 (用于回测)
    # A股是 T+1，今天出的信号只能明天买，收益也是明天的
    df['next_ret'] = df['price'].pct_change().shift(-1)

    # 3. 生成信号 (Signal)
    # 初始化仓位为 0
    df['position'] = 0.0

    # --- 策略核心逻辑 (JD: 动态参数调优) ---

    # 场景 A: 牛市 (State 2) -> 激进策略
    # 逻辑: 只要不是跌得太离谱(动量 > -5%)，就满仓，不怕回调
    mask_bull = (df['hidden_state'] == 2) & (df['Momentum'] > -0.05)
    df.loc[mask_bull, 'position'] = 1.0

    # 场景 B: 震荡市 (State 1) -> 稳健策略
    # 逻辑: 只有行业本身走强(动量 > 0)才买，否则空仓
    mask_osc = (df['hidden_state'] == 1) & (df['Momentum'] > 0)
    df.loc[mask_osc, 'position'] = 1.0

    # 场景 C: 熊市 (State 0) -> 宏观对冲
    # 逻辑: 强制空仓 (或者实务中可以配置国债/黄金，这里简化为0)
    mask_bear = (df['hidden_state'] == 0)
    df.loc[mask_bear, 'position'] = 0.0

    return df


def backtest_and_plot(df, sector_name="Semiconductor"):
    """
    回测与画图
    """
    print("📈 正在执行回测...")

    # 策略收益 = 今天的仓位 * 明天的涨跌幅
    df['strategy_ret'] = df['position'] * df['next_ret']

    # 累计净值 (Cumulative Returns)
    df['bench_wealth'] = (1 + df['next_ret'].fillna(0)).cumprod()
    df['strat_wealth'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    # 计算指标
    total_ret = df['strat_wealth'].iloc[-1] - 1
    bench_ret = df['bench_wealth'].iloc[-1] - 1

    print(f"   [{sector_name}] 基准收益: {bench_ret * 100:.2f}%")
    print(f"   [{sector_name}] 策略收益: {total_ret * 100:.2f}%")

    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['bench_wealth'], label='Buy & Hold (Benchmark)', color='gray', alpha=0.6)
    plt.plot(df['date'], df['strat_wealth'], label='Macro-Enhanced Strategy', color='red', linewidth=2)

    # 标记出牛市区域 (State 2) 用背景色
    # 这里用一个小技巧填充背景
    y_min, y_max = plt.ylim()
    # 找到状态变化的边界
    df['state_change'] = df['hidden_state'].diff()

    plt.title(f'Strategy Backtest: {sector_name} (HMM Regime + Momentum)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(IMG_PATH, 'strategy_performance.png')
    plt.savefig(save_path)
    print(f"   ✅ 回测净值图已保存: {save_path}")

    return df


if __name__ == "__main__":
    # 1. 加载
    df = load_and_merge_data("半导体")

    # 2. 计算信号
    df = calculate_strategy(df)

    # 3. 回测
    backtest_and_plot(df)

    # 4. 打印最后几天的操作建议
    last_day = df.iloc[-1]
    print("-" * 30)
    print(f"📅 最新日期: {last_day['date'].date()}")
    print(f"📊 宏观状态: {int(last_day['hidden_state'])} (0=熊, 1=震荡, 2=牛)")
    print(f"🚀 行业动量: {last_day['Momentum']:.2%}")
    print(f"💡 交易建议: {'【满仓买入/持有】' if last_day['position'] == 1 else '【空仓/卖出】'}")