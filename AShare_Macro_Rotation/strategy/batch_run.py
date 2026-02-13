import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import pandas as pd
import matplotlib.pyplot as plt

# === 插入这两行修复中文显示 (Windows专用) ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号



# === 引用之前的逻辑 ===
# 为了不重复造轮子，我们把 final_composite_backtest.py 当作库来调用
# 注意：你需要确保 final_composite_backtest.py 在 strategy 文件夹下
from final_composite_backtest import get_model_data, prepare_features, train_lgbm_for_strategy

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'plots')


def run_strategy_logic(df):
    """
    这是我们刚才验证过的[终极版]策略逻辑
    (MA20 + 流动性接管 + 优先级重排)
    """
    # 1. 计算 MA20 (灵敏均线)
    df['MA20'] = df['close'].rolling(window=20).mean()

    df['next_ret'] = df['close'].pct_change().shift(-1)
    df['position'] = 0.0

    for i in df.index:
        state = df.at[i, 'hidden_state']
        prob = df.at[i, 'prob']
        liquidity = df.at[i, 'Liquidity_Diff']
        close_price = df.at[i, 'close']
        ma20 = df.at[i, 'MA20']

        if pd.isna(ma20): continue

        is_trend_up = close_price > ma20

        # === 核心逻辑 ===
        # 1. 流动性共振 (钱多+趋势好 -> 猛干)
        if liquidity > 5.0 and is_trend_up:
            df.at[i, 'position'] = 1.0
        # 2. HMM 牛市
        elif state == 2:
            df.at[i, 'position'] = 1.0
        # 3. HMM 熊市 (且无流动性护盘)
        elif state == 0:
            df.at[i, 'position'] = 0.0
        # 4. 震荡市 (听 LightGBM)
        else:
            if prob > 0.52:
                df.at[i, 'position'] = 1.0
            else:
                df.at[i, 'position'] = 0.0

    # 计算策略净值
    df['strategy_ret'] = df['position'] * df['next_ret']
    df['strat_wealth'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    # 同时也返回基准净值，方便对比
    df['bench_wealth'] = (1 + df['next_ret'].fillna(0)).cumprod()

    return df


def run_batch():
    target_sectors = ["半导体", "白酒", "医疗", "新能源"]

    results = {}

    plt.figure(figsize=(14, 8))

    print("🚀 开始执行全行业轮动回测...")

    for sector in target_sectors:
        print(f"\n----------------------------")
        print(f"🧪 正在回测板块: 【{sector}】")

        try:
            # 1. 准备数据
            raw_df = get_model_data(sector)
            feat_df = prepare_features(raw_df)

            # 2. 独立训练 LightGBM (每个行业都有自己的微观特征)
            # 注意: 这里会打印训练日志，可以忽略
            backtest_df, split_idx = train_lgbm_for_strategy(feat_df)

            # 3. 跑策略
            res_df = run_strategy_logic(backtest_df)

            # 4. 记录结果
            final_ret = res_df['strat_wealth'].iloc[-1] - 1
            bench_ret = res_df['bench_wealth'].iloc[-1] - 1
            alpha = final_ret - bench_ret

            results[sector] = {
                'Strategy Return': final_ret,
                'Benchmark Return': bench_ret,
                'Alpha': alpha
            }

            print(f"   📊 [{sector}] 策略回报: {final_ret * 100:.2f}% (Alpha: {alpha * 100:.2f}%)")

            # 5. 画图 (画在一张大图上)
            plt.plot(res_df['date'], res_df['strat_wealth'], label=f'{sector} (Strategy)', linewidth=2)

        except Exception as e:
            print(f"   ❌ {sector} 回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 美化图表
    plt.title('Multi-Sector Rotation Strategy Performance (Liquidity + MA20)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Wealth (Normalized)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # 保存大乱斗图
    save_path = os.path.join(IMG_PATH, 'batch_sector_comparison.png')
    plt.savefig(save_path)
    print(f"\n✅ 全行业对比图已保存: {save_path}")

    # 打印最终排行榜
    print("\n🏆 === 最终战绩排行榜 (按 Alpha 排序) ===")
    sorted_res = sorted(results.items(), key=lambda x: x[1]['Alpha'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_res, 1):
        print(
            f"{rank}. {name}: 策略 {metrics['Strategy Return'] * 100:.2f}% | 基准 {metrics['Benchmark Return'] * 100:.2f}% | Alpha {metrics['Alpha'] * 100:.2f}%")


if __name__ == "__main__":
    run_batch()