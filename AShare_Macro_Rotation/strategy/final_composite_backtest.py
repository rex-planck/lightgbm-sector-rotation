import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import joblib  # 用于保存/加载模型

# === 配置区 ===
# 路径回退逻辑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # 回退到根目录
PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'plots')


# 1. 准备数据 (复用之前的逻辑)
def get_model_data(sector_name="半导体"):
    # 加载行情
    df_sector = pd.read_csv(os.path.join(RAW_PATH, f'sector_{sector_name}.csv'))
    # 列名标准化
    col_map = {c: 'date' for c in df_sector.columns if '日期' in c}
    col_map.update({c: 'close' for c in df_sector.columns if '收盘' in c})
    col_map.update({c: 'vol' for c in df_sector.columns if '成交量' in c})
    df_sector = df_sector.rename(columns=col_map)
    df_sector['date'] = pd.to_datetime(df_sector['date'])

    # 加载宏观
    df_hmm = pd.read_csv(os.path.join(PROCESSED_PATH, 'hmm_signals.csv'))
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])

    # 合并
    df = pd.merge(df_sector, df_hmm[['date', 'hidden_state', 'Liquidity_Diff']], on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df


# 2. 特征工程 (必须与训练时完全一致)
def prepare_features(df):
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['vol_change'] = df['vol'].pct_change()
    df['std_20'] = df['ret_1'].rolling(20).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    df['Macro_RSI_Interact'] = df['hidden_state'] * df['RSI']

    # 预测目标
    df['target'] = (df['close'].shift(-1) / df['close'] - 1 > 0.001).astype(int)
    return df.dropna().reset_index(drop=True)


# 3. 训练 LightGBM (内嵌)
def train_lgbm_for_strategy(df):
    print("🤖 正在重新训练 LightGBM 用于策略...")
    features = ['ret_1', 'ret_5', 'vol_change', 'std_20', 'RSI', 'hidden_state', 'Liquidity_Diff', 'Macro_RSI_Interact']

    # 使用前 70% 数据训练，后 30% 回测，避免未来函数
    split = int(len(df) * 0.7)
    train_data = df.iloc[:split]
    test_data = df.iloc[split:].copy()  # 这部分用于跑策略

    lgb_train = lgb.Dataset(train_data[features], label=train_data['target'])

    # 增加正则化防止过拟合
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 3,  # 限制树深
        'num_leaves': 8,  # 减少叶子
        'reg_alpha': 0.1,  # L1 正则
        'verbose': -1
    }

    model = lgb.train(params, lgb_train, num_boost_round=500)

    # 对测试集生成预测概率
    test_data['prob'] = model.predict(test_data[features])
    return test_data, split


def run_strategy(df):
    print("📈 执行最终复合策略 (Speed Fix: MA20 for Fast Entry)...")

    # === 修改点 1: 改用 MA20 (更灵敏) ===
    # 在流动性驱动的疯牛中，MA60 太慢了，我们要用 MA20 抢反弹
    df['MA20'] = df['close'].rolling(window=20).mean()

    df['next_ret'] = df['close'].pct_change().shift(-1)
    df['position'] = 0.0

    for i in df.index:
        state = df.at[i, 'hidden_state']
        prob = df.at[i, 'prob']
        liquidity = df.at[i, 'Liquidity_Diff']
        close_price = df.at[i, 'close']
        ma20 = df.at[i, 'MA20']  # 使用 MA20

        # 安全检查
        if pd.isna(ma20):
            df.at[i, 'position'] = 0.0
            continue

        # === 修改点 2: 判定条件改为 MA20 ===
        is_trend_up = close_price > ma20

        # === 优先级 1: 宏观流动性共振 ===
        # 逻辑: 钱多 + 站上月线 = 抢钱
        if liquidity > 5.0 and is_trend_up:
            df.at[i, 'position'] = 1.0

        # === 优先级 2: HMM 状态判断 ===
        elif state == 2:
            df.at[i, 'position'] = 1.0

        elif state == 0:
            df.at[i, 'position'] = 0.0

        # === 优先级 3: 震荡市微观择时 ===
        else:  # state == 1
            if prob > 0.52:
                df.at[i, 'position'] = 1.0
            else:
                df.at[i, 'position'] = 0.0

    # 计算净值
    df['strategy_ret'] = df['position'] * df['next_ret']
    df['bench_wealth'] = (1 + df['next_ret'].fillna(0)).cumprod()
    df['strat_wealth'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    return df



def evaluate(df):
    final_bench = df['bench_wealth'].iloc[-1]
    final_strat = df['strat_wealth'].iloc[-1]

    print("-" * 30)
    print(f"💰 [回测结果] 样本外区间")
    print(f"   基准总回报: {(final_bench - 1) * 100:.2f}%")
    print(f"   策略总回报: {(final_strat - 1) * 100:.2f}%")
    print(f"   超额收益 (Alpha): {(final_strat - final_bench) * 100:.2f}%")

    # === 画图优化版 ===
    plt.figure(figsize=(12, 6))

    # 1. 基准线 (实线，灰色，稍粗一点作为背景)
    plt.plot(df['date'], df['bench_wealth'], label='Benchmark (Buy & Hold)',
             color='gray', alpha=0.5, linewidth=3)

    # 2. 策略线 (红色，设为半透明或虚线，以便看出重合部分)
    # alpha=0.8 (不那么刺眼), linestyle='--' (虚线，表示这是我们的"操作")
    plt.plot(df['date'], df['strat_wealth'], label='Composite Strategy (Macro-Override)',
             color='red', linewidth=2, linestyle='--', alpha=0.9)

    # 3. 标注出"宏观接管"区域 (可选，由数据驱动)
    # 找出流动性 > 5 的区域并涂色，显得很专业
    # 填充背景色: 只要 Liquidity_Diff > 5，就涂成浅红色背景
    if 'Liquidity_Diff' in df.columns:
        # 为了画图，我们需要对齐索引
        y_min, y_max = plt.ylim()
        plt.fill_between(df['date'], y_min, y_max,
                         where=(df['Liquidity_Diff'] > 5.0),
                         color='red', alpha=0.1, label='Liquidity Driven (Beta Mode)')

    plt.title('Final Strategy: Avoiding Crash (2024) & Catching Bull (2025)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Wealth')
    plt.legend(loc='upper left')  # 图例放左上角
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(IMG_PATH, 'final_strategy_optimized.png')
    plt.savefig(save_file)
    print(f"✅ 优化版资金曲线已保存: {save_file}")

if __name__ == "__main__":
    # 1. 准备全量数据
    raw_df = get_model_data("半导体")
    feat_df = prepare_features(raw_df)

    # 2. 训练并获取测试集预测值
    backtest_df, split_idx = train_lgbm_for_strategy(feat_df)

    # 3. 跑策略
    result_df = run_strategy(backtest_df)

    # 4. 评估
    evaluate(result_df)