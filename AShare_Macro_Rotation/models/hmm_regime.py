import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 配置区 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 回退一级目录找到 data/processed
DATA_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'processed')
IMG_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'plots')  # 存放图片

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


def load_data():
    """加载之前清洗好的宽表"""
    file_path = os.path.join(DATA_PATH, 'hmm_input_matrix.csv')
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def feature_engineering(df):
    """
    构造 HMM 模型的输入特征
    实务经验：HMM 对 '波动率' 非常敏感，这是区分牛熊的关键
    """
    # 1. 对数收益率 (已有 Market_Return, 假设是 simple return, 转 log)
    df['log_ret'] = np.log(1 + df['Market_Return'])

    # 2. 滚动波动率 (20日标准差) -> 衡量恐慌程度
    df['volatility'] = df['log_ret'].rolling(window=20).std()

    # 3. 流动性剪刀差 (Macro Feature)
    # 归一化处理：因为 Liquidity_Diff 是绝对值(如 8.4)，而 ret 是小数(0.01)，量纲差距大
    # 这里做简单的 Z-Score 标准化
    df['liquidity_z'] = (df['Liquidity_Diff'] - df['Liquidity_Diff'].mean()) / df['Liquidity_Diff'].std()

    # 去除空值 (rolling 会产生 NaN)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def train_hmm(df, n_components=3):
    """
    训练 HMM 模型
    n_components=3 对应: [0: 下行震荡/熊, 1: 低波震荡, 2: 上行趋势/牛]
    """
    print(f"🧠 正在训练 HMM 模型 (N={n_components})...")

    # 准备训练数据: [Log收益率, 波动率, 流动性]
    # 这种组合既看"面子"(涨跌)，也看"里子"(波动)，还看"底气"(宏观)
    X = df[['log_ret', 'volatility', 'liquidity_z']].values

    # 建模拟合
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)

    # 预测隐状态
    hidden_states = model.predict(X)

    # === 关键步骤：状态对齐 (Reordering) ===
    # HMM 的状态 0,1,2 是随机分配的。我们需要根据"平均收益率"重新排序。
    # 目标: State 0 = 表现最差(熊), State 2 = 表现最好(牛)

    # 1. 计算每个状态的平均收益率
    state_means = []
    for i in range(n_components):
        mean_ret = df.loc[hidden_states == i, 'log_ret'].mean()
        state_means.append((i, mean_ret))

    # 2. 按收益率从小到大排序
    sorted_states = sorted(state_means, key=lambda x: x[1])
    # 建立映射字典: 旧ID -> 新ID (0=Low, 1=Mid, 2=High)
    mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_states)}

    print("   📊 状态重排映射 (Old -> New):", mapping)
    print("      (新定义: 0=熊市/恐慌, 1=震荡, 2=牛市/拉升)")

    # 3. 映射回 DataFrame
    df['hidden_state'] = [mapping[s] for s in hidden_states]

    return df, model


def plot_regimes(df):
    """可视化：给 K 线图上色"""
    print("🎨 正在绘制宏观状态图...")

    plt.figure(figsize=(15, 8))

    # 定义颜色: 0(熊)=绿(A股跌是绿), 1(震荡)=灰, 2(牛)=红
    # 适配 A 股习惯
    colors = ['green', 'gray', 'red']
    labels = ['Bear/Panic (State 0)', 'Oscillation (State 1)', 'Bull/Rally (State 2)']

    for i in range(3):
        state_data = df[df['hidden_state'] == i]
        # 散点图绘制 (用 close 价格)
        plt.scatter(state_data['date'], state_data['close'],
                    s=10, c=colors[i], label=labels[i], alpha=0.6)

    plt.title('A-Share Market Regimes Identified by HMM (Macro+Price)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('HS300 Index')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(IMG_PATH, 'hmm_market_regimes.png')
    plt.savefig(save_file)
    print(f"   ✅ 图片已保存至: {save_file}")

    # 额外：保存带状态的数据，供回测使用
    output_csv = os.path.join(DATA_PATH, 'hmm_signals.csv')
    df.to_csv(output_csv, index=False)
    print(f"   ✅ 信号数据已保存至: {output_csv}")


if __name__ == "__main__":
    # 1. 加载
    df_raw = load_data()

    # 2. 特征
    df_feat = feature_engineering(df_raw)

    # 3. 训练
    df_result, model = train_hmm(df_feat, n_components=3)

    # 4. 画图与保存
    plot_regimes(df_result)

    # 5. 打印最近几天的状态
    print("\n🔍 最近 5 个交易日的市场状态:")
    print(df_result[['date', 'close', 'Liquidity_Diff', 'hidden_state']].tail(5))