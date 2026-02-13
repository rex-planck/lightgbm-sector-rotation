import pandas as pd
import os

# === 配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


# 直接读取刚才回测生成的带预测值的数据 (我们需要修改一下 final_composite_backtest.py 让它保存中间结果，
# 或者这里简单起见，我们假设你刚才没有关掉 PyCharm，可以直接看日志。
# 但为了严谨，我们重新加载一下数据)

def diagnose_failure():
    print("🕵️‍♂️ 正在进行策略尸检 (Post-Mortem Analysis)...")

    # 我们重新运行一遍 strategy 的逻辑来获取数据，不画图
    from final_composite_backtest import get_model_data, prepare_features, train_lgbm_for_strategy

    raw_df = get_model_data("半导体")
    feat_df = prepare_features(raw_df)
    backtest_df, split = train_lgbm_for_strategy(feat_df)

    # 截取 2025 年后的数据 (踏空最为严重的区域)
    df_focus = backtest_df[backtest_df['date'] >= '2025-06-01'].copy()

    # 统计 HMM 状态分布
    print("\n1️⃣ HMM 状态分布 (2025-06 以来):")
    state_counts = df_focus['hidden_state'].value_counts().sort_index()
    print(state_counts)
    print("   (0=熊, 1=震荡, 2=牛)")

    # 统计 LightGBM 的信心
    print("\n2️⃣ LightGBM 预测分布 (在震荡市 State 1 中):")
    mask_osc = df_focus['hidden_state'] == 1
    probs = df_focus.loc[mask_osc, 'prob']
    print(f"   平均预测概率: {probs.mean():.4f}")
    print(f"   最大预测概率: {probs.max():.4f}")
    print(f"   超过阈值(0.53)的天数: {(probs > 0.53).sum()} / {len(probs)} 天")

    # 计算最大回撤 (Max Drawdown) - 这是你的遮羞布
    # 策略净值
    strat_curve = (1 + (df_focus['close'].pct_change().shift(-1) * (probs > 0.53).astype(int)).fillna(0)).cumprod()
    # 基准净值
    bench_curve = (1 + df_focus['close'].pct_change().shift(-1).fillna(0)).cumprod()

    def max_drawdown(series):
        return (series / series.cummax() - 1).min()

    print("\n3️⃣ 风险指标对比:")
    print(f"   基准最大回撤: {max_drawdown(bench_curve):.2%}")
    print(f"   策略最大回撤: {max_drawdown(strat_curve):.2%}")


if __name__ == "__main__":
    diagnose_failure()