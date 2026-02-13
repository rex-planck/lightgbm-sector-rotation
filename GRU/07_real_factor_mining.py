import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
import pandas as pd
import numpy as np
import gc
import os
from gplearn.genetic import SymbolicRegressor

# 1. 屏蔽警告 & 环境设置
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. 初始化 Qlib
print("🛠️ 初始化 Qlib 数据引擎...")
provider_uri = r"E:\Quant_program\Qlib-Cache\cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 3. 准备数据配置
market = "csi300"
benchmark = "SH000300"

# 我们使用 2018-2020 作为训练集（挖掘），2021 作为验证集
TRAIN_START = "2018-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2021-12-31"

data_handler_config = {
    "start_time": TRAIN_START,
    "end_time": TEST_END,
    "fit_start_time": TRAIN_START,
    "fit_end_time": TRAIN_END,
    "instruments": market,
    # 挖掘时不需要复杂的预处理，只需标准化和填充
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    # 预测目标：未来 5 日收益率
    "label": ["Ref($close, -5) / $close - 1"],
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": (TRAIN_START, TRAIN_END),
            "test": (TEST_START, TEST_END),
        },
    },
}


def clean_data(df):
    """暴力清洗数据：去除 Inf, NaN"""
    # 替换无穷大为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # 填充 NaN 为 0 (简单粗暴，但对 gplearn 很有效)
    df = df.fillna(0)
    return df


def run_mining():
    print("⏳ 正在加载 Alpha158 数据 (这可能需要 1-2 分钟)...")
    dataset = init_instance_by_config(dataset_config)

    # 获取 DataFrame
    # 这里的 'learn' 包含了 feature 和 label
    df_train = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
    df_test = dataset.prepare("test", col_set=["feature", "label"], data_key="learn")

    print(f"   原始训练集规模: {df_train.shape}")

    # --- 关键步骤：数据清洗 ---
    print("🧹 正在清洗数据 (Removing NaNs/Infs)...")
    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    # 分离特征 (X) 和 标签 (y)
    # feature 列名通常是 KMID, KLOW 等，label 是最后一列
    # 我们通过 level 0 来区分，或者直接切片
    # Alpha158 的特征列名比较长，这里我们假设前 158 列是特征
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # --- 关键步骤：降采样 (Downsampling) ---
    # 如果数据量超过 10万行，我们随机抽取 5万行来训练
    # 这样能极大加速遗传算法，同时不损失太多精度
    # 你的电脑有 32G 内存，我们可以稍微大胆点，用 10万行
    SAMPLE_SIZE = 100000

    if len(X_train) > SAMPLE_SIZE:
        print(f"✂️ 数据量过大，进行随机降采样至 {SAMPLE_SIZE} 行 (为了加速进化)...")
        # 保持随机种子一致
        sample_idx = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
        X_train_sample = X_train.iloc[sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    # 保存列名，方便后续查阅公式里的 X0, X1 是谁
    feature_names = X_train.columns.tolist()

    # 释放内存
    del df_train
    gc.collect()

    # 配置遗传规划
    print("\n🧬 配置遗传进化引擎 (Symbolic Regressor)...")
    # 我们定义一些适合金融的函数集
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min']

    print("\n🧬 配置遗传进化引擎 (Symbolic Regressor)...")
    # 我们定义一些适合金融的函数集
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min']

    est_gp = SymbolicRegressor(
        population_size=2000,
        generations=20,  # 🔥 增加到 20 代，给它更多时间进化
        tournament_size=20,
        stopping_criteria=1.0,  # 相关系数最大是1，设个达不到的值让它一直跑
        p_crossover=0.4,  # 降低杂交，增加突变
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.0001,  # 🔥 降低惩罚，允许公式变复杂一点，别老是用 0
        random_state=42,
        function_set=function_set,
        metric='spearman',  # 🔥🔥🔥 核心修改：直接优化 Rank IC，而不是误差！
        n_jobs=1
    )

    print("🚀 开始挖掘 (Mining Started)... 请耐心等待每一代的进度输出")
    est_gp.fit(X_train_sample, y_train_sample)

    # --- 结果分析 ---
    print("\n" + "=" * 50)
    print("🏆 挖掘结果 (Top Factor):")
    print("=" * 50)
    print(f"最强公式 (Raw): {est_gp._program}")

    # 尝试在测试集上验证效果
    print("\n📈 正在测试集上回测因子表现 (2021年)...")
    # 注意：预测时要用全量测试集，不要采样
    y_pred = est_gp.predict(X_test)

    # 计算 Rank IC
    res_df = pd.DataFrame({'pred': y_pred, 'label': y_test.values})
    rank_ic = res_df.rank().corr().iloc[0, 1]

    print("-" * 50)
    print(f"📊 因子样本外测试 (Out-of-Sample Test):")
    print(f"   Rank IC: {rank_ic:.4%}")
    print("-" * 50)

    if rank_ic > 0.03:
        print("🎉 恭喜！你挖到了一个有效的 Alpha 因子！")
        print("   (在纯机器挖掘中，OOS IC > 3% 已经非常不错了)")
    else:
        print("💪 效果一般，可能出现了过拟合。可以尝试调大 parsimony_coefficient 或增加 generations。")

    # 尝试解析公式中的 X
    print("\n🔍 公式特征解析提示:")
    print("   gplearn 输出的 X0, X1... 对应以下 Alpha158 特征:")
    for i in range(min(10, len(feature_names))):
        print(f"   X{i} -> {feature_names[i]}")
    print("   ... (更多请查阅 qlib Alpha158 文档)")


if __name__ == "__main__":
    run_mining()