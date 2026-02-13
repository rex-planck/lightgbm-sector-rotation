import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
import pandas as pd
import os

# ==========================================
# 1. 初始化环境 (关键配置)
# ==========================================
# 指向你的 E 盘数据仓库
provider_uri = r"E:\Quant_program\Qlib-Cache\cn_data"

# 容错检查
if not os.path.exists(provider_uri):
    raise FileNotFoundError(f"找不到数据路径: {provider_uri}，请检查解压是否成功！")

# 初始化 Qlib，指定为中国市场模式 (REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)

print(f"✅ Qlib 初始化完成！数据源: {provider_uri}")
print(f"🚀 正在准备训练：Alpha158 + LightGBM (目标: 沪深300)")
print("-" * 60)

# ==========================================
# 2. 市场与数据配置 (Data Handler)
# ==========================================
market = "csi300"  # 训练范围：沪深300成分股
benchmark = "SH000300"  # 哪怕不跑回测，通常也定义一个基准

data_handler_config = {
    # 训练时间段：2015年到2022年 (跨越了牛熊市)
    "start_time": "2015-01-01",
    "end_time": "2022-12-31",
    "fit_start_time": "2015-01-01",
    "fit_end_time": "2020-12-31",

    "instruments": market,

    # --- 核心黑科技：因子预处理 ---
    # infer_processors: 在生成因子后做什么？
    # 1. RobustZScoreNorm: 去极值并标准化 (处理财报暴雷等异常数据)
    # 2. Fillna: 填充缺失值 (防止模型报错)
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],

    # learn_processors: 训练前对“标签(Label)”做什么？
    # 1. DropnaLabel: 没标签的数据扔掉
    # 2. CSRankNorm: 截面排名标准化 (关键！让模型学习“谁涨得更好”，而不是“大盘涨了没”)
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],

    # 目标 Label: 预测未来 5 天的收益率 (Ref($close, -5) / $close - 1)
    "label": ["Ref($close, -5) / $close - 1"],
}

# ==========================================
# 3. 模型配置 (LightGBM)
# ==========================================
# 这是量化竞赛中常用的参数模版
task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,  # 开启多核加速
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2015-01-01", "2020-12-31"),  # 训练集 (5年)
                "valid": ("2021-01-01", "2021-12-31"),  # 验证集 (1年)
                "test": ("2022-01-01", "2022-12-31"),  # 测试集 (1年，这是真正的考场)
            },
        },
    },
}

# ... (前面的代码保持不变)

if __name__ == "__main__":
    # 启动一个实验记录
    with R.start(experiment_name="train_alpha158_lgb"):
        print("🛠️  正在构建数据集 (首次运行需要计算因子，可能需要 3-5 分钟)...")

        # 1. 初始化模型
        model = init_instance_by_config(task["model"])

        # 2. 初始化数据集
        dataset = init_instance_by_config(task["dataset"])

        print("📉 开始训练模型 (LightGBM Training)...")
        # 3. 训练
        model.fit(dataset)

        print("🔮 正在进行预测 (Inference)...")
        # 4. 预测
        recorder = R.get_recorder()

        # 生成预测结果 (Series 或 DataFrame)
        pred = model.predict(dataset)
        if isinstance(pred, list): pred = pred[0]
        if isinstance(pred, pd.DataFrame): pred = pred.iloc[:, 0]  # 确保是 Series

        print("\n📊 预测评分 Top 5 (Score 越高越好):")
        print(pred.head())

        # ==========================================
        # 5. 核心指标分析 (IC) - 修正版
        # ==========================================
        print("-" * 60)
        print("🏆 最终成绩单 (2022年测试集表现):")

        # --- 修正点：删除了 data_key="label" ---
        # Qlib 会自动根据 col_set="label" 找到对应的列
        label = dataset.prepare(segments="test", col_set="label")

        # 确保 label 也是 Series 格式
        if isinstance(label, pd.DataFrame): label = label.iloc[:, 0]

        # 数据对齐 (取交集，防止某些股票停牌导致行数不一致)
        common_index = pred.index.intersection(label.index)
        pred_test = pred.loc[common_index]
        label_test = label.loc[common_index]

        # 计算 IC (Pearson)
        ic = pred_test.corr(label_test)
        # 计算 Rank IC (Spearman)
        rank_ic = pred_test.rank().corr(label_test.rank())

        print(f"   IC (普通相关性): {ic:.4f}")
        print(f"   Rank IC (排名相关性): {rank_ic:.4f}")
        print("-" * 60)

        if rank_ic > 0.03:
            print("✅ 恭喜！模型有效！(Rank IC > 0.03)")
            print("这意味着你的 AI 确实从历史数据中学到了赚钱的规律。")
        else:
            print("⚠️ 模型效果一般。可能需要调整参数或更换因子。")