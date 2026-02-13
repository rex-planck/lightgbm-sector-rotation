import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
import sys
import os

# ==========================================
# 1. 核心配置
# ==========================================
provider_uri = r"E:\Quant_program\Qlib-Cache\cn_data"

if not os.path.exists(provider_uri):
    raise FileNotFoundError(f"找不到数据路径: {provider_uri}")

# 初始化 Qlib
qlib.init(provider_uri=provider_uri, region=REG_CN)

print(f"✅ 环境初始化完成！")
print(f"💰 开始全链路实战回测 (Train -> Predict -> Backtest)...")
print("-" * 60)

# ==========================================
# 2. 模型与数据配置 (Config)
# ==========================================
market = "csi300"
benchmark = "SH000300"

data_handler_config = {
    "start_time": "2015-01-01",
    "end_time": "2022-12-31",
    "fit_start_time": "2015-01-01",
    "fit_end_time": "2020-12-31",
    "instruments": market,
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "label": ["Ref($close, -5) / $close - 1"],
}

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
            "num_threads": 20,
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
                "train": ("2015-01-01", "2020-12-31"),
                "valid": ("2021-01-01", "2021-12-31"),
                "test": ("2022-01-01", "2022-12-31"),
            },
        },
    },
}

# ==========================================
# 3. 运行全流程 (Main Execution)
# ==========================================
if __name__ == "__main__":
    with R.start(experiment_name="backtest_sim_01"):
        # ----------------------------------------------------
        # 第一阶段：制造枪和子弹 (Train)
        # ----------------------------------------------------
        print("🛠️  阶段1/3: 训练模型 (Training)...")
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        model.fit(dataset)

        # ----------------------------------------------------
        # 第二阶段：生成信号 (Inference)
        # ----------------------------------------------------
        print("\n🔮 阶段2/3: 生成预测信号 (Inference)...")
        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()

        # ----------------------------------------------------
        # 第三阶段：配置并执行回测 (Backtest)
        # ----------------------------------------------------
        # 🔥 修正点：回测配置必须放在这里，因为这时候 model 和 dataset 才存在！
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "signal": (model, dataset),  # 👈 此时 model 已经训练好了，不会报错了
                    "topk": 50,
                    "n_drop": 5,
                    "only_tradable": True,
                },
            },
            "backtest": {
                "start_time": "2022-01-01",
                "end_time": "2022-12-31",
                "account": 10000000,
                "benchmark": benchmark,
                "exchange_kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        print("\n💸 阶段3/3: 资金回测 (Backtesting)...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        print("-" * 60)
        print(f"🎉 回测完成！")

        # ----------------------------------------------------
        # 第四阶段：读取战报 (Report)
        # ----------------------------------------------------
        try:
            # 自动寻找保存的 pkl 文件
            pa_path = os.path.join(recorder.save_dir, "portfolio_analysis")
            report_df = pd.read_pickle(os.path.join(pa_path, "port_analysis_1day.pkl"))
            indicator_df = pd.read_pickle(os.path.join(pa_path, "indicator_analysis_1day.pkl"))

            print("\n🏆 【最终战报 2022】")

            # 计算累计收益
            final_return = report_df['value'].iloc[-1] / report_df['value'].iloc[0] - 1

            # 安全获取指标（防止字典key报错）
            ann_ret = indicator_df.get('annualized_return_with_cost', 0)
            max_dd = indicator_df.get('max_drawdown', 0)
            ir = indicator_df.get('information_ratio', 0)

            print(f"💰 你的策略累计收益: {final_return:.2%}")
            print(f"📉 沪深300同期表现: 约 -21.6%")
            print("-" * 30)
            print(f"📈 年化收益 (含费): {ann_ret:.2%}")
            print(f"📉 最大回撤: {max_dd:.2%}")
            print(f"⚖️ 信息比率 (IR): {ir:.2f}")

            if final_return > -0.21:
                diff = final_return - (-0.216)
                print(f"\n✅ 恭喜！你跑赢了大盘约 {diff:.1%}！")
            else:
                print("\n⚠️ 跑输大盘，需要优化策略。")

        except Exception as e:
            print(f"⚠️ 战报读取失败 (但这不影响回测结果已保存): {e}")