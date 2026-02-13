import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import qlib
from qlib.data import D
from qlib.constant import REG_CN

# --- 配置 ---
# 你的实验记录路径 (保持不变)
BASE_DIR = r"E:\Quant_program\python_program\mlruns"
# 你的数据路径 (用于读取基准)
PROVIDER_URI = r"E:\Quant_program\Qlib-Cache\cn_data"


def find_and_plot():
    # 1. 初始化 Qlib
    if not os.path.exists(PROVIDER_URI):
        print(f"❌ 错误：找不到数据路径 {PROVIDER_URI}")
        return
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    print(f"🕵️‍♂️ 正在搜索每日账本文件 (report_normal_1day.pkl)...")

    # 🔥 核心修正：这里改为搜索 report_normal_1day.pkl
    search_pattern = os.path.join(BASE_DIR, "**", "report_normal_1day.pkl")
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        print("❌ 未找到账本文件！")
        print("请确认 03_backtest_simulation.py 是否成功运行并完成了回测。")
        return

    # 找到最新的文件
    latest_file = max(all_files, key=os.path.getmtime)
    print(f"✅ 锁定账本: {latest_file}")

    try:
        # 2. 读取回测数据
        df = pd.read_pickle(latest_file)

        # 🖨️ [调试] 看看这次读到了什么
        print("-" * 30)
        print(f"列名清单: {df.columns.tolist()}")
        # 预期应该包含: 'account', 'return', 'turnover', 'cost', 'bench' 等
        print("-" * 30)

        # 3. 准备绘图数据
        plt.figure(figsize=(12, 6))

        # 策略净值 (使用 'account' 列，如果没有就用 'return' 推算)
        if 'account' in df.columns:
            # 归一化：每天的账户余额 / 第一天的账户余额
            df['Strategy'] = df['account'] / df['account'].iloc[0]
        elif 'return' in df.columns:
            df['Strategy'] = (1 + df['return']).cumprod()
        else:
            print("❌ 错误：数据中没有 'account' 或 'return' 列，无法画图。")
            return

        # 基准净值 (使用 'bench' 列，Qlib 通常会自动记录基准收益)
        if 'bench' in df.columns:
            df['Benchmark'] = (1 + df['bench']).cumprod()
        else:
            # 如果没记录，手动去读沪深300
            print("📉 正在手动读取沪深300数据...")
            bench_df = D.features(['SH000300'], ['$close'], start_time=df.index[0], end_time=df.index[-1])
            df['Benchmark'] = bench_df['$close'] / bench_df['$close'].iloc[0]

        # 4. 绘图
        # 策略线 (红)
        plt.plot(df.index, df['Strategy'], label='My AI Strategy', color='#d62728', linewidth=2)
        # 基准线 (灰)
        plt.plot(df.index, df['Benchmark'], label='CSI 300', color='gray', linestyle='--', alpha=0.8)

        # 填充超额收益
        plt.fill_between(df.index, df['Strategy'], df['Benchmark'],
                         where=(df['Strategy'] >= df['Benchmark']),
                         facecolor='red', alpha=0.1, label='Alpha Gains')

        plt.title('Backtest Result 2022: Alpha158 + LightGBM', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (Net Value)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # 保存图片
        img_path = "backtest_final_success.png"
        plt.savefig(img_path)
        print(f"\n📊 绘图成功！图片已保存为: {img_path}")

        # 5. 打印最终收益
        strat_ret = df['Strategy'].iloc[-1] - 1
        bench_ret = df['Benchmark'].iloc[-1] - 1
        print(f"💰 策略最终收益: {strat_ret:.2%}")
        print(f"📉 基准最终收益: {bench_ret:.2%}")
        print(f"🚀 超额收益 (Alpha): {strat_ret - bench_ret:.2%}")

        plt.show()

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    find_and_plot()