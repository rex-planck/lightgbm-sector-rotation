import akshare as ak
import pandas as pd
import os
from datetime import datetime

# === 核心配置区 (改这里，所有函数都会自动生效) ===
# 建议: 2020-01-01 起，涵盖完整的牛熊周期
START_DATE = "20200101"
# 建议: 2026-01-01 (或者设为当前日期 datetime.now().strftime("%Y%m%d"))
END_DATE = "20260101"

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'raw')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def get_macro_data():
    """
    1. 获取宏观数据 (宏观数据通常不用传时间参数，取回来后再切片即可)
    """
    print("⏳ 正在获取中国宏观数据...")
    try:
        df_cpi = ak.macro_china_cpi_monthly()
        df_cpi.to_csv(os.path.join(DATA_PATH, "macro_cpi.csv"), index=False)

        df_money = ak.macro_china_money_supply()
        df_money.to_csv(os.path.join(DATA_PATH, "macro_money_supply.csv"), index=False)
        print(f"   ✅ 宏观数据更新完毕")
    except Exception as e:
        print(f"   ❌ 宏观数据获取失败: {e}")


def get_benchmark():
    """
    2. 获取沪深300基准 (使用全局时间配置)
    """
    print(f"⏳ [Plan B] 正在获取沪深300 ({START_DATE}-{END_DATE})...")
    try:
        # 使用新浪接口 (更稳定)
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.rename(columns={"date": "日期"})

        # 格式化时间并筛选
        df['日期'] = pd.to_datetime(df['日期'])
        # 将配置的字符串转为 datetime 进行比较
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)

        mask = (df['日期'] >= start_dt) & (df['日期'] <= end_dt)
        df_filtered = df.loc[mask]

        df_filtered.to_csv(os.path.join(DATA_PATH, "benchmark_hs300.csv"), index=False)
        print(f"   ✅ 基准数据已保存: {len(df_filtered)} 条")
    except Exception as e:
        print(f"   ❌ 基准获取失败: {e}")


def get_sector_index(sector_name="半导体"):
    """
    3. 获取行业数据 (使用全局时间配置)
    """
    print(f"⏳ 正在获取【{sector_name}】({START_DATE}-{END_DATE})...")
    try:
        # 东财接口可以直接传字符串参数
        df = ak.stock_board_industry_hist_em(
            symbol=sector_name,
            start_date=START_DATE,
            end_date=END_DATE,
            period="日k",
            adjust="qfq"
        )
        filename = f"sector_{sector_name}.csv"
        df.to_csv(os.path.join(DATA_PATH, filename), index=False)
        print(f"   ✅ {sector_name} 行情已保存: {len(df)} 条")
    except Exception as e:
        print(f"   ❌ {sector_name} 获取失败: {e}")


# ... (上面的函数定义保持不变) ...

if __name__ == "__main__":
    print(f"⚙️  当前设定时间窗口: {START_DATE} 至 {END_DATE}")
    print("-" * 30)

    # 1. 更新宏观和基准
    get_macro_data()
    get_benchmark()

    # 2. 批量下载行业数据
    sectors = {
        "半导体": "半导体",  # 别名 == 官方名
        "白酒": "酿酒行业",  # 别名 != 官方名
        "医疗": "医疗服务",
        "新能源": "光伏设备"
    }

    for alias, official_name in sectors.items():
        print(f"\n📥 正在下载板块: {alias} ({official_name})...")
        get_sector_index(official_name)

        # 构建路径
        src = os.path.join(DATA_PATH, f"sector_{official_name}.csv")
        dst = os.path.join(DATA_PATH, f"sector_{alias}.csv")

        # === 修复 Bug 的关键逻辑 ===
        if src == dst:
            print(f"   ✅ 文件名无需修改: sector_{alias}.csv")
            continue
        # =========================

        if os.path.exists(src):
            # 如果目标文件已存在，先删除，防止报错
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            print(f"   ✅ 已重命名为: sector_{alias}.csv")
        else:
            print(f"   ⚠️ 未找到源文件: {src}，可能下载失败")

    print("-" * 30)
    print("🎉 全行业数据更新完成！")