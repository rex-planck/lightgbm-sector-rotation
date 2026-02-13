import pandas as pd
import os

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(CURRENT_DIR, 'raw')
PROCESSED_PATH = os.path.join(CURRENT_DIR, 'processed')

if not os.path.exists(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)


def clean_macro_data():
    """
    1. 清洗宏观数据：生成宏观特征表 (月频)
    目标: 计算 [M2-CPI 剪刀差] -> 代表市场流动性剩余
    """
    print("🧹 [1/3] 开始清洗宏观数据...")

    try:
        # --- A. 处理 CPI ---
        df_cpi = pd.read_csv(os.path.join(RAW_PATH, 'macro_cpi.csv'))

        # 1. 识别列名
        date_col = [c for c in df_cpi.columns if '日期' in c or '月份' in c][0]
        if '今值' in df_cpi.columns:
            val_col = '今值'
        else:
            val_col = [c for c in df_cpi.columns if '全国' in c and '同比' in c][0]

        print(f"   [CPI] 日期列: {date_col}, 数值列: {val_col}")

        df_cpi = df_cpi[[date_col, val_col]].copy()
        df_cpi.columns = ['date', 'CPI_YoY']

        # --- B. 处理 M2 ---
        df_m2 = pd.read_csv(os.path.join(RAW_PATH, 'macro_money_supply.csv'))

        # 1. 识别列名
        date_col_m2 = [c for c in df_m2.columns if '时间' in c or '月份' in c or '日期' in c][0]
        # 模糊匹配数值列
        if '今值' in df_m2.columns:
            val_col_m2 = '今值'
        else:
            val_col_m2 = [c for c in df_m2.columns if 'M2' in c and '同比' in c][0]

        print(f"   [M2 ] 日期列: {date_col_m2}, 数值列: {val_col_m2}")

        df_m2 = df_m2[[date_col_m2, val_col_m2]].copy()
        df_m2.columns = ['date', 'M2_YoY']

        # --- C. 关键修复：清洗中文日期 ---
        # 报错原因：Pandas无法解析 "2025年12月份"
        # 解决方案：手动替换中文字符
        print("   ⚙️ 正在修复中文日期格式...")

        # 处理 CPI 日期 (以防万一也有中文)
        df_cpi['date'] = df_cpi['date'].astype(str).str.replace('年', '-').str.replace('月份', '').str.replace('月', '')

        # 处理 M2 日期 (重点修复对象)
        df_m2['date'] = df_m2['date'].astype(str).str.replace('年', '-').str.replace('月份', '').str.replace('月', '')

        # --- D. 格式统一与合并 ---
        # 强制转换为数值类型 (处理非数字字符)
        df_cpi['CPI_YoY'] = pd.to_numeric(df_cpi['CPI_YoY'], errors='coerce')
        df_m2['M2_YoY'] = pd.to_numeric(df_m2['M2_YoY'], errors='coerce')

        # 转为标准时间戳 (统一为当月1号)
        df_cpi['date'] = pd.to_datetime(df_cpi['date']).dt.to_period('M').dt.to_timestamp()
        df_m2['date'] = pd.to_datetime(df_m2['date']).dt.to_period('M').dt.to_timestamp()

        # 合并 (Merge)
        df_macro = pd.merge(df_cpi, df_m2, on='date', how='inner')

        # --- E. 特征工程 ---
        # M2 - CPI 剪刀差
        df_macro['Liquidity_Diff'] = df_macro['M2_YoY'] - df_macro['CPI_YoY']

        # 去除无效值
        df_macro = df_macro.dropna()

        print(f"   ✅ 宏观特征生成完毕: {len(df_macro)} 个月的数据")
        return df_macro

    except Exception as e:
        print(f"   ❌ 宏观清洗失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_benchmark_data():
    """
    2. 清洗基准数据 (沪深300)：计算市场收益率 (日频)
    """
    print("\n🧹 [2/3] 开始清洗基准数据(沪深300)...")
    try:
        df = pd.read_csv(os.path.join(RAW_PATH, 'benchmark_hs300.csv'))

        df['date'] = pd.to_datetime(df['日期'])
        df = df.sort_values('date')

        df['close'] = df['close'].astype(float)
        # 简单的日收益率
        df['Market_Return'] = df['close'].pct_change()

        df_clean = df[['date', 'close', 'Market_Return']].dropna()

        print(f"   ✅ 基准清洗完毕: {len(df_clean)} 个交易日")
        return df_clean
    except Exception as e:
        print(f"   ❌ 基准清洗失败: {e}")
        return None


def merge_data(df_macro, df_benchmark):
    """
    3. 数据对齐：将 [月频宏观] 映射到 [日频行情]
    """
    print("\n🔗 [3/3] 正在对齐宏观与行情数据...")

    if df_macro is None or df_benchmark is None:
        print("   ⚠️ 缺少前置数据，无法合并")
        return

    df_benchmark = df_benchmark.sort_values('date')
    df_macro = df_macro.sort_values('date')

    # 关键步骤：Merge Asof (Backward)
    # 对于每一天，找到最近一次发布的宏观数据
    df_merge = pd.merge_asof(
        df_benchmark,
        df_macro,
        on='date',
        direction='backward'
    )

    df_merge = df_merge.dropna()

    output_file = os.path.join(PROCESSED_PATH, 'hmm_input_matrix.csv')
    df_merge.to_csv(output_file, index=False)

    print(f"   🎉 最终宽表已保存: {output_file}")
    print(f"   📊 数据范围: {df_merge['date'].min().date()} 至 {df_merge['date'].max().date()}")
    print("   🔍 宽表预览 (Liquidity_Diff 即剪刀差因子):")
    print(df_merge[['date', 'close', 'CPI_YoY', 'M2_YoY', 'Liquidity_Diff']].tail(3))


if __name__ == "__main__":
    macro_df = clean_macro_data()
    bench_df = clean_benchmark_data()
    merge_data(macro_df, bench_df)