"""
低估值蓝筹策略

筛选条件：
- PE < 20 (市盈率低于20倍)
- PB < 2 (市净率低于2倍)
- 总市值 > 500亿 (大盘蓝筹股)
- 净利润增长率 > 10% (业绩稳定增长)
- ROE > 10% (股东回报优秀)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tushare_config import BLUE_CHIP_FILTER


class BlueChipStrategy:
    """低估值蓝筹策略"""
    
    def __init__(self, filters=None):
        """
        初始化策略
        
        Parameters:
        -----------
        filters : dict
            筛选条件配置
        """
        self.filters = filters or BLUE_CHIP_FILTER
        print("=" * 60)
        print("🏦 低估值蓝筹策略")
        print("=" * 60)
        print(f"筛选条件:")
        print(f"  • PE < {self.filters['max_pe']}")
        print(f"  • PB < {self.filters['max_pb']}")
        print(f"  • 总市值 > {self.filters['min_market_cap']}亿")
        print(f"  • 净利润增长率 > {self.filters['min_profit_growth']}%")
        print(f"  • ROE > {self.filters['min_roe']}%")
        print("=" * 60)
    
    def prepare_daily_data(self, df_daily):
        """
        准备每日指标数据
        
        Parameters:
        -----------
        df_daily : DataFrame
            daily_basic 原始数据
            
        Returns:
        --------
        DataFrame : 处理后的数据
        """
        if df_daily is None or df_daily.empty:
            return None
            
        df = df_daily.copy()
        
        # 转换单位：总市值从万元转为亿元
        df['total_mv'] = df['total_mv'] / 10000
        df['circ_mv'] = df['circ_mv'] / 10000
        
        # 数值类型转换
        numeric_cols = ['pe', 'pb', 'total_mv', 'circ_mv', 'turnover_rate', 'volume_ratio', 'div_yield']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def prepare_income_data(self, df_income):
        """
        准备利润表数据，计算净利润增长率
        
        Parameters:
        -----------
        df_income : DataFrame
            income 原始数据
            
        Returns:
        --------
        DataFrame : 包含净利润增长率的数据
        """
        if df_income is None or df_income.empty:
            return None
            
        df = df_income.copy()
        
        # 数值类型转换
        numeric_cols = ['n_income', 'n_income_attr_p', 'total_revenue', 'revenue', 'basic_eps']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 对于逐个获取的数据，直接使用已有数据，不计算同比
        # （因为逐股获取通常只获取最新一期）
        if 'n_income_yoy' not in df.columns:
            df['n_income_yoy'] = np.nan
        if 'revenue_yoy' not in df.columns:
            df['revenue_yoy'] = np.nan
            
        return df
    
    def prepare_fina_data(self, df_fina):
        """
        准备财务指标数据
        
        Parameters:
        -----------
        df_fina : DataFrame
            fina_indicator 原始数据
            
        Returns:
        --------
        DataFrame : 处理后的数据
        """
        if df_fina is None or df_fina.empty:
            return None
            
        df = df_fina.copy()
        
        # 数值类型转换
        numeric_cols = ['roe', 'roe_waa', 'roe_dt', 'roa', 'netprofit_margin', 'grossprofit_margin', 'debt_to_assets']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ROE 转为百分比形式（如果小于1）
        if 'roe' in df.columns:
            df['roe'] = df['roe'].apply(lambda x: x * 100 if pd.notna(x) and abs(x) < 1 else x)
        
        return df
    
    def filter_stocks(self, df_daily, df_income=None, df_fina=None, stock_basic=None):
        """
        执行低估值蓝筹筛选
        
        Parameters:
        -----------
        df_daily : DataFrame
            每日指标数据
        df_income : DataFrame
            利润表数据 (可选)
        df_fina : DataFrame
            财务指标数据 (可选)
        stock_basic : DataFrame
            股票基础信息 (可选)
            
        Returns:
        --------
        DataFrame : 筛选结果
        """
        print("\n🔍 开始筛选股票...")
        
        # 1. 准备每日指标数据
        df = self.prepare_daily_data(df_daily)
        if df is None or df.empty:
            print("❌ 无每日指标数据")
            return None
        
        initial_count = len(df)
        print(f"\n初始股票数: {initial_count}")
        
        # 2. 基础筛选：排除 PE/PB 为负或0的股票
        df = df[(df['pe'] > 0) & (df['pb'] > 0)]
        print(f"排除负PE/PB后: {len(df)}")
        
        # 3. 估值筛选：PE < 20, PB < 2
        df = df[(df['pe'] < self.filters['max_pe']) & (df['pb'] < self.filters['max_pb'])]
        print(f"PE < {self.filters['max_pe']}, PB < {self.filters['max_pb']}: {len(df)}")
        
        # 4. 市值筛选：总市值 > 50亿
        df = df[df['total_mv'] > self.filters['min_market_cap']]
        print(f"总市值 > {self.filters['min_market_cap']}亿: {len(df)}")
        
        # 5. 合并利润表数据 (如果有)
        if df_income is not None and not df_income.empty:
            df_income_processed = self.prepare_income_data(df_income)
            # 取每个股票最新的报告期数据
            latest_income = df_income_processed.sort_values('end_date').groupby('ts_code').last().reset_index()
            
            # 选择需要的列
            income_cols = ['ts_code', 'n_income', 'n_income_yoy', 'n_income_attr_p', 
                          'n_income_attr_p_yoy', 'revenue', 'revenue_yoy', 'basic_eps']
            available_income_cols = [c for c in income_cols if c in latest_income.columns]
            latest_income = latest_income[available_income_cols]
            
            df = df.merge(latest_income, on='ts_code', how='left')
            
            # 净利润增长率筛选（如果有数据）
            if 'n_income_yoy' in df.columns:
                valid_growth = df[df['n_income_yoy'].notna()]
                if len(valid_growth) > 0:
                    df_growth = df[df['n_income_yoy'] > self.filters['min_profit_growth']]
                    print(f"净利润增长率 > {self.filters['min_profit_growth']}%: {len(df_growth)}")
                    if len(df_growth) > 0:
                        df = df_growth
        
        # 6. 合并财务指标数据 (如果有)
        if df_fina is not None and not df_fina.empty:
            df_fina_processed = self.prepare_fina_data(df_fina)
            # 取每个股票最新的报告期数据
            latest_fina = df_fina_processed.sort_values('end_date').groupby('ts_code').last().reset_index()
            
            # 选择需要的列
            fina_cols = ['ts_code', 'roe', 'roe_waa', 'roa', 'netprofit_margin', 
                        'grossprofit_margin', 'debt_to_assets']
            available_fina_cols = [c for c in fina_cols if c in latest_fina.columns]
            latest_fina = latest_fina[available_fina_cols]
            
            df = df.merge(latest_fina, on='ts_code', how='left')
            
            # ROE 筛选（如果有数据）
            if 'roe' in df.columns:
                valid_roe = df[df['roe'].notna()]
                if len(valid_roe) > 0:
                    df_roe = df[df['roe'] > self.filters['min_roe']]
                    print(f"ROE > {self.filters['min_roe']}%: {len(df_roe)}")
                    if len(df_roe) > 0:
                        df = df_roe
        
        # 7. 合并股票基础信息 (如果有)
        if stock_basic is not None and not stock_basic.empty:
            df = df.merge(stock_basic[['ts_code', 'name', 'industry']], on='ts_code', how='left')
        
        # 8. 排序：按 PE 从小到大排序
        df = df.sort_values('pe')
        
        print(f"\n✅ 最终筛选结果: {len(df)} 只股票")
        return df
    
    def generate_report(self, result_df, top_n=30):
        """
        生成选股报告
        
        Parameters:
        -----------
        result_df : DataFrame
            筛选结果
        top_n : int
            显示前N只股票
            
        Returns:
        --------
        str : 报告文本
        """
        if result_df is None or result_df.empty:
            return "❌ 未找到符合条件的股票"
        
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("📊 低估值蓝筹策略 - 选股报告")
        lines.append("=" * 100)
        lines.append(f"筛选日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"筛选条件: PE < {self.filters['max_pe']}, PB < {self.filters['max_pb']}, "
                    f"市值 > {self.filters['min_market_cap']}亿, "
                    f"净利润增长 > {self.filters['min_profit_growth']}%, ROE > {self.filters['min_roe']}%")
        lines.append("=" * 100)
        
        # 选择展示列
        display_cols = ['ts_code', 'name', 'industry', 'close', 'pe', 'pb', 'total_mv', 
                       'n_income_yoy', 'roe', 'div_yield']
        available_cols = [c for c in display_cols if c in result_df.columns]
        
        display_df = result_df[available_cols].head(top_n)
        
        # 格式化输出
        lines.append(display_df.to_string(index=False))
        
        lines.append("=" * 100)
        lines.append(f"共筛选出 {len(result_df)} 只符合条件的股票，显示前 {min(top_n, len(result_df))} 只")
        
        # 统计信息
        lines.append("\n📈 统计信息:")
        lines.append(f"  平均 PE: {result_df['pe'].mean():.2f}")
        lines.append(f"  平均 PB: {result_df['pb'].mean():.2f}")
        lines.append(f"  平均市值: {result_df['total_mv'].mean():.2f} 亿")
        if 'n_income_yoy' in result_df.columns and result_df['n_income_yoy'].notna().any():
            avg_growth = result_df['n_income_yoy'].dropna().mean()
            lines.append(f"  平均净利润增长率: {avg_growth:.2f}%")
        if 'roe' in result_df.columns and result_df['roe'].notna().any():
            avg_roe = result_df['roe'].dropna().mean()
            lines.append(f"  平均 ROE: {avg_roe:.2f}%")
        if 'div_yield' in result_df.columns and result_df['div_yield'].notna().any():
            avg_div = result_df['div_yield'].dropna().mean()
            lines.append(f"  平均股息率: {avg_div:.2f}%")
        
        # 行业分布
        if 'industry' in result_df.columns:
            lines.append("\n🏭 行业分布 (Top 10):")
            industry_counts = result_df['industry'].value_counts().head(10)
            for industry, count in industry_counts.items():
                lines.append(f"  {industry}: {count} 只")
        
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def save_results(self, result_df, filename=None):
        """
        保存筛选结果
        
        Parameters:
        -----------
        result_df : DataFrame
            筛选结果
        filename : str
            文件名
        """
        if result_df is None or result_df.empty:
            print("❌ 无结果可保存")
            return
        
        from config.tushare_config import RESULTS_DIR
        
        if filename is None:
            filename = f"blue_chip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存: {filepath}")
        
        return filepath


if __name__ == "__main__":
    # 测试策略
    strategy = BlueChipStrategy()
