"""
数据获取模块 - 从 Tushare 获取股票数据
"""

import tushare as ts
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tushare_config import TUSHARE_TOKEN, START_DATE, END_DATE, DATA_DIR


class TushareDataFetcher:
    """Tushare 数据获取器"""
    
    def __init__(self, token=None):
        """初始化 API"""
        self.token = token or TUSHARE_TOKEN
        self.pro = ts.pro_api(self.token)
        print(f"✅ Tushare API 初始化成功")
        
    def get_trade_dates(self, start_date=None, end_date=None):
        """
        获取交易日历
        
        Parameters:
        -----------
        start_date : str
            开始日期 (YYYYMMDD)
        end_date : str
            结束日期 (YYYYMMDD)
            
        Returns:
        --------
        list : 交易日列表
        """
        start = start_date or START_DATE
        end = end_date or END_DATE
        
        print(f"📅 获取交易日历: {start} - {end}")
        
        df = self.pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1')
        trade_dates = df['cal_date'].tolist()
        print(f"✅ 共获取 {len(trade_dates)} 个交易日")
        return trade_dates
    
    def get_daily_basic(self, trade_date, retry=3):
        """
        获取个股每日指标 (PE, PB, 市值等)
        
        Parameters:
        -----------
        trade_date : str
            交易日期 (YYYYMMDD)
        retry : int
            重试次数
            
        Returns:
        --------
        DataFrame : 每日指标数据
        """
        for i in range(retry):
            try:
                df = self.pro.daily_basic(
                    trade_date=trade_date,
                    fields='ts_code,trade_date,close,pe,pb,total_mv,circ_mv,turnover_rate,turnover_rate_f,volume_ratio,div_yield'
                )
                return df
            except Exception as e:
                if i < retry - 1:
                    print(f"  ⚠️ 重试 {i+1}/{retry}: {e}")
                    time.sleep(1)
                else:
                    print(f"  ❌ 获取 {trade_date} 数据失败: {e}")
                    return None
    
    def get_daily_basic_monthly(self, start_date=None, end_date=None):
        """
        按月获取每日指标数据（每月最后一个交易日）
        
        Returns:
        --------
        DataFrame : 合并后的每月数据
        """
        trade_dates = self.get_trade_dates(start_date, end_date)
        
        # 获取每月最后一个交易日
        monthly_dates = []
        current_month = ""
        for date in trade_dates:
            month = date[:6]
            if month != current_month:
                if current_month != "":
                    monthly_dates.append(prev_date)
                current_month = month
            prev_date = date
        # 添加最后一个日期
        if trade_dates:
            monthly_dates.append(trade_dates[-1])
        
        print(f"📊 将获取 {len(monthly_dates)} 个月的月末数据")
        
        all_data = []
        for i, date in enumerate(monthly_dates):
            print(f"  [{i+1}/{len(monthly_dates)}] 获取 {date} 数据...", end=" ")
            df = self.get_daily_basic(date)
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"✓ {len(df)} 只股票")
            else:
                print(f"✗ 无数据")
            time.sleep(0.3)  # 限速
            
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            print(f"✅ 共获取 {len(result)} 条记录")
            return result
        return None
    
    def get_income_data(self, ts_code, period=None, fields=None, retry=3):
        """
        获取利润表数据 (普通接口)
        
        Parameters:
        -----------
        ts_code : str
            股票代码
        period : str
            报告期 (YYYYMMDD)
        fields : str
            指定字段
        retry : int
            重试次数
            
        Returns:
        --------
        DataFrame : 利润表数据
        """
        default_fields = 'ts_code,ann_date,end_date,report_type,basic_eps,total_revenue,revenue,n_income,n_income_attr_p'
        fields = fields or default_fields
        
        for i in range(retry):
            try:
                if period:
                    df = self.pro.income(ts_code=ts_code, period=period, fields=fields)
                else:
                    df = self.pro.income(ts_code=ts_code, fields=fields)
                return df
            except Exception as e:
                if i < retry - 1:
                    time.sleep(0.5)
                else:
                    return None
    
    def get_fina_indicator(self, ts_code, period=None, fields=None, retry=3):
        """
        获取财务指标数据 (普通接口)
        
        Parameters:
        -----------
        ts_code : str
            股票代码
        period : str
            报告期 (YYYYMMDD)
        fields : str
            指定字段
        retry : int
            重试次数
            
        Returns:
        --------
        DataFrame : 财务指标数据
        """
        default_fields = 'ts_code,ann_date,end_date,roe,roe_waa,roa,netprofit_margin,grossprofit_margin,debt_to_assets'
        fields = fields or default_fields
        
        for i in range(retry):
            try:
                if period:
                    df = self.pro.fina_indicator(ts_code=ts_code, period=period, fields=fields)
                else:
                    df = self.pro.fina_indicator(ts_code=ts_code, fields=fields)
                return df
            except Exception as e:
                if i < retry - 1:
                    time.sleep(0.5)
                else:
                    return None
    
    def get_latest_fina_data_for_stocks(self, ts_codes, max_stocks=200):
        """
        获取多只股票最新财务数据 (非VIP方式，逐个获取)
        
        Parameters:
        -----------
        ts_codes : list
            股票代码列表
        max_stocks : int
            最大获取股票数（控制API调用次数）
            
        Returns:
        --------
        dict : 包含最新利润表和财务指标数据的字典
        """
        print(f"\n📈 开始获取 {min(len(ts_codes), max_stocks)} 只股票的最新财务数据...")
        print("(非VIP用户，逐股获取，需要一定时间...)")
        
        income_data = []
        fina_data = []
        
        codes_to_fetch = ts_codes[:max_stocks]
        
        for i, code in enumerate(codes_to_fetch):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(codes_to_fetch)}...")
            
            # 获取最新利润表 (取最近一条)
            inc_df = self.get_income_data(code)
            if inc_df is not None and not inc_df.empty:
                inc_df = inc_df.sort_values('end_date', ascending=False).head(1)
                income_data.append(inc_df)
            
            # 获取最新财务指标 (取最近一条)
            fina_df = self.get_fina_indicator(code)
            if fina_df is not None and not fina_df.empty:
                fina_df = fina_df.sort_values('end_date', ascending=False).head(1)
                fina_data.append(fina_df)
            
            time.sleep(0.15)  # 限速，避免请求过快
        
        result = {}
        if income_data:
            result['income'] = pd.concat(income_data, ignore_index=True)
            print(f"✅ 获取到 {len(result['income'])} 只股票利润表数据")
        else:
            result['income'] = None
            
        if fina_data:
            result['fina'] = pd.concat(fina_data, ignore_index=True)
            print(f"✅ 获取到 {len(result['fina'])} 只股票财务指标数据")
        else:
            result['fina'] = None
            
        return result
    
    def get_stock_basic(self):
        """获取股票基础信息"""
        print("📋 获取股票基础信息...")
        try:
            df = self.pro.stock_basic(exchange='', list_status='L', 
                                       fields='ts_code,symbol,name,area,industry,market,list_date')
            print(f"✅ 共获取 {len(df)} 只股票")
            return df
        except Exception as e:
            print(f"❌ 获取股票基础信息失败: {e}")
            return None


def generate_quarterly_periods(start_year, end_year):
    """
    生成季度报告期列表
    
    Parameters:
    -----------
    start_year : int
        开始年份
    end_year : int
        结束年份
        
    Returns:
    --------
    list : 报告期列表
    """
    periods = []
    quarters = ['0331', '0630', '0930', '1231']
    for year in range(start_year, end_year + 1):
        for q in quarters:
            periods.append(f"{year}{q}")
    return periods


if __name__ == "__main__":
    # 测试数据获取
    fetcher = TushareDataFetcher()
    
    # 获取交易日历
    trade_dates = fetcher.get_trade_dates()
    print(f"\n交易日示例: {trade_dates[:5]} ... {trade_dates[-5:]}")
