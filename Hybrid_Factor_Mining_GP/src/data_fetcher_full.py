"""
全市场数据获取模块
支持沪深300 + 中证500 + 中证1000
覆盖3年+历史数据
"""
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Set
import sqlite3
import time
from tqdm import tqdm

from config_optimized import (
    TUSHARE_TOKEN, DB_PATH, INDEX_CODES,
    START_DATE, END_DATE
)


class RateLimiter:
    """API限速器 (2000积分: 500次/分钟)"""
    def __init__(self, max_calls=480, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def wait(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0]) + 1
            print(f"Rate limit hit, waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self.calls = []
        self.calls.append(time.time())


class FullMarketDataFetcher:
    """全市场数据获取器"""
    
    def __init__(self):
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self.limiter = RateLimiter()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        tables = [
            ("stock_basic", """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code TEXT PRIMARY KEY,
                    name TEXT,
                    industry TEXT,
                    market TEXT
                )
            """),
            ("daily_price", """
                CREATE TABLE IF NOT EXISTS daily_price (
                    ts_code TEXT,
                    trade_date TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    vol REAL, amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """),
            ("adj_factor", """
                CREATE TABLE IF NOT EXISTS adj_factor (
                    ts_code TEXT,
                    trade_date TEXT,
                    adj_factor REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """),
            ("daily_basic", """
                CREATE TABLE IF NOT EXISTS daily_basic (
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate REAL, turnover_rate_f REAL,
                    pe REAL, pe_ttm REAL, pb REAL,
                    total_mv REAL, circ_mv REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """),
            ("index_members", """
                CREATE TABLE IF NOT EXISTS index_members (
                    ts_code TEXT,
                    index_code TEXT,
                    in_date TEXT,
                    out_date TEXT
                )
            """),
        ]
        
        for name, sql in tables:
            cursor.execute(sql)
        
        conn.commit()
        conn.close()
        print(f"[OK] Database initialized: {DB_PATH}")
    
    def fetch_all_index_components(self) -> Set[str]:
        """获取所有指数成分股并合并"""
        all_stocks = set()
        
        for name, code in INDEX_CODES.items():
            print(f"\nFetching {name} ({code}) components...")
            
            # 获取季度末的成分股
            dates = pd.date_range(START_DATE, END_DATE, freq='Q')
            stocks = set()
            
            for date in dates:
                trade_date = date.strftime('%Y%m%d')
                try:
                    self.limiter.wait()
                    df = self.pro.index_weight(index_code=code, trade_date=trade_date)
                    if not df.empty:
                        stocks.update(df['con_code'].tolist())
                        print(f"  {trade_date}: {len(df)} stocks")
                except Exception as e:
                    print(f"  {trade_date}: Error - {e}")
            
            print(f"[OK] {name}: {len(stocks)} unique stocks")
            all_stocks.update(stocks)
        
        print(f"\n[OK] Total unique stocks across all indices: {len(all_stocks)}")
        return all_stocks
    
    def fetch_daily_data_batch(self, trade_date: str) -> Dict[str, pd.DataFrame]:
        """批量获取单日数据"""
        results = {}
        
        try:
            # 日线数据
            self.limiter.wait()
            df_price = self.pro.daily(trade_date=trade_date)
            if not df_price.empty:
                results['price'] = df_price
            
            # 复权因子
            self.limiter.wait()
            df_adj = self.pro.adj_factor(trade_date=trade_date)
            if not df_adj.empty:
                results['adj'] = df_adj
            
            # 每日指标
            self.limiter.wait()
            df_basic = self.pro.daily_basic(trade_date=trade_date)
            if not df_basic.empty:
                # 选择关键字段
                cols = ['ts_code', 'trade_date', 'turnover_rate', 'pe', 'pb', 'total_mv']
                df_basic = df_basic[[c for c in cols if c in df_basic.columns]]
                results['basic'] = df_basic
            
        except Exception as e:
            print(f"Error fetching {trade_date}: {e}")
        
        return results
    
    def fetch_all_data(self, stock_list: Set[str]):
        """获取所有历史数据"""
        stock_set = set(stock_list)
        
        # 获取交易日
        print("\nFetching trade calendar...")
        self.limiter.wait()
        df_cal = self.pro.trade_cal(exchange='', start_date=START_DATE, end_date=END_DATE, is_open='1')
        trade_dates = df_cal['cal_date'].tolist()
        print(f"[OK] {len(trade_dates)} trading days")
        
        conn = sqlite3.connect(DB_PATH)
        
        for i, trade_date in enumerate(tqdm(trade_dates, desc="Fetching data")):
            data = self.fetch_daily_data_batch(trade_date)
            
            if 'price' in data:
                df = data['price'][data['price']['ts_code'].isin(stock_set)]
                df.to_sql('daily_price', conn, if_exists='append', index=False)
            
            if 'adj' in data:
                df = data['adj'][data['adj']['ts_code'].isin(stock_set)]
                df.to_sql('adj_factor', conn, if_exists='append', index=False)
            
            if 'basic' in data:
                df = data['basic'][data['basic']['ts_code'].isin(stock_set)]
                df.to_sql('daily_basic', conn, if_exists='append', index=False)
            
            # 每10天保存一次
            if (i + 1) % 10 == 0:
                conn.commit()
        
        conn.commit()
        conn.close()
        print("\n[OK] All data fetched successfully!")


def main():
    """主函数"""
    print("="*70)
    print("FULL MARKET DATA FETCHER")
    print("="*70)
    
    fetcher = FullMarketDataFetcher()
    
    # 获取所有指数成分股
    stocks = fetcher.fetch_all_index_components()
    
    # 获取历史数据
    if stocks:
        fetcher.fetch_all_data(stocks)
    
    print("\n" + "="*70)
    print("DATA FETCHING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
