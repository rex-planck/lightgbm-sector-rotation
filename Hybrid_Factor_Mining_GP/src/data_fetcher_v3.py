"""
V3数据获取模块
支持：沪深300 + 中证500 + 中证1000，3年+历史数据
"""
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config_v3 import TUSHARE_TOKEN, DB_PATH_V3, INDEX_CODES_V3, START_DATE_V3, END_DATE_V3


class RateLimiter:
    """API限速器"""
    def __init__(self, max_calls=450, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def wait(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0]) + 1
            logger.info(f"Rate limit, sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self.calls = []
        self.calls.append(time.time())


class DataFetcherV3:
    """V3数据获取器"""
    
    def __init__(self):
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self.limiter = RateLimiter()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(DB_PATH_V3)
        cursor = conn.cursor()
        
        tables = {
            "stock_basic": """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code TEXT PRIMARY KEY,
                    name TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date TEXT
                )
            """,
            "daily_price": """
                CREATE TABLE IF NOT EXISTS daily_price (
                    ts_code TEXT,
                    trade_date TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    vol REAL, amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """,
            "adj_factor": """
                CREATE TABLE IF NOT EXISTS adj_factor (
                    ts_code TEXT,
                    trade_date TEXT,
                    adj_factor REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """,
            "daily_basic": """
                CREATE TABLE IF NOT EXISTS daily_basic (
                    ts_code TEXT,
                    trade_date TEXT,
                    turnover_rate REAL, turnover_rate_f REAL,
                    pe REAL, pe_ttm REAL, pb REAL, ps REAL,
                    total_mv REAL, circ_mv REAL, free_float REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """,
            "index_daily": """
                CREATE TABLE IF NOT EXISTS index_daily (
                    ts_code TEXT,
                    trade_date TEXT,
                    close REAL, open REAL, high REAL, low REAL,
                    vol REAL, amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """,
            "fetch_log": """
                CREATE TABLE IF NOT EXISTS fetch_log (
                    date TEXT PRIMARY KEY,
                    status TEXT,
                    records INTEGER
                )
            """
        }
        
        for name, sql in tables.items():
            cursor.execute(sql)
        
        conn.commit()
        conn.close()
        logger.info(f"[DB] Initialized: {DB_PATH_V3}")
    
    def fetch_stock_basic(self):
        """获取股票基础信息"""
        logger.info("Fetching stock basic info...")
        self.limiter.wait()
        df = self.pro.stock_basic(exchange='', list_status='L')
        
        conn = sqlite3.connect(DB_PATH_V3)
        df.to_sql('stock_basic', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"[OK] Fetched {len(df)} stocks")
        return df
    
    def fetch_index_components_all(self):
        """获取所有指数成分股"""
        all_stocks = set()
        
        for idx_name, idx_code in INDEX_CODES_V3.items():
            logger.info(f"\nFetching {idx_name} ({idx_code}) components...")
            
            # 获取每个季度的成分股
            dates = pd.date_range(START_DATE_V3, END_DATE_V3, freq='Q')
            stocks = set()
            
            for date in dates:
                trade_date = date.strftime('%Y%m%d')
                try:
                    self.limiter.wait()
                    df = self.pro.index_weight(index_code=idx_code, trade_date=trade_date)
                    if not df.empty:
                        stocks.update(df['con_code'].tolist())
                except Exception as e:
                    logger.warning(f"  {trade_date}: {e}")
            
            logger.info(f"  {idx_name}: {len(stocks)} unique stocks")
            all_stocks.update(stocks)
        
        logger.info(f"\n[OK] Total unique stocks: {len(all_stocks)}")
        return sorted(list(all_stocks))
    
    def fetch_daily_batch(self, trade_date):
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
                cols = ['ts_code', 'trade_date', 'turnover_rate', 'pe', 'pb', 'total_mv']
                df_basic = df_basic[[c for c in cols if c in df_basic.columns]]
                results['basic'] = df_basic
            
            # 指数日线（用于市场状态判断）
            for idx_code in INDEX_CODES_V3.values():
                self.limiter.wait()
                df_idx = self.pro.index_daily(ts_code=idx_code, trade_date=trade_date)
                if not df_idx.empty:
                    if 'index' not in results:
                        results['index'] = []
                    results['index'].append(df_idx)
            
        except Exception as e:
            logger.error(f"Error fetching {trade_date}: {e}")
        
        return results
    
    def fetch_all_historical_data(self, stock_list):
        """获取所有历史数据"""
        stock_set = set(stock_list)
        
        # 获取交易日历
        logger.info("\nFetching trade calendar...")
        self.limiter.wait()
        df_cal = self.pro.trade_cal(exchange='', start_date=START_DATE_V3, 
                                     end_date=END_DATE_V3, is_open='1')
        trade_dates = df_cal['cal_date'].tolist()
        logger.info(f"[OK] {len(trade_dates)} trading days from {START_DATE_V3} to {END_DATE_V3}")
        
        conn = sqlite3.connect(DB_PATH_V3)
        
        for i, trade_date in enumerate(tqdm(trade_dates, desc="Fetching data")):
            data = self.fetch_daily_batch(trade_date)
            
            if 'price' in data:
                df = data['price'][data['price']['ts_code'].isin(stock_set)]
                df.to_sql('daily_price', conn, if_exists='append', index=False)
            
            if 'adj' in data:
                df = data['adj'][data['adj']['ts_code'].isin(stock_set)]
                df.to_sql('adj_factor', conn, if_exists='append', index=False)
            
            if 'basic' in data:
                df = data['basic'][data['basic']['ts_code'].isin(stock_set)]
                df.to_sql('daily_basic', conn, if_exists='append', index=False)
            
            if 'index' in data:
                df_idx = pd.concat(data['index'], ignore_index=True)
                df_idx.to_sql('index_daily', conn, if_exists='append', index=False)
            
            # 每10天提交一次
            if (i + 1) % 10 == 0:
                conn.commit()
        
        conn.commit()
        conn.close()
        logger.info("\n[OK] All data fetched successfully!")
    
    def verify_data(self):
        """验证数据完整性"""
        conn = sqlite3.connect(DB_PATH_V3)
        cursor = conn.cursor()
        
        stats = {}
        for table in ['daily_price', 'adj_factor', 'daily_basic', 'index_daily']:
            try:
                cursor.execute(f"SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM {table}")
                row = cursor.fetchone()
                stats[table] = row
            except:
                stats[table] = (0, None, None)
        
        conn.close()
        
        print("\n[Data Verification]")
        for table, (count, min_date, max_date) in stats.items():
            print(f"  {table:20s}: {count:8d} records ({min_date} to {max_date})")
        
        return stats


def main():
    """主函数"""
    print("="*70)
    print("V3 DATA FETCHER - 3 Years Full Market Data")
    print("="*70)
    
    fetcher = DataFetcherV3()
    
    # 1. 获取股票基础信息
    fetcher.fetch_stock_basic()
    
    # 2. 获取所有指数成分股
    stocks = fetcher.fetch_index_components_all()
    
    # 3. 获取历史数据
    if stocks:
        fetcher.fetch_all_historical_data(stocks)
    
    # 4. 验证数据
    fetcher.verify_data()
    
    print("\n" + "="*70)
    print("DATA FETCHING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
