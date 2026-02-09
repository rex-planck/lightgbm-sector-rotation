"""
Tushare 数据获取模块
负责从 Tushare API 获取原始数据并存储到 SQLite
"""
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import sqlite3
from tqdm import tqdm

from config import (
    TUSHARE_TOKEN, DB_PATH, INDEX_CODE,
    START_DATE, END_DATE, BASE_FEATURES
)


class TushareDataFetcher:
    """Tushare 数据获取器"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化 Tushare API
        
        Args:
            token: Tushare API Token，如果为 None 则使用 config 中的配置
        """
        self.token = token or TUSHARE_TOKEN
        if not self.token:
            raise ValueError("请提供 Tushare Token（2000积分账户）")
        
        self.pro = ts.pro_api(self.token)
        self._init_database()
    
    def _init_database(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 创建股票列表表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_basic (
                ts_code TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                market TEXT,
                list_date TEXT
            )
        """)
        
        # 创建日线数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_price (
                ts_code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                vol REAL,
                amount REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        
        # 创建复权因子表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adj_factor (
                ts_code TEXT,
                trade_date TEXT,
                adj_factor REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        
        # 创建每日指标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_basic (
                ts_code TEXT,
                trade_date TEXT,
                turnover_rate REAL,
                turnover_rate_f REAL,
                volume_ratio REAL,
                pe REAL,
                pe_ttm REAL,
                pb REAL,
                ps REAL,
                ps_ttm REAL,
                dv_ratio REAL,
                dv_ttm REAL,
                total_share REAL,
                float_share REAL,
                free_share REAL,
                total_mv REAL,
                circ_mv REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        
        # 创建指数成分股表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_weight (
                index_code TEXT,
                con_code TEXT,
                trade_date TEXT,
                weight REAL,
                PRIMARY KEY (index_code, con_code, trade_date)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ 数据库初始化完成: {DB_PATH}")
    
    def fetch_stock_basic(self) -> pd.DataFrame:
        """获取股票基础信息"""
        print("📥 获取股票基础信息...")
        df = self.pro.stock_basic(exchange='', list_status='L')
        
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('stock_basic', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"   共获取 {len(df)} 只股票")
        return df
    
    def fetch_index_components(self, index_code: str = INDEX_CODE) -> List[str]:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码，如 '000300.SH'
            
        Returns:
            成分股代码列表
        """
        print(f"📥 获取指数 {index_code} 成分股...")
        
        # 获取每个月最后一个交易日的成分股
        df_weights = []
        
        # 生成月份列表
        start_dt = datetime.strptime(START_DATE, "%Y%m%d")
        end_dt = datetime.strptime(END_DATE, "%Y%m%d")
        current = start_dt
        
        while current <= end_dt:
            # 获取该月最后一天
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
            trade_date = last_day.strftime("%Y%m%d")
            
            try:
                df = self.pro.index_weight(index_code=index_code, trade_date=trade_date)
                if not df.empty:
                    df_weights.append(df)
                    print(f"   {trade_date}: {len(df)} 只成分股")
            except Exception as e:
                print(f"   {trade_date}: 获取失败 - {e}")
            
            # 移动到下一个月
            current = next_month
        
        if df_weights:
            df_all = pd.concat(df_weights, ignore_index=True)
            conn = sqlite3.connect(DB_PATH)
            df_all.to_sql('index_weight', conn, if_exists='replace', index=False)
            conn.close()
            
            # 返回所有出现过的成分股
            all_stocks = df_all['con_code'].unique().tolist()
            print(f"✅ 共获取 {len(all_stocks)} 只不同的成分股")
            return all_stocks
        else:
            print("⚠️ 未获取到成分股数据")
            return []
    
    def fetch_daily_price(self, ts_code: str, start_date: str = START_DATE, 
                          end_date: str = END_DATE) -> Optional[pd.DataFrame]:
        """
        获取单只股票日线数据
        
        Args:
            ts_code: 股票代码，如 '000001.SZ'
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            DataFrame 或 None
        """
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df if not df.empty else None
        except Exception as e:
            print(f"   获取 {ts_code} 日线数据失败: {e}")
            return None
    
    def fetch_adj_factor(self, ts_code: str, start_date: str = START_DATE,
                         end_date: str = END_DATE) -> Optional[pd.DataFrame]:
        """获取复权因子"""
        try:
            df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df if not df.empty else None
        except Exception as e:
            print(f"   获取 {ts_code} 复权因子失败: {e}")
            return None
    
    def fetch_daily_basic(self, ts_code: str, start_date: str = START_DATE,
                          end_date: str = END_DATE) -> Optional[pd.DataFrame]:
        """获取每日指标"""
        try:
            df = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df if not df.empty else None
        except Exception as e:
            print(f"   获取 {ts_code} 每日指标失败: {e}")
            return None
    
    def fetch_all_stocks_data(self, stock_list: List[str], batch_size: int = 100):
        """
        批量获取多只股票的全部数据
        
        Args:
            stock_list: 股票代码列表
            batch_size: 每批处理的股票数量
        """
        print(f"\n📥 开始获取 {len(stock_list)} 只股票的数据...")
        
        conn = sqlite3.connect(DB_PATH)
        
        all_daily = []
        all_adj = []
        all_basic = []
        
        for i, ts_code in enumerate(tqdm(stock_list, desc="获取数据")):
            # 获取日线数据
            df_daily = self.fetch_daily_price(ts_code)
            if df_daily is not None:
                all_daily.append(df_daily)
            
            # 获取复权因子
            df_adj = self.fetch_adj_factor(ts_code)
            if df_adj is not None:
                all_adj.append(df_adj)
            
            # 获取每日指标（有积分限制，可能较慢）
            df_basic = self.fetch_daily_basic(ts_code)
            if df_basic is not None:
                all_basic.append(df_basic)
            
            # 批量写入，避免内存溢出
            if (i + 1) % batch_size == 0:
                self._batch_save(conn, all_daily, all_adj, all_basic)
                all_daily, all_adj, all_basic = [], [], []
        
        # 保存剩余数据
        if all_daily or all_adj or all_basic:
            self._batch_save(conn, all_daily, all_adj, all_basic)
        
        conn.close()
        print("✅ 数据获取完成！")
    
    def _batch_save(self, conn: sqlite3.Connection, 
                    all_daily: List[pd.DataFrame],
                    all_adj: List[pd.DataFrame],
                    all_basic: List[pd.DataFrame]):
        """批量保存数据到数据库"""
        if all_daily:
            df_daily = pd.concat(all_daily, ignore_index=True)
            df_daily.to_sql('daily_price', conn, if_exists='append', index=False)
        
        if all_adj:
            df_adj = pd.concat(all_adj, ignore_index=True)
            df_adj.to_sql('adj_factor', conn, if_exists='append', index=False)
        
        if all_basic:
            df_basic = pd.concat(all_basic, ignore_index=True)
            df_basic.to_sql('daily_basic', conn, if_exists='append', index=False)


def main():
    """主函数：运行数据获取流程"""
    print("=" * 60)
    print("🚀 Tushare 数据获取工具")
    print("=" * 60)
    
    # 初始化获取器
    fetcher = TushareDataFetcher()
    
    # 1. 获取股票基础信息
    fetcher.fetch_stock_basic()
    
    # 2. 获取指数成分股（沪深300）
    stock_list = fetcher.fetch_index_components(INDEX_CODE)
    
    if not stock_list:
        print("❌ 未获取到成分股，退出")
        return
    
    # 3. 获取所有成分股的日线数据
    fetcher.fetch_all_stocks_data(stock_list, batch_size=50)
    
    print("\n" + "=" * 60)
    print("✅ 数据获取完成！")
    print(f"📁 数据存储在: {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
