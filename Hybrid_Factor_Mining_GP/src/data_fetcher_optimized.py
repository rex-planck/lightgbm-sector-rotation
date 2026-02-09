"""
优化版 Tushare 数据获取模块
针对 2000 积分账户优化：
1. 使用批量接口减少 API 调用次数
2. 添加智能限流控制
3. 支持断点续传
"""
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Set
import sqlite3
import time
from tqdm import tqdm
import logging

from config import (
    TUSHARE_TOKEN, DB_PATH, INDEX_CODE,
    START_DATE, END_DATE
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Tushare API 限速器（2000积分：每分钟500次）"""
    
    def __init__(self, max_calls: int = 480, period: int = 60):
        """
        Args:
            max_calls: 每周期最大调用次数（留20次缓冲）
            period: 周期（秒）
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def wait_if_needed(self):
        """检查是否需要等待"""
        now = time.time()
        # 清理过期记录
        self.calls = [c for c in self.calls if now - c < self.period]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0]) + 1
            logger.info(f"⏳ 限速等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
            self.calls = []
        
        self.calls.append(time.time())


class TushareDataFetcherOptimized:
    """优化版 Tushare 数据获取器"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or TUSHARE_TOKEN
        if not self.token:
            raise ValueError("请提供 Tushare Token")
        
        self.pro = ts.pro_api(self.token)
        self.rate_limiter = RateLimiter()
        self._init_database()
        
        # 测试 API
        self._test_api()
    
    def _test_api(self):
        """测试 API 连接"""
        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.trade_cal(exchange='', start_date='20240101', end_date='20240105')
            logger.info(f"✅ API 连接成功，剩余积分: {self._get_remaining_points()}")
        except Exception as e:
            logger.error(f"❌ API 连接失败: {e}")
            raise
    
    def _get_remaining_points(self) -> int:
        """获取剩余积分"""
        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.user()
            return df['remaining'].values[0] if 'remaining' in df.columns else -1
        except:
            return -1
    
    def _init_database(self):
        """初始化数据库（添加元数据表）"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 原有表结构
        tables = [
            ("stock_basic", """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date TEXT
                )
            """),
            ("daily_price", """
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
                    turnover_rate REAL,
                    turnover_rate_f REAL,
                    pe REAL,
                    pe_ttm REAL,
                    pb REAL,
                    total_mv REAL,
                    circ_mv REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """),
            ("index_weight", """
                CREATE TABLE IF NOT EXISTS index_weight (
                    index_code TEXT,
                    con_code TEXT,
                    trade_date TEXT,
                    weight REAL,
                    PRIMARY KEY (index_code, con_code, trade_date)
                )
            """),
            # 新增：元数据表（用于断点续传）
            ("fetch_meta", """
                CREATE TABLE IF NOT EXISTS fetch_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        ]
        
        for name, sql in tables:
            cursor.execute(sql)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ 数据库初始化完成: {DB_PATH}")
    
    def _get_meta(self, key: str) -> Optional[str]:
        """获取元数据"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM fetch_meta WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def _set_meta(self, key: str, value: str):
        """设置元数据"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO fetch_meta (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        conn.commit()
        conn.close()
    
    def fetch_stock_basic(self) -> pd.DataFrame:
        """获取股票基础信息"""
        logger.info("📥 获取股票基础信息...")
        self.rate_limiter.wait_if_needed()
        df = self.pro.stock_basic(exchange='', list_status='L')
        
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('stock_basic', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"   共获取 {len(df)} 只股票")
        self._set_meta('stock_basic_count', str(len(df)))
        return df
    
    def fetch_index_components(self, index_code: str = INDEX_CODE) -> List[str]:
        """获取指数成分股历史（按季度采样减少调用）"""
        logger.info(f"📥 获取指数 {index_code} 成分股...")
        
        cached = self._get_meta(f'index_components_{index_code}')
        if cached:
            stock_list = cached.split(',')
            logger.info(f"   从缓存加载 {len(stock_list)} 只成分股")
            return stock_list
        
        df_weights = []
        start_dt = datetime.strptime(START_DATE, "%Y%m%d")
        end_dt = datetime.strptime(END_DATE, "%Y%m%d")
        
        # 按季度采样（每年3、6、9、12月）
        current = start_dt
        while current <= end_dt:
            # 找到最近的季末日
            quarter_end_month = ((current.month - 1) // 3 + 1) * 3
            if current.month == quarter_end_month:
                trade_date = current.strftime("%Y%m%d")
            else:
                next_quarter = current.replace(day=1)
                if quarter_end_month > 12:
                    next_quarter = next_quarter.replace(year=current.year+1, month=3)
                else:
                    next_quarter = next_quarter.replace(month=quarter_end_month+1)
                last_day = next_quarter - timedelta(days=1)
                trade_date = last_day.strftime("%Y%m%d")
            
            try:
                self.rate_limiter.wait_if_needed()
                df = self.pro.index_weight(index_code=index_code, trade_date=trade_date)
                if not df.empty:
                    df_weights.append(df)
                    logger.info(f"   {trade_date}: {len(df)} 只成分股")
            except Exception as e:
                logger.warning(f"   {trade_date}: 获取失败 - {e}")
            
            # 跳到下个季度
            if quarter_end_month == 12:
                current = current.replace(year=current.year+1, month=3, day=31)
            else:
                current = current.replace(month=quarter_end_month+3, day=1)
        
        if df_weights:
            df_all = pd.concat(df_weights, ignore_index=True)
            conn = sqlite3.connect(DB_PATH)
            df_all.to_sql('index_weight', conn, if_exists='replace', index=False)
            conn.close()
            
            all_stocks = sorted(df_all['con_code'].unique().tolist())
            self._set_meta(f'index_components_{index_code}', ','.join(all_stocks))
            logger.info(f"✅ 共获取 {len(all_stocks)} 只不同的成分股")
            return all_stocks
        else:
            logger.error("⚠️ 未获取到成分股数据")
            return []
    
    def fetch_daily_price_batch(self, trade_date: str) -> Optional[pd.DataFrame]:
        """批量获取单日所有股票日线数据（高效）"""
        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.daily(trade_date=trade_date)
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"   获取 {trade_date} 日线数据失败: {e}")
            return None
    
    def fetch_adj_factor_batch(self, trade_date: str) -> Optional[pd.DataFrame]:
        """批量获取单日复权因子"""
        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.adj_factor(trade_date=trade_date)
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"   获取 {trade_date} 复权因子失败: {e}")
            return None
    
    def fetch_daily_basic_batch(self, trade_date: str) -> Optional[pd.DataFrame]:
        """批量获取单日每日指标"""
        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.daily_basic(trade_date=trade_date)
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"   获取 {trade_date} 每日指标失败: {e}")
            return None
    
    def get_trade_dates(self) -> List[str]:
        """获取交易日列表"""
        self.rate_limiter.wait_if_needed()
        df = self.pro.trade_cal(exchange='', start_date=START_DATE, end_date=END_DATE, is_open='1')
        return df['cal_date'].tolist()
    
    def fetch_all_data_by_date(self, stock_list: List[str]):
        """
        按日期批量获取所有数据（高效模式）
        
        Args:
            stock_list: 股票代码列表（用于过滤）
        """
        stock_set = set(stock_list)
        
        # 获取交易日列表
        logger.info("📅 获取交易日列表...")
        trade_dates = self.get_trade_dates()
        logger.info(f"   共 {len(trade_dates)} 个交易日")
        
        # 检查断点
        last_date = self._get_meta('last_fetched_date')
        if last_date:
            trade_dates = [d for d in trade_dates if d > last_date]
            logger.info(f"   从断点 {last_date} 继续，剩余 {len(trade_dates)} 天")
        
        conn = sqlite3.connect(DB_PATH)
        
        for i, trade_date in enumerate(tqdm(trade_dates, desc="获取数据")):
            # 批量获取日线
            df_daily = self.fetch_daily_price_batch(trade_date)
            if df_daily is not None:
                df_daily = df_daily[df_daily['ts_code'].isin(stock_set)]
                df_daily.to_sql('daily_price', conn, if_exists='append', index=False)
            
            # 批量获取复权因子
            df_adj = self.fetch_adj_factor_batch(trade_date)
            if df_adj is not None:
                df_adj = df_adj[df_adj['ts_code'].isin(stock_set)]
                df_adj.to_sql('adj_factor', conn, if_exists='append', index=False)
            
            # 批量获取每日指标（2000积分支持）
            df_basic = self.fetch_daily_basic_batch(trade_date)
            if df_basic is not None:
                # 选择关键字段
                cols = ['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f',
                       'pe', 'pe_ttm', 'pb', 'total_mv', 'circ_mv']
                df_basic = df_basic[df_basic['ts_code'].isin(stock_set)]
                df_basic = df_basic[[c for c in cols if c in df_basic.columns]]
                df_basic.to_sql('daily_basic', conn, if_exists='append', index=False)
            
            # 每 10 天保存一次断点
            if (i + 1) % 10 == 0:
                self._set_meta('last_fetched_date', trade_date)
                logger.info(f"   已保存断点: {trade_date}")
        
        conn.close()
        self._set_meta('last_fetched_date', trade_dates[-1] if trade_dates else '')
        logger.info("✅ 数据获取完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Tushare 数据获取工具（优化版）")
    print("=" * 60)
    
    fetcher = TushareDataFetcherOptimized()
    
    # 1. 获取股票基础信息
    fetcher.fetch_stock_basic()
    
    # 2. 获取指数成分股
    stock_list = fetcher.fetch_index_components(INDEX_CODE)
    
    if not stock_list:
        print("❌ 未获取到成分股，退出")
        return
    
    # 3. 按日期批量获取数据（高效）
    fetcher.fetch_all_data_by_date(stock_list)
    
    print("\n" + "=" * 60)
    print("✅ 数据获取完成！")
    print(f"📁 数据存储在: {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
