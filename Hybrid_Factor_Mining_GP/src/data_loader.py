"""
数据加载模块
从 SQLite 加载数据并进行预处理
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime

from config import DB_PATH, START_DATE, END_DATE, TRAIN_END, VALID_END


class DataLoader:
    """数据加载器"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        query = "SELECT DISTINCT ts_code FROM daily_price ORDER BY ts_code"
        df = pd.read_sql(query, self.conn)
        return df['ts_code'].tolist()
    
    def get_trade_dates(self) -> List[str]:
        """获取交易日列表"""
        query = """
            SELECT DISTINCT trade_date 
            FROM daily_price 
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        df = pd.read_sql(query, self.conn, params=(START_DATE, END_DATE))
        return df['trade_date'].tolist()
    
    def load_all_data(self) -> pd.DataFrame:
        """
        加载所有数据并合并
        
        Returns:
            DataFrame: 包含 OHLCV、复权因子、每日指标的合并数据
        """
        print("[DataLoader] Loading data...")
        
        # 1. 加载日线数据
        query_price = """
            SELECT * FROM daily_price 
            WHERE trade_date BETWEEN ? AND ?
        """
        df_price = pd.read_sql(query_price, self.conn, params=(START_DATE, END_DATE))
        print(f"   Daily price: {len(df_price)} records")
        
        # 2. 加载复权因子
        query_adj = """
            SELECT ts_code, trade_date, adj_factor FROM adj_factor
            WHERE trade_date BETWEEN ? AND ?
        """
        df_adj = pd.read_sql(query_adj, self.conn, params=(START_DATE, END_DATE))
        print(f"   Adj factor: {len(df_adj)} records")
        
        # 3. 加载每日指标
        query_basic = """
            SELECT ts_code, trade_date, turnover_rate, pe, pb 
            FROM daily_basic
            WHERE trade_date BETWEEN ? AND ?
        """
        df_basic = pd.read_sql(query_basic, self.conn, params=(START_DATE, END_DATE))
        print(f"   Daily basic: {len(df_basic)} records")
        
        # 4. 合并数据
        df = df_price.merge(
            df_adj, on=['ts_code', 'trade_date'], how='left'
        ).merge(
            df_basic, on=['ts_code', 'trade_date'], how='left'
        )
        
        print(f"[OK] Merged data: {len(df)} records, {df['ts_code'].nunique()} stocks")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备基础特征
        
        Args:
            df: 原始数据 DataFrame
            
        Returns:
            添加特征后的 DataFrame
        """
        print("[DataLoader] Building features...")
        
        # 确保数据按股票和日期排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        
        # 计算复权价格
        df['adj_close'] = df['close'] * df['adj_factor']
        df['adj_open'] = df['open'] * df['adj_factor']
        df['adj_high'] = df['high'] * df['adj_factor']
        df['adj_low'] = df['low'] * df['adj_factor']
        
        # 收益率特征
        df['returns_1d'] = df.groupby('ts_code')['adj_close'].pct_change()
        df['returns_5d'] = df.groupby('ts_code')['adj_close'].pct_change(5)
        df['returns_20d'] = df.groupby('ts_code')['adj_close'].pct_change(20)
        
        # 波动率特征
        df['volatility_20d'] = df.groupby('ts_code')['returns_1d'].rolling(20).std().values
        
        # 成交量特征
        df['volume_ma5'] = df.groupby('ts_code')['vol'].rolling(5).mean().values
        df['volume_ma20'] = df.groupby('ts_code')['vol'].rolling(20).mean().values
        df['volume_ratio'] = df['vol'] / df['volume_ma20']
        
        # 价格位置特征
        df['price_position'] = (df['adj_close'] - df['adj_low']) / (df['adj_high'] - df['adj_low'] + 1e-8)
        
        # 技术指标简化版
        # RSI
        delta = df.groupby('ts_code')['adj_close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.groupby(df['ts_code']).rolling(14).mean().values
        avg_loss = loss.groupby(df['ts_code']).rolling(14).mean().values
        df['rsi_14'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
        
        # MACD 简化
        ema12 = df.groupby('ts_code')['adj_close'].ewm(span=12).mean().values
        ema26 = df.groupby('ts_code')['adj_close'].ewm(span=26).mean().values
        df['macd'] = ema12 - ema26
        
        # 清洗数据
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        print(f"[OK] Features built: {len(df)} records")
        return df
    
    def prepare_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        准备标签（未来收益率）
        
        Args:
            df: 特征数据
            horizon: 预测周期（天）
            
        Returns:
            添加标签后的 DataFrame
        """
        print(f"[DataLoader] Building labels ({horizon}-day return)...")
        
        # 计算未来收益率
        df = df.sort_values(['ts_code', 'trade_date'])
        df['future_return'] = df.groupby('ts_code')['adj_close'].shift(-horizon) / df['adj_close'] - 1
        
        # 截面标准化（Rank Norm）
        df['label'] = df.groupby('trade_date')['future_return'].transform(
            lambda x: (x.rank() - 0.5) / len(x) - 0.5 if len(x) > 1 else 0
        )
        
        # 删除无法计算标签的行
        df = df.dropna(subset=['label'])
        
        print(f"[OK] Labels built: {len(df)} valid records")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练/验证/测试集
        
        Args:
            df: 完整数据
            
        Returns:
            (train_df, valid_df, test_df)
        """
        train_df = df[df['trade_date'] <= TRAIN_END]
        valid_df = df[(df['trade_date'] > TRAIN_END) & (df['trade_date'] <= VALID_END)]
        test_df = df[df['trade_date'] > VALID_END]
        
        print(f"\n[DataLoader] Dataset split:")
        print(f"   Train: {train_df['trade_date'].min()} ~ {train_df['trade_date'].max()} ({len(train_df)})")
        print(f"   Valid: {valid_df['trade_date'].min()} ~ {valid_df['trade_date'].max()} ({len(valid_df)})")
        print(f"   Test: {test_df['trade_date'].min()} ~ {test_df['trade_date'].max()} ({len(test_df)})")
        
        return train_df, valid_df, test_df


def main():
    """测试数据加载"""
    loader = DataLoader()
    
    # 加载数据
    df_raw = loader.load_all_data()
    
    # 构建特征
    df_features = loader.prepare_features(df_raw)
    
    # 构建标签
    df_labeled = loader.prepare_labels(df_features)
    
    # 划分数据集
    train_df, valid_df, test_df = loader.split_data(df_labeled)
    
    loader.close()
    
    print("\n[OK] Data loading test completed!")


if __name__ == "__main__":
    main()
