"""
增强版特征工程模块
包含更多技术指标、基本面特征和行业特征
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
import sqlite3

from config_optimized import DB_PATH, START_DATE, END_DATE, FEATURE_GROUPS


class EnhancedFeatureEngineer:
    """增强版特征工程器"""
    
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
    
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        print("[FeatureEngineering] Loading data from database...")
        
        query = """
            SELECT d.*, a.adj_factor, b.turnover_rate, b.pe, b.pb, b.total_mv
            FROM daily_price d
            LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
            WHERE d.trade_date BETWEEN ? AND ?
        """
        df = pd.read_sql(query, self.conn, params=(START_DATE, END_DATE))
        print(f"  Loaded {len(df)} records")
        return df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        print("[FeatureEngineering] Calculating enhanced features...")
        df = df.sort_values(['ts_code', 'trade_date']).copy()
        
        # 1. 复权价格
        df['adj_close'] = df['close'] * df['adj_factor']
        df['adj_open'] = df['open'] * df['adj_factor']
        df['adj_high'] = df['high'] * df['adj_factor']
        df['adj_low'] = df['low'] * df['adj_factor']
        
        # 2. 收益率特征 (多周期)
        for period in [1, 5, 10, 20, 60]:
            df[f'ret_{period}d'] = df.groupby('ts_code')['adj_close'].pct_change(period)
        
        # 3. 对数收益率
        df['log_ret'] = np.log(df['adj_close'] / df.groupby('ts_code')['adj_close'].shift(1))
        
        # 4. 波动率特征 (多周期)
        for period in [20, 60]:
            df[f'volatility_{period}d'] = df.groupby('ts_code')['log_ret'].rolling(period).std().values
        
        # 5. 成交量特征
        df['volume_ma5'] = df.groupby('ts_code')['vol'].rolling(5).mean().values
        df['volume_ma20'] = df.groupby('ts_code')['vol'].rolling(20).mean().values
        df['volume_ratio'] = df['vol'] / df['volume_ma20']
        df['volume_change'] = df['vol'].pct_change()
        
        # 6. 价格形态特征
        df['price_range'] = (df['adj_high'] - df['adj_low']) / df['adj_close']
        df['body_ratio'] = abs(df['adj_close'] - df['adj_open']) / (df['adj_high'] - df['adj_low'] + 1e-8)
        df['upper_shadow'] = (df['adj_high'] - df[['adj_close', 'adj_open']].max(axis=1)) / df['adj_close']
        df['lower_shadow'] = (df[['adj_close', 'adj_open']].min(axis=1) - df['adj_low']) / df['adj_close']
        
        # 7. 技术指标 - RSI
        for period in [6, 14, 28]:
            delta = df['adj_close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.groupby(df['ts_code']).rolling(period).mean().values
            avg_loss = loss.groupby(df['ts_code']).rolling(period).mean().values
            df[f'rsi_{period}'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
        
        # 8. 技术指标 - MACD
        ema12 = df.groupby('ts_code')['adj_close'].ewm(span=12).mean().values
        ema26 = df.groupby('ts_code')['adj_close'].ewm(span=26).mean().values
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df.groupby('ts_code')['macd'].ewm(span=9).mean().values
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 9. 布林带
        for period in [20, 60]:
            rolling_mean = df.groupby('ts_code')['adj_close'].rolling(period).mean().values
            rolling_std = df.groupby('ts_code')['adj_close'].rolling(period).std().values
            df[f'bb_middle_{period}'] = rolling_mean
            df[f'bb_upper_{period}'] = rolling_mean + 2 * rolling_std
            df[f'bb_lower_{period}'] = rolling_mean - 2 * rolling_std
            df[f'bb_position_{period}'] = (df['adj_close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
        
        # 10. 动量指标
        for period in [10, 20, 60]:
            df[f'momentum_{period}'] = df['adj_close'] / df.groupby('ts_code')['adj_close'].shift(period) - 1
        
        # 11. 基本面特征处理
        df['pe'] = df['pe'].fillna(df.groupby('trade_date')['pe'].transform('median'))
        df['pb'] = df['pb'].fillna(df.groupby('trade_date')['pb'].transform('median'))
        df['turnover_rate'] = df['turnover_rate'].fillna(0)
        
        # 12. 市值特征
        df['log_mv'] = np.log(df['total_mv'] + 1)
        
        # 13. 截面排名特征
        rank_features = ['ret_5d', 'volatility_20d', 'volume_ratio', 'rsi_14']
        for feat in rank_features:
            if feat in df.columns:
                df[f'{feat}_rank'] = df.groupby('trade_date')[feat].rank(pct=True)
        
        # 14. 时序Z-Score
        for feat in ['adj_close', 'vol']:
            for period in [20, 60]:
                rolling_mean = df.groupby('ts_code')[feat].rolling(period).mean().values
                rolling_std = df.groupby('ts_code')[feat].rolling(period).std().values
                df[f'{feat}_zscore_{period}'] = (df[feat] - rolling_mean) / (rolling_std + 1e-8)
        
        print(f"  Total features: {len([c for c in df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'adj_factor']])}")
        return df
    
    def prepare_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """准备标签"""
        print(f"[FeatureEngineering] Preparing labels (horizon={horizon} days)...")
        
        df = df.sort_values(['ts_code', 'trade_date'])
        
        # 未来收益率
        df[f'future_ret_{horizon}d'] = df.groupby('ts_code')['adj_close'].shift(-horizon) / df['adj_close'] - 1
        
        # 截面标准化 (Rank Norm)
        df['label'] = df.groupby('trade_date')[f'future_ret_{horizon}d'].transform(
            lambda x: (x.rank() - 0.5) / len(x) - 0.5 if len(x) > 1 else 0
        )
        
        # 删除无法计算标签的行
        df = df.dropna(subset=['label'])
        
        print(f"  Valid samples: {len(df)}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列列表"""
        exclude = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                   'vol', 'amount', 'adj_factor', 'label', 'future_ret_5d']
        return [c for c in df.columns if c not in exclude]
    
    def close(self):
        self.conn.close()


def main():
    """测试"""
    engineer = EnhancedFeatureEngineer()
    df_raw = engineer.load_data()
    df_features = engineer.calculate_features(df_raw)
    df_labeled = engineer.prepare_labels(df_features)
    feature_cols = engineer.get_feature_columns(df_labeled)
    
    print(f"\n[OK] Feature engineering completed!")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(df_labeled)}")
    
    engineer.close()


if __name__ == "__main__":
    main()
