"""
市场状态判断 (HMM) + 行业中性约束
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Dict, List, Tuple
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config_v3 import MARKET_REGIME_CONFIG, SECTOR_NEUTRAL_CONFIG, DB_PATH_V3


class MarketRegimeDetector:
    """市场状态检测器 (HMM)"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = {0: 'bear', 1: 'sideways', 2: 'bull'}
    
    def prepare_features(self, index_df: pd.DataFrame) -> np.ndarray:
        """准备市场特征"""
        # 计算收益率
        index_df['return'] = index_df['close'].pct_change()
        
        # 计算波动率 (20日)
        index_df['volatility'] = index_df['return'].rolling(20).std()
        
        # 计算成交量变化
        index_df['volume_ma'] = index_df['vol'].rolling(20).mean()
        index_df['volume_ratio'] = index_df['vol'] / index_df['volume_ma']
        
        # 计算趋势 (价格在MA上方/下方)
        index_df['ma60'] = index_df['close'].rolling(60).mean()
        index_df['trend'] = (index_df['close'] > index_df['ma60']).astype(int)
        
        # 特征矩阵
        features = index_df[['return', 'volatility', 'volume_ratio', 'trend']].dropna()
        return features.values
    
    def fit(self, index_df: pd.DataFrame) -> pd.DataFrame:
        """训练HMM模型"""
        logger.info("Training HMM market regime model...")
        
        X = self.prepare_features(index_df)
        
        # 训练HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.model.fit(X)
        
        # 预测状态
        hidden_states = self.model.predict(X)
        
        # 创建结果DataFrame
        result_df = index_df.dropna(subset=['return']).copy()
        result_df['market_regime'] = hidden_states
        
        # 根据收益率均值标记状态
        regime_stats = []
        for i in range(self.n_regimes):
            mask = hidden_states == i
            if mask.sum() > 0:
                avg_return = X[mask, 0].mean()  # 平均收益率
                regime_stats.append((i, avg_return))
        
        # 排序：高收益=牛市，低收益=熊市，中收益=震荡
        regime_stats.sort(key=lambda x: x[1], reverse=True)
        self.regime_map = {
            regime_stats[0][0]: 'bull',
            regime_stats[1][0]: 'sideways',
            regime_stats[2][0]: 'bear'
        }
        
        result_df['market_state'] = result_df['market_regime'].map(self.regime_map)
        
        logger.info(f"[OK] Market regimes identified: {self.regime_map}")
        for state in ['bull', 'bear', 'sideways']:
            count = (result_df['market_state'] == state).sum()
            logger.info(f"  {state}: {count} days")
        
        return result_df
    
    def get_current_regime(self, recent_data: pd.DataFrame) -> str:
        """获取当前市场状态"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(recent_data)
        state = self.model.predict(X[-1:])[0]
        return self.regime_map.get(state, 'unknown')


class SectorNeutralConstraint:
    """行业中性约束"""
    
    def __init__(self, max_deviation: float = 0.05):
        self.max_deviation = max_deviation
    
    def calculate_sector_weights(self, df: pd.DataFrame, 
                                  selected_stocks: List[str],
                                  date: str) -> Dict[str, float]:
        """计算行业权重"""
        day_data = df[(df['trade_date'] == date) & 
                      (df['ts_code'].isin(selected_stocks))].copy()
        
        if 'industry' not in day_data.columns:
            logger.warning("Industry column not found, skipping sector neutral")
            return {}
        
        # 按行业分组
        sector_weights = {}
        for industry, group in day_data.groupby('industry'):
            weight = len(group) / len(day_data) if len(day_data) > 0 else 0
            sector_weights[industry] = weight
        
        return sector_weights
    
    def adjust_weights(self, df: pd.DataFrame, 
                       selected_stocks: List[str],
                       date: str,
                       base_weights: Dict[str, float]) -> Dict[str, float]:
        """调整权重以满足行业中性"""
        if not SECTOR_NEUTRAL_CONFIG['enabled']:
            return base_weights
        
        # 获取基准行业分布（使用全市场）
        day_data = df[df['trade_date'] == date].copy()
        if 'industry' not in day_data.columns:
            return base_weights
        
        benchmark_sector = {}
        for industry, group in day_data.groupby('industry'):
            benchmark_sector[industry] = len(group) / len(day_data)
        
        # 计算当前选中的行业分布
        selected_data = day_data[day_data['ts_code'].isin(selected_stocks)]
        current_sector = {}
        for industry, group in selected_data.groupby('industry'):
            current_sector[industry] = sum(base_weights.get(ts, 0) for ts in group['ts_code'])
        
        # 调整权重
        adjusted_weights = base_weights.copy()
        
        for industry, benchmark_weight in benchmark_sector.items():
            current_weight = current_sector.get(industry, 0)
            
            # 检查是否偏离过多
            if abs(current_weight - benchmark_weight) > self.max_deviation:
                # 需要调整该行业内的股票权重
                industry_stocks = selected_data[selected_data['industry'] == industry]['ts_code'].tolist()
                
                # 目标权重
                target_weight = benchmark_weight
                
                if len(industry_stocks) > 0:
                    # 均匀分配新权重
                    per_stock_weight = target_weight / len(industry_stocks)
                    for ts in industry_stocks:
                        adjusted_weights[ts] = per_stock_weight
        
        # 归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights


class IntegratedFeatureEngineer:
    """集成特征工程（包含市场状态）"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector(n_regimes=MARKET_REGIME_CONFIG['n_regimes'])
        self.conn = sqlite3.connect(DB_PATH_V3)
    
    def load_and_enhance_data(self) -> pd.DataFrame:
        """加载数据并增强"""
        logger.info("Loading data from database...")
        
        query = """
            SELECT d.*, a.adj_factor, b.turnover_rate, b.pe, b.pb, s.industry
            FROM daily_price d
            LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            LEFT JOIN daily_basic b ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
            LEFT JOIN stock_basic s ON d.ts_code = s.ts_code
        """
        df = pd.read_sql(query, self.conn)
        
        logger.info(f"Loaded {len(df)} records")
        
        # 加载指数数据用于市场状态
        try:
            query_idx = "SELECT * FROM index_daily WHERE ts_code = '000300.SH'"
            df_index = pd.read_sql(query_idx, self.conn)
            
            # 检测市场状态
            if len(df_index) > 100:
                regime_df = self.regime_detector.fit(df_index)
                # 合并到主数据
                df = df.merge(regime_df[['trade_date', 'market_state']], 
                            on='trade_date', how='left')
        except Exception as e:
            logger.warning(f"Failed to load index data or calculate market regime: {e}")
            logger.info("Proceeding without market regime features.")
        
        return df
    
    def close(self):
        self.conn.close()


def main():
    """测试"""
    print("Market Regime + Sector Neutral Module")
    print("Ready for integration")


if __name__ == "__main__":
    main()
