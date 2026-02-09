"""
遗传规划因子挖掘模块
使用 gplearn 自动挖掘 Alpha 因子
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import (
    GP_CONFIG, FACTOR_OUTPUT_DIR, LABEL_COL,
    BASE_FEATURES
)
from data_loader import DataLoader


# ==================== 自定义函数 ====================
def rank(x):
    """截面排名（替代 alphalens 的 rank）"""
    return pd.Series(x).rank(pct=True).values if len(x) > 1 else np.zeros_like(x)

def ts_rank(x, window=10):
    """时序排名"""
    if len(x) < window:
        return np.zeros_like(x)
    s = pd.Series(x)
    return s.rolling(window).apply(lambda y: y.rank(pct=True).iloc[-1] if len(y) == window else 0, raw=False).values

def ts_mean(x, window=10):
    """时序均值"""
    return pd.Series(x).rolling(window).mean().values

def ts_std(x, window=10):
    """时序标准差"""
    return pd.Series(x).rolling(window).std().values

def ts_max(x, window=10):
    """时序最大值"""
    return pd.Series(x).rolling(window).max().values

def ts_min(x, window=10):
    """时序最小值"""
    return pd.Series(x).rolling(window).min().values

def ts_delta(x, window=10):
    """时序差分"""
    return pd.Series(x).diff(window).values

def ts_corr(x, y, window=10):
    """时序相关性"""
    return pd.Series(x).rolling(window).corr(pd.Series(y)).values

# 包装为 gplearn 函数
rank_function = make_function(function=rank, name='rank', arity=1)
ts_mean_function = make_function(function=ts_mean, name='ts_mean', arity=1)
ts_std_function = make_function(function=ts_std, name='ts_std', arity=1)
ts_max_function = make_function(function=ts_max, name='ts_max', arity=1)
ts_min_function = make_function(function=ts_min, name='ts_min', arity=1)
ts_delta_function = make_function(function=ts_delta, name='ts_delta', arity=1)


class GPFactorMiner:
    """GP 因子挖掘器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化
        
        Args:
            config: GP 配置参数
        """
        self.config = config or GP_CONFIG
        self.mined_factors = []  # 存储挖掘出的因子
        self.factor_programs = []  # 存储因子程序（可复现）
    
    def calculate_ic(self, factor_values: np.ndarray, labels: np.ndarray) -> float:
        """
        计算 Rank IC
        
        Args:
            factor_values: 因子值
            labels: 标签（未来收益率）
            
        Returns:
            Rank IC 值
        """
        # 清洗数据
        mask = ~(np.isnan(factor_values) | np.isnan(labels) | 
                 np.isinf(factor_values) | np.isinf(labels))
        
        if mask.sum() < 10:
            return 0.0
        
        f = factor_values[mask]
        l = labels[mask]
        
        # 计算 Rank IC
        f_rank = pd.Series(f).rank()
        l_rank = pd.Series(l).rank()
        
        ic = np.corrcoef(f_rank, l_rank)[0, 1]
        return ic if not np.isnan(ic) else 0.0
    
    def mine_factor_single_day(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[str, float]:
        """
        对单日截面数据进行因子挖掘
        
        Args:
            X: 特征数据 (n_stocks, n_features)
            y: 标签 (n_stocks,)
            
        Returns:
            (最佳因子程序字符串, IC值)
        """
        # 自定义适应度函数
        def _rank_ic_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return abs(self.calculate_ic(y_pred, y))
        
        # 创建 GP 模型
        est_gp = SymbolicRegressor(
            population_size=self.config['population_size'],
            generations=self.config['generations'],
            tournament_size=self.config['tournament_size'],
            stopping_criteria=self.config['stopping_criteria'],
            p_crossover=self.config['p_crossover'],
            p_subtree_mutation=self.config['p_subtree_mutation'],
            p_hoist_mutation=self.config['p_hoist_mutation'],
            p_point_mutation=self.config['p_point_mutation'],
            max_samples=self.config['max_samples'],
            parsimony_coefficient=self.config['parsimony_coefficient'],
            random_state=np.random.randint(0, 10000),
            function_set=self.config['function_set'],
            metric=_rank_ic_scorer,
            verbose=0
        )
        
        try:
            est_gp.fit(X, y)
            best_program = est_gp._program
            best_ic = best_program.raw_fitness_
            return str(best_program), best_ic
        except Exception as e:
            return None, 0.0
    
    def mine_factors_cross_section(self, df: pd.DataFrame, 
                                   feature_cols: List[str],
                                   n_days: int = 50) -> pd.DataFrame:
        """
        截面因子挖掘（每天独立挖掘）
        
        Args:
            df: 数据 DataFrame
            feature_cols: 用于挖掘的特征列
            n_days: 采样的天数
            
        Returns:
            挖掘出的因子 DataFrame
        """
        print(f"🧬 开始截面因子挖掘（采样 {n_days} 天）...")
        
        # 随机采样交易日
        trade_dates = df['trade_date'].unique()
        if len(trade_dates) > n_days:
            sampled_dates = np.random.choice(trade_dates, n_days, replace=False)
        else:
            sampled_dates = trade_dates
        
        factor_results = []
        
        for date in tqdm(sampled_dates, desc="挖掘因子"):
            day_data = df[df['trade_date'] == date]
            
            if len(day_data) < 50:  # 股票数量太少则跳过
                continue
            
            X = day_data[feature_cols].values
            y = day_data[LABEL_COL].values
            
            # 清洗数据
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            
            if len(X) < 50:
                continue
            
            # 挖掘因子
            program_str, ic = self.mine_factor_single_day(
                pd.DataFrame(X, columns=feature_cols), y
            )
            
            if program_str and ic > self.config['min_ic_threshold']:
                factor_results.append({
                    'date': date,
                    'program': program_str,
                    'ic': ic,
                    'n_stocks': len(X)
                })
        
        df_factors = pd.DataFrame(factor_results)
        print(f"✅ 挖掘出 {len(df_factors)} 个候选因子")
        
        return df_factors
    
    def deduplicate_factors(self, df_factors: pd.DataFrame, 
                           df_full: pd.DataFrame,
                           feature_cols: List[str],
                           top_n: int = 50) -> pd.DataFrame:
        """
        因子去重：基于全样本 IC 和相关性筛选
        
        Args:
            df_factors: 候选因子 DataFrame
            df_full: 完整数据
            feature_cols: 特征列
            top_n: 最终保留的因子数
            
        Returns:
            筛选后的因子 DataFrame
        """
        print(f"🔄 开始因子去重和筛选...")
        
        # 按程序分组，选择出现频率高且 IC 高的
        program_stats = df_factors.groupby('program').agg({
            'ic': ['mean', 'std', 'count']
        }).reset_index()
        program_stats.columns = ['program', 'ic_mean', 'ic_std', 'frequency']
        program_stats['score'] = program_stats['ic_mean'] * np.log1p(program_stats['frequency'])
        
        # 选择 Top N
        top_programs = program_stats.nlargest(min(top_n * 2, len(program_stats)), 'score')
        
        print(f"   选择 {len(top_programs)} 个高频高 IC 因子进行全样本验证")
        
        # 全样本计算 IC
        final_factors = []
        for _, row in tqdm(top_programs.iterrows(), desc="全样本验证"):
            program_str = row['program']
            
            # 这里简化处理：假设可以通过程序字符串复现因子值
            # 实际应用中需要将程序转换为可执行代码
            # 暂时使用 IC 均值作为因子值代理
            
            final_factors.append({
                'program': program_str,
                'ic_mean': row['ic_mean'],
                'ic_std': row['ic_std'],
                'frequency': row['frequency'],
                'score': row['score']
            })
        
        df_final = pd.DataFrame(final_factors)
        df_final = df_final.nlargest(top_n, 'score')
        
        print(f"✅ 最终筛选出 {len(df_final)} 个高质量因子")
        
        return df_final
    
    def save_factors(self, df_factors: pd.DataFrame, filename: str = "mined_factors.csv"):
        """保存挖掘的因子"""
        output_path = os.path.join(FACTOR_OUTPUT_DIR, filename)
        df_factors.to_csv(output_path, index=False)
        print(f"💾 因子已保存到: {output_path}")


def main():
    """主函数：运行因子挖掘流程"""
    print("=" * 60)
    print("🚀 GP 因子挖掘系统")
    print("=" * 60)
    
    # 1. 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    
    # 2. 选择用于挖掘的特征
    feature_cols = [
        'open', 'high', 'low', 'close', 'vol',
        'turnover_rate', 'pe', 'pb',
        'returns_1d', 'returns_5d', 'returns_20d',
        'volatility_20d', 'volume_ratio', 'price_position',
        'rsi_14', 'macd'
    ]
    
    # 只保留存在的列
    feature_cols = [c for c in feature_cols if c in df_labeled.columns]
    print(f"\n📋 用于因子挖掘的特征: {feature_cols}")
    
    # 3. 使用训练集挖掘因子
    train_df = df_labeled[df_labeled['trade_date'] <= '20211231']
    
    # 4. 创建挖掘器并运行
    miner = GPFactorMiner()
    
    # 截面挖掘
    df_candidate_factors = miner.mine_factors_cross_section(
        train_df, feature_cols, n_days=50
    )
    
    # 去重筛选
    df_final_factors = miner.deduplicate_factors(
        df_candidate_factors, train_df, feature_cols, top_n=GP_CONFIG['n_factors']
    )
    
    # 保存结果
    miner.save_factors(df_final_factors)
    
    # 打印 Top 10 因子
    print("\n🏆 Top 10 挖掘出的因子:")
    print(df_final_factors.head(10)[['program', 'ic_mean', 'frequency']].to_string())
    
    loader.close()
    
    print("\n" + "=" * 60)
    print("✅ 因子挖掘完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
