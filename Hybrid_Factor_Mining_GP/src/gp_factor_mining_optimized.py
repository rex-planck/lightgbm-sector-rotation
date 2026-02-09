"""
优化版GP因子挖掘
种群1500，进化30代
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
import pickle
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config_optimized import GP_CONFIG_OPTIMIZED, FACTOR_OUTPUT_DIR


# 自定义函数
def rank(x):
    return pd.Series(x).rank(pct=True).fillna(0.5).values if len(x) > 1 else np.zeros_like(x)

def ts_mean(x, d=10):
    return pd.Series(x).rolling(window=int(d), min_periods=1).mean().values

def ts_std(x, d=10):
    return pd.Series(x).rolling(window=int(d), min_periods=1).std().fillna(0).values

def safe_div(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(y) > 1e-8, x / y, 0.0)

rank_func = make_function(function=rank, name='rank', arity=1)


class OptimizedGPFactorMiner:
    """优化版GP因子挖掘器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or GP_CONFIG_OPTIMIZED
        self.programs = []
        self.executors = []
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], label_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """准备数据"""
        df_clean = df[feature_cols + [label_col]].dropna()
        
        # 限制样本量以控制计算时间
        max_samples = 50000
        if len(df_clean) > max_samples:
            df_clean = df_clean.sample(n=max_samples, random_state=42)
        
        X = df_clean[feature_cols].values
        y = df_clean[label_col].values
        
        # 清洗异常值
        X = np.clip(X, -5, 5)
        y = np.clip(y, -5, 5)
        
        return X, y
    
    def mine_factors(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """挖掘因子"""
        print(f"\n[GP Mining] Starting genetic programming...")
        print(f"  Population: {self.config['population_size']}")
        print(f"  Generations: {self.config['generations']}")
        print(f"  Training samples: {len(X)}")
        
        # 配置函数集
        function_set = self.config['function_set'] + [rank_func]
        
        # 创建GP模型
        gp = SymbolicTransformer(
            generations=self.config['generations'],
            population_size=self.config['population_size'],
            hall_of_fame=self.config['n_factors'],
            n_components=self.config['n_factors'],
            function_set=function_set,
            parsimony_coefficient=self.config['parsimony_coefficient'],
            max_samples=self.config['max_samples'],
            stopping_criteria=self.config['stopping_criteria'],
            p_crossover=self.config['p_crossover'],
            p_subtree_mutation=self.config['p_subtree_mutation'],
            p_hoist_mutation=self.config['p_hoist_mutation'],
            p_point_mutation=self.config['p_point_mutation'],
            tournament_size=self.config['tournament_size'],
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
        
        print("  Evolving (this may take 20-40 minutes)...")
        gp.fit(X, y)
        
        # 提取程序
        try:
            self.programs = [str(p) for p in gp._best_programs[:self.config['n_factors']]]
        except:
            # 如果_best_programs不存在，尝试其他方式
            self.programs = [f'Factor_{i}' for i in range(self.config['n_factors'])]
        
        print(f"\n[OK] Mined {len(self.programs)} factors")
        return self.programs, gp
    
    def compute_factor_values(self, df: pd.DataFrame, gp_model, feature_cols: List[str]) -> pd.DataFrame:
        """计算因子值"""
        print("\n[GP Mining] Computing factor values...")
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X = np.clip(X, -5, 5)
        
        # 使用GP模型转换
        factors = gp_model.transform(X)
        
        # 添加因子列
        n_factors = min(factors.shape[1], self.config['n_factors'])
        for i in range(n_factors):
            df[f'gp_factor_{i}'] = factors[:, i]
        
        print(f"  Added {n_factors} GP factor columns")
        return df
    
    def validate_factors(self, df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
        """验证因子有效性"""
        print("\n[GP Mining] Validating factors...")
        
        gp_cols = [c for c in df.columns if c.startswith('gp_factor_')]
        results = []
        
        for col in tqdm(gp_cols, desc="Calculating IC"):
            # 计算IC
            ic_list = []
            for date, group in df.groupby('trade_date'):
                if len(group) < 30:
                    continue
                f = group[col].values
                l = group[label_col].values
                
                mask = ~(np.isnan(f) | np.isnan(l))
                if mask.sum() < 20:
                    continue
                
                ic = np.corrcoef(pd.Series(f[mask]).rank(), pd.Series(l[mask]).rank())[0, 1]
                if not np.isnan(ic):
                    ic_list.append(ic)
            
            if ic_list:
                results.append({
                    'factor': col,
                    'ic_mean': np.mean(ic_list),
                    'ic_std': np.std(ic_list),
                    'ir': np.mean(ic_list) / (np.std(ic_list) + 1e-8),
                    'ic_positive_ratio': np.mean([ic > 0 for ic in ic_list]),
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('ir', key=abs, ascending=False)
        
        print(f"\n  Validation completed for {len(results)} factors")
        print(f"  Best factor IR: {df_results.iloc[0]['ir']:.4f}" if len(df_results) > 0 else "  No valid factors")
        
        return df_results
    
    def select_diverse_factors(self, df_results: pd.DataFrame, df: pd.DataFrame, 
                               top_n: int = 30, max_corr: float = 0.6) -> List[str]:
        """选择低相关性因子"""
        print(f"\n[GP Mining] Selecting {top_n} diverse factors...")
        
        # 筛选高质量因子
        good_factors = df_results[abs(df_results['ic_mean']) > self.config['min_ic_threshold']]
        
        if len(good_factors) == 0:
            print("  Warning: No factors meet IC threshold, using top by IR")
            good_factors = df_results.head(top_n)
        
        gp_cols = good_factors['factor'].tolist()
        
        # 计算相关性矩阵
        corr_matrix = df[gp_cols].corr().abs()
        
        # 贪心选择
        selected = []
        for col in gp_cols:
            if len(selected) == 0:
                selected.append(col)
            else:
                max_correlation = max([corr_matrix.loc[col, s] for s in selected])
                if max_correlation < max_corr:
                    selected.append(col)
            
            if len(selected) >= top_n:
                break
        
        print(f"  Selected {len(selected)} factors")
        return selected
    
    def save_results(self, df_results: pd.DataFrame, selected_factors: List[str], 
                     train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        """保存结果"""
        print("\n[GP Mining] Saving results...")
        
        # 保存因子统计
        df_results.to_csv(f'{FACTOR_OUTPUT_DIR}/gp_factor_stats_optimized.csv', index=False)
        
        # 保存选中因子列表
        with open(f'{FACTOR_OUTPUT_DIR}/selected_factors.txt', 'w') as f:
            for factor in selected_factors:
                f.write(factor + '\n')
        
        # 保存数据
        train_df.to_pickle(f'{FACTOR_OUTPUT_DIR}/train_data_optimized.pkl')
        valid_df.to_pickle(f'{FACTOR_OUTPUT_DIR}/valid_data_optimized.pkl')
        test_df.to_pickle(f'{FACTOR_OUTPUT_DIR}/test_data_optimized.pkl')
        
        print(f"  Results saved to {FACTOR_OUTPUT_DIR}")


def main():
    """测试"""
    from feature_engineering_enhanced import EnhancedFeatureEngineer
    from config_optimized import TRAIN_END, VALID_END
    
    print("="*70)
    print("OPTIMIZED GP FACTOR MINING")
    print("="*70)
    
    # 加载数据
    engineer = EnhancedFeatureEngineer()
    df_raw = engineer.load_data()
    df_features = engineer.calculate_features(df_raw)
    df_labeled = engineer.prepare_labels(df_features)
    feature_cols = engineer.get_feature_columns(df_labeled)
    
    # 划分数据集
    train_df = df_labeled[df_labeled['trade_date'] <= TRAIN_END].copy()
    valid_df = df_labeled[(df_labeled['trade_date'] > TRAIN_END) & 
                          (df_labeled['trade_date'] <= VALID_END)].copy()
    test_df = df_labeled[df_labeled['trade_date'] > VALID_END].copy()
    
    print(f"\nDataset split: Train {len(train_df)}, Valid {len(valid_df)}, Test {len(test_df)}")
    print(f"Features: {len(feature_cols)}")
    
    # GP挖掘
    miner = OptimizedGPFactorMiner()
    X_train, y_train = miner.prepare_data(train_df, feature_cols)
    programs, gp_model = miner.mine_factors(X_train, y_train, feature_cols)
    
    # 计算因子值
    train_df = miner.compute_factor_values(train_df, gp_model, feature_cols)
    valid_df = miner.compute_factor_values(valid_df, gp_model, feature_cols)
    test_df = miner.compute_factor_values(test_df, gp_model, feature_cols)
    
    # 验证
    df_results = miner.validate_factors(train_df)
    selected = miner.select_diverse_factors(df_results, train_df)
    
    # 保存
    miner.save_results(df_results, selected, train_df, valid_df, test_df)
    
    engineer.close()
    
    print("\n" + "="*70)
    print("GP FACTOR MINING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
