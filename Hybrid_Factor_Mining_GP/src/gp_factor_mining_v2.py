"""
GP 因子挖掘模块 V2
支持真正的因子公式解析与执行
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.functions import make_function
import pickle
import os
import re
from typing import List, Dict, Tuple, Callable
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import GP_CONFIG, FACTOR_OUTPUT_DIR, LABEL_COL
from data_loader import DataLoader


# ==================== 自定义 GP 函数 ====================

def _safe_div(x, y):
    """安全除法"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(y) > 1e-8, x / y, 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_log(x):
    """安全对数"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x > 1e-8, np.log(x), 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_sqrt(x):
    """安全开方"""
    with np.errstate(invalid='ignore'):
        result = np.where(x >= 0, np.sqrt(x), 0.0)
    return np.nan_to_num(result, nan=0.0)

def _rank(x):
    """截面排名（0-1）"""
    if len(x) == 0:
        return x
    return pd.Series(x).rank(pct=True).fillna(0.5).values

def _ts_mean(x, d):
    """时序均值"""
    if len(x) < d:
        return np.zeros_like(x)
    return pd.Series(x).rolling(window=int(d), min_periods=1).mean().values

def _ts_std(x, d):
    """时序标准差"""
    if len(x) < d:
        return np.zeros_like(x)
    return pd.Series(x).rolling(window=int(d), min_periods=1).std().fillna(0).values

def _ts_max(x, d):
    """时序最大值"""
    if len(x) < d:
        return np.zeros_like(x)
    return pd.Series(x).rolling(window=int(d), min_periods=1).max().values

def _ts_min(x, d):
    """时序最小值"""
    if len(x) < d:
        return np.zeros_like(x)
    return pd.Series(x).rolling(window=int(d), min_periods=1).min().values

def _ts_delta(x, d):
    """时序差分"""
    if len(x) < d:
        return np.zeros_like(x)
    return pd.Series(x).diff(periods=int(d)).fillna(0).values

def _ts_returns(x, d):
    """时序收益率"""
    if len(x) < d + 1:
        return np.zeros_like(x)
    x_prev = pd.Series(x).shift(periods=int(d))
    return _safe_div(x - x_prev, x_prev)

def _ts_corr(x, y, d):
    """时序相关性"""
    if len(x) < d:
        return np.zeros_like(x)
    sx = pd.Series(x)
    sy = pd.Series(y)
    return sx.rolling(window=int(d), min_periods=2).corr(sy).fillna(0).values

def _ts_rank(x, d):
    """时序排名"""
    if len(x) < d:
        return np.zeros_like(x)
    s = pd.Series(x)
    return s.rolling(window=int(d), min_periods=1).apply(
        lambda y: y.rank(pct=True).iloc[-1] if len(y) > 0 else 0.5, raw=False
    ).fillna(0.5).values

def _sign(x):
    """符号函数"""
    return np.sign(x)

def _sigmoid(x):
    """Sigmoid 函数"""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

# 创建 GP 函数
rank_func = make_function(function=_rank, name='rank', arity=1)
sign_func = make_function(function=_sign, name='sign', arity=1)
sigmoid_func = make_function(function=_sigmoid, name='sigmoid', arity=1)


class FactorProgramExecutor:
    """因子程序执行器 - 将 GP 程序字符串转为可执行函数"""
    
    # 函数映射
    FUNCTION_MAP = {
        'add': lambda x, y: np.add(x, y),
        'sub': lambda x, y: np.subtract(x, y),
        'mul': lambda x, y: np.multiply(x, y),
        'div': lambda x, y: _safe_div(x, y),
        'sqrt': lambda x: _safe_sqrt(x),
        'log': lambda x: _safe_log(x),
        'abs': lambda x: np.abs(x),
        'neg': lambda x: np.negative(x),
        'inv': lambda x: _safe_div(1.0, x),
        'max': lambda x, y: np.maximum(x, y),
        'min': lambda x, y: np.minimum(x, y),
        'rank': lambda x: _rank(x),
        'sign': lambda x: _sign(x),
        'sigmoid': lambda x: _sigmoid(x),
    }
    
    def __init__(self, program_str: str, feature_names: List[str]):
        """
        Args:
            program_str: GP 程序字符串，如 "mul(X0, sub(X1, X2))"
            feature_names: 特征名列表，如 ['open', 'close', 'high']
        """
        self.program_str = program_str
        self.feature_names = feature_names
        self.program_tree = self._parse_program(program_str)
    
    def _parse_program(self, s: str) -> dict:
        """解析程序字符串为树结构"""
        s = s.strip()
        
        # 检查是否是函数调用
        match = re.match(r'(\w+)\((.*)\)', s)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            # 分割参数（处理嵌套括号）
            args = self._split_args(args_str)
            return {
                'type': 'function',
                'name': func_name,
                'args': [self._parse_program(a) for a in args]
            }
        else:
            # 是变量或常数
            return {'type': 'variable', 'name': s}
    
    def _split_args(self, s: str) -> List[str]:
        """分割函数参数"""
        args = []
        depth = 0
n        current = []
        
        for char in s:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return args
    
    def execute(self, df: pd.DataFrame, groupby_col: str = 'ts_code') -> pd.Series:
        """
        执行程序计算因子值
        
        Args:
            df: 数据 DataFrame
            groupby_col: 分组列（用于截面计算）
            
        Returns:
            因子值 Series
        """
        results = []
        
        for name, group in df.groupby(groupby_col):
            group_values = self._eval_node(self.program_tree, group)
            results.append(pd.Series(group_values, index=group.index))
        
        return pd.concat(results)
    
    def _eval_node(self, node: dict, df: pd.DataFrame) -> np.ndarray:
        """递归求值节点"""
        if node['type'] == 'variable':
            name = node['name']
            # 检查是否是特征
            if name in self.feature_names:
                return df[name].values
            # 检查是否是 X0, X1, ...
            elif name.startswith('X') and name[1:].isdigit():
                idx = int(name[1:])
                if idx < len(self.feature_names):
                    return df[self.feature_names[idx]].values
                else:
                    return np.zeros(len(df))
            # 常数
            else:
                try:
                    return np.full(len(df), float(name))
                except:
                    return np.zeros(len(df))
        
        elif node['type'] == 'function':
            func_name = node['name']
            args = [self._eval_node(a, df) for a in node['args']]
            
            if func_name in self.FUNCTION_MAP:
                return self.FUNCTION_MAP[func_name](*args)
            else:
                return np.zeros(len(df))
        
        return np.zeros(len(df))


class GPFactorMinerV2:
    """GP 因子挖掘器 V2"""
    
    def __init__(self, config: Dict = None):
        self.config = config or GP_CONFIG
        self.factor_programs = []
        self.executors = []
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        准备用于 GP 的特征
        
        Returns:
            (处理后的 DataFrame, 特征名列表)
        """
        # 基础特征
        base_features = ['open', 'high', 'low', 'close', 'vol', 'amount']
        derived_features = []
        
        # 计算衍生特征
        df = df.copy()
        
        # 收益率特征
        df['ret_1d'] = df.groupby('ts_code')['close'].pct_change()
        df['ret_5d'] = df.groupby('ts_code')['close'].pct_change(5)
        derived_features.extend(['ret_1d', 'ret_5d'])
        
        # 价格形态特征
        df['high_low_ratio'] = _safe_div(df['high'], df['low'])
        df['close_open_ratio'] = _safe_div(df['close'], df['open'])
        df['price_range'] = _safe_div(df['high'] - df['low'], df['close'])
        derived_features.extend(['high_low_ratio', 'close_open_ratio', 'price_range'])
        
        # 成交量特征
        df['vol_ma5'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())
        df['vol_ratio'] = _safe_div(df['vol'], df['vol_ma5'])
        derived_features.extend(['vol_ma5', 'vol_ratio'])
        
        # 技术指标（如果有）
        if 'turnover_rate' in df.columns:
            df['turnover'] = df['turnover_rate'] / 100
            derived_features.append('turnover')
        
        # 填充缺失值
        all_features = base_features + derived_features
        all_features = [f for f in all_features if f in df.columns]
        
        for f in all_features:
            df[f] = df[f].fillna(0).replace([np.inf, -np.inf], 0)
        
        return df, all_features
    
    def calculate_ic(self, factor_values: np.ndarray, labels: np.ndarray) -> float:
        """计算 Rank IC"""
        mask = ~(np.isnan(factor_values) | np.isnan(labels) | 
                 np.isinf(factor_values) | np.isinf(labels))
        
        if mask.sum() < 10:
            return 0.0
        
        f = pd.Series(factor_values[mask]).rank()
        l = pd.Series(labels[mask]).rank()
        
        ic = np.corrcoef(f, l)[0, 1]
        return ic if not np.isnan(ic) else 0.0
    
    def mine_factors_symbolic_transformer(self, df: pd.DataFrame, 
                                          feature_cols: List[str],
                                          n_factors: int = 50) -> List[str]:
        """
        使用 SymbolicTransformer 挖掘因子
        比 SymbolicRegressor 更适合多因子挖掘
        """
        logger.info(f"🧬 开始 GP 因子挖掘（目标 {n_factors} 个）...")
        
        # 准备数据
        df_clean = df[feature_cols + [LABEL_COL, 'ts_code', 'trade_date']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean[LABEL_COL].values
        
        # 清洗数据
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        logger.info(f"   使用 {len(X)} 条样本进行挖掘")
        
        # 配置函数集
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min']
        function_set += [rank_func, sign_func]  # 添加自定义函数
        
        # 创建 SymbolicTransformer
        gp = SymbolicTransformer(
            generations=self.config['generations'],
            population_size=self.config['population_size'],
            hall_of_fame=n_factors,
            n_components=n_factors,
            function_set=function_set,
            parsimony_coefficient=self.config['parsimony_coefficient'],
            max_samples=self.config['max_samples'],
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("   开始进化...")
        gp.fit(X, y)
        
        # 提取程序字符串
        programs = []
        for i, program in enumerate(gp.best_programs_):
            if i >= n_factors:
                break
            program_str = str(program)
            programs.append(program_str)
        
        logger.info(f"✅ 挖掘出 {len(programs)} 个因子程序")
        return programs
    
    def validate_factors(self, programs: List[str], df: pd.DataFrame,
                        feature_cols: List[str], min_ic: float = 0.02) -> pd.DataFrame:
        """
        验证因子有效性
        
        Returns:
            DataFrame 包含因子信息和 IC 统计
        """
        logger.info(f"🔍 验证 {len(programs)} 个因子...")
        
        results = []
        
        for i, program_str in enumerate(tqdm(programs, desc="验证因子")):
            try:
                # 创建执行器
                executor = FactorProgramExecutor(program_str, feature_cols)
                
                # 计算因子值
                factor_values = executor.execute(df)
                
                # 计算每日 IC
                ic_list = []
                for date, group in df.groupby('trade_date'):
                    if len(group) < 20:
                        continue
                    ic = self.calculate_ic(
                        factor_values[group.index].values,
                        group[LABEL_COL].values
                    )
                    if not np.isnan(ic):
                        ic_list.append(ic)
                
                if len(ic_list) > 0:
                    ic_mean = np.mean(ic_list)
                    ic_std = np.std(ic_list)
                    ir = ic_mean / (ic_std + 1e-8)
                    
                    results.append({
                        'program': program_str,
                        'ic_mean': ic_mean,
                        'ic_std': ic_std,
                        'ir': ir,
                        'ic_positive_ratio': np.mean([ic > 0 for ic in ic_list]),
                        'valid': abs(ic_mean) >= min_ic
                    })
                else:
                    results.append({
                        'program': program_str,
                        'ic_mean': 0,
                        'ic_std': 0,
                        'ir': 0,
                        'ic_positive_ratio': 0,
                        'valid': False
                    })
                    
            except Exception as e:
                logger.warning(f"   验证因子 {i} 失败: {e}")
                results.append({
                    'program': program_str,
                    'ic_mean': 0,
                    'ic_std': 0,
                    'ir': 0,
                    'valid': False
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('ir', ascending=False)
        
        valid_count = df_results['valid'].sum()
        logger.info(f"✅ 验证完成，{valid_count}/{len(programs)} 个因子通过 IC 阈值")
        
        return df_results
    
    def select_diverse_factors(self, df_results: pd.DataFrame, 
                               df: pd.DataFrame,
                               feature_cols: List[str],
                               top_n: int = 30,
                               max_corr: float = 0.7) -> pd.DataFrame:
        """
        选择相关性低的多样化因子
        """
        logger.info(f"🔄 选择低相关性因子（目标 {top_n} 个，最大相关性 {max_corr}）...")
        
        # 只选择有效的因子
        df_valid = df_results[df_results['valid']].copy()
        
        if len(df_valid) == 0:
            logger.warning("⚠️ 没有有效因子")
            return df_results.head(top_n)
        
        # 计算因子值
        factor_values_list = []
        for _, row in df_valid.iterrows():
            try:
                executor = FactorProgramExecutor(row['program'], feature_cols)
                values = executor.execute(df).values
                factor_values_list.append(values)
            except:
                factor_values_list.append(np.zeros(len(df)))
        
        # 计算相关性矩阵
        factor_matrix = np.column_stack(factor_values_list)
        corr_matrix = np.corrcoef(factor_matrix.T)
        
        # 贪心选择低相关性因子
        selected_indices = []
        for i in range(len(df_valid)):
            if len(selected_indices) == 0:
                selected_indices.append(i)
            else:
                # 检查与已选因子的最大相关性
                max_correlation = max([abs(corr_matrix[i, j]) for j in selected_indices])
                if max_correlation < max_corr:
                    selected_indices.append(i)
            
            if len(selected_indices) >= top_n:
                break
        
        selected_programs = df_valid.iloc[selected_indices]['program'].tolist()
        
        # 标记选中的因子
        df_results['selected'] = df_results['program'].isin(selected_programs)
        
        logger.info(f"✅ 选中 {len(selected_indices)} 个低相关性因子")
        return df_results
    
    def save_factors(self, df_results: pd.DataFrame, filename: str = "mined_factors_v2.csv"):
        """保存因子"""
        output_path = os.path.join(FACTOR_OUTPUT_DIR, filename)
        df_results.to_csv(output_path, index=False)
        logger.info(f"💾 因子已保存: {output_path}")


def main():
    """主函数"""
    import logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("🚀 GP 因子挖掘系统 V2")
    print("=" * 60)
    
    # 1. 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    loader.close()
    
    # 2. 初始化挖掘器
    miner = GPFactorMinerV2()
    
    # 3. 准备特征
    df_features, feature_cols = miner.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    
    # 只使用训练集挖掘
    train_df = df_labeled[df_labeled['trade_date'] <= '20211231']
    
    logger.info(f"\n📋 使用特征: {feature_cols}")
    logger.info(f"   训练集样本: {len(train_df)}")
    
    # 4. 挖掘因子
    programs = miner.mine_factors_symbolic_transformer(
        train_df, feature_cols, n_factors=GP_CONFIG['n_factors']
    )
    
    # 5. 验证因子
    df_results = miner.validate_factors(
        programs, train_df, feature_cols, min_ic=GP_CONFIG['min_ic_threshold']
    )
    
    # 6. 选择多样化因子
    df_results = miner.select_diverse_factors(
        df_results, train_df, feature_cols,
        top_n=30, max_corr=GP_CONFIG['max_correlation']
    )
    
    # 7. 保存结果
    miner.save_factors(df_results)
    
    # 8. 打印结果
    print("\n" + "=" * 60)
    print("🏆 Top 10 因子（按 IR 排序）:")
    print("=" * 60)
    top10 = df_results.head(10)
    for idx, row in top10.iterrows():
        print(f"\n[{idx+1}] IR={row['ir']:.3f}, IC={row['ic_mean']:.4f}")
        print(f"    程序: {row['program'][:80]}...")
        print(f"    选中: {'✅' if row.get('selected', False) else '❌'}")
    
    print("\n" + "=" * 60)
    print("✅ 因子挖掘完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
