"""
因子分析模块
计算 IC、换手率、相关性等指标
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

from config import OUTPUT_DIR, LABEL_COL


class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def calculate_ic(factor_values: pd.Series, labels: pd.Series) -> float:
        """
        计算 Rank IC
        
        Args:
            factor_values: 因子值
            labels: 标签（未来收益率）
            
        Returns:
            Rank IC
        """
        df = pd.DataFrame({'factor': factor_values, 'label': labels})
        df = df.dropna()
        
        if len(df) < 10:
            return np.nan
        
        return df['factor'].rank().corr(df['label'].rank())
    
    @staticmethod
    def calculate_turnover(factor_values: pd.Series, 
                          dates: pd.Series,
                          ts_codes: pd.Series) -> float:
        """
        计算因子换手率（时序稳定性）
        
        Args:
            factor_values: 因子值
            dates: 日期
            ts_codes: 股票代码
            
        Returns:
            平均换手率
        """
        df = pd.DataFrame({
            'factor': factor_values,
            'date': dates,
            'ts_code': ts_codes
        })
        
        # 按日期计算截面排名变化
        turnover_list = []
        for date, group in df.groupby('date'):
            if len(group) < 10:
                continue
            # 换手率 = 排名变化的绝对值之和 / 2
            # 简化：用标准差衡量
            turnover_list.append(group['factor'].std())
        
        return np.mean(turnover_list) if turnover_list else np.nan
    
    def analyze_factor(self, df: pd.DataFrame, factor_col: str) -> Dict:
        """
        分析单个因子
        
        Args:
            df: 数据 DataFrame
            factor_col: 因子列名
            
        Returns:
            分析结果字典
        """
        results = {
            'factor_name': factor_col,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'ir': np.nan,  # IR = IC_mean / IC_std
            'turnover': np.nan,
            'coverage': 0,  # 覆盖率
        }
        
        # 计算每日 IC
        ic_list = []
        for date, group in df.groupby('trade_date'):
            if factor_col not in group.columns or LABEL_COL not in group.columns:
                continue
            
            ic = self.calculate_ic(group[factor_col], group[LABEL_COL])
            if not np.isnan(ic):
                ic_list.append(ic)
        
        if ic_list:
            results['ic_mean'] = np.mean(ic_list)
            results['ic_std'] = np.std(ic_list)
            results['ir'] = results['ic_mean'] / (results['ic_std'] + 1e-8)
        
        # 计算换手率
        if factor_col in df.columns:
            results['turnover'] = self.calculate_turnover(
                df[factor_col], df['trade_date'], df['ts_code']
            )
            results['coverage'] = df[factor_col].notna().mean()
        
        return results
    
    def analyze_all_factors(self, df: pd.DataFrame, 
                           factor_cols: List[str]) -> pd.DataFrame:
        """
        批量分析多个因子
        
        Args:
            df: 数据 DataFrame
            factor_cols: 因子列名列表
            
        Returns:
            分析结果 DataFrame
        """
        print(f"🔍 分析 {len(factor_cols)} 个因子...")
        
        results = []
        for factor_col in factor_cols:
            result = self.analyze_factor(df, factor_col)
            results.append(result)
        
        df_results = pd.DataFrame(results)
        return df_results.sort_values('ir', ascending=False)
    
    @staticmethod
    def calculate_correlation(df: pd.DataFrame, 
                             factor_cols: List[str]) -> pd.DataFrame:
        """
        计算因子间相关性
        
        Args:
            df: 数据 DataFrame
            factor_cols: 因子列名列表
            
        Returns:
            相关性矩阵
        """
        # 按日期计算截面平均相关性
        corr_list = []
        
        for date, group in df.groupby('trade_date'):
            factor_data = group[factor_cols].dropna()
            if len(factor_data) < 10:
                continue
            corr_list.append(factor_data.corr().values)
        
        if corr_list:
            avg_corr = np.mean(corr_list, axis=0)
            return pd.DataFrame(avg_corr, index=factor_cols, columns=factor_cols)
        else:
            return pd.DataFrame(np.eye(len(factor_cols)), 
                               index=factor_cols, columns=factor_cols)
    
    def plot_ic_distribution(self, df: pd.DataFrame, factor_cols: List[str],
                            save_path: str = None):
        """
        绘制 IC 分布图
        
        Args:
            df: 数据 DataFrame
            factor_cols: 因子列名列表
            save_path: 保存路径
        """
        # 计算每个因子的每日 IC
        ic_data = {}
        for factor_col in factor_cols[:10]:  # 只画前 10 个
            ic_list = []
            dates = []
            for date, group in df.groupby('trade_date'):
                ic = self.calculate_ic(group[factor_col], group[LABEL_COL])
                if not np.isnan(ic):
                    ic_list.append(ic)
                    dates.append(date)
            ic_data[factor_col] = ic_list
        
        # 绘图
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # IC 时序图
        for factor_col, ic_values in ic_data.items():
            axes[0].plot(ic_values, label=factor_col, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Rank IC')
        axes[0].set_title('Factor IC Time Series')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # IC 分布图
        ic_means = [np.mean(v) for v in ic_data.values()]
        ic_stds = [np.std(v) for v in ic_data.values()]
        axes[1].bar(range(len(ic_means)), ic_means, yerr=ic_stds, capsize=5)
        axes[1].set_xticks(range(len(ic_means)))
        axes[1].set_xticklabels(ic_data.keys(), rotation=45, ha='right')
        axes[1].set_ylabel('Mean IC')
        axes[1].set_title('Factor IC Mean and Std')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 图表已保存: {save_path}")
        else:
            plt.savefig(os.path.join(OUTPUT_DIR, 'ic_distribution.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame,
                                save_path: str = None):
        """
        绘制相关性热力图
        
        Args:
            corr_matrix: 相关性矩阵
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Factor Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, df_analysis: pd.DataFrame,
                       corr_matrix: pd.DataFrame,
                       save_path: str = None):
        """
        生成因子分析报告
        
        Args:
            df_analysis: 因子分析结果
            corr_matrix: 相关性矩阵
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, 'factor_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("📊 因子分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 总体统计
            f.write("【总体统计】\n")
            f.write(f"因子数量: {len(df_analysis)}\n")
            f.write(f"平均 IC: {df_analysis['ic_mean'].mean():.4f}\n")
            f.write(f"平均 IR: {df_analysis['ir'].mean():.4f}\n")
            f.write(f"高 IC 因子数 (|IC|>0.03): {(df_analysis['ic_mean'].abs() > 0.03).sum()}\n\n")
            
            # Top 10 因子
            f.write("【Top 10 因子】\n")
            top10 = df_analysis.head(10)
            for idx, row in top10.iterrows():
                f.write(f"\n{row['factor_name']}:\n")
                f.write(f"  IC Mean: {row['ic_mean']:.4f}\n")
                f.write(f"  IC Std:  {row['ic_std']:.4f}\n")
                f.write(f"  IR:      {row['ir']:.4f}\n")
                f.write(f"  Turnover:{row['turnover']:.4f}\n")
            
            # 高相关性因子对
            f.write("\n【高相关性因子对 (|corr| > 0.7)】\n")
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'factor1': corr_matrix.index[i],
                            'factor2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                for pair in high_corr_pairs:
                    f.write(f"  {pair['factor1']} - {pair['factor2']}: {pair['correlation']:.4f}\n")
            else:
                f.write("  无高相关性因子对\n")
        
        print(f"💾 分析报告已保存: {save_path}")


def main():
    """测试因子分析"""
    from data_loader import DataLoader
    
    print("=" * 60)
    print("🔍 因子分析测试")
    print("=" * 60)
    
    # 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    # 选择要分析的因子
    factor_cols = [c for c in df_labeled.columns 
                  if c in ['returns_1d', 'returns_5d', 'volatility_20d',
                          'volume_ratio', 'rsi_14', 'macd']]
    
    print(f"\n分析因子: {factor_cols}")
    
    # 创建分析器
    analyzer = FactorAnalyzer()
    
    # 分析所有因子
    df_analysis = analyzer.analyze_all_factors(df_labeled, factor_cols)
    
    print("\n📊 分析结果:")
    print(df_analysis.to_string())
    
    # 计算相关性
    corr_matrix = analyzer.calculate_correlation(df_labeled, factor_cols)
    
    # 生成图表和报告
    analyzer.plot_ic_distribution(df_labeled, factor_cols)
    analyzer.plot_correlation_matrix(corr_matrix)
    analyzer.generate_report(df_analysis, corr_matrix)
    
    print("\n✅ 因子分析完成！")


if __name__ == "__main__":
    main()
