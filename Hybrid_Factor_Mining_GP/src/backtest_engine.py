"""
回测引擎
基于 GP 因子 + GRU 预测进行回测
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Optional
import os
import logging

from config import OUTPUT_DIR, LABEL_COL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBacktest:
    """简化版回测引擎"""
    
    def __init__(self, initial_capital: float = 10000000.0,
                 commission: float = 0.001,  # 手续费
                 slippage: float = 0.002):   # 滑点
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.history = []
    
    def reset(self):
        """重置回测状态"""
        self.positions = {}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.history = []
    
    def run_backtest(self, df: pd.DataFrame, 
                     signal_col: str = 'pred',
                     price_col: str = 'close',
                     date_col: str = 'trade_date',
                     top_n: int = 20,
                     rebalance_freq: int = 5) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            df: 包含预测信号的数据
            signal_col: 信号列名
            price_col: 价格列名
            date_col: 日期列名
            top_n: 每日持仓数量
            rebalance_freq: 调仓频率（天）
            
        Returns:
            回测结果 DataFrame
        """
        logger.info("\n📊 开始回测...")
        
        self.reset()
        results = []
        
        # 按日期分组
        dates = sorted(df[date_col].unique())
        last_rebalance = -rebalance_freq
        current_holdings = []
        
        for i, date in enumerate(dates):
            day_data = df[df[date_col] == date].copy()
            
            if len(day_data) < top_n * 2:
                continue
            
            # 获取当日预测值
            day_data = day_data.dropna(subset=[signal_col, price_col])
            
            if len(day_data) < top_n:
                continue
            
            # 计算当前持仓市值
            portfolio_value = self.cash
            for ts_code, shares in self.positions.items():
                price_data = day_data[day_data['ts_code'] == ts_code]
                if len(price_data) > 0:
                    price = price_data[price_col].values[0]
                    portfolio_value += shares * price
            
            self.portfolio_value = portfolio_value
            
            # 记录历史
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions)
            })
            
            # 调仓日
            if i - last_rebalance >= rebalance_freq:
                # 清仓
                for ts_code, shares in list(self.positions.items()):
                    price_data = day_data[day_data['ts_code'] == ts_code]
                    if len(price_data) > 0:
                        price = price_data[price_col].values[0]
                        sell_price = price * (1 - self.slippage)
                        self.cash += shares * sell_price * (1 - self.commission)
                
                self.positions = {}
                
                # 选新标的（按预测值排序）
                day_data_sorted = day_data.sort_values(signal_col, ascending=False)
                selected = day_data_sorted.head(top_n)
                
                # 等权重买入
                weight = 1.0 / top_n
                for _, row in selected.iterrows():
                    ts_code = row['ts_code']
                    price = row[price_col]
                    buy_price = price * (1 + self.slippage)
                    
                    invest_amount = portfolio_value * weight * (1 - self.commission)
                    shares = int(invest_amount / buy_price)
                    
                    if shares > 0:
                        self.positions[ts_code] = shares
                        self.cash -= shares * buy_price
                
                last_rebalance = i
                logger.info(f"   {date}: 调仓，持有 {len(self.positions)} 只股票，"
                           f"净值: {portfolio_value/self.initial_capital:.4f}")
        
        df_results = pd.DataFrame(results)
        return df_results
    
    def calculate_metrics(self, df_results: pd.DataFrame, 
                         benchmark: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算回测指标
        
        Args:
            df_results: 回测结果
            benchmark: 基准数据（可选）
            
        Returns:
            指标字典
        """
        df = df_results.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 计算收益率
        df['returns'] = df['portfolio_value'].pct_change()
        
        # 年化收益率
        total_return = df['portfolio_value'].iloc[-1] / self.initial_capital - 1
        n_years = len(df) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 波动率
        volatility = df['returns'].std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率 2%）
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (df['returns'] > 0).mean()
        
        # 盈亏比
        avg_gain = df[df['returns'] > 0]['returns'].mean()
        avg_loss = abs(df[df['returns'] < 0]['returns'].mean())
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # 与基准比较（如果有）
        if benchmark is not None:
            benchmark['date'] = pd.to_datetime(benchmark['date'])
            benchmark = benchmark.set_index('date').sort_index()
            
            # 计算基准收益率
            common_dates = df.index.intersection(benchmark.index)
            if len(common_dates) > 0:
                strategy_returns = df.loc[common_dates, 'returns']
                benchmark_returns = benchmark.loc[common_dates, 'returns']
                
                # 信息比率
                tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
                excess_return = (strategy_returns - benchmark_returns).mean() * 252
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            else:
                information_ratio = 0
        else:
            information_ratio = 0
        
        metrics = {
            '总收益率': f"{total_return:.2%}",
            '年化收益率': f"{annual_return:.2%}",
            '年化波动率': f"{volatility:.2%}",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{profit_loss_ratio:.2f}",
            '信息比率': f"{information_ratio:.2f}",
            '交易天数': len(df)
        }
        
        return metrics
    
    def plot_results(self, df_results: pd.DataFrame, 
                     save_path: Optional[str] = None):
        """
        绘制回测结果
        """
        df = df_results.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['nav'] = df['portfolio_value'] / self.initial_capital
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 净值曲线
        axes[0].plot(df['date'], df['nav'], label='Strategy', linewidth=2)
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Net Asset Value')
        axes[0].set_title('Backtest Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 回撤曲线
        cummax = df['nav'].cummax()
        drawdown = (df['nav'] - cummax) / cummax
        axes[1].fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
        axes[1].plot(df['date'], drawdown, color='red', linewidth=1)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Drawdown')
        axes[1].set_title('Drawdown')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 回测图表已保存: {save_path}")
        else:
            save_path = os.path.join(OUTPUT_DIR, 'backtest_result.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


def run_backtest_analysis(pred_df: pd.DataFrame, save_results: bool = True):
    """
    运行回测分析
    
    Args:
        pred_df: 包含预测值的 DataFrame
        save_results: 是否保存结果
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 回测分析")
    logger.info("=" * 60)
    
    engine = SimpleBacktest(
        initial_capital=10000000.0,
        commission=0.001,
        slippage=0.002
    )
    
    # 运行回测
    df_results = engine.run_backtest(
        pred_df,
        signal_col='pred',
        top_n=20,
        rebalance_freq=5
    )
    
    if len(df_results) == 0:
        logger.error("❌ 回测结果为空")
        return
    
    # 计算指标
    metrics = engine.calculate_metrics(df_results)
    
    logger.info("\n📈 回测指标:")
    for key, value in metrics.items():
        logger.info(f"   {key}: {value}")
    
    # 绘图
    engine.plot_results(df_results)
    
    # 保存结果
    if save_results:
        output_path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
        df_results.to_csv(output_path, index=False)
        logger.info(f"💾 回测结果已保存: {output_path}")
        
        # 保存指标
        metrics_path = os.path.join(OUTPUT_DIR, 'backtest_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("回测指标\n")
            f.write("=" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"💾 指标已保存: {metrics_path}")
    
    return df_results, metrics


def main():
    """主函数"""
    print("=" * 60)
    print("📊 回测引擎")
    print("=" * 60)
    
    # 加载预测结果
    from data_loader import DataLoader
    from two_stage_model_v2 import TwoStageModelV2
    
    # 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    # 只使用测试集
    test_df = df_labeled[df_labeled['trade_date'] > '20221231']
    
    # 加载模型并预测
    model = TwoStageModelV2()
    if model.load_model("gru_best_v2.pth"):
        # 计算 GP 因子
        base_features = ['open', 'high', 'low', 'close', 'vol', 'ret_1d', 'ret_5d']
        base_features = [c for c in base_features if c in test_df.columns]
        test_df = model.compute_gp_factors(test_df, base_features)
        
        # 预测
        pred_df = model.predict(test_df)
        
        # 运行回测
        run_backtest_analysis(pred_df)
    else:
        print("⚠️ 未找到训练好的模型，请先运行训练")


if __name__ == "__main__":
    main()
