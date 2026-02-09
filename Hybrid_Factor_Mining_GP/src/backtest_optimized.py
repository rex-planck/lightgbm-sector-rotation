"""
优化版回测引擎
包含：仓位管理、止损止盈、动态权重
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

from config_optimized import BACKTEST_CONFIG, OUTPUT_DIR


class OptimizedBacktest:
    """优化版回测引擎"""
    
    def __init__(self, config: Dict = None):
        self.config = config or BACKTEST_CONFIG
        self.initial_capital = self.config['initial_capital']
        self.commission = self.config['commission']
        self.slippage = self.config['slippage']
        self.stop_loss = self.config['stop_loss']
        self.take_profit = self.config['take_profit']
        self.max_weight = self.config['max_position_weight']
        
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.positions = {}  # {ts_code: {'shares': x, 'cost': y, 'high': z}}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.history = []
        self.trades = []
    
    def calculate_position_weights(self, signals: pd.Series, method: str = 'rank') -> pd.Series:
        """计算仓位权重"""
        if method == 'equal':
            # 等权重
            n = len(signals)
            weights = pd.Series(1.0/n, index=signals.index)
        elif method == 'rank':
            # 按排名加权 (信号越好权重越高)
            ranks = signals.rank()
            weights = ranks / ranks.sum()
        elif method == 'signal':
            # 按信号强度加权
            abs_signals = signals.abs()
            weights = abs_signals / abs_signals.sum()
        else:
            weights = pd.Series(1.0/len(signals), index=signals.index)
        
        # 限制最大权重
        weights = weights.clip(upper=self.max_weight)
        # 重新归一化
        weights = weights / weights.sum()
        return weights
    
    def check_stop_loss_take_profit(self, current_price: float, position: Dict) -> Tuple[bool, str]:
        """检查止损止盈"""
        if position['shares'] == 0:
            return False, ''
        
        cost = position['cost']
        high = position.get('high', cost)
        
        # 计算收益率
        pnl_ratio = (current_price - cost) / cost
        
        # 止损
        if pnl_ratio < -self.stop_loss:
            return True, 'stop_loss'
        
        # 止盈 (从最高点回落)
        if high > cost:
            drawdown_from_high = (high - current_price) / (high - cost)
            if drawdown_from_high > 0.5:  # 从最高点回落50%
                return True, 'take_profit'
        
        return False, ''
    
    def run_backtest(self, df: pd.DataFrame, signal_col: str = 'pred',
                     price_col: str = 'close', date_col: str = 'trade_date',
                     top_n: int = 30, rebalance_freq: int = 5) -> pd.DataFrame:
        """运行回测"""
        print("\n[Backtest] Running optimized backtest...")
        
        self.reset()
        dates = sorted(df[date_col].unique())
        
        last_rebalance = -rebalance_freq
        
        for i, date in enumerate(dates):
            day_data = df[df[date_col] == date].copy()
            
            if len(day_data) < top_n:
                continue
            
            # 更新持仓最高价
            for ts_code, pos in self.positions.items():
                stock_data = day_data[day_data['ts_code'] == ts_code]
                if len(stock_data) > 0:
                    current_price = stock_data[price_col].values[0]
                    pos['high'] = max(pos.get('high', pos['cost']), current_price)
            
            # 检查止损止盈
            for ts_code in list(self.positions.keys()):
                stock_data = day_data[day_data['ts_code'] == ts_code]
                if len(stock_data) > 0:
                    current_price = stock_data[price_col].values[0]
                    should_exit, reason = self.check_stop_loss_take_profit(current_price, self.positions[ts_code])
                    
                    if should_exit:
                        # 平仓
                        shares = self.positions[ts_code]['shares']
                        sell_price = current_price * (1 - self.slippage)
                        proceeds = shares * sell_price * (1 - self.commission)
                        self.cash += proceeds
                        
                        self.trades.append({
                            'date': date,
                            'ts_code': ts_code,
                            'action': f'sell_{reason}',
                            'shares': shares,
                            'price': sell_price,
                            'proceeds': proceeds
                        })
                        
                        del self.positions[ts_code]
                        print(f"  {date}: {ts_code} {reason} at {sell_price:.2f}")
            
            # 计算当前净值
            portfolio_value = self.cash
            for ts_code, pos in self.positions.items():
                stock_data = day_data[day_data['ts_code'] == ts_code]
                if len(stock_data) > 0:
                    price = stock_data[price_col].values[0]
                    portfolio_value += pos['shares'] * price
            
            self.portfolio_value = portfolio_value
            
            # 记录历史
            self.history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions),
                'market_exposure': sum(pos['shares'] * day_data[day_data['ts_code']==ts_code][price_col].values[0] 
                                      for ts_code, pos in self.positions.items() 
                                      if len(day_data[day_data['ts_code']==ts_code]) > 0) / portfolio_value if portfolio_value > 0 else 0
            })
            
            # 调仓日
            if i - last_rebalance >= rebalance_freq:
                # 清仓现有持仓
                for ts_code, pos in list(self.positions.items()):
                    stock_data = day_data[day_data['ts_code'] == ts_code]
                    if len(stock_data) > 0:
                        price = stock_data[price_col].values[0]
                        sell_price = price * (1 - self.slippage)
                        proceeds = pos['shares'] * sell_price * (1 - self.commission)
                        self.cash += proceeds
                
                self.positions = {}
                
                # 选新标的
                day_data_with_signal = day_data.dropna(subset=[signal_col, price_col])
                if len(day_data_with_signal) < top_n:
                    continue
                
                # 按信号排序
                day_data_sorted = day_data_with_signal.sort_values(signal_col, ascending=False)
                selected = day_data_sorted.head(top_n)
                
                # 计算权重
                weights = self.calculate_position_weights(selected[signal_col], method='rank')
                
                # 买入
                for _, row in selected.iterrows():
                    ts_code = row['ts_code']
                    price = row[price_col]
                    buy_price = price * (1 + self.slippage)
                    
                    weight = weights.get(ts_code, 1.0/top_n)
                    target_value = portfolio_value * weight
                    invest_amount = target_value * (1 - self.commission)
                    shares = int(invest_amount / buy_price)
                    
                    if shares > 0 and self.cash >= shares * buy_price:
                        self.positions[ts_code] = {
                            'shares': shares,
                            'cost': buy_price,
                            'high': buy_price
                        }
                        self.cash -= shares * buy_price
                
                last_rebalance = i
                
                if (i + 1) % 10 == 0:
                    print(f"  {date}: Rebalanced, NAV={portfolio_value/self.initial_capital:.4f}, "
                          f"Positions={len(self.positions)}")
        
        df_results = pd.DataFrame(self.history)
        return df_results
    
    def calculate_metrics(self, df_results: pd.DataFrame) -> Dict:
        """计算回测指标"""
        df = df_results.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 收益率
        df['returns'] = df['portfolio_value'].pct_change()
        
        # 总收益率
        total_return = df['portfolio_value'].iloc[-1] / self.initial_capital - 1
        
        # 年化收益率
        n_years = len(df) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 波动率
        volatility = df['returns'].std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = (annual_return - self.config['risk_free_rate']) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = (df['returns'] > 0).mean()
        
        # 盈亏比
        gains = df[df['returns'] > 0]['returns'].mean()
        losses = abs(df[df['returns'] < 0]['returns'].mean())
        profit_loss_ratio = gains / losses if losses > 0 else 0
        
        # 最大连续亏损天数
        df['is_loss'] = df['returns'] < 0
        consecutive_losses = df['is_loss'].groupby((df['is_loss'] != df['is_loss'].shift()).cumsum()).sum()
        max_consecutive_losses = consecutive_losses.max()
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Calmar Ratio': f"{calmar:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Profit/Loss Ratio': f"{profit_loss_ratio:.2f}",
            'Max Consecutive Losses': f"{int(max_consecutive_losses)} days",
            'Trading Days': len(df),
            'Final Value': f"{df['portfolio_value'].iloc[-1]:,.0f}"
        }
    
    def plot_results(self, df_results: pd.DataFrame, save_path: str = None):
        """绘制回测结果"""
        df = df_results.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['nav'] = df['portfolio_value'] / self.initial_capital
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 净值曲线
        axes[0].plot(df['date'], df['nav'], label='Strategy', linewidth=2, color='blue')
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('NAV')
        axes[0].set_title('Optimized Backtest Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 回撤
        cummax = df['nav'].cummax()
        drawdown = (df['nav'] - cummax) / cummax
        axes[1].fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
        axes[1].plot(df['date'], drawdown, color='red', linewidth=1)
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True, alpha=0.3)
        
        # 仓位暴露
        axes[2].plot(df['date'], df['market_exposure'], color='green', linewidth=1)
        axes[2].set_ylabel('Market Exposure')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Plot saved: {save_path}")
        
        plt.close()


def main():
    """测试"""
    print("="*70)
    print("OPTIMIZED BACKTEST ENGINE")
    print("="*70)
    
    # 加载预测结果
    pred_df = pd.read_csv(f'{OUTPUT_DIR}/models/ensemble_predictions.csv')
    
    # 这里需要简化，实际应该与原始数据对齐
    print(f"Loaded {len(pred_df)} predictions")
    
    # 运行回测
    engine = OptimizedBacktest()
    
    # 需要完整的price数据，这里简化处理
    print("\n[Note] Full backtest requires complete price data alignment")
    print("[OK] Backtest engine ready for use")


if __name__ == "__main__":
    main()
