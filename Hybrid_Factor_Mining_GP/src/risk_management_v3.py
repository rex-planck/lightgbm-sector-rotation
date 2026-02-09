"""
V3风控管理模块
动态仓位管理 + 自适应止损
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config_v3 import RISK_CONFIG_V3


class DynamicPositionSizer:
    """动态仓位管理器"""
    
    def __init__(self, method: str = "kelly"):
        self.method = method
        self.config = RISK_CONFIG_V3
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly公式计算最优仓位"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # 赔率
        p = win_rate            # 胜率
        q = 1 - p               # 败率
        
        kelly_f = (b * p - q) / b if b != 0 else 0
        return max(0, min(kelly_f, self.config['max_position_weight']))
    
    def calculate_position_sizes(self, signals: pd.Series, 
                                  historical_returns: pd.Series = None) -> Dict[str, float]:
        """计算仓位大小"""
        if self.method == "equal":
            # 等权重
            n = len(signals)
            weights = {ts: 1.0/n for ts in signals.index}
        
        elif self.method == "kelly" and historical_returns is not None:
            # Kelly公式
            win_rate = (historical_returns > 0).mean()
            avg_win = historical_returns[historical_returns > 0].mean()
            avg_loss = abs(historical_returns[historical_returns < 0].mean())
            
            kelly_weight = self.kelly_criterion(win_rate, avg_win, avg_loss)
            
            # 按信号强度分配
            total_signal = signals.abs().sum()
            if total_signal > 0:
                weights = {ts: kelly_weight * abs(sig) / total_signal 
                          for ts, sig in signals.items()}
            else:
                n = len(signals)
                weights = {ts: kelly_weight / n for ts in signals.index}
        
        elif self.method == "risk_parity":
            # 风险平价 (简化版，按波动率反比)
            volatilities = signals.rolling(20).std() if hasattr(signals, 'rolling') else pd.Series(1, index=signals.index)
            inv_vol = 1.0 / (volatilities + 1e-8)
            total_inv_vol = inv_vol.sum()
            weights = {ts: inv_vol[ts] / total_inv_vol if total_inv_vol > 0 else 1.0/len(signals)
                      for ts in signals.index}
        
        else:
            # 默认等权重
            n = len(signals)
            weights = {ts: 1.0/n for ts in signals.index}
        
        # 限制最大/最小权重
        for ts in weights:
            weights[ts] = max(self.config['min_position_weight'],
                            min(self.config['max_position_weight'], weights[ts]))
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def calculate_portfolio_var(self, returns_df: pd.DataFrame, 
                                weights: Dict[str, float]) -> float:
        """计算组合VaR"""
        # 简化计算：假设独立同分布
        portfolio_returns = sum(returns_df[ts] * w for ts, w in weights.items() if ts in returns_df.columns)
        var = portfolio_returns.std() * np.sqrt(self.config['var_lookback'])
        return var


class AdaptiveStopLoss:
    """自适应止损器"""
    
    def __init__(self):
        self.config = RISK_CONFIG_V3
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """计算ATR (平均真实波幅)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    def check_stop_loss(self, position: Dict, current_data: pd.Series) -> Tuple[bool, str]:
        """检查是否需要止损"""
        cost = position['cost']
        high = position.get('high', cost)
        current_price = current_data['close']
        
        # 当前收益率
        pnl_pct = (current_price - cost) / cost
        
        # 1. 硬止损 (固定百分比)
        if self.config['stop_loss_method'] == 'fixed':
            if pnl_pct < -self.config['stop_loss_atr_mult']:
                return True, f"hard_stop ({pnl_pct:.2%})"
        
        # 2. ATR动态止损
        elif self.config['stop_loss_method'] == 'atr':
            if 'high' in current_data and 'low' in current_data:
                atr = self.calculate_atr(
                    pd.Series([current_data['high']]),
                    pd.Series([current_data['low']]),
                    pd.Series([current_data['close']])
                )
                atr_stop = cost - (atr * self.config['stop_loss_atr_mult'])
                if current_price < atr_stop:
                    return True, f"atr_stop ({atr:.4f})"
        
        # 3. 追踪止损
        if self.config['trailing_stop']:
            # 从最高点回落一定比例
            if high > cost:
                drawdown_from_high = (high - current_price) / (high - cost)
                if drawdown_from_high > self.config['trailing_stop_pct']:
                    return True, f"trailing_stop ({drawdown_from_high:.2%})"
        
        # 4. 最大回撤限制
        portfolio_value = position.get('portfolio_value', cost)
        if portfolio_value > 0:
            current_value = position['shares'] * current_price
            drawdown = (current_value - portfolio_value) / portfolio_value
            if drawdown < -self.config['max_drawdown_limit']:
                return True, f"max_dd_limit ({drawdown:.2%})"
        
        return False, ""
    
    def update_trailing_high(self, position: Dict, current_price: float):
        """更新追踪最高价"""
        if 'high' not in position:
            position['high'] = position['cost']
        position['high'] = max(position['high'], current_price)


class RiskManagedBacktest:
    """V3风控管理回测引擎"""
    
    def __init__(self):
        self.config = RISK_CONFIG_V3
        self.position_sizer = DynamicPositionSizer(method=self.config['position_sizing'])
        self.stop_loss_manager = AdaptiveStopLoss()
        
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.positions = {}  # {ts_code: {'shares': x, 'cost': y, 'high': z, 'weight': w}}
        self.cash = self.config['initial_capital']
        self.portfolio_value = self.config['initial_capital']
        self.history = []
        self.daily_returns = []
    
    def run_backtest(self, df: pd.DataFrame, signal_col: str = 'pred',
                     price_col: str = 'close', date_col: str = 'trade_date',
                     top_n: int = 30, rebalance_freq: int = 5) -> pd.DataFrame:
        """运行风控优化回测"""
        logger.info("Running V3 Risk-Managed Backtest...")
        
        dates = sorted(df[date_col].unique())
        last_rebalance = -rebalance_freq
        
        for i, date in enumerate(dates):
            day_data = df[df[date_col] == date].copy()
            if len(day_data) < top_n:
                continue
            
            # 更新追踪最高价
            for ts_code in self.positions:
                stock = day_data[day_data['ts_code'] == ts_code]
                if len(stock) > 0:
                    self.stop_loss_manager.update_trailing_high(
                        self.positions[ts_code], 
                        stock[price_col].values[0]
                    )
            
            # 检查止损
            for ts_code in list(self.positions.keys()):
                stock = day_data[day_data['ts_code'] == ts_code]
                if len(stock) > 0:
                    should_exit, reason = self.stop_loss_manager.check_stop_loss(
                        self.positions[ts_code], 
                        stock.iloc[0]
                    )
                    
                    if should_exit:
                        shares = self.positions[ts_code]['shares']
                        price = stock[price_col].values[0]
                        sell_price = price * (1 - self.config['slippage'])
                        proceeds = shares * sell_price * (1 - self.config['commission'])
                        self.cash += proceeds
                        
                        logger.info(f"  {date}: {ts_code} STOP {reason}")
                        del self.positions[ts_code]
            
            # 计算当前净值
            portfolio = self.cash
            for ts_code, pos in self.positions.items():
                stock = day_data[day_data['ts_code'] == ts_code]
                if len(stock) > 0:
                    portfolio += pos['shares'] * stock[price_col].values[0]
            
            self.portfolio_value = portfolio
            
            # 记录
            self.history.append({
                'date': date,
                'portfolio_value': portfolio,
                'cash': self.cash,
                'n_positions': len(self.positions),
                'market_value': portfolio - self.cash
            })
            
            # 调仓日
            if i - last_rebalance >= rebalance_freq:
                # 清仓
                for ts_code, pos in list(self.positions.items()):
                    stock = day_data[day_data['ts_code'] == ts_code]
                    if len(stock) > 0:
                        price = stock[price_col].values[0]
                        sell_price = price * (1 - self.config['slippage'])
                        self.cash += pos['shares'] * sell_price * (1 - self.config['commission'])
                
                self.positions = {}
                
                # 选新标的
                day_data = day_data.dropna(subset=[signal_col, price_col])
                if len(day_data) < top_n:
                    continue
                
                selected = day_data.sort_values(signal_col, ascending=False).head(top_n)
                signals = selected.set_index('ts_code')[signal_col]
                
                # 动态仓位
                weights = self.position_sizer.calculate_position_sizes(signals)
                
                # 买入
                for _, row in selected.iterrows():
                    ts_code = row['ts_code']
                    price = row[price_col]
                    buy_price = price * (1 + self.config['slippage'])
                    
                    weight = weights.get(ts_code, 1.0/top_n)
                    target_value = portfolio * weight
                    shares = int(target_value * (1 - self.config['commission']) / buy_price)
                    
                    if shares > 0 and self.cash >= shares * buy_price:
                        self.positions[ts_code] = {
                            'shares': shares,
                            'cost': buy_price,
                            'high': buy_price,
                            'weight': weight
                        }
                        self.cash -= shares * buy_price
                
                last_rebalance = i
                
                if (i+1) % 20 == 0:
                    logger.info(f"{date}: NAV={portfolio/self.config['initial_capital']:.4f}, "
                               f"Pos={len(self.positions)}, Cash={self.cash/self.config['initial_capital']:.2%}")
        
        return pd.DataFrame(self.history)
    
    def calculate_metrics(self, df_results: pd.DataFrame) -> Dict:
        """计算回测指标"""
        df = df_results.copy()
        df['returns'] = df['portfolio_value'].pct_change()
        
        total_return = df['portfolio_value'].iloc[-1] / self.config['initial_capital'] - 1
        n_years = len(df) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = df['returns'].std() * np.sqrt(252)
        sharpe = (annual_return - self.config.get('risk_free_rate', 0.03)) / volatility if volatility > 0 else 0
        
        cummax = df['portfolio_value'].cummax()
        max_dd = ((df['portfolio_value'] - cummax) / cummax).min()
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Win Rate': f"{(df['returns'] > 0).mean():.1%}",
            'Trading Days': len(df)
        }


def main():
    """测试"""
    print("V3 Risk Management Module")
    print("Ready for integration")


if __name__ == "__main__":
    main()
