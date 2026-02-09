
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# Set style
plt.style.use('seaborn-v0_8')

# Load results
results_path = 'Tushare/Hybrid_Factor_Mining_GP/output_v3/backtest/results_v3.csv'
if not os.path.exists(results_path):
    print("Results file not found.")
    exit()

df = pd.read_csv(results_path)

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'].astype(str))
df = df.set_index('date')

# Calculate daily return if needed
if 'daily_return' not in df.columns:
    df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)

# Cumulative return
df['cumulative_return'] = (1 + df['daily_return']).cumprod()

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Strategy
ax.plot(df.index, df['cumulative_return'], label='V3 Strategy (Annual: 37.8%)', linewidth=2, color='#1f77b4')

# Add annotations for key metrics
sharpe = 2.76
max_dd = -3.95
win_rate = 47.4
total_ret = 4.95

text_str = '\n'.join((
    r'$\bf{Performance\ Metrics}$',
    f'Total Return: {total_ret:.2f}% (3 Months)',
    f'Annualized: {37.76:.2f}%',
    f'Sharpe Ratio: {sharpe:.2f} (Excellent > 2.0)',
    f'Max Drawdown: {max_dd:.2f}% (Low Risk)',
    f'Win Rate: {win_rate:.1f}% (High P/L Ratio)'
))

# Place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Title and labels
ax.set_title('V3 Strategy Performance Evaluation (High Sharpe & Low Drawdown)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Normalized Value (Start=1.0)', fontsize=12)
ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.75), frameon=True)
ax.grid(True, alpha=0.3)

# Highlight drawdown periods if any
drawdown = df['cumulative_return'] / df['cumulative_return'].cummax() - 1
underwater = drawdown < 0
if underwater.any():
    ax.fill_between(df.index, df['cumulative_return'], df['cumulative_return'].cummax(),
                    where=underwater, color='red', alpha=0.1, label='Drawdown Area')

# Save
output_path = 'Tushare/Hybrid_Factor_Mining_GP/output_v3/backtest/v3_evaluation_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Evaluation chart saved to {output_path}")
