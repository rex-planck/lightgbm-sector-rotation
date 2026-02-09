
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8')

# Load results
results_path = 'Tushare/Hybrid_Factor_Mining_GP/output_v3/backtest/results_v3.csv'
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    print("Columns:", df.columns)
    
    # Calculate daily return from portfolio value if return column is missing
    if 'portfolio_value' in df.columns:
        df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
        df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
    else:
        print("Error: 'portfolio_value' column not found.")
        exit()
        
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df = df.set_index('date')
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_return'], label='V3 Strategy (Fast Run)')
    plt.title('V3 Strategy Backtest Performance (Fast Run)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = 'Tushare/Hybrid_Factor_Mining_GP/output_v3/backtest/v3_fast_performance.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
else:
    print("Results file not found.")
