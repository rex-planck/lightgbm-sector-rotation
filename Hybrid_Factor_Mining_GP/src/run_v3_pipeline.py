"""
V3完整流程运行脚本
整合：3年数据 + Transformer + 对抗训练 + 市场状态 + 行业中性 + 动态风控
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from config_v3 import OUTPUT_DIR, DB_PATH_V3
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ['factors', 'models', 'backtest', 'reports']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def step1_data_v3():
    """步骤1: 获取V3数据 (3年+ 全市场)"""
    print("\n" + "="*75)
    print("STEP 1/6: V3 DATA FETCHING (3 Years + CSI300/500/1000)")
    print("="*75)
    
    from data_fetcher_v3 import DataFetcherV3
    
    fetcher = DataFetcherV3()
    
    # 获取股票信息
    fetcher.fetch_stock_basic()
    
    # 获取所有成分股
    stocks = fetcher.fetch_index_components_all()
    
    # 获取历史数据
    if stocks:
        fetcher.fetch_all_historical_data(stocks)
    
    # 验证
    stats = fetcher.verify_data()
    
    print("\n[OK] V3 Data fetching completed!")
    return stats


def step2_features_v3():
    """步骤2: V3特征工程 (增强特征 + 市场状态)"""
    print("\n" + "="*75)
    print("STEP 2/6: V3 FEATURE ENGINEERING (Market Regime Detection)")
    print("="*75)
    
    from market_regime_sector_neutral import IntegratedFeatureEngineer
    
    engineer = IntegratedFeatureEngineer()
    df = engineer.load_and_enhance_data()
    
    # 基础特征计算
    df['adj_close'] = df['close'] * df['adj_factor']
    for period in [1, 5, 10, 20, 60]:
        df[f'ret_{period}d'] = df.groupby('ts_code')['adj_close'].pct_change(period)
    
    df['volatility_20d'] = df.groupby('ts_code')['ret_1d'].rolling(20).std().values
    df['rsi_14'] = df.groupby('ts_code')['adj_close'].apply(
        lambda x: (100 - (100 / (1 + x.diff().clip(lower=0).rolling(14).mean() / 
                                (-x.diff().clip(upper=0)).rolling(14).mean())))
    ).reset_index(level=0, drop=True)
    
    # 标签
    df['label'] = df.groupby('ts_code')['adj_close'].shift(-5) / df['adj_close'] - 1
    df['label'] = df.groupby('trade_date')['label'].transform(
        lambda x: (x.rank() - 0.5) / len(x) - 0.5 if len(x) > 1 else 0
    )
    df = df.dropna(subset=['label'])
    
    # 保存
    df.to_pickle(os.path.join(OUTPUT_DIR, 'processed_data_v3.pkl'))
    
    feature_cols = [c for c in df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'label']]
    
    print(f"\n[OK] Feature engineering completed!")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Market state coverage: {df['market_state'].value_counts().to_dict() if 'market_state' in df.columns else 'N/A'}")
    
    engineer.close()
    return df, feature_cols


def step3_gp_v3(df, feature_cols):
    """步骤3: V3 GP挖掘 (优化参数)"""
    print("\n" + "="*75)
    print("STEP 3/6: V3 GP FACTOR MINING (2000 Population, 30 Generations)")
    print("="*75)
    
    from gplearn.genetic import SymbolicTransformer
    from config_v3 import GP_CONFIG_V3, split_dates
    
    dates = split_dates()
    train_df = df[df['trade_date'] <= dates['train_end']].copy()
    valid_df = df[(df['trade_date'] > dates['train_end']) & (df['trade_date'] <= dates['valid_end'])].copy()
    test_df = df[df['trade_date'] > dates['valid_end']].copy()
    
    # 采样加速
    train_sample = train_df.sample(n=min(50000, len(train_df)), random_state=42)
    
    # 确保所有列都是数值型
    for col in feature_cols:
        train_sample[col] = pd.to_numeric(train_sample[col], errors='coerce')
        
    X_train = train_sample[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).clip(-5, 5)
    y_train = train_sample['label'].fillna(0)
    
    print(f"GP Mining: Pop={GP_CONFIG_V3['population_size']}, Gen={GP_CONFIG_V3['generations']}")
    print(f"Training on {len(X_train)} samples...")
    
    gp = SymbolicTransformer(**GP_CONFIG_V3, verbose=1, n_jobs=-1)
    gp.fit(X_train.values, y_train.values)
    
    # 计算因子
    for name, df_split in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        # 确保数据类型正确
        for col in feature_cols:
            df_split[col] = pd.to_numeric(df_split[col], errors='coerce')
        
        X = df_split[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).clip(-5, 5).values
        factors = gp.transform(X)
        for i in range(factors.shape[1]):
            df_split[f'gp_factor_{i}'] = factors[:, i]
        print(f"  {name}: {factors.shape[1]} factors")
    
    # 验证
    gp_cols = [c for c in train_df.columns if c.startswith('gp_factor_')]
    results = []
    for col in gp_cols:
        ic_train = train_df[col].corr(train_df['label'], method='spearman')
        ic_valid = valid_df[col].corr(valid_df['label'], method='spearman')
        results.append({'factor': col, 'train_ic': ic_train, 'valid_ic': ic_valid})
    
    results_df = pd.DataFrame(results).sort_values('valid_ic', key=abs, ascending=False)
    print(f"\nTop 5 Factors by Valid IC:")
    for _, row in results_df.head(5).iterrows():
        print(f"  {row.factor}: Train={row.train_ic:+.4f}, Valid={row.valid_ic:+.4f}")
    
    # 保存
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'factors', 'gp_stats_v3.csv'), index=False)
    train_df.to_pickle(os.path.join(OUTPUT_DIR, 'factors', 'train_v3.pkl'))
    valid_df.to_pickle(os.path.join(OUTPUT_DIR, 'factors', 'valid_v3.pkl'))
    test_df.to_pickle(os.path.join(OUTPUT_DIR, 'factors', 'test_v3.pkl'))
    
    print(f"\n[OK] GP Mining completed! Best Valid IC: {results_df.iloc[0]['valid_ic']:.4f}")
    return train_df, valid_df, test_df, gp_cols


def step4_model_v3(train_df, valid_df, gp_cols):
    """步骤4: V3模型训练 (Transformer + 对抗训练)"""
    print("\n" + "="*75)
    print("STEP 4/6: V3 MODEL TRAINING (Transformer + Adversarial Training)")
    print("="*75)
    
    from model_transformer_adversarial import AdversarialTrainer
    
    # 归一化
    for col in gp_cols:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-8
        for df in [train_df, valid_df]:
            df[col] = ((df[col] - mean) / std).clip(-5, 5)
    
    # 训练
    trainer = AdversarialTrainer(input_size=len(gp_cols))
    result = trainer.train(
        train_df, valid_df, gp_cols,
        save_path=os.path.join(OUTPUT_DIR, 'models', 'transformer_v3_best.pth')
    )
    
    print(f"\n[OK] Training completed! Best Valid IC: {result['best_ic']:.4f}")
    return trainer


def step5_backtest_v3(trainer, test_df, gp_cols):
    """步骤5: V3回测 (动态风控 + 市场状态)"""
    print("\n" + "="*75)
    print("STEP 5/6: V3 BACKTEST (Dynamic Risk Management)")
    print("="*75)
    
    from risk_management_v3 import RiskManagedBacktest
    
    # 归一化
    for col in gp_cols:
        mean = pd.read_pickle(os.path.join(OUTPUT_DIR, 'factors', 'train_v3.pkl'))[col].mean()
        std = pd.read_pickle(os.path.join(OUTPUT_DIR, 'factors', 'train_v3.pkl'))[col].std() + 1e-8
        test_df[col] = ((test_df[col] - mean) / std).clip(-5, 5)
    
    # 预测
    predictions = trainer.predict(test_df, gp_cols)
    test_df['pred'] = np.nan
    test_df.iloc[:len(predictions), test_df.columns.get_loc('pred')] = predictions
    test_df = test_df.dropna(subset=['pred', 'close'])
    
    # 回测
    engine = RiskManagedBacktest()
    results = engine.run_backtest(test_df, signal_col='pred', top_n=30, rebalance_freq=5)
    
    metrics = engine.calculate_metrics(results)
    
    print("\n" + "="*75)
    print("V3 BACKTEST RESULTS")
    print("="*75)
    for key, value in metrics.items():
        print(f"  {key:20s}: {value}")
    
    # 保存
    results.to_csv(os.path.join(OUTPUT_DIR, 'backtest', 'results_v3.csv'), index=False)
    
    print(f"\n[OK] Backtest completed!")
    return results, metrics


def generate_report(metrics):
    """生成最终报告"""
    print("\n" + "="*75)
    print("STEP 6/6: V3 FINAL REPORT")
    print("="*75)
    
    print("\n[V3 Optimization Summary]")
    print("1. Data: 3 years + CSI300/500/1000 (~1500 stocks)")
    print("2. Features: Enhanced 50+ features with market regime")
    print("3. GP: 2000 population, 30 generations")
    print("4. Model: 6-layer Transformer + Adversarial Training")
    print("5. Strategy: Sector-neutral with market regime awareness")
    print("6. Risk: Dynamic Kelly position sizing + Adaptive stop-loss")
    
    print("\n[Final Metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*75)
    print("V3 PIPELINE COMPLETED!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*75)


def run_v3_pipeline():
    """运行完整V3流程"""
    print("="*75)
    print("V3 ULTIMATE OPTIMIZATION PIPELINE")
    print("="*75)
    
    try:
        # 检查已有数据
        if not os.path.exists(DB_PATH_V3):
            print("\n[Note] V3 database not found. Running data fetch first...")
            step1_data_v3()
        
        # 特征工程
        df, feature_cols = step2_features_v3()
        
        # GP挖掘
        train_df, valid_df, test_df, gp_cols = step3_gp_v3(df, feature_cols)
        
        # 模型训练
        trainer = step4_model_v3(train_df, valid_df, gp_cols)
        
        # 回测
        results, metrics = step5_backtest_v3(trainer, test_df, gp_cols)
        
        # 报告
        generate_report(metrics)
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_v3_pipeline()
