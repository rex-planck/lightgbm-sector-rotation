"""
优化版完整流程运行脚本
整合所有优化模块
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
from config_optimized import OUTPUT_DIR, FACTOR_OUTPUT_DIR, MODEL_OUTPUT_DIR, DB_PATH
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FACTOR_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def step1_fetch_data():
    """步骤1: 获取全市场数据"""
    print("\n" + "="*70)
    print("STEP 1: FETCHING FULL MARKET DATA")
    print("="*70)
    
    from data_fetcher_full import FullMarketDataFetcher
    
    fetcher = FullMarketDataFetcher()
    
    # 获取所有指数成分股
    stocks = fetcher.fetch_all_index_components()
    
    # 获取历史数据
    if stocks:
        fetcher.fetch_all_data(stocks)
    
    print("\n[OK] Data fetching completed!")


def step2_feature_engineering():
    """步骤2: 增强特征工程"""
    print("\n" + "="*70)
    print("STEP 2: ENHANCED FEATURE ENGINEERING")
    print("="*70)
    
    from feature_engineering_enhanced import EnhancedFeatureEngineer
    
    engineer = EnhancedFeatureEngineer()
    df_raw = engineer.load_data()
    df_features = engineer.calculate_features(df_raw)
    df_labeled = engineer.prepare_labels(df_features, horizon=5)
    
    # 保存处理后的数据
    df_labeled.to_pickle(f'{OUTPUT_DIR}/processed_data.pkl')
    
    feature_cols = engineer.get_feature_columns(df_labeled)
    print(f"\n[OK] Feature engineering completed!")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Samples: {len(df_labeled)}")
    
    engineer.close()
    return df_labeled, feature_cols


def step3_gp_mining(df_labeled, feature_cols):
    """步骤3: 优化版GP因子挖掘"""
    print("\n" + "="*70)
    print("STEP 3: OPTIMIZED GP FACTOR MINING")
    print("="*70)
    
    from gp_factor_mining_optimized import OptimizedGPFactorMiner
    from config_optimized import TRAIN_END, VALID_END
    
    # 划分数据集
    train_df = df_labeled[df_labeled['trade_date'] <= TRAIN_END].copy()
    valid_df = df_labeled[(df_labeled['trade_date'] > TRAIN_END) & 
                          (df_labeled['trade_date'] <= VALID_END)].copy()
    test_df = df_labeled[df_labeled['trade_date'] > VALID_END].copy()
    
    print(f"Dataset split: Train {len(train_df)}, Valid {len(valid_df)}, Test {len(test_df)}")
    
    # GP挖掘
    miner = OptimizedGPFactorMiner()
    X_train, y_train = miner.prepare_data(train_df, feature_cols)
    programs, gp_model = miner.mine_factors(X_train, y_train, feature_cols)
    
    # 计算因子值
    print("\nComputing factor values for all datasets...")
    train_df = miner.compute_factor_values(train_df, gp_model, feature_cols)
    valid_df = miner.compute_factor_values(valid_df, gp_model, feature_cols)
    test_df = miner.compute_factor_values(test_df, gp_model, feature_cols)
    
    # 验证和筛选
    df_results = miner.validate_factors(train_df)
    selected = miner.select_diverse_factors(df_results, train_df)
    
    # 保存结果
    miner.save_results(df_results, selected, train_df, valid_df, test_df)
    
    print("\n[OK] GP mining completed!")
    return train_df, valid_df, test_df, selected


def step4_train_ensemble(train_df, valid_df, selected_factors):
    """步骤4: 训练集成模型"""
    print("\n" + "="*70)
    print("STEP 4: TRAINING ENSEMBLE MODEL (Transformer + LSTM + GRU)")
    print("="*70)
    
    from transformer_model import EnsembleTrainer
    
    # 归一化
    print("Normalizing features...")
    for col in selected_factors:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-8
        train_df[col] = ((train_df[col] - mean) / std).clip(-5, 5)
        valid_df[col] = ((valid_df[col] - mean) / std).clip(-5, 5)
    
    # 训练集成模型
    trainer = EnsembleTrainer(input_size=len(selected_factors))
    trainer.train_ensemble(train_df, valid_df, selected_factors, n_models=5)
    
    print("\n[OK] Ensemble training completed!")
    return trainer


def step5_backtest(trainer, test_df, selected_factors):
    """步骤5: 优化版回测"""
    print("\n" + "="*70)
    print("STEP 5: OPTIMIZED BACKTEST")
    print("="*70)
    
    from backtest_optimized import OptimizedBacktest
    
    # 生成预测
    print("Generating predictions...")
    test_pred = trainer.predict(test_df, selected_factors)
    
    # 对齐数据
    test_df_aligned = test_df.iloc[:len(test_pred)].copy()
    test_df_aligned['pred'] = test_pred
    
    # 计算IC
    test_ic = test_df_aligned['pred'].corr(test_df_aligned['label'], method='spearman')
    print(f"Test Rank IC: {test_ic:.4f}")
    
    # 运行回测
    engine = OptimizedBacktest()
    df_results = engine.run_backtest(
        test_df_aligned,
        signal_col='pred',
        top_n=30,
        rebalance_freq=5
    )
    
    # 计算指标
    metrics = engine.calculate_metrics(df_results)
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    for key, value in metrics.items():
        print(f"  {key:25s}: {value}")
    
    # 保存结果
    df_results.to_csv(f'{OUTPUT_DIR}/backtest_results.csv', index=False)
    engine.plot_results(df_results, save_path=f'{OUTPUT_DIR}/backtest_plot.png')
    
    print("\n[OK] Backtest completed!")


def run_full_pipeline():
    """运行完整流程"""
    print("="*70)
    print("OPTIMIZED HYBRID FACTOR MINING SYSTEM - FULL PIPELINE")
    print("="*70)
    print("\nOptimizations:")
    print("  - Extended data: CSI300 + CSI500 + CSI1000, 3+ years")
    print("  - Enhanced features: 50+ technical & fundamental indicators")
    print("  - Optimized GP: Population 1500, Generations 30")
    print("  - Transformer Ensemble: 5 models with regularization")
    print("  - Advanced backtest: Position sizing, stop-loss, take-profit")
    
    try:
        # 检查是否已有数据
        if not os.path.exists(DB_PATH):
            print("\n[Warning] Database not found. Please run data fetching first:")
            print("  python run_optimized_pipeline.py --step fetch")
            return
        
        # 步骤2: 特征工程
        df_labeled, feature_cols = step2_feature_engineering()
        
        # 步骤3: GP挖掘
        train_df, valid_df, test_df, selected_factors = step3_gp_mining(df_labeled, feature_cols)
        
        # 步骤4: 训练集成模型
        trainer = step4_train_ensemble(train_df, valid_df, selected_factors)
        
        # 步骤5: 回测
        step5_backtest(trainer, test_df, selected_factors)
        
        print("\n" + "="*70)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("="*70)
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Hybrid Factor Mining Pipeline")
    parser.add_argument('--step', choices=['all', 'fetch', 'features', 'gp', 'train', 'backtest'],
                       default='all', help='Which step to run')
    
    args = parser.parse_args()
    
    if args.step == 'fetch':
        step1_fetch_data()
    elif args.step == 'features':
        step2_feature_engineering()
    elif args.step == 'gp':
        df_labeled, feature_cols = step2_feature_engineering()
        step3_gp_mining(df_labeled, feature_cols)
    elif args.step == 'train':
        df_labeled, feature_cols = step2_feature_engineering()
        train_df, valid_df, test_df, selected = step3_gp_mining(df_labeled, feature_cols)
        step4_train_ensemble(train_df, valid_df, selected)
    elif args.step == 'backtest':
        df_labeled, feature_cols = step2_feature_engineering()
        train_df, valid_df, test_df, selected = step3_gp_mining(df_labeled, feature_cols)
        trainer = step4_train_ensemble(train_df, valid_df, selected)
        step5_backtest(trainer, test_df, selected)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
