"""
项目主入口 V2
整合优化后的所有模块
"""
import argparse
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import TUSHARE_TOKEN, DB_PATH, OUTPUT_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'pipeline.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def check_config():
    """检查配置"""
    if not TUSHARE_TOKEN:
        logger.error("❌ 请在 config.py 中设置 TUSHARE_TOKEN")
        return False
    logger.info(f"✅ Token 配置正常: {TUSHARE_TOKEN[:10]}...")
    return True


def step1_fetch_data(force: bool = False):
    """步骤1: 获取数据"""
    logger.info("\n" + "=" * 60)
    logger.info("📥 步骤 1/4: 获取 Tushare 数据")
    logger.info("=" * 60)
    
    from data_fetcher_optimized import TushareDataFetcherOptimized
    
    fetcher = TushareDataFetcherOptimized()
    
    # 检查是否需要重新获取
    if not force and os.path.exists(DB_PATH):
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_price")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            logger.info(f"   数据库已有 {count} 条日线数据")
            response = input("   是否重新获取？(y/N): ").strip().lower()
            if response != 'y':
                logger.info("   跳过数据获取")
                return
    
    # 获取数据
    fetcher.fetch_stock_basic()
    stock_list = fetcher.fetch_index_components()
    
    if stock_list:
        fetcher.fetch_all_data_by_date(stock_list)
    
    logger.info("✅ 数据获取完成")


def step2_prepare_data():
    """步骤2: 数据预处理"""
    logger.info("\n" + "=" * 60)
    logger.info("🔧 步骤 2/4: 数据预处理")
    logger.info("=" * 60)
    
    from data_loader import DataLoader
    
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    logger.info(f"✅ 数据预处理完成: {len(df_labeled)} 条有效数据")
    return df_labeled


def step3_mine_factors():
    """步骤3: GP 因子挖掘"""
    logger.info("\n" + "=" * 60)
    logger.info("🧬 步骤 3/4: GP 因子挖掘")
    logger.info("=" * 60)
    
    from data_loader import DataLoader
    from gp_factor_mining_v2 import GPFactorMinerV2
    from config import TRAIN_END, GP_CONFIG
    
    # 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    loader.close()
    
    # 准备特征
    miner = GPFactorMinerV2()
    df_features, feature_cols = miner.prepare_features(df_raw)
    
    # 这里需要重新加载 DataLoader 来准备标签
    loader = DataLoader()
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    # 只使用训练集
    train_df = df_labeled[df_labeled['trade_date'] <= TRAIN_END]
    
    logger.info(f"   训练集样本: {len(train_df)}")
    logger.info(f"   使用特征: {feature_cols}")
    
    # 挖掘因子
    programs = miner.mine_factors_symbolic_transformer(
        train_df, feature_cols, n_factors=GP_CONFIG['n_factors']
    )
    
    # 验证因子
    df_results = miner.validate_factors(
        programs, train_df, feature_cols, min_ic=GP_CONFIG['min_ic_threshold']
    )
    
    # 选择多样化因子
    df_results = miner.select_diverse_factors(
        df_results, train_df, feature_cols,
        top_n=30, max_corr=GP_CONFIG['max_correlation']
    )
    
    # 保存结果
    miner.save_factors(df_results)
    
    # 打印结果
    valid_count = df_results['valid'].sum() if 'valid' in df_results.columns else len(df_results)
    selected_count = df_results['selected'].sum() if 'selected' in df_results.columns else 0
    
    logger.info(f"✅ 因子挖掘完成: {valid_count} 个有效，{selected_count} 个被选中")
    
    # 打印 Top 5
    if 'ir' in df_results.columns:
        logger.info("\n🏆 Top 5 因子:")
        for idx, row in df_results.head(5).iterrows():
            logger.info(f"   [{idx+1}] IR={row.get('ir', 0):.3f}, IC={row.get('ic_mean', 0):.4f}")


def step4_train_model():
    """步骤4: 训练两阶段模型"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 步骤 4/4: 两阶段模型训练 (GP + GRU)")
    logger.info("=" * 60)
    
    from data_loader import DataLoader
    from two_stage_model_v2 import TwoStageModelV2
    from config import TRAIN_END, VALID_END
    
    # 加载数据
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    # 划分数据集
    train_df = df_labeled[df_labeled['trade_date'] <= TRAIN_END]
    valid_df = df_labeled[(df_labeled['trade_date'] > TRAIN_END) & 
                           (df_labeled['trade_date'] <= VALID_END)]
    test_df = df_labeled[df_labeled['trade_date'] > VALID_END]
    
    logger.info(f"   训练: {len(train_df)}, 验证: {len(valid_df)}, 测试: {len(test_df)}")
    
    # 创建模型
    model = TwoStageModelV2()
    
    # 加载/计算 GP 因子
    base_features = ['open', 'high', 'low', 'close', 'vol', 'ret_1d', 'ret_5d']
    base_features = [c for c in base_features if c in train_df.columns]
    
    if not model.load_gp_factors():
        logger.warning("⚠️ 未找到 GP 因子，将只使用 Alpha 因子")
    
    # 计算因子
    logger.info("🔧 计算 GP 因子...")
    train_df = model.compute_gp_factors(train_df, base_features)
    valid_df = model.compute_gp_factors(valid_df, base_features)
    test_df = model.compute_gp_factors(test_df, base_features)
    
    # 训练 GRU
    model.train(train_df, valid_df)
    
    # 测试集评估
    logger.info("\n📊 测试集评估...")
    model.load_model("gru_best_v2.pth")
    test_result = model.predict(test_df)
    
    # 计算测试集 IC
    test_mask = test_result['pred'].notna()
    if test_mask.sum() > 100:
        import numpy as np
        test_ic = np.corrcoef(
            test_result.loc[test_mask, 'pred'].rank(),
            test_result.loc[test_mask, 'label'].rank()
        )[0, 1]
        logger.info(f"   测试集 Rank IC: {test_ic:.4f}")
    
    logger.info("✅ 模型训练完成")


def run_full_pipeline(force_fetch: bool = False):
    """运行完整流程"""
    start_time = datetime.now()
    
    if not check_config():
        return
    
    try:
        step1_fetch_data(force=force_fetch)
    except Exception as e:
        logger.error(f"❌ 数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        step2_prepare_data()
    except Exception as e:
        logger.error(f"❌ 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        step3_mine_factors()
    except Exception as e:
        logger.error(f"❌ 因子挖掘失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        step4_train_model()
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 全流程完成！耗时: {elapsed:.1f} 分钟")
    logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于 GPlearn 的'机理+数据'混合因子挖掘系统 V2"
    )
    parser.add_argument(
        "--step",
        choices=["all", "fetch", "prepare", "mine", "train"],
        default="all",
        help="选择运行步骤"
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="强制重新获取数据"
    )
    
    args = parser.parse_args()
    
    if args.step == "all":
        run_full_pipeline(force_fetch=args.force_fetch)
    elif args.step == "fetch":
        if check_config():
            step1_fetch_data(force=args.force_fetch)
    elif args.step == "prepare":
        step2_prepare_data()
    elif args.step == "mine":
        step3_mine_factors()
    elif args.step == "train":
        step4_train_model()


if __name__ == "__main__":
    main()
