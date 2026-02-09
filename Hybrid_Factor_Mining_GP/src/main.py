"""
项目主入口
整合数据获取、因子挖掘、模型训练的全流程
"""
import argparse
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import TUSHARE_TOKEN


def check_config():
    """检查配置是否完整"""
    if not TUSHARE_TOKEN:
        print("❌ 错误：请在 src/config.py 中设置 TUSHARE_TOKEN")
        print("   获取方式：https://tushare.pro/register")
        return False
    return True


def run_data_fetch():
    """运行数据获取"""
    print("\n" + "=" * 60)
    print("📥 步骤 1/4: 获取 Tushare 数据")
    print("=" * 60)
    from data_fetcher import main as fetch_main
    fetch_main()


def run_data_prepare():
    """运行数据预处理"""
    print("\n" + "=" * 60)
    print("🔧 步骤 2/4: 数据预处理")
    print("=" * 60)
    from data_loader import DataLoader
    
    loader = DataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    
    # 保存处理后的数据
    output_path = os.path.join(loader.db_path.replace('.db', '_processed.csv'))
    df_labeled.to_csv(output_path.replace('.db', '_processed.csv'), index=False)
    
    loader.close()
    print(f"✅ 数据预处理完成")


def run_factor_mining():
    """运行因子挖掘"""
    print("\n" + "=" * 60)
    print("🧬 步骤 3/4: GP 因子挖掘")
    print("=" * 60)
    from gp_factor_mining import main as gp_main
    gp_main()


def run_two_stage_model():
    """运行两阶段模型"""
    print("\n" + "=" * 60)
    print("🚀 步骤 4/4: 两阶段模型训练 (GP因子 + GRU)")
    print("=" * 60)
    from two_stage_gru import main as gru_main
    gru_main()


def run_full_pipeline():
    """运行完整流程"""
    if not check_config():
        return
    
    print("\n" + "=" * 60)
    print("🎯 启动完整流程: 数据 → 因子挖掘 → GRU训练")
    print("=" * 60)
    
    try:
        run_data_fetch()
    except Exception as e:
        print(f"⚠️ 数据获取步骤出错（可能已有数据）: {e}")
    
    try:
        run_data_prepare()
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return
    
    try:
        run_factor_mining()
    except Exception as e:
        print(f"❌ 因子挖掘失败: {e}")
        return
    
    try:
        run_two_stage_model()
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 全流程完成！")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于 GPlearn 的'机理+数据'混合因子挖掘系统"
    )
    parser.add_argument(
        "--step",
        choices=["all", "fetch", "prepare", "mine", "train"],
        default="all",
        help="选择运行步骤: all=全流程, fetch=获取数据, prepare=预处理, mine=因子挖掘, train=模型训练"
    )
    
    args = parser.parse_args()
    
    if args.step == "all":
        run_full_pipeline()
    elif args.step == "fetch":
        if check_config():
            run_data_fetch()
    elif args.step == "prepare":
        run_data_prepare()
    elif args.step == "mine":
        run_factor_mining()
    elif args.step == "train":
        run_two_stage_model()


if __name__ == "__main__":
    main()
