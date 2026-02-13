import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# === 配置区 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 关键修改：因为文件在 models/lgbm/ 下，需要往上退两层才能回到项目根目录
# level 1: models/lgbm -> models
# level 2: models -> AShare_Macro_Rotation (根目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')

# 打印一下路径，确保没错
print(f"📂 数据读取路径: {RAW_PATH}")


def load_data(sector_name="半导体"):
    # 1. 加载行业数据
    df_sector = pd.read_csv(os.path.join(RAW_PATH, f'sector_{sector_name}.csv'))
    # 简单的列名清洗
    col_map = {c: 'date' for c in df_sector.columns if '日期' in c}
    col_map.update({c: 'close' for c in df_sector.columns if '收盘' in c})
    col_map.update({c: 'open' for c in df_sector.columns if '开盘' in c})
    col_map.update({c: 'high' for c in df_sector.columns if '最高' in c})
    col_map.update({c: 'low' for c in df_sector.columns if '最低' in c})
    col_map.update({c: 'vol' for c in df_sector.columns if '成交量' in c})

    df_sector = df_sector.rename(columns=col_map)
    df_sector['date'] = pd.to_datetime(df_sector['date'])

    # 2. 加载 HMM 宏观状态
    df_hmm = pd.read_csv(os.path.join(PROCESSED_PATH, 'hmm_signals.csv'))
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])

    # 合并
    df = pd.merge(df_sector, df_hmm[['date', 'hidden_state', 'Liquidity_Diff']], on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def feature_engineering(df):
    """
    构建符合 LightGBM 输入的特征 (JD: 特征工程)
    """
    # 1. 基础量价特征
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['vol_change'] = df['vol'].pct_change()

    # 2. 波动率特征
    df['std_20'] = df['ret_1'].rolling(20).std()

    # 3. 技术指标 (手动计算 RSI, 免去安装 TA-Lib 的麻烦)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # 4. 宏观交互特征 (关键创新点!)
    # 逻辑: 在牛市(State 2)中，RSI超买可能不是顶；在熊市(State 0)中，RSI超卖可能不是底
    df['Macro_RSI_Interact'] = df['hidden_state'] * df['RSI']

    # 5. 标签 (Label): 预测明天涨(1)还是跌(0)
    # A股实务: 预测收益率 > 0.1% (扣除手续费)
    df['target'] = (df['close'].shift(-1) / df['close'] - 1 > 0.001).astype(int)

    # 去除空值
    df = df.dropna()
    return df


def train_lgbm(df):
    """
    使用 LightGBM 进行滚动训练 (Walk-Forward)
    """
    print("🤖 正在训练 LightGBM 模型...")

    features = ['ret_1', 'ret_5', 'vol_change', 'std_20', 'RSI', 'hidden_state', 'Liquidity_Diff', 'Macro_RSI_Interact']
    target = 'target'

    # 时间序列分割 (防止未来函数)
    # 以前 80% 做训练，后 20% 做测试
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"   训练集: {len(train_df)} 天 | 测试集: {len(test_df)} 天")

    # 创建 LGB 数据集
    lgb_train = lgb.Dataset(train_df[features], label=train_df[target])
    lgb_eval = lgb.Dataset(test_df[features], label=test_df[target], reference=lgb_train)

    # 参数 (针对金融时序微调)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )

    # 预测
    y_pred_prob = model.predict(test_df[features])
    y_pred_class = (y_pred_prob > 0.52).astype(int)  # 阈值设高一点(0.52)提高胜率

    # 评估
    acc = accuracy_score(test_df[target], y_pred_class)
    auc = roc_auc_score(test_df[target], y_pred_prob)

    print("-" * 30)
    print(f"   ✅ 预测准确率 (Accuracy): {acc:.2%}")
    print(f"   ✅ AUC 得分: {auc:.4f}")

    # 特征重要性
    print("\n   🔍 特征重要性排序:")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importance()
    }).sort_values('Importance', ascending=False)
    print(importance)

    return model, importance


if __name__ == "__main__":
    df_raw = load_data("半导体")
    df_feat = feature_engineering(df_raw)
    model, imp = train_lgbm(df_feat)