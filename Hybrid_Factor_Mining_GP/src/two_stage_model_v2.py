"""
两阶段模型 V2
阶段1: GP 挖掘因子 + 执行计算
阶段2: GRU 时序建模
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging

from config import (
    GRU_CONFIG, MODEL_OUTPUT_DIR, FACTOR_OUTPUT_DIR,
    LABEL_COL, TRAIN_END, VALID_END, START_DATE, END_DATE
)
from data_loader import DataLoader as StockDataLoader
from gp_factor_mining_v2 import FactorProgramExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== GRU 模型 ====================

class GRUModel(nn.Module):
    """GRU 预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最后时刻
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze()


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 seq_len: int = 20, step: int = 1):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.step = step
        
        # 按股票分组构建样本
        self.samples = []
        
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            values = group[feature_cols + [LABEL_COL]].values
            
            if len(values) <= seq_len:
                continue
            
            # 构建滑动窗口
            for i in range(0, len(values) - seq_len, step):
                seq = values[i:i+seq_len]
                if not np.isnan(seq).any():
                    self.samples.append(seq)
        
        logger.info(f"   创建 {len(self.samples)} 个序列样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-self.step, :-1], dtype=torch.float32)
        y = torch.tensor(seq[-1, -1], dtype=torch.float32)
        return x, y


# ==================== 两阶段模型 ====================

class TwoStageModelV2:
    """两阶段模型 V2"""
    
    def __init__(self):
        self.gp_programs = []
        self.gp_executors = []
        self.feature_cols = []
        self.gru_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.factor_stats = {}
        
        logger.info(f"🖥️ 使用设备: {self.device}")
    
    def load_gp_factors(self, filepath: str = None) -> bool:
        """加载 GP 因子"""
        if filepath is None:
            filepath = os.path.join(FACTOR_OUTPUT_DIR, "mined_factors_v2.csv")
        
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ 因子文件不存在: {filepath}")
            return False
        
        df_factors = pd.read_csv(filepath)
        
        # 选择有效的因子
        if 'selected' in df_factors.columns:
            df_selected = df_factors[df_factors['selected'] == True]
        elif 'valid' in df_factors.columns:
            df_selected = df_factors[df_factors['valid'] == True]
        else:
            df_selected = df_factors.head(30)
        
        if len(df_selected) == 0:
            logger.warning("⚠️ 没有选中的因子")
            return False
        
        self.gp_programs = df_selected['program'].tolist()
        logger.info(f"📥 加载了 {len(self.gp_programs)} 个 GP 因子")
        
        # 打印统计
        if 'ic_mean' in df_selected.columns:
            logger.info(f"   平均 IC: {df_selected['ic_mean'].abs().mean():.4f}")
            logger.info(f"   平均 IR: {df_selected['ir'].abs().mean():.4f}")
        
        return True
    
    def compute_gp_factors(self, df: pd.DataFrame, 
                          base_features: List[str]) -> pd.DataFrame:
        """
        计算 GP 因子值
        """
        logger.info("🔧 计算 GP 因子值...")
        
        df_result = df.copy()
        
        # 如果没有 GP 程序，使用基础 Alpha 因子
        if not self.gp_programs:
            logger.info("   使用预设 Alpha 因子")
            return self._compute_alpha_factors(df_result, base_features)
        
        # 创建执行器并计算
        self.gp_executors = []
        valid_count = 0
        
        for i, program in enumerate(tqdm(self.gp_programs, desc="计算 GP 因子")):
            try:
                executor = FactorProgramExecutor(program, base_features)
                self.gp_executors.append(executor)
                
                # 计算因子值
                factor_values = executor.execute(df_result)
                col_name = f'gp_factor_{i}'
                df_result[col_name] = factor_values
                
                # 记录统计信息
                self.factor_stats[col_name] = {
                    'mean': float(factor_values.mean()),
                    'std': float(factor_values.std()),
                    'program': program
                }
                
                valid_count += 1
                
            except Exception as e:
                logger.warning(f"   因子 {i} 计算失败: {e}")
                continue
        
        # 添加基础 Alpha 因子作为补充
        df_result = self._compute_alpha_factors(df_result, base_features)
        
        # 更新特征列表
        self.feature_cols = [c for c in df_result.columns 
                            if c.startswith('gp_factor_') or c.startswith('alpha_')]
        
        logger.info(f"✅ 共计算 {valid_count} 个 GP 因子 + Alpha 因子")
        logger.info(f"   总特征数: {len(self.feature_cols)}")
        
        return df_result
    
    def _compute_alpha_factors(self, df: pd.DataFrame, 
                               base_features: List[str]) -> pd.DataFrame:
        """计算经典 Alpha 因子"""
        df = df.copy()
        
        # Alpha 001: (close - open) / ((high - low) + 0.001)
        if all(c in df.columns for c in ['close', 'open', 'high', 'low']):
            df['alpha_001'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
        
        # Alpha 002: rank(vol) * rank(returns)
        if 'vol' in df.columns and 'ret_1d' in df.columns:
            df['alpha_002'] = df.groupby('trade_date')['vol'].rank(pct=True) * \
                             df.groupby('trade_date')['ret_1d'].rank(pct=True)
        
        # Alpha 003: -1 * correlation(rank(close), rank(vol), 5)
        if 'close' in df.columns and 'vol' in df.columns:
            df['rank_close'] = df.groupby('trade_date')['close'].rank(pct=True)
            df['rank_vol'] = df.groupby('trade_date')['vol'].rank(pct=True)
            df['alpha_003'] = -df.groupby('ts_code').apply(
                lambda x: x['rank_close'].rolling(5).corr(x['rank_vol'])
            ).reset_index(level=0, drop=True).fillna(0)
        
        # Alpha 004: -1 * ts_rank(low, 10)
        if 'low' in df.columns:
            df['alpha_004'] = -df.groupby('ts_code')['low'].transform(
                lambda x: x.rolling(10).apply(lambda y: y.rank(pct=True).iloc[-1] if len(y) > 0 else 0.5)
            ).fillna(0.5)
        
        # Alpha 005: rank(open - close) / rank(open + close)
        if all(c in df.columns for c in ['open', 'close']):
            df['alpha_005'] = df.groupby('trade_date')['open'].apply(
                lambda x: (x - df.loc[x.index, 'close']).rank(pct=True)
            ) / (df['open'] + df['close'] + 0.001)
        
        # 清理临时列
        for col in ['rank_close', 'rank_vol']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          fit_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        截面标准化（Z-Score）
        """
        df = df.copy()
        
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            
            if fit_df is not None and col in fit_df.columns:
                # 使用训练集的统计量
                mean = fit_df.groupby('trade_date')[col].mean().mean()
                std = fit_df.groupby('trade_date')[col].std().mean()
            else:
                # 使用当前截面
                mean = df.groupby('trade_date')[col].transform('mean')
                std = df.groupby('trade_date')[col].transform('std') + 1e-8
            
            df[col] = (df[col] - mean) / std
            df[col] = df[col].clip(-5, 5)  # 截断极端值
            df[col] = df[col].fillna(0)
        
        return df
    
    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        """
        训练 GRU 模型
        """
        logger.info("\n🚀 开始训练 GRU 模型...")
        
        # 标准化
        train_df = self.normalize_features(train_df)
        valid_df = self.normalize_features(valid_df, fit_df=train_df)
        
        # 创建数据集
        train_dataset = TimeSeriesDataset(
            train_df, self.feature_cols, 
            seq_len=GRU_CONFIG['step_len'],
            step=1
        )
        valid_dataset = TimeSeriesDataset(
            valid_df, self.feature_cols,
            seq_len=GRU_CONFIG['step_len'],
            step=5  # 验证集步长更大，加快训练
        )
        
        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            logger.error("❌ 数据集为空")
            return
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=GRU_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=GRU_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"   训练批次: {len(train_loader)}, 验证批次: {len(valid_loader)}")
        
        # 创建模型
        self.gru_model = GRUModel(
            input_size=len(self.feature_cols),
            hidden_size=GRU_CONFIG['hidden_size'],
            num_layers=GRU_CONFIG['num_layers'],
            dropout=GRU_CONFIG['dropout']
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.gru_model.parameters(),
            lr=GRU_CONFIG['learning_rate'],
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        best_ic = -1
        patience_counter = 0
        
        # 训练循环
        for epoch in range(GRU_CONFIG['epochs']):
            # 训练
            self.gru_model.train()
            train_loss = 0
            train_count = 0
            
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                x = x.to(self.device)
                y = y.to(self.device)
                
                # 数据清洗
                x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
                y = torch.nan_to_num(y, nan=0.0)
                
                optimizer.zero_grad()
                pred = self.gru_model(x)
                loss = criterion(pred, y)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gru_model.parameters(), max_norm=5.0
                )
                optimizer.step()
                
                train_loss += loss.item()
                train_count += 1
            
            avg_train_loss = train_loss / max(train_count, 1)
            
            # 验证
            valid_ic = self._evaluate(valid_loader)
            scheduler.step(valid_ic)
            
            logger.info(f"   Epoch {epoch+1}/{GRU_CONFIG['epochs']} | "
                       f"Loss: {avg_train_loss:.6f} | Valid IC: {valid_ic:.4f}")
            
            # 早停
            if valid_ic > best_ic:
                best_ic = valid_ic
                patience_counter = 0
                self.save_model("gru_best_v2.pth")
                logger.info(f"   💾 保存最佳模型 (IC: {best_ic:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= GRU_CONFIG['early_stopping_patience']:
                    logger.info(f"   ⏹️ 早停触发，最佳 Valid IC: {best_ic:.4f}")
                    break
        
        logger.info(f"\n✅ 训练完成，最佳 Valid IC: {best_ic:.4f}")
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """评估模型"""
        self.gru_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                
                pred = self.gru_model(x)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.numpy())
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        # 计算 Rank IC
        mask = ~(np.isnan(preds) | np.isnan(labels))
        if mask.sum() < 100:
            return 0.0
        
        p_rank = pd.Series(preds[mask]).rank()
        l_rank = pd.Series(labels[mask]).rank()
        
        ic = np.corrcoef(p_rank, l_rank)[0, 1]
        return ic if not np.isnan(ic) else 0.0
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """预测"""
        df = self.normalize_features(df)
        
        dataset = TimeSeriesDataset(
            df, self.feature_cols,
            seq_len=GRU_CONFIG['step_len'],
            step=1
        )
        loader = DataLoader(dataset, batch_size=GRU_CONFIG['batch_size'],
                           shuffle=False, num_workers=0)
        
        self.gru_model.eval()
        all_preds = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                x = torch.nan_to_num(x, nan=0.0)
                pred = self.gru_model(x)
                all_preds.extend(pred.cpu().numpy())
        
        # 创建结果 DataFrame
        # 简化处理：预测值对应到原始数据的末尾
        df_result = df.copy()
        df_result['pred'] = np.nan
        
        # 这里简化处理，实际需要按日期对齐
        if len(all_preds) > 0:
            df_result.loc[df_result.index[-len(all_preds):], 'pred'] = all_preds
        
        return df_result
    
    def save_model(self, filename: str):
        """保存模型"""
        if self.gru_model is None:
            return
        
        save_dict = {
            'model_state': self.gru_model.state_dict(),
            'config': GRU_CONFIG,
            'feature_cols': self.feature_cols,
            'gp_programs': self.gp_programs,
            'factor_stats': self.factor_stats
        }
        
        filepath = os.path.join(MODEL_OUTPUT_DIR, filename)
        torch.save(save_dict, filepath)
        logger.info(f"💾 模型已保存: {filepath}")
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(MODEL_OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ 模型文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_cols = checkpoint['feature_cols']
        self.gp_programs = checkpoint.get('gp_programs', [])
        self.factor_stats = checkpoint.get('factor_stats', {})
        
        self.gru_model = GRUModel(
            input_size=len(self.feature_cols),
            hidden_size=GRU_CONFIG['hidden_size'],
            num_layers=GRU_CONFIG['num_layers'],
            dropout=GRU_CONFIG['dropout']
        ).to(self.device)
        
        self.gru_model.load_state_dict(checkpoint['model_state'])
        logger.info(f"📥 模型已加载: {filepath}")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 两阶段模型 V2: GP因子 + GRU")
    print("=" * 60)
    
    # 1. 加载数据
    loader = StockDataLoader()
    df_raw = loader.load_all_data()
    df_features = loader.prepare_features(df_raw)
    df_labeled = loader.prepare_labels(df_features)
    loader.close()
    
    # 2. 划分数据集
    train_df = df_labeled[df_labeled['trade_date'] <= TRAIN_END]
    valid_df = df_labeled[(df_labeled['trade_date'] > TRAIN_END) & 
                           (df_labeled['trade_date'] <= VALID_END)]
    test_df = df_labeled[df_labeled['trade_date'] > VALID_END]
    
    logger.info(f"\n📊 数据集:")
    logger.info(f"   训练: {len(train_df)} 条")
    logger.info(f"   验证: {len(valid_df)} 条")
    logger.info(f"   测试: {len(test_df)} 条")
    
    # 3. 创建模型
    model = TwoStageModelV2()
    
    # 4. 加载或计算 GP 因子
    base_features = ['open', 'high', 'low', 'close', 'vol', 'ret_1d', 'ret_5d']
    base_features = [c for c in base_features if c in train_df.columns]
    
    if not model.load_gp_factors():
        logger.info("⚠️ 未找到 GP 因子，将只使用 Alpha 因子")
    
    # 计算因子
    train_df = model.compute_gp_factors(train_df, base_features)
    valid_df = model.compute_gp_factors(valid_df, base_features)
    test_df = model.compute_gp_factors(test_df, base_features)
    
    # 5. 训练 GRU
    model.train(train_df, valid_df)
    
    # 6. 测试集评估
    logger.info("\n📊 测试集评估...")
    model.load_model("gru_best_v2.pth")
    test_result = model.predict(test_df)
    
    # 计算测试集 IC
    test_mask = test_result['pred'].notna()
    if test_mask.sum() > 100:
        test_ic = np.corrcoef(
            test_result.loc[test_mask, 'pred'].rank(),
            test_result.loc[test_mask, LABEL_COL].rank()
        )[0, 1]
        logger.info(f"   测试集 Rank IC: {test_ic:.4f}")
    
    # 保存因子统计
    stats_path = os.path.join(MODEL_OUTPUT_DIR, "factor_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(model.factor_stats, f, indent=2)
    logger.info(f"💾 因子统计已保存: {stats_path}")
    
    print("\n" + "=" * 60)
    print("✅ 两阶段模型训练和评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
