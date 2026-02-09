"""
Transformer模型 + 对抗训练 (Adversarial Training)
提升模型鲁棒性和泛化能力
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config_v3 import TRANSFORMER_CONFIG_V3, ADVERSARIAL_CONFIG


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerPredictor(nn.Module):
    """Transformer预测器"""
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 多层归一化和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model // 2)
        self.dropout = nn.Dropout(dropout)
        
        # 输出头
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x, return_features=False):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        features = x[:, -1, :]  # 取最后时刻
        x = self.norm1(features)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        if return_features:
            return x.squeeze(), features
        return x.squeeze()


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, df: pd.DataFrame, features: List[str], seq_len: int = 30):
        self.seq_len = seq_len
        self.features = features
        self.samples = []
        
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')
            vals = group[features + ['label']].values
            if len(vals) > seq_len:
                for i in range(len(vals) - seq_len):
                    if not np.isnan(vals[i:i+seq_len]).any():
                        self.samples.append(vals[i:i+seq_len])
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq = self.samples[idx]
        return (torch.tensor(seq[:-1, :-1], dtype=torch.float32),
                torch.tensor(seq[-1, -1], dtype=torch.float32))


class AdversarialTrainer:
    """对抗训练器"""
    
    def __init__(self, input_size: int, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerPredictor(
            input_size=input_size,
            d_model=TRANSFORMER_CONFIG_V3['d_model'],
            nhead=TRANSFORMER_CONFIG_V3['nhead'],
            num_layers=TRANSFORMER_CONFIG_V3['num_layers'],
            dim_feedforward=TRANSFORMER_CONFIG_V3['dim_feedforward'],
            dropout=TRANSFORMER_CONFIG_V3['dropout']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRANSFORMER_CONFIG_V3['learning_rate'],
            weight_decay=TRANSFORMER_CONFIG_V3['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TRANSFORMER_CONFIG_V3['epochs']
        )
        
        self.criterion = nn.MSELoss()
        logger.info(f"[AdversarialTrainer] Device: {self.device}")
    
    def generate_adversarial_examples(self, x: torch.Tensor, y: torch.Tensor,
                                      epsilon: float = 0.01, pgd_steps: int = 3) -> torch.Tensor:
        """生成对抗样本 (PGD攻击)"""
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(pgd_steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            self.model.train()
            pred = self.model(x_adv)
            loss = self.criterion(pred, y)
            loss.backward()
            
            # 生成扰动
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv.detach() + epsilon * grad_sign
            
            # 裁剪到有效范围
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, -5, 5)  # 特征范围
            x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv.detach()
    
    def train_epoch(self, train_loader: DataLoader, adversarial: bool = True) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_adv_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # 正常前向传播
            pred = self.model(x)
            clean_loss = self.criterion(pred, y)
            
            if adversarial and ADVERSARIAL_CONFIG['enabled']:
                # 生成对抗样本
                x_adv = self.generate_adversarial_examples(
                    x, y,
                    epsilon=ADVERSARIAL_CONFIG['epsilon'],
                    pgd_steps=ADVERSARIAL_CONFIG['pgd_steps']
                )
                
                # 对抗损失
                pred_adv = self.model(x_adv)
                adv_loss = self.criterion(pred_adv, y)
                
                # 总损失 = 清洁损失 + α * 对抗损失
                alpha = ADVERSARIAL_CONFIG['alpha']
                loss = clean_loss + alpha * adv_loss
                total_adv_loss += adv_loss.item()
            else:
                loss = clean_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_adv_loss = total_adv_loss / len(train_loader) if adversarial else 0
        
        return avg_loss, avg_adv_loss
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        all_pred, all_true = [], []
        total_loss = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        
        # 计算Rank IC
        mask = ~(np.isnan(all_pred) | np.isnan(all_true))
        if mask.sum() < 100:
            return 0, total_loss / len(data_loader)
        
        ic = np.corrcoef(
            pd.Series(np.array(all_pred)[mask]).rank(),
            pd.Series(np.array(all_true)[mask]).rank()
        )[0, 1]
        
        return ic if not np.isnan(ic) else 0, total_loss / len(data_loader)
    
    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
              features: List[str], save_path: str = None) -> dict:
        """完整训练流程"""
        # 创建数据集
        train_dataset = TimeSeriesDataset(train_df, features, TRANSFORMER_CONFIG_V3['seq_len'])
        valid_dataset = TimeSeriesDataset(valid_df, features, TRANSFORMER_CONFIG_V3['seq_len'])
        
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_CONFIG_V3['batch_size'],
                                 shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=TRANSFORMER_CONFIG_V3['batch_size'],
                                 num_workers=0)
        
        logger.info(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
        
        best_ic = -1
        patience_counter = 0
        history = {'train_loss': [], 'valid_ic': [], 'valid_loss': []}
        
        for epoch in range(TRANSFORMER_CONFIG_V3['epochs']):
            # 训练
            train_loss, adv_loss = self.train_epoch(train_loader, adversarial=True)
            
            # 验证
            valid_ic, valid_loss = self.evaluate(valid_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['valid_ic'].append(valid_ic)
            history['valid_loss'].append(valid_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{TRANSFORMER_CONFIG_V3['epochs']} - "
                           f"Train Loss: {train_loss:.4f} (Adv: {adv_loss:.4f}), "
                           f"Valid IC: {valid_ic:.4f}, Valid Loss: {valid_loss:.4f}")
            
            # 早停
            if valid_ic > best_ic:
                best_ic = valid_ic
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"  Saved best model (IC: {best_ic:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= TRANSFORMER_CONFIG_V3['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best Valid IC: {best_ic:.4f}")
        return {'best_ic': best_ic, 'history': history}
    
    def predict(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """预测"""
        dataset = TimeSeriesDataset(df, features, TRANSFORMER_CONFIG_V3['seq_len'])
        loader = DataLoader(dataset, batch_size=TRANSFORMER_CONFIG_V3['batch_size'], shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                pred = self.model(x)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions)


def main():
    """测试"""
    print("Transformer + Adversarial Training Module")
    print("Ready for integration")


if __name__ == "__main__":
    main()
