"""
Transformer模型 + 集成学习框架
包含正则化和学习率调度
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict
from tqdm import tqdm

from config_optimized import TRANSFORMER_CONFIG, ENSEMBLE_CONFIG, MODEL_OUTPUT_DIR


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


class TransformerModel(nn.Module):
    """Transformer预测模型"""
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 正则化层
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后时刻
        x = self.norm1(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()


class LSTMModel(nn.Module):
    """LSTM备选模型"""
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()


class GRUModel(nn.Module):
    """GRU备选模型"""
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()


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
                    seq = vals[i:i+seq_len]
                    if not np.isnan(seq).any():
                        self.samples.append(seq)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1, :-1], dtype=torch.float32)
        y = torch.tensor(seq[-1, -1], dtype=torch.float32)
        return x, y


class EnsembleTrainer:
    """集成学习训练器"""
    
    def __init__(self, input_size: int, device: str = None):
        self.input_size = input_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.model_types = []
        print(f"[Ensemble] Using device: {self.device}")
    
    def create_model(self, model_type: str):
        """创建单个模型"""
        if model_type == 'transformer':
            return TransformerModel(
                input_size=self.input_size,
                d_model=TRANSFORMER_CONFIG['d_model'],
                nhead=TRANSFORMER_CONFIG['nhead'],
                num_layers=TRANSFORMER_CONFIG['num_layers'],
                dim_feedforward=TRANSFORMER_CONFIG['dim_feedforward'],
                dropout=TRANSFORMER_CONFIG['dropout']
            )
        elif model_type == 'lstm':
            return LSTMModel(self.input_size)
        elif model_type == 'gru':
            return GRUModel(self.input_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_single_model(self, model: nn.Module, train_loader: DataLoader, 
                          valid_loader: DataLoader, model_idx: int) -> Dict:
        """训练单个模型"""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRANSFORMER_CONFIG['learning_rate'],
            weight_decay=TRANSFORMER_CONFIG['weight_decay']
        )
        
        # 学习率调度
        if TRANSFORMER_CONFIG['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TRANSFORMER_CONFIG['epochs']
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3
            )
        
        best_ic = -1
        patience_counter = 0
        history = {'train_loss': [], 'valid_ic': []}
        
        for epoch in range(TRANSFORMER_CONFIG['epochs']):
            # 训练
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            valid_ic = self.evaluate(model, valid_loader)
            
            if TRANSFORMER_CONFIG['lr_scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(valid_ic)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['valid_ic'].append(valid_ic)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Model {model_idx} Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Valid IC={valid_ic:.4f}")
            
            # 早停
            if valid_ic > best_ic:
                best_ic = valid_ic
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f"{MODEL_OUTPUT_DIR}/model_{model_idx}_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= TRANSFORMER_CONFIG['early_stopping_patience']:
                    print(f"  Model {model_idx} early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳权重
        model.load_state_dict(torch.load(f"{MODEL_OUTPUT_DIR}/model_{model_idx}_best.pth"))
        
        return {'model': model, 'best_ic': best_ic, 'history': history}
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader) -> float:
        """评估模型"""
        model.eval()
        all_pred, all_true = [], []
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                pred = model(x).cpu().numpy()
                all_pred.extend(pred)
                all_true.extend(y.numpy())
        
        # 计算Rank IC
        mask = ~(np.isnan(all_pred) | np.isnan(all_true))
        if mask.sum() < 100:
            return 0.0
        
        ic = np.corrcoef(
            pd.Series(np.array(all_pred)[mask]).rank(),
            pd.Series(np.array(all_true)[mask]).rank()
        )[0, 1]
        
        return ic if not np.isnan(ic) else 0.0
    
    def train_ensemble(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                       features: List[str], n_models: int = 5):
        """训练集成模型"""
        print(f"\n[Ensemble] Training {n_models} models...")
        
        # 模型类型轮换
        model_types = ['transformer', 'lstm', 'gru', 'transformer', 'lstm']
        
        for i in range(n_models):
            print(f"\nTraining model {i+1}/{n_models} ({model_types[i]})...")
            
            # Bagging: 采样部分数据
            if ENSEMBLE_CONFIG['bagging_ratio'] < 1.0:
                sample_df = train_df.sample(frac=ENSEMBLE_CONFIG['bagging_ratio'], random_state=i*42)
            else:
                sample_df = train_df
            
            # 创建数据集
            train_dataset = TimeSeriesDataset(sample_df, features, TRANSFORMER_CONFIG['step_len'])
            valid_dataset = TimeSeriesDataset(valid_df, features, TRANSFORMER_CONFIG['step_len'])
            
            train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_CONFIG['batch_size'], shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=TRANSFORMER_CONFIG['batch_size'])
            
            # 创建并训练模型
            model = self.create_model(model_types[i])
            result = self.train_single_model(model, train_loader, valid_loader, i)
            
            self.models.append(result['model'])
            self.model_types.append(model_types[i])
            
            print(f"  Model {i+1} best Valid IC: {result['best_ic']:.4f}")
        
        print(f"\n[OK] Ensemble training completed. {len(self.models)} models ready.")
    
    def predict(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """集成预测"""
        dataset = TimeSeriesDataset(df, features, TRANSFORMER_CONFIG['step_len'])
        loader = DataLoader(dataset, batch_size=TRANSFORMER_CONFIG['batch_size'], shuffle=False)
        
        all_preds = []
        
        for model in self.models:
            model.eval()
            preds = []
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(self.device)
                    pred = model(x).cpu().numpy()
                    preds.extend(pred)
            all_preds.append(preds)
        
        # 平均集成
        ensemble_pred = np.mean(all_preds, axis=0)
        return ensemble_pred


def main():
    """测试"""
    print("="*70)
    print("TRANSFORMER ENSEMBLE MODEL")
    print("="*70)
    
    # 加载数据
    train_df = pd.read_pickle(f'{MODEL_OUTPUT_DIR}/../factors/train_data_optimized.pkl')
    valid_df = pd.read_pickle(f'{MODEL_OUTPUT_DIR}/../factors/valid_data_optimized.pkl')
    test_df = pd.read_pickle(f'{MODEL_OUTPUT_DIR}/../factors/test_data_optimized.pkl')
    
    # 获取GP因子
    gp_features = [c for c in train_df.columns if c.startswith('gp_factor_')]
    print(f"GP features: {len(gp_features)}")
    
    # 归一化
    for col in gp_features:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-8
        train_df[col] = ((train_df[col] - mean) / std).clip(-5, 5)
        valid_df[col] = ((valid_df[col] - mean) / std).clip(-5, 5)
        test_df[col] = ((test_df[col] - mean) / std).clip(-5, 5)
    
    # 训练集成模型
    trainer = EnsembleTrainer(input_size=len(gp_features))
    trainer.train_ensemble(train_df, valid_df, gp_features, n_models=ENSEMBLE_CONFIG['n_models'])
    
    # 测试
    print("\n[Testing]...")
    test_pred = trainer.predict(test_df, gp_features)
    test_ic = np.corrcoef(pd.Series(test_pred).rank(), test_df['label'].rank())[0, 1]
    print(f"Test IC: {test_ic:.4f}")
    
    # 保存预测
    results = pd.DataFrame({'pred': test_pred, 'label': test_df['label'].values[:len(test_pred)]})
    results.to_csv(f'{MODEL_OUTPUT_DIR}/ensemble_predictions.csv', index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
