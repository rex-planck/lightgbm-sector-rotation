import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# 1. 初始化
provider_uri = r"E:\Quant_program\Qlib-Cache\cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)


# 2. PyTorch GRU 模型 (增加一点稳定性)
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)  # 加点 Dropout
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred.squeeze()


# 3. 滑窗 Dataset (保持不变，逻辑没问题)
class RollingDataset(Dataset):
    def __init__(self, df, step_len=20):
        self.step_len = step_len
        self.data_values = df.values
        self.index_map = []

        # 按 instrument 分组计算切片索引
        # 这里为了防止 df index 不规范，我们尝试 reset_index 再 groupby
        # 但 Qlib 数据通常是 MultiIndex，直接 groupby(level='instrument') 即可
        try:
            grouped = df.groupby(level='instrument')
        except TypeError:
            # 备用方案：如果索引有问题，强制重置
            df_temp = df.reset_index()
            grouped = df_temp.groupby('instrument')

        current_idx = 0
        for name, group in grouped:
            group_len = len(group)
            if group_len > step_len:
                valid_starts = np.arange(current_idx, current_idx + group_len - step_len + 1)
                self.index_map.append(valid_starts)
            current_idx += group_len

        if len(self.index_map) > 0:
            self.index_map = np.concatenate(self.index_map)
        else:
            self.index_map = np.array([])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        start_row = self.index_map[idx]
        end_row = start_row + self.step_len
        window = self.data_values[start_row:end_row]

        feature = window[:, :-1]
        label = window[-1, -1]

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# 4. 配置
market = "csi300"
data_handler_config = {
    "start_time": "2015-01-01",
    "end_time": "2022-12-31",
    "fit_start_time": "2015-01-01",
    "fit_end_time": "2020-12-31",
    "instruments": market,
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "label": ["Ref($close, -5) / $close - 1"],
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2015-01-01", "2020-12-31"),
            "valid": ("2021-01-01", "2021-12-31"),
            "test": ("2022-01-01", "2022-12-31"),
        },
    },
}


def run_training():
    with R.start(experiment_name="gru_rolling_window_stable"):
        print("🛠️ 初始化数据...")
        qlib_dataset = init_instance_by_config(dataset_config)
        df_train = qlib_dataset.prepare("train", col_set=["feature", "label"], data_key="infer")
        df_valid = qlib_dataset.prepare("valid", col_set=["feature", "label"], data_key="infer")
        df_test = qlib_dataset.prepare("test", col_set=["feature", "label"], data_key="infer")

        print(f"   Train shape: {df_train.shape}")

        print("🔄 构建数据集...")
        train_set = RollingDataset(df_train, step_len=20)
        valid_set = RollingDataset(df_valid, step_len=20)
        test_set = RollingDataset(df_test, step_len=20)

        # 增大 Batch Size 也能稍微稳一点
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 使用设备: {device}")

        model = SimpleGRU(input_size=158, hidden_size=64).to(device)
        criterion = nn.MSELoss()
        # 🔥 降低学习率，防止步子迈太大扯到蛋
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        print("\n🔥 开始训练 (Training)...")
        epochs = 5
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            count = 0

            for feature, label in train_loader:
                # 🔥 终极防爆 1: 强制清洗数据，把 NaN 变成 0
                feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
                label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)

                feature, label = feature.to(device), label.to(device)

                optimizer.zero_grad()
                pred = model(feature)
                loss = criterion(pred, label)

                # 如果这一步 Loss 还是 NaN，就跳过，别更新权重毁了模型
                if torch.isnan(loss):
                    continue

                loss.backward()

                # 🔥 终极防爆 2: 梯度裁剪 (Gradient Clipping)
                # 这行代码能把所有超过 5.0 的梯度强行拉回来，解决 NaN 的核心
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / count if count > 0 else 0
            print(f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

        print("\n🔮 开始回测 (Backtest)...")
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for feature, label in test_loader:
                # 预测时也要清洗，不然预测出来全是 NaN
                feature = torch.nan_to_num(feature, nan=0.0).to(device)
                pred = model(feature)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label.numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        df_res = pd.DataFrame({"pred": preds, "label": labels})
        # 再次清洗结果，防止计算相关性时报错
        df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_res) > 0:
            ic = df_res.corr().iloc[0, 1]
            rank_ic = df_res.rank().corr().iloc[0, 1]

            print("-" * 50)
            print(f"📊 实验结果 (Stable GRU):")
            print(f"   Samples: {len(df_res)}")
            print(f"   Rank IC: {rank_ic:.4f}")
            print("-" * 50)

            if rank_ic > 0.02:
                torch.save(model.state_dict(), 'gru_best.pth')
                print("💾 模型已保存为 gru_best.pth")
                print("✅ 深度学习模型构建成功！")
        else:
            print("❌ 数据全被过滤掉了，请检查数据源。")


if __name__ == "__main__":
    run_training()