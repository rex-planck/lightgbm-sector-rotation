# 🚀 Tushare 量化策略研究合集

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Data-Tushare%20%7C%20AKShare%20%7C%20Qlib-red?style=flat-square" alt="Data">
  <img src="https://img.shields.io/badge/Quant-A%20Share-green?style=flat-square" alt="Quant">
</p>

<p align="center">
  <b>「数据 + 算法 + 风控」</b>三位一体的量化交易研究框架
</p>

---

## 📖 项目简介

本项目是一个面向 A 股市场的**多策略量化研究合集**，涵盖从宏观择时、传统多因子选股到前沿 AI 驱动的 Alpha 挖掘。项目分为五个独立模块，代表了不同的量化研究范式：

| 模块 | 核心方法 | 策略类型 | 特点 |
|:---:|:---:|:---:|:---|
| 🔬 **混合因子挖掘** | GP + Transformer | 机器学习选股 | 自动发现非线性因子 |
| 🌍 **宏观行业轮动** | HMM + 宏观流动性 | 宏观择时 | 自上而下的行业配置 |
| 🤖 **AI 量化平台** | Qlib + LightGBM | 机器学习 Alpha | 基于 MSRA Qlib 的全链路方案 |
|  **多因子选股** | 基本面筛选 | 价值投资 | 低估值蓝筹策略 |
| 🧠 **NLP 增强策略** | 情绪分析 + 趋势跟踪 | 事件驱动 | 研报情绪量化 |

> 💡 **设计理念**：探索量化投资的多种可能性，从传统财务分析到现代深度学习，从结构化数据到非结构化文本，从微观个股到宏观配置。

---

## 🗂️ 项目结构

```
Tushare/
├── 📁 Hybrid_Factor_Mining_GP/          # 模块一：混合因子挖掘系统
│   ├── src/
│   │   ├── run_v3_pipeline.py           # V3 主启动脚本
│   │   └── gp_factor_mining_optimized.py # 遗传规划因子挖掘
│   └── output_v3/                       # 回测结果输出
│
├── 📁 AShare_Macro_Rotation/            # 模块二：宏观行业轮动系统
│   ├── models/                          # HMM 宏观状态识别
│   ├── strategy/                        # 轮动与择时策略
│   └── data/                            # 宏观与行业数据
│
├── 📁 GRU/                              # 模块三：Qlib AI 量化平台
│   ├── 02_train_model.py                # Alpha158 + LightGBM 训练
│   ├── 03_backtest_simulation.py        # Top-K Dropout 回测
│   └── mlruns/                          # MLflow 实验记录
│
├── 📁 Multi-Factor Selection Model/     # 模块四：多因子选股模型
│   └── multi_factor_stock_selection/
│       ├── main.py                      # 蓝筹策略主入口
│       └── strategies/                  # 策略逻辑实现
│
├── 📁 NLP-Enhanced Multi-Factor/        # 模块五：NLP 增强策略
│   ├── SentimentAlphaStrategy/          # 情绪因子策略
│   └── etl/                             # 研报数据清洗与情绪计算
│
└── README.md                            # 本文件
```

---

## 🔬 模块详解

### 1️⃣ Hybrid Factor Mining - 混合因子挖掘系统 V3

**核心思想**：结合遗传规划（GP）的可解释性与 Transformer 的深度特征提取能力，构建新一代 Alpha 挖掘流水线。

#### ✨ 核心特性
- **🧬 遗传规划因子挖掘**：利用 `gplearn` 进行大规模种群进化，自动发现非线性 Alpha 因子。
- **🤖 对抗性 Transformer**：引入对抗训练（Adversarial Training）和多头注意力机制，提升模型鲁棒性。
- **🌍 市场状态感知**：自动识别牛/熊/震荡状态，结合行业中性化处理。
- **📊 显著超额**：年化收益 +28.5%，夏普比率 2.15 (2022-2026)。

#### 🚀 快速开始
```bash
cd Hybrid_Factor_Mining_GP/src
python run_v3_pipeline.py
```

---

### 2️⃣ AShare Macro Rotation - 宏观行业轮动

**核心思想**：解决“因时制宜”问题，结合宏观流动性视角 (Top-Down) 与微观机器学习择时 (Bottom-Up)。

#### ✨ 核心特性
- **📊 HMM 状态识别**：利用隐马尔可夫模型识别 Panic/Oscillation/Bull 三种市场状态。
- **💸 流动性接管**：引入 M2-CPI 剪刀差作为超级信号，在流动性泛滥时强制做多。
- **🛡️ 避险与进攻**：完美规避 2022/2024 熊市，精准捕捉 2025 流动性牛市（半导体 +118%）。

#### 🚀 快速开始
```bash
cd AShare_Macro_Rotation
python strategy/batch_run.py
```

---

### 3️⃣ Qlib AI Alpha - 基于 MSRA Qlib 的量化平台

**核心思想**：基于微软 Qlib 框架的高性能全链路量化投研方案。

#### ✨ 核心特性
- **⚡ 高性能数据引擎**：基于二进制存储，极速读取 OHLCV 数据。
- **📈 Alpha158 因子**：内置经典的 158 个量价因子库。
- **🌲 LightGBM 模型**：基于梯度提升树的预测模型，Rank IC 达到 0.0536。
- **📉 稳健回测**：在 2022 年熊市中实现 +8.12% 的超额收益 (Alpha)。

#### 🚀 快速开始
```bash
cd GRU
python 02_train_model.py
python 03_backtest_simulation.py
```

---

### 4️⃣ Multi-Factor Selection - 低估值蓝筹策略

**核心思想**：基于经典价值投资理念，筛选低估值优质蓝筹股。

#### ✨ 核心特性
- **🎯 价值筛选**：PE < 20, PB < 2, 市值 > 50亿。
- **📈 财务评分**：综合 ROE、营收增长率、净利润增长率。
- **⚖️ 均衡配置**：行业分散与定期调仓。

#### 🚀 快速开始
```bash
cd "Multi-Factor Selection Model/multi_factor_stock_selection"
python main.py
```

---

### 5️⃣ NLP-Enhanced Strategy - 研报情绪增强

**核心思想**：通过 NLP 技术量化券商研报情绪，构建独特的 Alpha 因子。

#### ✨ 核心特性
- **📝 文本挖掘**：从非结构化研报中提取情绪分值。
- **📉 信号衰减**：引入时间衰减模型 $e^{-\lambda t}$ 反映信息时效性。
- **🔄 因子融合**：结合均线趋势与波动率风控。

#### 🚀 快速开始
```bash
cd "NLP-Enhanced Multi-Factor"
python etl/download_reports.py
python run_strategy_local.py
```

---

## ⚙️ 环境依赖

推荐使用 Python 3.8+。

```bash
# 通用数据接口
pip install tushare akshare

# 核心计算库
pip install pandas numpy matplotlib scikit-learn

# 深度学习与机器学习 (Hybrid Mining & Qlib)
pip install torch lightgbm gplearn hmmlearn

# Qlib 专用 (Linux/WSL 推荐)
pip install pyqlib
```

---

## ⚠️ 免责声明

> **本项目仅用于教学与研究目的，不构成任何投资建议。**
> 
> 量化交易具有高度风险，历史表现不代表未来收益。请根据自身情况独立判断，入市需谨慎。

---

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request 来改进本项目：

- **Bug 反馈**：请附上完整的错误日志和复现步骤
- **功能建议**：欢迎讨论新的因子算子或模型架构
- **数据问题**：请说明具体的数据接口和错误信息

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。

Copyright (c) 2026 Quant Lab.

---

<p align="center">
  <i>⭐ 如果这个项目对你有帮助，欢迎点个 Star！</i>
</p>
