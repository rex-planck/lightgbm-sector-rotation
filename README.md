# 🚀 Tushare 量化策略研究合集

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Data-Tushare%20%7C%20AKShare-red?style=flat-square" alt="Data">
  <img src="https://img.shields.io/badge/Quant-A%20Share-green?style=flat-square" alt="Quant">
</p>

<p align="center">
  <b>「数据 + 算法 + 风控」</b>三位一体的量化交易研究框架
</p>

---

## 📖 项目简介

本项目是一个面向 A 股市场的**多策略量化研究合集**，涵盖从传统多因子选股到前沿 AI 驱动的 Alpha 挖掘。项目分为三个独立模块，每个模块代表一种不同的量化研究范式：

| 模块 | 核心方法 | 策略类型 | 特点 |
|:---:|:---:|:---:|:---|
| 🔬 **混合因子挖掘** | GP + Transformer | 机器学习选股 | 自动发现非线性因子 |
| 🏦 **多因子选股** | 基本面筛选 | 价值投资 | 低估值蓝筹策略 |
| 🧠 **NLP 增强策略** | 情绪分析 + 趋势跟踪 | 事件驱动 | 研报情绪量化 |

> 💡 **设计理念**：探索量化投资的多种可能性，从传统财务分析到现代深度学习，从结构化数据到非结构化文本。

---

## 🗂️ 项目结构

```
Tushare/
├── 📁 Hybrid_Factor_Mining_GP/          # 模块一：混合因子挖掘系统
│   ├── src/
│   │   ├── run_v3_pipeline.py           # V3 主启动脚本
│   │   ├── gp_factor_mining_optimized.py # 遗传规划因子挖掘
│   │   ├── model_transformer_adversarial.py # 对抗性 Transformer
│   │   ├── market_regime_sector_neutral.py  # 市场状态检测
│   │   └── risk_management_v3.py        # 动态风控引擎
│   ├── data/                            # 行情数据存储
│   ├── output_v3/                       # 回测结果输出
│   └── README.md                        # 详细文档
│
├── 📁 Multi-Factor Selection Model/     # 模块二：多因子选股模型
│   └── multi_factor_stock_selection/
│       ├── main.py                      # 主程序入口
│       ├── strategies/
│       │   └── blue_chip_strategy.py    # 低估值蓝筹策略
│       ├── data/                        # 数据获取与存储
│       └── backtest/                    # 回测结果
│
├── 📁 NLP-Enhanced Multi-Factor/        # 模块三：NLP 增强策略
│   ├── SentimentAlphaStrategy/
│   │   └── main.py                      # 策略核心逻辑
│   ├── etl/
│   │   ├── download_reports.py          # 数据获取
│   │   └── calc_report_sentiment.py     # 情绪评分引擎
│   ├── backtest_results/                # 回测图表
│   └── README.md
│
└── README.md                            # 本文件
```

---

## 🔬 模块详解

### 1️⃣ Hybrid Factor Mining - 混合因子挖掘系统 V3

**核心思想**：结合遗传规划（GP）的可解释性与 Transformer 的深度特征提取能力，构建新一代 Alpha 挖掘流水线。

#### 🏗️ 系统架构

```
数据获取 → 特征工程 → GP 因子挖掘 → 对抗性 Transformer → 动态风控回测
```

#### ✨ 核心特性

- **🧬 遗传规划因子挖掘**
  - 大规模种群（1500+）与多代进化（30+ 代）
  - 多样性保护机制，强制挖掘低相关性因子
  - 支持时序专用算子：`ts_rank`, `ts_corr`, `decay_linear`

- **🤖 对抗性 Transformer 模型**
  - 8头自注意力机制，捕捉因子间非线性交互
  - 对抗训练（FGSM）提升模型鲁棒性
  - 集成学习框架（5 模型集成）

- **🌍 市场状态感知**
  - 自动识别牛/熊/震荡市场状态
  - 行业中性化处理（申万一级行业分类）

- **🛡️ 动态风控**
  - 基于 Kelly 公式的动态仓位管理
  - 硬止损（8%）+ 追踪止盈（15%回撤）

#### 📊 回测表现（2022-2026）

| 指标 | 策略表现 | 基准表现 |
|:---:|:---:|:---:|
| 年化收益 | **+28.5%** | -5.2% |
| 夏普比率 | **2.15** | -0.3 |
| 最大回撤 | **-12.4%** | -35.6% |
| Rank IC | **0.092** | - |

#### 🚀 快速开始

```bash
cd Hybrid_Factor_Mining_GP
pip install -r requirements.txt

# 配置 Tushare Token（src/config_v3.py）
# TUSHARE_TOKEN = "your_token_here"

# 运行完整流水线
cd src
python run_v3_pipeline.py
```

---

### 2️⃣ Multi-Factor Selection Model - 低估值蓝筹策略

**核心思想**：基于经典价值投资理念，通过市盈率（PE）、市净率（PB）、市值等多维度因子，筛选出被低估的优质蓝筹股。

#### 🎯 策略逻辑

1. **初步筛选**：PE < 20，PB < 2，市值 > 50亿
2. **财务评分**：ROE、营收增长率、净利润增长率综合打分
3. **行业分散**：避免单一行业过度集中
4. **定期调仓**：月度/季度再平衡

#### 📋 数据流程

```
股票基础信息 → 每日指标数据（PE/PB/市值）→ 财务数据 → 综合评分 → 选股
```

#### 🚀 快速开始

```bash
cd "Multi-Factor Selection Model/multi_factor_stock_selection"
pip install -r requirements.txt

# 运行策略
python main.py
```

---

### 3️⃣ NLP-Enhanced Multi-Factor - NLP 增强策略

**核心思想**：市场价格不仅由财务数据驱动，还受到投资者情绪（舆情）的强烈影响。通过 NLP 技术量化券商研报情绪，构建独特的 Alpha 因子。

#### 🧠 核心策略逻辑

**情绪量化公式**：
```
S_effective = S_initial × e^(-λt)
```
其中 λ 为日度衰减率（默认 0.1），反映信息随时间的影响力衰减。

#### 🔀 因子组合

| 因子类型 | 核心指标 | 作用 |
|:---:|:---:|:---|
| **Alpha** | NLP 情绪评分 | 决定入场/出场信心的主要信号 |
| **Beta** | 均线系统 (5/20) | 趋势确认与择时过滤 |
| **Risk** | 动态回撤限制 | 基于波动率的仓位管理 |

#### 📊 回测标的示例

策略在以下标的展现出稳定的 Alpha 获利能力：

- **000630** - 铜陵有色
- **000875** - 吉电股份  
- **002202** - 金风科技
- **601615** - 明阳智能
- **603067** - 振华股份

#### 🚀 快速开始

```bash
cd "NLP-Enhanced Multi-Factor"
pip install pandas akshare matplotlib

# 1. 同步数据
python etl/download_reports.py

# 2. 生成情绪因子
python etl/calc_report_sentiment.py

# 3. 启动回测
python run_strategy_local.py
```

---

## ⚙️ 环境依赖

### 基础依赖

```bash
# 所有模块通用
pip install pandas numpy matplotlib

# 数据接口
pip install tushare akshare

# 机器学习（混合因子挖掘模块）
pip install torch scikit-learn gplearn tqdm
```

### 数据接口配置

| 数据源 | 配置方式 | 适用模块 |
|:---:|:---:|:---:|
| **Tushare** | `TUSHARE_TOKEN` 环境变量或配置文件 | 模块一、二 |
| **AKShare** | 开箱即用 | 模块三 |

---

## 🗺️ 路线图

- [ ] **LLM 升级**：从词典评分转向 FinBERT/GPT-4 架构
- [ ] **多源舆情**：引入雪球、股吧等社交媒体情绪
- [ ] **实盘接口**：对接 QMT/Ptrade 实现自动化交易
- [ ] **在线学习**：模型增量更新与自适应优化
- [ ] **风险平价**：基于实时因子波动率的动态仓位配比

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
