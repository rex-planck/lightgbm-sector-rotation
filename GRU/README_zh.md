
<div align="center">
  <a href="./README_zh.md">简体中文</a> | 
  <a href="./README.md">English</a>
</div>

<br />

# 基于 MSRA Qlib 的 AI Alpha 量化投研平台 🚀

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Qlib](https://img.shields.io/badge/Quant-MSRA%20Qlib-orange.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![Status](https://img.shields.io/badge/Status-Backtest%20Passed-success.svg)

## 📖 项目简介 (Introduction)

本项目是一个基于微软 [Qlib](https://github.com/microsoft/qlib) 开发的**全链路量化投研框架**。项目实现了从**高性能二进制数据存储**到 **Alpha 因子挖掘**、**机器学习建模 (GBDT)** 以及**考虑交易成本的回测**的端到端工作流。

系统主要针对 **沪深300 (CSI 300)** 市场。通过利用 **Alpha158** 因子库和 **LightGBM** 模型，该策略在 2022 年的市场剧烈波动中成功捕捉到了显著的 Alpha，展现了稳健的选股能力。

---

## 📊 核心绩效 (2022 样本外)

> **"量化建模不仅仅是预测股价，更是在严格成本约束下的盈亏博弈。"**

模型训练数据区间为 **2015-2020**，验证集为 **2021**，测试集为 **2022**（中国市场的典型熊市年份）。

### 1. 模型预测能力
- **Rank IC (信息系数):** `0.0536` (Pearson: `0.0540`)
- *解读：在日频交易中，Rank IC > 0.05 表明模型具有较强的截面预测能力。*

### 2. 回测结果 (Top-K Dropout 策略)
- **基准指数:** 沪深300指数 (SH000300)
- **策略累计收益:** `~ -13.9%` (在暴跌中表现出显著韧性)
- **基准累计收益:** `~ -21.6%`
- **超额收益 (Alpha):** `+8.12%` (年化，扣除成本后)
- **信息比率 (IR):** `1.04`
- **交易成本:** 包含佣金 `0.05%` + 印花税 `0.1%`。

![Backtest Curve](backtest_final_success.png)
*(图：2022年累计净值曲线。红色区域代表策略积累的 Alpha)*

---

## 🛠️ 项目架构

项目结构遵循专业的量化工程工作流：

```text
Python-Quant-Practice
├── python_program/
│   ├── 00_unpack_data.py          # 数据工程: 部署高性能二进制数据引擎
│   ├── 01_test_setup.py           # 环境: 验证 Qlib 初始化
│   ├── 02_train_model.py          # 建模: Alpha158 因子提取与 LightGBM 训练
│   ├── 03_backtest_simulation.py  # 回测: 包含交易摩擦成本的 Top-K 策略模拟
│   ├── 04_analysis.py             # 分析: 绩效可视化与归因分析
│   ├── mlruns/                    # 日志: 实验记录 (由 MLflow 管理)
│   └── backtest_final_success.png # 结果可视化图表
├── .gitignore
└── README_zh.md

```

---

## 🔬 方法论 (Methodology)

### 1. 数据工程 (High-Performance)

* **引擎:** Qlib Binary Storage (`.bin` 文件)，读取速度远超 CSV/Database。
* **特征:** 日频 OHLCV + VWAP。
* **股票池:** 沪深300成分股。
* **预处理:** `RobustZScoreNorm` (去极值) + `Fillna`。

### 2. Alpha 研究 (Factor Mining)

* **因子库:** **Alpha158** (包含动量、波动率、量价相关性等 158 个量价因子)。
* **标签 (Label):** 未来 5 日超额收益 (`Ref($close, -5) / $close - 1`)。
* **处理:** `CSRankNorm` (截面排序标准化)，确保标签的市场中性，消除大盘涨跌影响。

### 3. 机器学习建模 (The Engine)

* **模型:** **LightGBM** (梯度提升决策树)。
* **核心超参:**
* `learning_rate`: 0.0421
* `num_leaves`: 210
* `max_depth`: 8
* `early_stopping`: 启用以防止过拟合。



### 4. 策略与执行

* **策略:** `TopkDropoutStrategy` (优胜劣汰策略)。
* **逻辑:** 每日持有预测分数最高的 50 只股票。
* **换手控制:** 每日仅替换排名跌出前 50 的最后 5 只股票 (`n_drop=5`)，极大地降低了换手率和交易成本。
* **风控:** 剔除停牌股及涨跌停股票。

---

## 🚀 快速开始 (Quick Start)

### 前置要求

* Python 3.8+
* Microsoft Qlib
* Pandas, NumPy, Matplotlib

### 安装

```bash
# 克隆项目
git clone [https://github.com/rex-planck/Python-Quant-Practice.git](https://github.com/rex-planck/Python-Quant-Practice.git)

# 安装依赖
pip install pyqlib pandas numpy matplotlib scikit-learn

```

### 使用步骤

1. **初始化数据:** 运行 `00_unpack_data.py` 将二进制数据集部署到本地工作区。
2. **训练模型:** 运行 `02_train_model.py`。这将自动生成因子数据并训练 LightGBM 模型。
3. **执行回测:** 运行 `03_backtest_simulation.py` 模拟 2022 年的交易。
4. **可视化:** 运行 `04_analysis.py` 生成盈亏曲线图。

---

## 👨‍💻 作者 (Author)

**王宏然 (Rex)**

* **学校:** 山东大学 (Shandong University)
* **专业:** 国际经济与贸易 / 区块链 (辅修)
* **研究方向:** 量化金融, AI 因子挖掘, 密码学
* **联系方式:** [202200150202@mail.sdu.edu.cn](mailto:202200150202@mail.sdu.edu.cn)

---

*免责声明: 本项目仅供学术研究与求职展示使用。投资有风险，入市需谨慎。*

```

```
