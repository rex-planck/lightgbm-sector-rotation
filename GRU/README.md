<div align="center">
  <a href="./README_zh.md">简体中文</a> | 
  <a href="./README.md">English</a>
</div>

<br />


# AI-Driven Alpha Research Platform based on MSRA Qlib 🚀

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Qlib](https://img.shields.io/badge/Quant-MSRA%20Qlib-orange.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![Status](https://img.shields.io/badge/Status-Backtest%20Passed-success.svg)

## 📖 Introduction

This project is a **full-cycle quantitative research framework** developed based on Microsoft's [Qlib](https://github.com/microsoft/qlib). It implements an end-to-end workflow from **high-performance data storage** (Binary) to **Alpha factor mining**, **Machine Learning modeling** (GBDT), and **transaction-cost-aware backtesting**.

The system targets the **CSI 300 (A-Share)** market. By utilizing the **Alpha158** factor library and **LightGBM** model, it successfully captured significant Alpha during the 2022 bear market, demonstrating robust stock selection capabilities.

---

## 📊 Key Performance (2022 Out-of-Sample)

> **"Modeling is not just about prediction, but about PnL under strict cost constraints."**

The model was trained on data from **2015-2020**, validated in **2021**, and tested in **2022** (a major bear market in China).

### 1. Model Prediction Power
- **Rank IC (Information Coefficient):** `0.0536` (Pearson: `0.0540`)
- *Interpretation: A Rank IC > 0.05 indicates strong predictive power in daily frequency trading.*

### 2. Backtest Result (Top-K Dropout Strategy)
- **Benchmark:** CSI 300 Index (SH000300)
- **Cumulative Return (Strategy):** `~ -13.9%` (Resilient in crash)
- **Cumulative Return (Benchmark):** `~ -21.6%`
- **Excess Return (Alpha):** `+8.12%` (Annualized, after cost)
- **Information Ratio (IR):** `1.04`
- **Trading Cost:** Commission `0.05%` + Tax `0.1%` included.

![Backtest Curve](backtest_final_success.png)
*(Figure: Cumulative Net Value Curve in 2022. The Red area represents the accumulated Alpha)*

---

## 🛠️ Project Architecture

The repository is structured to reflect a professional quantitative workflow:

```text
Python-Quant-Practice
├── python_program/
│   ├── 00_unpack_data.py          # Data Engineering: Deploy binary data engine
│   ├── 01_test_setup.py           # Environment: Verify Qlib initialization
│   ├── 02_train_model.py          # Modeling: Alpha158 extraction & LightGBM training
│   ├── 03_backtest_simulation.py  # Backtest: Top-K Strategy with cost simulation
│   ├── 04_analysis.py             # Analysis: Visualization & Attribution
│   ├── mlruns/                    # Logs: Experiment records (managed by MLflow)
│   └── backtest_final_success.png # Result Visualization
├── .gitignore
└── README.md

```

---

## 🔬 Methodology

### 1. Data Engineering (High-Performance)

* **Engine:** Qlib Binary Storage (`.bin` files).
* **Features:** OHLCV + VWAP (Day Frequency).
* **Universe:** CSI 300 Constituents.
* **Normalization:** `RobustZScoreNorm` (removes outliers) + `Fillna`.

### 2. Alpha Research (Factor Mining)

* **Factor Library:** **Alpha158** (158 factors including Momentum, Volatility, and Volume-Price correlations).
* **Labeling:** 5-day future excess return (`Ref($close, -5) / $close - 1`).
* **Processing:** `CSRankNorm` (Cross-Sectional Rank Normalization) to make labels market-neutral.

### 3. Machine Learning (The Engine)

* **Model:** **LightGBM** (Gradient Boosting Decision Tree).
* **Hyperparameters:**
* `learning_rate`: 0.0421
* `num_leaves`: 210
* `max_depth`: 8
* `early_stopping`: Enabled to prevent overfitting.



### 4. Strategy & Execution

* **Strategy:** `TopkDropoutStrategy`.
* **Logic:** Hold Top 50 stocks with highest scores.
* **Turnover Control:** Only replace the bottom 5 stocks daily (`n_drop=5`) to minimize transaction costs.
* **Risk Control:** Filter out suspended stocks and limit-up/down stocks.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* Microsoft Qlib
* Pandas, NumPy, Matplotlib

### Installation

```bash
# Clone the repository
git clone [https://github.com/rex-planck/Python-Quant-Practice.git](https://github.com/rex-planck/Python-Quant-Practice.git)

# Install dependencies
pip install pyqlib pandas numpy matplotlib scikit-learn

```

### Usage

1. **Initialize Data:** Run `00_unpack_data.py` to deploy the binary dataset to your local workspace.
2. **Train Model:** Run `02_train_model.py`. This will generate the factors and train the LightGBM model.
3. **Run Backtest:** Run `03_backtest_simulation.py` to simulate trading in 2022.
4. **Visualize:** Run `04_analysis.py` to plot the PnL curve.

---

## 👨‍💻 Author

**Wang Hongran (Rex)**

* **University:** Shandong University 
* **Major:** International Economics & Trade / Blockchain (Minor)
* **Focus:** Quantitative Finance, AI Factors, Cryptography
* **Contact:** [202200150202@mail.sdu.edu.cn](mailto:202200150202@mail.sdu.edu.cn)

---

*Disclaimer: This project is for research and recruitment demonstration purposes only. Investment involves risks.*

```

```
