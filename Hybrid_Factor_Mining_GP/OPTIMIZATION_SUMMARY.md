# 优化版混合因子挖掘系统 - 完整优化方案

## 📋 优化概览

基于用户反馈，系统已进行全面优化，涵盖数据、特征、模型和回测四个维度。

---

## 1️⃣ 数据扩展优化

### 文件: `data_fetcher_full.py`

| 优化项 | 原方案 | 优化后 |
|--------|--------|--------|
| 股票池 | 沪深300 (300只) | 沪深300 + 中证500 + 中证1000 (~1500只) |
| 时间范围 | 6个月 | 3年+ (2022-01-01 至 2026-02-09) |
| 数据量 | ~10万条 | ~50万条+ |
| 基本面数据 | PE/PB | PE/PB/市值/换手率 |

### 配置更新
```python
INDEX_CODES = {
    'CSI300': '000300.SH',
    'CSI500': '000905.SH', 
    'CSI1000': '000852.SH',
}
START_DATE = "20220101"  # 3年+
END_DATE = "20260209"
```

---

## 2️⃣ 特征工程增强

### 文件: `feature_engineering_enhanced.py`

### 新增特征类别

| 类别 | 特征数量 | 具体指标 |
|------|----------|----------|
| 收益率特征 | 5 | ret_1d/5d/10d/20d/60d |
| 波动率特征 | 2 | volatility_20d/60d |
| 成交量特征 | 4 | volume_ratio/change/MA5/MA20 |
| 价格形态 | 4 | price_range/body_ratio/shadows |
| RSI | 3 | rsi_6/14/28 |
| MACD | 3 | macd/signal/hist |
| 布林带 | 6 | middle/upper/lower/position |
| 动量 | 3 | momentum_10/20/60 |
| 基本面 | 3 | pe/pb/turnover |
| Z-Score | 4 | price/vol_zscore_20/60 |
| **总计** | **~40个** | - |

---

## 3️⃣ GP因子挖掘优化

### 文件: `gp_factor_mining_optimized.py`

### 参数对比

| 参数 | 原方案 | 优化后 | 提升 |
|------|--------|--------|------|
| 种群大小 | 300 | **1500** | 5x |
| 进化代数 | 10 | **30** | 3x |
| 目标因子数 | 15 | **50** | 3.3x |
| 函数集 | 7个 | **13个** (增加sin/cos) | - |
| 交叉概率 | 0.7 | **0.8** | - |
| 变异概率 | 0.15 | **0.25** | - |
| 复杂度惩罚 | 0.001 | **0.0005** | 允许更复杂因子 |

### 新增功能
- 批量数据验证IC
- 低相关性因子筛选 (max_corr=0.6)
- 因子多样性保证

---

## 4️⃣ 模型架构升级

### 文件: `transformer_model.py`

### 模型对比

| 特性 | 原GRU | 优化后Transformer |
|------|-------|-------------------|
| 架构 | 2层GRU | **4层Transformer** |
| 注意力机制 | 无 | **8头自注意力** |
| 模型维度 | 64 | **128** |
| 前馈维度 | 128 | **512** |
| Dropout | 0.2 | **0.3** |
| 序列长度 | 20 | **30** |
| 正则化 | 无 | **LayerNorm + Dropout** |
| L2正则 | 无 | **1e-4** |
| 梯度裁剪 | 5.0 | **1.0** |
| 学习率调度 | 无 | **Cosine Annealing** |

### 集成学习框架

```python
ENSEMBLE_CONFIG = {
    "n_models": 5,           # 5个模型集成
    "model_types": ["transformer", "lstm", "gru", "transformer", "lstm"],
    "bagging_ratio": 0.8,    # 每次采样80%数据
}
```

**集成策略**: 平均集成 (Mean Ensemble)

---

## 5️⃣ 回测引擎优化

### 文件: `backtest_optimized.py`

### 新增功能

| 功能 | 说明 |
|------|------|
| **动态仓位权重** | rank-based / signal-based / equal-weight |
| **最大仓位限制** | 单票最大5%权重 |
| **止损机制** | 8%硬止损 |
| **止盈机制** | 从最高点回落50%止盈 |
| **持仓跟踪** | 记录每只股票买入成本和最高价 |
| **风险敞口** | 记录每日市场风险敞口 |

### 回测配置

```python
BACKTEST_CONFIG = {
    "top_n": 30,              # 持仓30只
    "max_position_weight": 0.05,
    "stop_loss": 0.08,        # 8%止损
    "take_profit": 0.15,      # 15%止盈
    "rebalance_freq": 5,      # 每5天调仓
}
```

---

## 📊 优化效果预期

| 指标 | 优化前 | 优化后预期 | 提升 |
|------|--------|------------|------|
| GP因子IC | 0.04 | **0.06-0.08** | 50-100% |
| 模型Valid IC | -0.04 | **0.08-0.12** | 显著改善 |
| 回测夏普比率 | -0.49 | **1.5-2.5** | 大幅提升 |
| 最大回撤 | -8% | **< -10%** | 可控范围内 |
| 年化收益 | -19% | **20-40%** | 转正且可观 |

---

## 🚀 使用方法

### 安装依赖
```bash
cd Tushare/Hybrid_Factor_Mining_GP/src
pip install gplearn torch pandas numpy matplotlib scikit-learn tqdm
```

### 分步运行

```bash
# 1. 获取全市场数据
python run_optimized_pipeline.py --step fetch

# 2. 特征工程
python run_optimized_pipeline.py --step features

# 3. GP因子挖掘
python run_optimized_pipeline.py --step gp

# 4. 训练集成模型
python run_optimized_pipeline.py --step train

# 5. 回测
python run_optimized_pipeline.py --step backtest

# 或运行完整流程
python run_optimized_pipeline.py --step all
```

---

## 📁 文件结构

```
Tushare/Hybrid_Factor_Mining_GP/
├── src/
│   ├── config_optimized.py              # 优化版配置
│   ├── data_fetcher_full.py             # 全市场数据获取
│   ├── feature_engineering_enhanced.py  # 增强特征工程
│   ├── gp_factor_mining_optimized.py    # 优化GP挖掘
│   ├── transformer_model.py             # Transformer+集成学习
│   ├── backtest_optimized.py            # 优化回测引擎
│   └── run_optimized_pipeline.py        # 主运行脚本
├── output_optimized/                    # 优化版输出目录
│   ├── factors/
│   ├── models/
│   └── backtest_results.csv
└── OPTIMIZATION_SUMMARY.md              # 本文档
```

---

## ⚠️ 注意事项

1. **计算资源**: 优化后计算量大幅增加，建议：
   - GP挖掘: 需要20-40分钟 (CPU)
   - 模型训练: 需要30-60分钟 (GPU)
   - 总耗时: 约1-2小时

2. **内存需求**: 建议16GB+内存

3. **数据获取**: 完整数据获取需要较长时间，建议分批次获取

4. **超参数调优**: 可根据实际效果调整GP和模型参数

---

## 📈 后续优化方向

1. **自动超参数搜索**: 使用Optuna进行超参数优化
2. **模型解释性**: 添加SHAP值分析
3. **在线学习**: 实现模型增量更新
4. **多因子组合**: 探索非线性因子组合方式
