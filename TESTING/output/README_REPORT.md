# 2016-2026 长周期参数优化与回测报告

生成时间：2026-02-18

## 1) 本轮优化范围

- 数据下载期：2016-01-01 ~ 2026-02-18（用于特征与训练）
- 回测执行期：2020-01-01 ~ 2026-02-18（在回测引擎中强制截断）
- 优化方式：仅使用 `output/predictions.pkl` 做参数网格，不重复训练模型

### 第一轮粗网格

- 参数：`top_k` × `prob_threshold`
- 产物：`output/grid_full.csv`

### 第二轮聚焦网格

- 参数：`top_k in [9,10,11]`、`threshold in [0.55..0.75]`、`trailing_stop in [0.00,0.03,0.05,0.08,0.10]`
- 产物：`output/refine_grid.csv`

## 2) 最终选定参数（已回填 config.py）

- `top_k = 10`
- `prob_threshold = 0.58`
- `trailing_stop_pct = 0.10`
- `stop_loss_pct = 0.05`（保持不变）

## 3) 最终回测结果（2020-01-01 ~ 2026-02-18）

来自 `output/final_backtest_summary.json`：

- 初始资金：10,000,000
- 期末资金：12,867,563.67
- 总收益：+28.68%
- 年化收益：+4.37%
- Sharpe：0.212
- 最大回撤：41.44%

## 4) 图表与文件

- 综合图：`output/phase4_report.png`
- 净值曲线：`output/equity_curve.png`
- 回撤图：`output/drawdown.png`
- 月收益热力图：`output/monthly_heatmap.png`
- 回测摘要：`output/final_backtest_summary.json`

## 5) 结论

在当前信号质量与交易规则下，本轮自迭代后年化提升到 **4.37%**，但仍显著低于“>50%（熊市除外）”目标。主要瓶颈在于：

1. 预测概率分布对阈值不敏感区间较大；
2. 高于 10 持仓时出现明显风险放大与资金曲线失稳；
3. 交易日覆盖较高但选股 alpha 强度不足。

建议下一轮优化转向“信号端升级”而非继续微调执行层参数（例如：标签定义、特征集、行业/市值中性约束、交易成本建模与仓位上限机制）。
