import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import graphviz # 用于画图，可选

# 设置随机种子，保证结果可复现
np.random.seed(42)

print("🧪 启动遗传规划因子挖掘系统 (Alpha Mining)...")

# 1. 模拟量化数据 (Mock Data)
# 在实战中，这里替换为你从 Qlib 导出的 DataFrame
print("1. 生成模拟量化数据 (Open, High, Low, Close, Volume)...")
n_samples = 5000
X = pd.DataFrame()
# 构造一些基本的量价数据
X['open'] = np.random.random(n_samples) * 100
X['close'] = X['open'] * (1 + np.random.normal(0, 0.02, n_samples))
X['high'] = np.maximum(X['open'], X['close']) * (1 + np.random.random(n_samples) * 0.01)
X['low'] = np.minimum(X['open'], X['close']) * (1 - np.random.random(n_samples) * 0.01)
X['volume'] = np.random.random(n_samples) * 1000000

# 构造一个真实的 Alpha 逻辑（作为上帝视角的“答案”）
# 假设真实规律是：(收盘价 - 开盘价) * 成交量
# 我们的目标是看 gplearn 能不能反向把这个公式“猜”出来
true_alpha = (X['close'] - X['open']) * np.log(X['volume'])
y = true_alpha + np.random.normal(0, 0.1, n_samples) # 加上一些市场噪音

# 2. 定义函数集 (Function Set)
# 这是构建因子的积木：加减乘除、三角函数、逻辑判断等
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

# 3. 配置遗传规划模型 (Symbolic Regressor)
print("2. 配置遗传进化引擎...")
est_gp = SymbolicRegressor(
    population_size=1000,   # 种群大小：每一代有多少个公式
    generations=10,         # 进化代数：繁衍多少代
    tournament_size=20,     # 锦标赛大小：每次选多少个出来打架
    stopping_criteria=0.01, # 停止条件
    p_crossover=0.7,        # 杂交概率：两个公式互换基因
    p_subtree_mutation=0.1, # 突变概率：公式发生变异
    p_hoist_mutation=0.05,  # 提升突变
    p_point_mutation=0.1,   # 点突变
    max_samples=0.9,        # 每次训练用的数据比例
    verbose=1,              # 打印进度
    parsimony_coefficient=0.001, # 惩罚系数：惩罚过于复杂的公式（防过拟合）
    random_state=0,
    function_set=function_set
)

# 4. 开始挖掘 (Training)
print("3. 开始挖掘 Alpha 因子 (Evolution Start)...")
print("   目标: 从噪音中找回规律 y = (close - open) * log(volume)")
est_gp.fit(X, y)

# 5. 分析结果
print("\n" + "="*50)
print("🏆 挖掘结果 (Top Factor Found):")
print("="*50)
best_program = est_gp._program
print(f"最强公式: {best_program}")
print(f"公式深度: {best_program.depth_}")
print(f"适应度 (Fitness): {best_program.raw_fitness_:.4f}")

# 看看它主要用了哪些变量
print("\n📊 变量重要性 (Feature Importance):")
# gplearn 没有直接的 feature_importance，我们通过肉眼观察公式
# 或者计算 Rank IC

# 6. 验证因子效果 (Rank IC)
# 使用模型生成的公式预测出的值
y_pred = est_gp.predict(X)
# 计算 Rank IC
df_res = pd.DataFrame({'pred': y_pred, 'label': y})
rank_ic = df_res.rank().corr().iloc[0, 1]

print("-" * 50)
print(f"📈 因子评价:")
print(f"   Rank IC: {rank_ic:.4%}")
print("-" * 50)

if rank_ic > 0.8: # 因为是模拟数据，应该很高
    print("✅ 成功！AI 成功还原了市场逻辑！")
else:
    print("🤔 还需要继续进化。")