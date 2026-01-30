# 基础数据处理和科学计算
import numpy as np
import pandas as pd

# 数据可视化
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# 机器学习和建模
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import (StandardScaler, KBinsDiscretizer,
                                  FunctionTransformer)
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import (r2_score, mean_squared_error,
                            mean_absolute_error)
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

# 模型持久化
import joblib

# 不平衡数据处理（已注释，按需启用）
# from imblearn.over_sampling import RandomOverSampler

r_result = pd.read_csv('r_database.csv')
z_result = pd.read_csv('z_database.csv')

r_result = r_result.drop(index=[142, 145, 167])
z_result = z_result.drop(index=[147, 148, 149, 123, 124, 125, 128])

r_result =r_result.reset_index(drop=True)
z_result =z_result.reset_index(drop=True)


r_X = r_result.drop(columns=['inter_r', 'SampleName', 'RAFT', 'Gβ/_add', 'Gp/re'])
r_X_filter = r_X.drop(columns=['E4', 'V1', 'Q3', 'P1', 'E2', 'P4', 'P2', 'V2', 'E11', 'Q11', 'E7', 'V5', 'E8', 'Q15',
                               'Q7', 'V6', 'B10', 'V7', 'Q13', 'V8', 'E10'])
r_y1 = r_result['Gβ/_add']
r_y2 = r_result['Gp/re']
z_X = z_result.drop(columns=['inter_z', 'SampleName', 'RAFT', 'Gadd/p', 'Gadd/_add'])
z_X_filter = z_X.drop(columns=['E2', 'E4', 'P1', 'P4', 'V2', 'V1', 'P2', 'Q3', 'E8', 'Q15', 'E10', 'B10', 'B9', 'Q4',
                               'V8', 'Q13', 'V5', 'V6', 'Q10', 'Q14', 'E11', 'Q9', 'Q12', 'B13'])
z_y1 = z_result['Gadd/p']
z_y2 = z_result['Gadd/_add']

r_X_y1_selected = r_X_filter[['E3', 'B1', 'PEOE_VSA12_RAFT', 'B7', 'MinAbsEStateIndex', 'VSA_EState7_RAFT', 'E6', 'PEOE_VSA10', 'BalabanJ', 'VSA_EState5', 'E1', 'PEOE_VSA9_RAFT', 'B5', 'B12', 'E5', 'VSA_EState3_RAFT', 'Q5', 'Q12', 'VSA_EState9_RAFT', 'Q8', 'V3', 'Kappa1', 'Q2', 'Q4', 'P5', 'SMR_VSA6', 'BCUT2D_MWLOW', 'B11', 'P9']
]
r_X_y2_selected = r_X_filter[['B12', 'Kappa3_RAFT', 'E5', 'B3', 'MaxAbsEStateIndex', 'Q12', 'EState_VSA3_RAFT', 'B5', 'qed_RAFT', 'PEOE_VSA7_RAFT', 'FpDensityMorgan2', 'PEOE_VSA10_RAFT', 'MaxPartialCharge_RAFT', 'P6', 'AvgIpc_RAFT', 'PEOE_VSA4_RAFT', 'BCUT2D_MWLOW_RAFT', 'E1', 'B8', 'EState_VSA9', 'Q10', 'AvgIpc', 'E6', 'PEOE_VSA8_RAFT', 'E9', 'FractionCSP3', 'B1', 'B2']
]

z_X_y1_selected = z_X_filter[['PEOE_VSA2_RAFT', 'E3', 'EState_VSA9', 'Kappa3_RAFT', 'B7', 'E9', 'E1', 'BCUT2D_CHGHI', 'EState_VSA4_RAFT', 'B4', 'EState_VSA3_RAFT', 'BCUT2D_LOGPLOW', 'MinPartialCharge_RAFT', 'Chi4v_RAFT', 'Q5', 'fr_imidazole_RAFT', 'E7', 'B1', 'E6', 'MaxAbsEStateIndex_RAFT', 'PEOE_VSA8', 'VSA_EState5_RAFT', 'MinAbsEStateIndex', 'MinAbsEStateIndex_RAFT', 'SMR_VSA6_RAFT', 'MinAbsPartialCharge_RAFT', 'P6', 'E5']
]

z_X_y2_selected = z_X_filter[['B6', 'EState_VSA4_RAFT', 'Chi2n_RAFT', 'SlogP_VSA4_RAFT', 'SMR_VSA6_RAFT', 'PEOE_VSA10', 'PEOE_VSA10_RAFT', 'PEOE_VSA7', 'E3', 'B1', 'E9', 'VSA_EState1_RAFT', 'fr_imidazole_RAFT', 'E7', 'EState_VSA8_RAFT', 'SMR_VSA2_RAFT', 'Q2', 'P6', 'E6', 'E1', 'MaxAbsEStateIndex', 'Chi4v_RAFT', 'Q5', 'V3', 'FpDensityMorgan2', 'BCUT2D_MWHI_RAFT', 'B5', 'MinAbsEStateIndex_RAFT', 'BCUT2D_LOGPLOW']
]

X = r_X_y1_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class AverageRegressor(BaseEstimator, RegressorMixin):
    """严格等权回归器，直接对输入取平均"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # 无需训练，直接返回 self
        return self

    def predict(self, X):
        # 对基模型的预测结果取平均
        return np.mean(X, axis=1)

# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_r_y1.pkl')

# 预测结果
predictions = stacking_model.predict(X_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
predictions_df = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)

r2 = r2_score(r_y1, predictions)
mse = mean_squared_error(r_y1, predictions)
mae = mean_absolute_error(r_y1, predictions)
print(f'r2:{r2},mse:{mse},mae:{mae}')

plt.figure(figsize=(6, 5))

# 创建颜色映射条件
colors = []
for x_val, y_val in zip(r_y1, predictions_df['Predictions']):
    if y_val <= 0 and x_val <= 0:
        colors.append('#008600')  # y<=0且x<=0的点为绿色
    elif y_val <= 0 and x_val > 0:
        colors.append('#7A37AC')   # y<=0且x>0的点为蓝色
    elif y_val > 0 and x_val <= 0:
        colors.append('#7A37AC')    # y>0且x<=0的点为红色
    else:
        colors.append('#D9D9D9')  # 其他情况保持原色

# 绘制散点图
ax = sns.scatterplot(x=r_y1, y=predictions_df['Predictions'],
                    color=colors, alpha=0.7, s=90, edgecolor='black', linewidth=0.5)

# 添加标准预测线（y = x）
plt.plot([r_y1.min(), r_y1.max()], [r_y1.min(), r_y1.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = r_y1 + 4  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = r_y1 - 4  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(r_y1, mae_line_y_p, color='#7F7F7F', linestyle='-', label='MAE = 2.0 line', linewidth=2)  # 橙色
plt.plot(r_y1, mae_line_y_n, color='#7F7F7F', linestyle='-', linewidth=2)  # 橙色

# 添加参考线
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)  # y=0的灰色虚线

x_major_locator = MultipleLocator(4)
y_major_locator = MultipleLocator(4)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# 调整坐标轴字体大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度字体
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5, color='#888888')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

X = r_X_y2_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_r_y2.pkl')

# 预测结果
predictions = stacking_model.predict(X_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
predictions_df = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
r2 = r2_score(r_y2, predictions)
mse = mean_squared_error(r_y2, predictions)
mae = mean_absolute_error(r_y2, predictions)
print(f'r2:{r2},mse:{mse},mae:{mae}')

plt.figure(figsize=(6, 5))

Y_scaled = pd.DataFrame(r_y2).squeeze()

# 创建颜色映射条件
colors = []
for x_val, y_val in zip(r_y2, predictions_df['Predictions']):
    if y_val > -9 and x_val > -9:
        colors.append('#008600')  # y<=0且x<=0的点为绿色
    elif y_val <= -9 and x_val > -9:
        colors.append('#7A37AC')   # y<=0且x>0的点为蓝色
    elif y_val > -9 and x_val <= -9:
        colors.append('#7A37AC')    # y>0且x<=0的点为红色
    else:
        colors.append('#D9D9D9')  # 其他情况保持原色

# 绘制散点图
ax = sns.scatterplot(x=r_y2, y=predictions_df['Predictions'],
                    color=colors, alpha=0.7, s=90, edgecolor='black', linewidth=0.5)

# 添加标准预测线（y = x）
plt.plot([r_y2.min(), r_y2.max()], [r_y2.min(), r_y2.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = r_y2 + 2  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = r_y2 - 2  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(r_y2, mae_line_y_p, color='#7F7F7F', linestyle='-', label='MAE = 2.0 line', linewidth=2)  # 橙色
plt.plot(r_y2, mae_line_y_n, color='#7F7F7F', linestyle='-', linewidth=2)  # 橙色

# 添加参考线
ax.axhline(y=-9, color='black', linestyle='--', linewidth=1)  # y=0的灰色虚线

x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# 调整坐标轴字体大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度字体
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5, color='#888888')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

X = z_X_y1_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_z_y1.pkl')

# 预测结果
predictions = stacking_model.predict(X_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
predictions_df = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
r2 = r2_score(z_y1, predictions)
mse = mean_squared_error(z_y1, predictions)
mae = mean_absolute_error(z_y1, predictions)
print(f'r2:{r2},mse:{mse},mae:{mae}')

plt.figure(figsize=(6, 5))

Y_scaled = pd.DataFrame(z_y1).squeeze()

# 创建颜色映射条件
colors = []
for x_val, y_val in zip(z_y1, predictions_df['Predictions']):
    if y_val <= 0 and x_val <= 0:
        colors.append('#008600')  # y<=0且x<=0的点为绿色
    elif y_val <= 0 and x_val > 0:
        colors.append('#7A37AC')   # y<=0且x>0的点为蓝色
    elif y_val > 0 and x_val <= 0:
        colors.append('#7A37AC')    # y>0且x<=0的点为红色
    else:
        colors.append('#D9D9D9')  # 其他情况保持原色

# 绘制散点图
ax = sns.scatterplot(x=z_y1, y=predictions_df['Predictions'],
                    color=colors, alpha=0.7, s=90, edgecolor='black', linewidth=0.5)

# 添加标准预测线（y = x）
plt.plot([z_y1.min(), z_y1.max()], [z_y1.min(), z_y1.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = z_y1 + 2  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = z_y1 - 2  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(z_y1, mae_line_y_p, color='#7F7F7F', linestyle='-', label='MAE = 2.0 line', linewidth=2)  # 橙色
plt.plot(z_y1, mae_line_y_n, color='#7F7F7F', linestyle='-', linewidth=2)  # 橙色

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)  # y=0的灰色虚线

x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# 调整坐标轴字体大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度字体
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5, color='#888888')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

X = z_X_y2_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_z_y2.pkl')

# 预测结果
predictions = stacking_model.predict(X_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
predictions_df = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)

r2 = r2_score(z_y2, predictions)
mse = mean_squared_error(z_y2, predictions)
mae = mean_absolute_error(z_y2, predictions)
print(f'r2:{r2},mse:{mse},mae:{mae}')

plt.figure(figsize=(6, 5))

# 创建颜色映射条件
colors = []
for x_val, y_val in zip(z_y2, predictions_df['Predictions']):
    if y_val > -2 and x_val > -2:
        colors.append('#008600')  # y<=0且x<=0的点为绿色
    elif y_val <= -2 and x_val > -2:
        colors.append('#7A37AC')   # y<=0且x>0的点为蓝色
    elif y_val > -2 and x_val <= -2:
        colors.append('#7A37AC')    # y>0且x<=0的点为红色
    else:
        colors.append('#D9D9D9')  # 其他情况保持原色

# 绘制散点图
ax = sns.scatterplot(x=z_y2, y=predictions_df['Predictions'],
                    color=colors, alpha=0.7, s=90, edgecolor='black', linewidth=0.5)

# 添加标准预测线（y = x）
plt.plot([z_y2.min(), z_y2.max()], [z_y2.min(), z_y2.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = z_y2 + 4  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = z_y2 - 4  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(z_y2, mae_line_y_p, color='#7F7F7F', linestyle='-', label='MAE = 2.0 line', linewidth=2)  # 橙色
plt.plot(z_y2, mae_line_y_n, color='#7F7F7F', linestyle='-', linewidth=2)  # 橙色

# 添加参考线
ax.axhline(y=-2, color='black', linestyle='--', linewidth=1)  # y=0的灰色虚线

x_major_locator = MultipleLocator(4)
y_major_locator = MultipleLocator(4)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# 调整坐标轴字体大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 增大刻度字体
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5, color='#888888')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()



# Set Calibri font globally
rcParams['font.family'] = 'Calibri'

# Create a new column for coloring points based on the condition
r_result['judgement'] = np.where(
    (r_result['Gβ/_add'] < 0) & (r_result['Gp/re'] > -9),
    'feasible',
    'no-feasible'
)

# 统计符合条件的数据个数
feasible_count = r_result['judgement'].value_counts().get('feasible', 0)
# 总数据个数
total_count = len(r_result)
# 计算百分比
feasible_percentage = (feasible_count / total_count) * 100


