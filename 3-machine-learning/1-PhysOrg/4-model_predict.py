import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
# from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from sklearn.decomposition import PCA
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
# from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

r_result = pd.read_csv('r_database.csv')
z_result = pd.read_csv('z_database.csv')

r_result = r_result.drop(index=[142, 145, 167])
z_result = z_result.drop(index=[147, 148, 149, 123, 124, 125, 128])

r_result =r_result.reset_index(drop=True)
z_result =z_result.reset_index(drop=True)

r_X = r_result.drop(columns=['inter_r', 'SampleName', 'RAFT', 'Gβ/_add', 'Gp/re'])
r_X_filter = r_X[
    ['E1', 'E3', 'E5', 'B1', 'B2', 'B3', 'Q1', 'Q2', 'V3', 'P3', 'E6', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11',
     'B12', 'B13', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9', 'Q10', 'Q12', 'Q14', 'V4', 'V9', 'P5', 'P6', 'P7', 'P9']]
r_y1 = r_result['Gβ/_add']
r_y2 = r_result['Gp/re']
z_X = z_result.drop(columns=['inter_z', 'SampleName', 'RAFT', 'Gadd/p', 'Gadd/_add'])
z_X_filter = z_X[
    ['E1', 'E3', 'E5', 'B1', 'B2', 'B3', 'Q1', 'Q2', 'V3', 'P3', 'E6', 'E7', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11',
     'B12', 'Q5', 'Q6', 'Q7', 'Q8', 'Q11', 'V4', 'V7', 'V9', 'P5', 'P6', 'P7', 'P9']]
z_y1 = z_result['Gadd/p']
z_y2 = z_result['Gadd/_add']

X = r_X_filter
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

# 绘制散点图
ax = sns.scatterplot(x=r_y1, y=predictions_df['Predictions'],
                color='#D9D9D9', alpha=0.7, s=90, edgecolor='black', linewidth=0.5)  # 深蓝色

# 添加标准预测线（y = x）
plt.plot([r_y1.min(), r_y1.max()], [r_y1.min(), r_y1.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = r_y1 + 4  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = r_y1 - 4  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(r_y1, mae_line_y_p, color='#7F7F7F', linestyle=':', label='MAE = 2.0 line', linewidth=3)  # 橙色
plt.plot(r_y1, mae_line_y_n, color='#7F7F7F', linestyle=':', linewidth=3)  # 橙色

# 设置标题和标签
# plt.title(f"Stacking Model Predictions", fontsize=14, weight='bold', color='#333333')
# plt.xlabel("True Values", fontsize=12, color='#555555')
# plt.ylabel("Predictions", fontsize=12, color='#555555')

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

# 绘制散点图
ax = sns.scatterplot(x=r_y2, y=predictions_df['Predictions'],
                color='#D9D9D9', alpha=0.7, s=90, edgecolor='black', linewidth=0.5)  # 深蓝色

# 添加标准预测线（y = x）
plt.plot([r_y2.min(), r_y2.max()], [r_y2.min(), r_y2.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = r_y2 + 2  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = r_y2 - 2  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(r_y2, mae_line_y_p, color='#7F7F7F', linestyle=':', label='MAE = 2.0 line', linewidth=3)  # 橙色
plt.plot(r_y2, mae_line_y_n, color='#7F7F7F', linestyle=':', linewidth=3)  # 橙色

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

X = z_X_filter
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

# 绘制散点图
ax = sns.scatterplot(x=z_y1, y=predictions_df['Predictions'],
                color='#D9D9D9', alpha=0.7, s=90, edgecolor='black', linewidth=0.5)  # 深蓝色

# 添加标准预测线（y = x）
plt.plot([z_y1.min(), z_y1.max()], [z_y1.min(), z_y1.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = z_y1 + 2  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = z_y1 - 2  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(z_y1, mae_line_y_p, color='#7F7F7F', linestyle=':', label='MAE = 2.0 line', linewidth=3)  # 橙色
plt.plot(z_y1, mae_line_y_n, color='#7F7F7F', linestyle=':', linewidth=3)  # 橙色

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

# base_learners = [
#     ('Support Vector Regression', SVR(kernel='linear', C=10.0, gamma='scale')),
#     ('Linear Regression', LinearRegression()),
#     ('XGBoost', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, colsample_bytree=0.8, subsample=0.6)),
#     ('Ridge Regression', Ridge(alpha=0.1)),
#     ('neural_network', MLPRegressor(activation='tanh', alpha=0.001, hidden_layer_sizes=(50, 50), learning_rate_init=0.001))
# ]
# stacking_model = StackingRegressor(
#             estimators=base_learners,
#             final_estimator=AverageRegressor(),  # 严格等权平均
#         )
#
# stacking_models.fit(X_scaled, z_y2)

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

# 绘制散点图
ax = sns.scatterplot(x=z_y2, y=predictions_df['Predictions'],
                color='#D9D9D9', alpha=0.7, s=90, edgecolor='black', linewidth=0.5)  # 深蓝色

# 添加标准预测线（y = x）
plt.plot([z_y2.min(), z_y2.max()], [z_y2.min(), z_y2.max()], color='#000000', linestyle='-', label='Perfect Prediction (y = x)', linewidth=3)  # 红色

# 计算误差 MAE = 2.0 的线
mae_line_y_p = z_y2 + 4  # 真实值 + 2 生成误差为 2 的线
mae_line_y_n = z_y2 - 4  # 真实值 - 2 生成误差为 2 的线

# 绘制 MAE = 2.0 的线
plt.plot(z_y2, mae_line_y_p, color='#7F7F7F', linestyle=':', label='MAE = 2.0 line', linewidth=3)  # 橙色
plt.plot(z_y2, mae_line_y_n, color='#7F7F7F', linestyle=':', linewidth=3)  # 橙色


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