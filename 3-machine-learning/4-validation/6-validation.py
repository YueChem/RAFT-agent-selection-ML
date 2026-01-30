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


# # 读取CSV文件
# result = pd.read_csv('validation_descriptor.csv')
#
# # 修正1: 过滤DataFrame的正确语法是 result[result['列名'].条件]
# # 修正2: 添加正则表达式参数regex=True
# z_result = result[result['RAFT'].str.contains('z25|z26', na=False, regex=True)]
# r_result = result[result['RAFT'].str.contains('r26', na=False, regex=True)]
#
# # 保存结果
# z_result.to_csv('validation_z.csv', index=False)  # 添加index=False避免保存索引列
# r_result.to_csv('validation_r.csv', index=False)

valid_r = pd.read_csv('validation_r.csv')
valid_z = pd.read_csv('validation_z.csv')

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

r_y1_filter = ['E3', 'B1', 'PEOE_VSA12_RAFT', 'B7', 'MinAbsEStateIndex', 'VSA_EState7_RAFT', 'E6', 'PEOE_VSA10', 'BalabanJ', 'VSA_EState5', 'E1', 'PEOE_VSA9_RAFT', 'B5', 'B12', 'E5', 'VSA_EState3_RAFT', 'Q5', 'Q12', 'VSA_EState9_RAFT', 'Q8', 'V3', 'Kappa1', 'Q2', 'Q4', 'P5', 'SMR_VSA6', 'BCUT2D_MWLOW', 'B11', 'P9']
r_y2_filter = ['B12', 'Kappa3_RAFT', 'E5', 'B3', 'MaxAbsEStateIndex', 'Q12', 'EState_VSA3_RAFT', 'B5', 'qed_RAFT', 'PEOE_VSA7_RAFT', 'FpDensityMorgan2', 'PEOE_VSA10_RAFT', 'MaxPartialCharge_RAFT', 'P6', 'AvgIpc_RAFT', 'PEOE_VSA4_RAFT', 'BCUT2D_MWLOW_RAFT', 'E1', 'B8', 'EState_VSA9', 'Q10', 'AvgIpc', 'E6', 'PEOE_VSA8_RAFT', 'E9', 'FractionCSP3', 'B1', 'B2']
z_y1_filter = ['PEOE_VSA2_RAFT', 'E3', 'EState_VSA9', 'Kappa3_RAFT', 'B7', 'E9', 'E1', 'BCUT2D_CHGHI', 'EState_VSA4_RAFT', 'B4', 'EState_VSA3_RAFT', 'BCUT2D_LOGPLOW', 'MinPartialCharge_RAFT', 'Chi4v_RAFT', 'Q5', 'fr_imidazole_RAFT', 'E7', 'B1', 'E6', 'MaxAbsEStateIndex_RAFT', 'PEOE_VSA8', 'VSA_EState5_RAFT', 'MinAbsEStateIndex', 'MinAbsEStateIndex_RAFT', 'SMR_VSA6_RAFT', 'MinAbsPartialCharge_RAFT', 'P6', 'E5']
z_y2_filter = ['B6', 'EState_VSA4_RAFT', 'Chi2n_RAFT', 'SlogP_VSA4_RAFT', 'SMR_VSA6_RAFT', 'PEOE_VSA10', 'PEOE_VSA10_RAFT', 'PEOE_VSA7', 'E3', 'B1', 'E9', 'VSA_EState1_RAFT', 'fr_imidazole_RAFT', 'E7', 'EState_VSA8_RAFT', 'SMR_VSA2_RAFT', 'Q2', 'P6', 'E6', 'E1', 'MaxAbsEStateIndex', 'Chi4v_RAFT', 'Q5', 'V3', 'FpDensityMorgan2', 'BCUT2D_MWHI_RAFT', 'B5', 'MinAbsEStateIndex_RAFT', 'BCUT2D_LOGPLOW']


r_X_y1_selected = r_X_filter[r_y1_filter]
r_X_y2_selected = r_X_filter[r_y2_filter]

z_X_y1_selected = z_X_filter[z_y1_filter]
z_X_y2_selected = z_X_filter[z_y2_filter]

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


X = r_X_y1_selected
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_valid = valid_r[r_y1_filter]
X_valid_scaled = scaler.transform(X_valid)
# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_r_y1.pkl')
# 预测结果
predictions = stacking_model.predict(X_valid_scaled)
valid_r['predictions_r_y1']  = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
print(valid_r[['SampleName','RAFT','predictions_r_y1']])


X = r_X_y2_selected
X_scaled = scaler.fit_transform(X)
X_valid = valid_r[r_y2_filter]
X_valid_scaled = scaler.transform(X_valid)
# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_r_y2.pkl')
# 预测结果
predictions = stacking_model.predict(X_valid_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
valid_r['predictions_r_y2']  = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
print(valid_r[['SampleName','RAFT','predictions_r_y2']])


X = z_X_y1_selected
X_scaled = scaler.fit_transform(X)
X_valid = valid_z[z_y1_filter]
X_valid_scaled = scaler.transform(X_valid)
# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_z_y1.pkl')
# 预测结果
predictions = stacking_model.predict(X_valid_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
valid_z['predictions_z_y1']  = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
print(valid_z[['SampleName','RAFT','predictions_z_y1']])

X = z_X_y2_selected
X_scaled = scaler.fit_transform(X)
X_valid = valid_z[z_y2_filter]
X_valid_scaled = scaler.transform(X_valid)
# 假设 'stacking_model' 是已经加载好的模型
stacking_model = joblib.load('best_equal_stacking_model_z_y2.pkl')
# 预测结果
predictions = stacking_model.predict(X_valid_scaled)
# 将 predictions 转换为 pandas DataFrame，并确保数据类型为 float
valid_z['predictions_z_y2'] = pd.DataFrame(predictions, columns=['Predictions'], dtype=float)
print(valid_z[['SampleName','RAFT','predictions_z_y2']])

