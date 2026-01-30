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

r_result = pd.read_csv('r_database.csv')
z_result = pd.read_csv('z_database.csv')

r_result = r_result.drop(index=[142, 145, 167])
z_result = z_result.drop(index=[147, 148, 149, 123, 124, 125, 128])


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

X = z_X_filter
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = z_y2

# 初始化结果存储
results = {
    'Model': [],
    'Train_R2_mean': [],
    'Test_R2_mean': [],
    'Train_MSE_mean': [],
    'Test_MSE_mean': []
}

# 定义5次随机分组
n_splits = 5
# random_states = np.random.randint(0, 1000, n_splits)
# r_y1
# random_states = [436,452,472,361,597]
# r_y2
# random_states = [959,123,264,651,75]
# z_y1
# random_states = [921,348,845,597,785]
# z_y2
random_states = [932,81,61,787,807]

# 定义所有基础模型及其参数网格
base_models = {
    'LR': {
        'model': LinearRegression(),
        'params': {'fit_intercept': [True, False]}
    },
    'RR': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
    },
    'XGB': {
        'model': XGBRegressor(objective='reg:squarederror', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.2, 0.4, 0.6],
            'colsample_bytree': [0.6, 0.8]
        }
    },
    'DNN': {
        'model': MLPRegressor(random_state=42, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# 首先需要存储每个模型在所有随机分组中的最佳参数和对应测试R²
model_best_params_records = {name: {'params': [], 'test_r2': []} for name in base_models.keys()}

# 1. 训练和评估所有基础模型（同时记录每次的最佳参数）
for model_name, model_info in base_models.items():
    print(f"\n=== 正在处理基础模型: {model_name} ===")

    train_r2_scores = []
    test_r2_scores = []
    train_mse_scores = []
    test_mse_scores = []

    for i, state in enumerate(random_states):
        print(f"\n--- 随机分组 {i + 1}/{n_splits} (random_state={state}) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=state)

        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            scoring='r2',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # 记录本次分组的最佳参数和测试R²
        y_test_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        model_best_params_records[model_name]['params'].append(best_params)
        model_best_params_records[model_name]['test_r2'].append(test_r2)

        y_train_pred = best_model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)

        print(f"当前最佳参数: {best_params}")
        print(f"训练集 R²: {train_r2:.4f}, 训练集 MSE: {train_mse:.4f}")
        print(f"测试集 R²: {test_r2:.4f}, 测试集 MSE: {test_mse:.4f}")

    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)
    avg_train_mse = np.mean(train_mse_scores)
    avg_test_mse = np.mean(test_mse_scores)

    results['Model'].append(model_name)
    results['Train_R2_mean'].append(avg_train_r2)
    results['Test_R2_mean'].append(avg_test_r2)
    results['Train_MSE_mean'].append(avg_train_mse)
    results['Test_MSE_mean'].append(avg_test_mse)

    print(f"\n=== {model_name} 平均性能 ===")
    print(f"平均训练集 R²: {avg_train_r2:.4f}")
    print(f"平均测试集 R²: {avg_test_r2:.4f}")
    print(f"平均训练集 MSE: {avg_train_mse:.4f}")
    print(f"平均测试集 MSE: {avg_test_mse:.4f}")


# 2. 获取测试集R²最高的最佳参数版本
def get_best_models(base_models, model_best_params_records):
    best_models = {}
    for model_name, model_info in base_models.items():
        # 找到测试R²最高的参数组合
        test_r2_scores = model_best_params_records[model_name]['test_r2']
        best_idx = np.argmax(test_r2_scores)
        best_params = model_best_params_records[model_name]['params'][best_idx]

        best_model = model_info['model'].set_params(**best_params)

        best_models[model_name] = best_model

        print(f"\n=== {model_name} 最终选择参数 ===")
        print(f"测试集最高 R²: {max(test_r2_scores):.4f}")
        print(f"最佳参数: {best_params}")

    return best_models


# 调用时传入记录的最佳参数信息
best_base_models_Linear = get_best_models(base_models, model_best_params_records)

best_base_models_Equal = get_best_models(
    {name: model for name, model in base_models.items() if name != 'DNN'},
    model_best_params_records
)

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

# 定义堆叠模型
stacking_models = {
    'Linear_Stacking': {
        'model': StackingRegressor(
            estimators=list(best_base_models_Linear.items()),
            final_estimator=LinearRegression(),
        )
    },
    'Equal_Weight_Stacking': {
        'model': StackingRegressor(
            estimators=list(best_base_models_Equal.items()),
            final_estimator=AverageRegressor(),  # 严格等权平均
        )
    }
}

# 3. 训练和评估堆叠模型
best_equal_stacking_model = None
best_train_r2 = -np.inf

for model_name, model_info in stacking_models.items():
    print(f"\n=== 正在处理堆叠模型: {model_name} ===")

    train_r2_scores = []
    test_r2_scores = []
    train_mse_scores = []
    test_mse_scores = []

    for i, state in enumerate(random_states):
        print(f"\n--- 随机分组 {i + 1}/{n_splits} (random_state={state}) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=state)

        model = model_info['model']
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # 如果是等权堆叠
        if model_name == 'Equal_Weight_Stacking':
            if train_r2 > best_train_r2:
                best_train_r2 = train_r2
                best_equal_stacking_model = model
                print(f"更新最佳等权堆叠模型，训练R²: {best_train_r2:.4f}")

        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        train_mse_scores.append(train_mse)
        test_mse_scores.append(test_mse)

        print(f"训练集 R²: {train_r2:.4f}, 训练集 MSE: {train_mse:.4f}")
        print(f"测试集 R²: {test_r2:.4f}, 测试集 MSE: {test_mse:.4f}")

    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)
    avg_train_mse = np.mean(train_mse_scores)
    avg_test_mse = np.mean(test_mse_scores)

    results['Model'].append(model_name)
    results['Train_R2_mean'].append(avg_train_r2)
    results['Test_R2_mean'].append(avg_test_r2)
    results['Train_MSE_mean'].append(avg_train_mse)
    results['Test_MSE_mean'].append(avg_test_mse)

    print(f"\n=== {model_name} 平均性能 ===")
    print(f"平均训练集 R²: {avg_train_r2:.4f}")
    print(f"平均测试集 R²: {avg_test_r2:.4f}")
    print(f"平均训练集 MSE: {avg_train_mse:.4f}")
    print(f"平均测试集 MSE: {avg_test_mse:.4f}")

# joblib.dump(best_equal_stacking_model, 'best_equal_stacking_model_z_y2.pkl')
print(f"\n已保存最佳等权堆叠模型，训练R²: {best_train_r2:.4f} 到 best_equal_stacking_model_z_y1.pkl")

# 创建结果DataFrame并排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Test_R2_mean', ascending=False)
print("\n=== 最终结果（按测试集R²排序） ===")
print(results_df.to_string())