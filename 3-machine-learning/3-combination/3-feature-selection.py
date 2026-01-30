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
r_X_filter = r_X.drop(columns=['E4', 'V1', 'Q3', 'P1', 'E2', 'P4', 'P2', 'V2', 'E11', 'Q11', 'E7', 'V5', 'E8', 'Q15',
                               'Q7', 'V6', 'B10', 'V7', 'Q13', 'V8', 'E10'])
r_y1 = r_result['Gβ/_add']
r_y2 = r_result['Gp/re']
z_X = z_result.drop(columns=['inter_z', 'SampleName', 'RAFT', 'Gadd/p', 'Gadd/_add'])
z_X_filter = z_X.drop(columns=['E2', 'E4', 'P1', 'P4', 'V2', 'V1', 'P2', 'Q3', 'E8', 'Q15', 'E10', 'B10', 'B9', 'Q4',
                               'V8', 'Q13', 'V5', 'V6', 'Q10', 'Q14', 'E11', 'Q9', 'Q12', 'B13'])
z_y1 = z_result['Gadd/p']
z_y2 = z_result['Gadd/_add']

X = z_X_filter
# 在缩放前保存特征名
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 然后将特征名重新附加回去
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

y = z_y2

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from collections import defaultdict


def feature_selection_ensemble(X, y, n_splits=5, top_n=20, random_state=None):
    """
    基于5次随机分组的综合特征筛选过程
    返回：
    - lgb_consensus_features: 5次分组中至少出现3次的前top_n个LGB重要特征
    - lasso_consensus_features: 5次分组中至少出现3次的Lasso非零系数特征
    - final_intersection: 两种方法共同选中的特征
    - final_union: 两种方法选中的特征并集（去除高相关性特征）
    """
    # 初始化特征统计字典
    lgb_feature_counts = defaultdict(int)
    lasso_feature_counts = defaultdict(int)

    # 用于存储每次LGB的特征重要性
    lgb_importance_dfs = []

    for i in range(n_splits):
        # 每次使用不同的随机种子
        current_seed = random_state + i if random_state is not None else None

        # 划分训练集和测试集 (这里用train_test_split代替KFold更符合原文描述)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=current_seed)

        # ============= LightGBM 特征选择 =============
        lgb_model = LGBMRegressor(n_estimators=100, max_depth=3,
                                  min_child_samples=5,  # 默认20，减小此值允许更细的分割
                                  min_split_gain=0.001,  # 默认0，设为小正数避免无意义分割
                                  random_state=current_seed)
        lgb_model.fit(X_train, y_train)

        # 获取重要性并排序
        lgb_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': lgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # 保存本次的重要性
        lgb_importance_dfs.append(lgb_importances.set_index('Feature'))

        # 记录本次分组的top_n重要特征
        top_features = lgb_importances.head(top_n)['Feature'].tolist()
        for feat in top_features:
            lgb_feature_counts[feat] += 1

        # ============= Lasso 特征选择 =============
        lasso_model = Lasso(alpha=0.01, random_state=current_seed)
        lasso_model.fit(X_train, y_train)

        # 获取非零系数特征
        lasso_coef = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': abs(lasso_model.coef_)
        }).sort_values('Coefficient', ascending=False)
        selected_features = lasso_coef.head(top_n)['Feature'].tolist()

        for feat in selected_features:
            lasso_feature_counts[feat] += 1

    # ============= 综合筛选结果 =============
    # LGB共识特征 (至少出现3次)
    lgb_consensus = [feat for feat, count in lgb_feature_counts.items()
                     if count >= (n_splits // 2 + 1)]  # 多数出现
    lgb_consensus_features = sorted(lgb_consensus,
                                    key=lambda x: -lgb_feature_counts[x])[:top_n]

    # Lasso共识特征 (至少出现3次)
    lasso_consensus_features = [feat for feat, count in lasso_feature_counts.items()
                                if count >= (n_splits // 2 + 1)]

    # 最终特征组合
    final_intersection = list(set(lgb_consensus_features) & set(lasso_consensus_features))
    final_union = list(set(lgb_consensus_features) | set(lasso_consensus_features))

    # ============= 计算平均LGB特征重要性 =============
    # 合并所有重要性数据
    avg_lgb_importance = pd.concat(lgb_importance_dfs, axis=1)
    avg_lgb_importance = avg_lgb_importance.mean(axis=1).to_frame('Avg_Importance')

    # ============= 相关性分析 =============
    # 计算并集特征的相关系数矩阵
    union_df = X[final_union]
    corr_matrix = union_df.corr().abs()

    # 找出高相关性特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    # 对于每对高相关性特征，保留平均LGB重要性较高的那个
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        # 获取两个特征的平均LGB重要性
        imp1 = avg_lgb_importance.loc[feat1, 'Avg_Importance'] if feat1 in avg_lgb_importance.index else 0
        imp2 = avg_lgb_importance.loc[feat2, 'Avg_Importance'] if feat2 in avg_lgb_importance.index else 0

        # 比较重要性，决定保留哪个
        if imp1 > imp2:
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)

    # 从并集中移除选定的特征
    final_union_filtered = [feat for feat in final_union if feat not in features_to_remove]

    return {
        'lgb_consensus': lgb_consensus_features,
        'lasso_consensus': lasso_consensus_features,
        'intersection': final_intersection,
        'union': final_union,
        'union_filtered': final_union_filtered,
        'high_corr_pairs': high_corr_pairs,
        'features_removed': list(features_to_remove),
        'lgb_feature_counts': lgb_feature_counts,
        'lasso_feature_counts': lasso_feature_counts,
        'avg_lgb_importance': avg_lgb_importance
    }


# 示例用法
if __name__ == "__main__":
    results = feature_selection_ensemble(
        X_scaled_df, y,
        n_splits=5,
        top_n=20,
        random_state=42
    )

    # 打印结果
    print("LightGBM共识特征 (出现次数≥3的前10个):")
    print(results['lgb_consensus'])
    print("\nLasso共识特征 (出现次数≥3的前10个):")
    print(results['lasso_consensus'])
    print("\n两种方法共同选中的特征:")
    print(results['intersection'])
    print("\n两种方法选中的特征并集(原始):")
    print(results['union'])
    print("\n两种方法选中的特征并集(去除高相关性特征后):")
    print(results['union_filtered'])
    print("\n检测到的高相关性特征对:")
    print(results['high_corr_pairs'])
    print("\n被移除的特征:")
    print(results['features_removed'])
    print("\n平均LGB特征重要性:")
    print(results['avg_lgb_importance'].sort_values('Avg_Importance', ascending=False))

    # # 可选：保存特征出现频次
    # pd.DataFrame.from_dict(results['lgb_feature_counts'],
    #                        orient='index',
    #                        columns=['Count']).sort_values('Count', ascending=False).to_csv('lgb_feature_counts_z_y2.csv')
    #
    # pd.DataFrame.from_dict(results['lasso_feature_counts'],
    #                        orient='index',
    #                        columns=['Count']).sort_values('Count', ascending=False).to_csv('lasso_feature_counts_z_y2.csv')

