import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, LassoLarsCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             make_scorer, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.decomposition import PCA
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

r_result = pd.read_csv('r_database.csv')
z_result = pd.read_csv('z_database.csv')


r_X = r_result.drop(columns=['inter_r', 'SampleName', 'RAFT', 'Gβ/_add', 'Gp/re'])
z_X = z_result.drop(columns=['inter_z', 'SampleName', 'RAFT', 'Gadd/p', 'Gadd/_add'])

r_X_filter = r_X[['E1', 'E3', 'E5', 'B1', 'B2', 'B3', 'Q1', 'Q2', 'V3', 'P3', 'E6', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12', 'B13', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9', 'Q10', 'Q12', 'Q14', 'V4', 'V9', 'P5', 'P6', 'P7', 'P9']]
z_X_filter = z_X[['E1', 'E3', 'E5', 'B1', 'B2', 'B3', 'Q1', 'Q2', 'V3', 'P3', 'E6', 'E7', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'Q5', 'Q6', 'Q7', 'Q8', 'Q11', 'V4', 'V7', 'V9', 'P5', 'P6', 'P7', 'P9']]

r_y1 = r_result['Gβ/_add'].drop(index=[142, 145, 167])
r_y2 = r_result['Gp/re'].drop(index=[142, 145, 167])
z_y1 = z_result['Gadd/p'].drop(index=[147, 148, 149, 123, 124, 125, 128])
z_y2 = z_result['Gadd/_add'].drop(index=[147, 148, 149, 123, 124, 125, 128])

X_r_elec = r_X.iloc[:, 1:][['E6', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12', 'B13', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9', 'Q10', 'Q12', 'Q14']]
X_r_ster = r_X.iloc[:, 1:][['V4', 'V9', 'P5', 'P6', 'P7', 'P9']]
X_z_elec = z_X.iloc[:, 1:][['E6', 'E7', 'E9', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'Q5', 'Q6', 'Q7', 'Q8', 'Q11']]
X_z_ster = z_X.iloc[:, 1:][['V4', 'V7', 'V9', 'P5', 'P6', 'P7', 'P9']]


# 数据标准化
scaler = StandardScaler()
data_scaled_e = scaler.fit_transform(X_r_elec)
data_scaled_v = scaler.fit_transform(X_r_ster)
# PCA分析
pca = PCA(n_components=1)  # 只提取第一个主成分
pc1_e = pca.fit_transform(data_scaled_e)
pc1_composition_e = pd.Series(pca.components_[0], index=X_r_elec.columns, name="PC1 Composition")
top5_components = pc1_composition_e.abs().nlargest(5)
print(f'r-e:{top5_components}')
pc1_v = pca.fit_transform(data_scaled_v)
pc1_composition_v = pd.Series(pca.components_[0], index=X_r_ster.columns, name="PC1 Composition")
top5_components_v = pc1_composition_v.abs().nlargest(5)
print(top5_components_v.to_string())

# 将 pc1_e 和 pc1_v 转换为 pandas Series
pc1_e_series = pd.Series(pc1_e.flatten(), name="PC1_electronic")
pc1_v_series = pd.Series(pc1_v.flatten(), name="PC1_steric")
# pearson_corr = pc1_e_series.corr(X_r_elec['B6'])
pc1_e_series_cleaned = pc1_e_series.drop(index=[142, 145, 167])
pc1_v_series_cleaned = pc1_v_series.drop(index=[142, 145, 167])
descriptor_data_m_r_cleaned = r_X

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


# Create the jointplot
g = sns.jointplot(
    x=r_y1,
    y=r_y2,
    data=r_result,
    kind='scatter',
    hue='judgement',
    palette={'feasible': 'green', 'no-feasible': 'blue'},
    s=90,  # Increase marker size for better visibility
    legend=False,  # Disable the legend on the plot
    alpha=0.4,
    marginal_kws=dict(fill=True, alpha=0.2)  # Set marginal KDE plots with transparency
)

# Add black border to the first 25 points
first_25 = r_result.iloc[:25]
for i in range(len(first_25)):
    g.ax_joint.scatter(
        x=first_25['Gβ/_add'].iloc[i],
        y=first_25['Gp/re'].iloc[i],
        s=150,  # Adjust size of the border
        facecolors='none',  # Ensure the inner part of the point is empty
        edgecolors='black',  # Set border color to black
        linewidths=1.5  # Make the border thicker
    )


# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gβ/_add vs Gp/re', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(4)
y_major_locator = MultipleLocator(2)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = r_y1.min(), r_y1.max()
ymin, ymax = r_y2.min(), r_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# # Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()

# Create the scatter plot with 'Q10' for coloring
g = sns.jointplot(
                  x=r_y1,
                  y=r_y2,
                  kind='scatter',
                  hue=pc1_e_series_cleaned,  # Use 'Q10' for coloring
                  palette='viridis',  # Choose a color palette
                  data=r_result,  # Use the cleaned DataFrame
                  s=70,  # Increase marker size for better visibility
                  legend=False,
                  marker='s',  # Change marker shape to square
                  edgecolor='black',  # Set black border
                  linewidth=1.5)  # Set border width

# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gβ/_add vs Gp/re', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(4)
y_major_locator = MultipleLocator(2)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = r_y1.min(), r_y1.max()
ymin, ymax = r_y2.min(), r_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()


# Create the scatter plot with 'Q10' for coloring
g = sns.jointplot(
                  x=r_y1,
                  y=r_y2,
                  kind='scatter',
                  hue=pc1_v_series_cleaned,  # Use 'Q10' for coloring
                  palette='viridis',  # Choose a color palette
                  data=r_result,  # Use the cleaned DataFrame
                  s=70,  # Increase marker size for better visibility
                  legend=False,
                  marker='s',  # Change marker shape to square
                  edgecolor='black',  # Set black border
                  linewidth=1.5)  # Set border width

# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gβ/_add vs Gp/re', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(4)
y_major_locator = MultipleLocator(2)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = r_y1.min(), r_y1.max()
ymin, ymax = r_y2.min(), r_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()

data_scaled_e = scaler.fit_transform(X_z_elec)
data_scaled_v = scaler.fit_transform(X_z_ster)
# PCA分析
pca = PCA(n_components=1)  # 只提取第一个主成分
pc1_e = pca.fit_transform(data_scaled_e)
pc1_composition_e = pd.Series(pca.components_[0], index=X_z_elec.columns, name="PC1 Composition")
top5_components = pc1_composition_e.abs().nlargest(5)
print(top5_components.to_string())

pc1_v = pca.fit_transform(data_scaled_v)
pc1_composition_v = pd.Series(pca.components_[0], index=X_z_ster.columns, name="PC1 Composition")
top5_components_v = pc1_composition_v.abs().nlargest(5)
print(top5_components_v.to_string())

# 将 pc1_e 和 pc1_v 转换为 pandas Series
pc1_e_series = pd.Series(pc1_e.flatten(), name="PC1_electronic")
pc1_v_series = pd.Series(pc1_v.flatten(), name="PC1_steric")

pc1_e_series_cleaned = pc1_e_series.drop(index=[147, 148, 149, 123, 124, 125, 128])
pc1_v_series_cleaned = pc1_v_series.drop(index=[147, 148, 149, 123, 124, 125, 128])
descriptor_data_m_r_cleaned = z_X

# Create a new column for coloring points based on the condition
z_result['judgement'] = np.where((z_result['Gadd/p'] < 0) & (z_result['Gadd/_add'] > -2), 'feasible', 'no-feasible')

# 统计符合条件的数据个数
feasible_count = z_result['judgement'].value_counts().get('feasible', 0)
# 总数据个数
total_count = len(z_result)
# 计算百分比
feasible_percentage = (feasible_count / total_count) * 100


# Create the jointplot
g = sns.jointplot(
    x=z_y1,
    y=z_y2,
    data=z_result,
    kind='scatter',
    hue='judgement',
    palette={'feasible': 'green', 'no-feasible': 'blue'},
    s=90,  # Increase marker size for better visibility
    legend=False,  # Disable the legend on the plot
    alpha=0.4,
    marginal_kws=dict(fill=True, alpha=0.2)  # Set marginal KDE plots with transparency
)

# Add black border to the first 24 points
first_24 = z_result.iloc[:25]
for i in range(len(first_24)):
    g.ax_joint.scatter(
        x=first_24['Gadd/p'].iloc[i],
        y=first_24['Gadd/_add'].iloc[i],
        s=150,  # Adjust size of the border
        facecolors='none',  # Ensure the inner part of the point is empty
        edgecolors='black',  # Set border color to black
        linewidths=1.5  # Make the border thicker
    )


# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gadd/p vs Gadd/_add', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(4)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = z_y1.min(), z_y1.max()
ymin, ymax = z_y2.min(), z_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()


# Create the scatter plot with 'B4m' for coloring
g = sns.jointplot(x=z_y1,
                  y=z_y2,
                  kind='scatter',
                  hue=pc1_e_series_cleaned,  # Use 'Q10' for coloring
                  palette='viridis',  # Choose a color palette
                  data=z_result,  # Use the cleaned DataFrame
                  s=70,  # Increase marker size for better visibility
                  legend=False,
                  marker='s',  # Change marker shape to square
                  edgecolor='black',  # Set black border
                  linewidth=1.5)  # Set border width

# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gadd/p vs Gadd/_add', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(4)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = z_y1.min(), z_y1.max()
ymin, ymax = z_y2.min(), z_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()


# Create the scatter plot with 'B4m' for coloring
g = sns.jointplot(x=z_y1,
                  y=z_y2,
                  kind='scatter',
                  hue=pc1_v_series_cleaned,  # Use 'Q10' for coloring
                  palette='viridis',  # Choose a color palette
                  data=z_result,  # Use the cleaned DataFrame
                  s=70,  # Increase marker size for better visibility
                  legend=False,
                  marker='s',  # Change marker shape to square
                  edgecolor='black',  # Set black border
                  linewidth=1.5)  # Set border width

# Set the labels and title with bold font
g.set_axis_labels('', '')  # Empty strings for axis labels
plt.suptitle('Gadd/p vs Gadd/_add', fontsize=15, weight='bold')  # Set title with bold font


# Set major tick intervals for both x and y axes to 2
x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(4)

# Apply the locators to both axes
g.ax_joint.xaxis.set_major_locator(x_major_locator)
g.ax_joint.yaxis.set_major_locator(y_major_locator)

# Set the x and y axis limits
xmin, xmax = z_y1.min(), z_y1.max()
ymin, ymax = z_y2.min(), z_y2.max()

plt.xlim([xmin-2, xmax+2])
plt.ylim([ymin-2, ymax+2])


# Set the font size and font family for the tick labels
g.ax_joint.tick_params(axis='both', labelsize=18)  # Increase tick label size
g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontsize=18,weight='bold')
g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontsize=18,weight='bold')

# Show the plot
plt.show()