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

def combine_descriptors():
    # 读取两个CSV文件
    monomer_df = pd.read_csv('monomer_descriptor.csv')
    raft_df = pd.read_csv('RAFT_descriptor.csv')

    # 创建一个空的DataFrame来存储结果
    combined_df = pd.DataFrame()

    # 遍历monomer文件的每一行
    for i, monomer_row in monomer_df.iterrows():
        # 对当前monomer行，与RAFT文件的每一行合并
        for _, raft_row in raft_df.iterrows():
            # 合并两行数据
            combined_row = pd.concat([monomer_row, raft_row], axis=0)
            # 转置并添加到结果DataFrame中
            combined_df = pd.concat([combined_df, combined_row.to_frame().T], axis=0)

    # 重置索引并保存结果
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv('total_descriptor.csv', index=False)
    print(f"成功生成total_descriptor.csv，共{len(combined_df)}行数据")


import pandas as pd


def filter_raft_descriptors(raft,valid):
    # 读取total_descriptor.csv文件
    total_df = pd.read_csv('total_descriptor.csv')

    # 筛选出RAFT列中含有'z07'的行
    descriptor = total_df[total_df['RAFT'].str.contains(raft, na=False)].copy()

    # 删除含有'r26'或'r27'的行
    descriptor = descriptor[~descriptor['RAFT'].str.contains(valid, na=False)]

    # 重置索引
    descriptor.reset_index(drop=True, inplace=True)

    # 打印结果信息
    print(f"筛选后剩余{descriptor.shape[0]}行数据")

    return descriptor


def merge_with_database(descriptor, inter, y1, y2):
    """
    读取database.xlsx第一个表的前3列，与r_descriptor合并

    参数:
    r_descriptor: pd.DataFrame - 要合并的r_descriptor数据框

    返回:
    pd.DataFrame - 合并后的数据框
    """
    try:
        # 读取database.xlsx第一个表的前3列
        db_data = pd.read_excel('database.xlsx', usecols=[inter, y1, y2], sheet_name=0, header=0)
        # 确保r_descriptor和db_data行数相同
        if len(descriptor) != len(db_data):
            # 取最小行数以匹配
            min_rows = min(len(descriptor), len(db_data))
            descriptor = descriptor.iloc[:min_rows]
            db_data = db_data.iloc[:min_rows]

        # 合并数据
        merged_df = pd.concat([db_data, descriptor], axis=1)

        return merged_df

    except FileNotFoundError:
        print("错误: 未找到database.xlsx文件")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    # combine_descriptors()
    r_descriptor = filter_raft_descriptors('z07', 'r26')
    z_descriptor = filter_raft_descriptors('r01', 'z25|z26')

    r_result = merge_with_database(r_descriptor, 0, 1, 2)
    r_result.to_csv('r_database.csv')
    z_result = merge_with_database(z_descriptor, 3, 4, 5)
    z_result.to_csv('z_database.csv')






