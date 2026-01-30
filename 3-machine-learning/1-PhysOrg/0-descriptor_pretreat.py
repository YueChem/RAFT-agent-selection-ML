import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps, gridspec
import sys
from datetime import datetime


def filter_descriptors(monomer_df, raft_df, corr_threshold):
    """
    分析并筛选monomer和RAFT描述符数据

    参数:
    monomer_df: pd.DataFrame - monomer描述符数据
    raft_df: pd.DataFrame - RAFT描述符数据
    corr_threshold: float - 相关性阈值(默认0.9)
    返回:
    tuple: (筛选后的monomer_df, 筛选后的raft_df, 删除的monomer描述符, 删除的raft描述符)
    """
    # 1. 计算相关性矩阵
    print("=" * 50 + "\nMonomer描述符分析:\n" + "=" * 50)
    dcorr_monomer = monomer_df.corr(method='pearson')
    filtered_monomer, dropped_monomer = process_descriptors(monomer_df,
                                                            dcorr_monomer,
                                                            corr_threshold,
                                                            "Monomer")

    print("\n" + "=" * 50 + "\nRAFT描述符分析:\n" + "=" * 50)
    dcorr_raft = raft_df.corr(method='pearson')
    filtered_raft, dropped_raft = process_descriptors(raft_df,
                                                      dcorr_raft,
                                                      corr_threshold,
                                                      "RAFT")
    plt.figure(figsize=(11, 9), dpi=100)
    ax_monomer = sns.heatmap(data=dcorr_monomer, cmap='YlGnBu', center=0.5, linewidths=1, annot=False,
                             annot_kws={'size': 5, 'weight': 'bold', 'color': 'black'})

    plt.figure(figsize=(11, 9), dpi=100)
    ax_RAFT = sns.heatmap(data=dcorr_raft, cmap='YlGnBu', center=0.5, linewidths=1, annot=False,
                          annot_kws={'size': 3, 'weight': 'bold', 'color': 'black'})

    plt.show()

    return filtered_monomer, filtered_raft, dropped_monomer, dropped_raft


def process_descriptors(df, corr_matrix, corr_threshold, descriptor_type):
    """处理单个描述符数据集"""
    # 打印相关性排序
    print(f"\n{descriptor_type}描述符相关性(从大到小排序):")
    print_corr_sorted(corr_matrix)

    # 找出需要删除的描述符
    to_drop = find_redundant_descriptors(df, corr_threshold)

    # 删除选定的描述符
    filtered_df = df.drop(columns=to_drop)
    print(f"\n删除的{descriptor_type}描述符列:", to_drop)
    print(f"\n筛选后的{descriptor_type}描述符列:", filtered_df.columns.tolist())

    return filtered_df, to_drop


def print_corr_sorted(corr_matrix):
    """打印排序后的相关性矩阵"""
    # 获取上三角矩阵(不含对角线)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 展平并排序
    sorted_corr = upper.stack().sort_values(ascending=False)

    # 打印前20个最高相关性
    print(sorted_corr.head(30))

from sklearn.preprocessing import StandardScaler

def find_redundant_descriptors(df, corr_threshold):
    """找出需要删除的描述符（包含标准化步骤）"""
    # 1. 数据标准化
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # 2. 计算相关矩阵
    corr_matrix = df_scaled.corr().abs()
    to_drop = set()
    dropped_in_corr = set()  # 记录因相关性被删除的列

    # 3. 找出高相关性对（直接删除colname_i，不比较方差）
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > corr_threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]

                # 如果任一列已被处理，则跳过
                if colname_i in dropped_in_corr or colname_j in dropped_in_corr:
                    continue

                # 直接删除colname_i（不比较方差）
                to_drop.add(colname_i)
                dropped_in_corr.add(colname_i)

    return list(to_drop)

def filter_raft_descriptors(total_df, raft,valid):

    # 筛选出RAFT列中含有'z07'的行
    descriptor = total_df[total_df['SampleName'].str.contains(raft, na=False)].copy()

    # 删除含有'r26'或'r27'的行
    descriptor = descriptor[~descriptor['SampleName'].str.contains(valid, na=False)]

    # 重置索引
    descriptor.reset_index(drop=True, inplace=True)

    # 打印结果信息
    print(f"筛选后剩余{descriptor.shape[0]}行数据")

    return descriptor


# 使用示例
if __name__ == "__main__":
    # 重定向标准输出到文件和控制台
    class DualOutput:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()


    # 创建带有时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"descriptor_analysis_{timestamp}.txt"

    # 开始重定向输出
    sys.stdout = DualOutput(output_filename)

    try:
        # 读取数据（跳过第一列，假设是ID）
        monomer_df = pd.read_csv('monomer_descriptor.csv').iloc[:, 1:]
        raft_df = pd.read_csv('RAFT_descriptor.csv')
        r_df = filter_raft_descriptors(raft_df, 'z07', 'r26').iloc[:, 1:]
        z_df = filter_raft_descriptors(raft_df, 'r01', 'z25|z26').iloc[:, 1:]

        print("=" * 60)
        print("描述符筛选分析报告")
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print("\n数据概览:")
        print(f"Monomer描述符数量: {monomer_df.shape[1]}")
        print(f"RAFT描述符数量: {raft_df.shape[1]}")
        print(f"样本数量: {monomer_df.shape[0]}")

        # 运行分析
        filtered_monomer, filtered_raft, dropped_monomer, dropped_raft = filter_descriptors(
            monomer_df, z_df,
            corr_threshold=0.9,
        )

        # 最终结果汇总
        print("\n" + "=" * 60)
        print("最终筛选结果汇总")
        print("=" * 60)
        print("\nMonomer描述符:")
        print(f"保留数量: {filtered_monomer.shape[1]}")
        print(f"删除数量: {len(dropped_monomer)}")
        print("删除的描述符列表:")
        for desc in dropped_monomer:
            print(f"- {desc}")

        print("\nRAFT描述符:")
        print(f"保留数量: {filtered_raft.shape[1]}")
        print(f"删除数量: {len(dropped_raft)}")
        print("删除的描述符列表:")
        for desc in dropped_raft:
            print(f"- {desc}")

        print("\n" + "=" * 60)
        print("分析完成！结果已保存至:", output_filename)
        print("=" * 60)

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
    finally:
        # 恢复标准输出
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal