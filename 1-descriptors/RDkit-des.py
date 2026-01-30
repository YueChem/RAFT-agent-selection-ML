import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def calculate_rdkit_descriptors(mol):
    """计算RDKit描述符"""
    if mol is None:
        return None

    desc_dict = {}
    for desc_name, desc_func in Descriptors.descList:
        try:
            desc_dict[desc_name] = desc_func(mol)
        except:
            desc_dict[desc_name] = None
    return desc_dict


def process_mol_files(directory='.'):
    """处理目录中的所有.mol文件"""
    results = []

    # 获取当前目录下所有.mol文件
    mol_files = [f for f in os.listdir(directory) if f.endswith('.mol')]

    for filename in mol_files:
        filepath = os.path.join(directory, filename)
        try:
            # 读取.mol文件
            mol = Chem.MolFromMolFile(filepath)
            if mol is not None:
                # 计算描述符
                descriptors = calculate_rdkit_descriptors(mol)
                if descriptors:
                    # 添加文件名作为第一列
                    descriptors['SampleName'] = filename
                    results.append(descriptors)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    # 转换为DataFrame
    if results:
        df = pd.DataFrame(results)
        # 将SampleName移到第一列
        cols = ['SampleName'] + [col for col in df.columns if col != 'SampleName']
        df = df[cols]
        return df
    else:
        return pd.DataFrame()


# 处理当前目录下的.mol文件
descriptors_df = process_mol_files()

# 保存到CSV文件
if not descriptors_df.empty:
    output_file = 'RAFT_descriptors.csv'
    descriptors_df.to_csv(output_file, index=False)
    print(f"描述符已保存到 {output_file}")
else:
    print("未找到有效的.mol文件或无法计算描述符")