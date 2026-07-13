import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr,shapiro, spearmanr, rankdata
import dcor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import mne
import pickle
import copy
from sklearn.feature_selection import mutual_info_regression
from statsmodels.nonparametric.smoothers_lowess import lowess
from pygam import LinearGAM, s, f, te
from sklearn.utils import shuffle
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind
from scipy.stats import false_discovery_control
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

fature_data = np.load('EEG_source_power_conn.npz')
suvr_values = np.load('abeta_suvr.npy')

source_pd=fature_data['source_pd'];source_pt=fature_data['source_pt']
source_pa=fature_data['source_pa'];source_pb=fature_data['source_pb']
delta_conn_ave=fature_data['delta_conn_ave'];theta_conn_ave=fature_data['theta_conn_ave']
alpha_conn_ave=fature_data['alpha_conn_ave'];beta_conn_ave=fature_data['beta_conn_ave']

feat_7network = np.load('ADPET_73_7network_power_conn.npz')
net7_rpd=feat_7network['net7_rpd'];net7_rpt=feat_7network['net7_rpt']
net7_rpa=feat_7network['net7_rpa'];net7_rpb=feat_7network['net7_rpb']

def create_features_dict_from_arrays(arrays, feature_names):
    """
    从多个数组创建特征字典
    
    参数:
    arrays: 包含多个特征数组的列表，每个数组形状为(n,68)
    feature_names: 对应的特征名称列表
    
    返回:
    dict: 特征名称到特征数据的字典
    """
    # 验证输入
    if len(arrays) != len(feature_names):
        raise ValueError("数组数量与特征名称数量不一致")
    
    # 创建字典
    features_dict = {}
    for i, name in enumerate(feature_names):
        features_dict[name] = arrays[i]
    
    return features_dict
    
if __name__ == "__main__":
    
    # 特征名称列表
    feature_names = [
        'source_pd',
        'source_pt',
        'source_pa',
        'source_pb',
        'delta_conn_ave',
        'theta_conn_ave',
        'alpha_conn_ave',
        'beta_conn_ave',
        'entropy'
    ]
    
    # 创建特征数组列表
    feature_arrays = [
        source_pd,
        source_pt,
        source_pa,
        source_pb,
        delta_conn_ave,
        theta_conn_ave,
        alpha_conn_ave,
        beta_conn_ave,
        entropy
    ]

    feat_names = [
        'net7_rpd',
        'net7_rpt',
        'net7_rpa',
        'net7_rpb']

    feat_arrays = [
        net7_rpd,
        net7_rpt,
        net7_rpa,
        net7_rpb]
    # 使用新函数创建特征字典
    features_dict = create_features_dict_from_arrays(feature_arrays, feature_names)
    feat_7network_dict = create_features_dict_from_arrays(feat_arrays, feat_names)
    # 输出目录
    output_dir = " "


def read_data_file(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    读取数据文件（支持Excel和CSV格式）
    
    参数:
        file_path: 文件路径
        encoding: CSV文件编码（默认为utf-8）
    
    返回:
        pandas DataFrame
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    file_ext = file_path_obj.suffix.lower()
    
    if file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_ext in ['.csv']:
        # 尝试多种编码读取CSV文件
        encodings_to_try = [encoding, 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
        for enc in encodings_to_try:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        # 如果所有编码都失败，尝试读取前几行检测编码
        try:
            with open(file_path, 'rb') as f:
                rawdata = f.read(10000)
                result = chardet.detect(rawdata)
                return pd.read_csv(file_path, encoding=result['encoding'])
        except:
            raise ValueError(f"无法读取CSV文件: {file_path}。请检查文件编码。")
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}。请使用Excel(.xlsx, .xls)或CSV(.csv)格式")

def match_eeg_files_to_subjects(
    eeg_file_list_path: str,
    subject_info_path: str,
    eeg_column: str = "EEG",
    age_column: str = "age",
    gender_column: str = "gender",
    education_column: str = "eduction",
    tau_status_column: str = "tau_status",
    abeta_status_column: str = "Abeta_status",
    eeg_file_column: str = None
) -> Dict[str, Any]:
    """
    根据重排序的EEG文件名列表，在subject信息表中匹配对应的age、gender、tau_status和Abeta_status信息
    
    参数:
        eeg_file_list_path: 包含重排序EEG文件名的文件路径（支持Excel和CSV）
        subject_info_path: 包含所有subject信息的文件路径（支持Excel和CSV）
        eeg_column: subject信息表中EEG文件名列名（默认为"EEG"）
        age_column: subject信息表中年龄列名（默认为"age"）
        gender_column: subject信息表中性别列名（默认为"gender"）
        education_column: subject信息表中性别列名（默认为"education"）
        tau_status_column: subject信息表中tau状态列名（默认为"tau_status"）
        abeta_status_column: subject信息表中Abeta状态列名（默认为"Abeta_status"）
        eeg_file_column: 重排序文件中EEG文件名所在列名（如为None则读取第一列）
    
    返回:
        包含匹配结果的字典，包括：
        - age_list: 年龄列表
        - gender_list: 性别列表
        - education_list: 教育年龄列表
        - tau_status_list: tau状态列表
        - abeta_status_list: Abeta状态列表
        - matched_indices: 匹配成功的索引列表
        - unmatched_files: 未匹配的文件名列表
        - matched_subjects: 匹配的subject信息DataFrame
    """
    print(f"开始匹配EEG文件名与subject信息...")
    
    # 1. 读取重排序的EEG文件名列表
    print(f"读取重排序EEG文件名列表: {eeg_file_list_path}")
    try:
        eeg_list_df = read_data_file(eeg_file_list_path)
    except Exception as e:
        raise ValueError(f"读取重排序文件失败: {e}")
    
    # 确定EEG文件名所在列
    if eeg_file_column is None:
        # 默认使用第一列
        eeg_file_column = eeg_list_df.columns[0]
        print(f"使用默认列名: {eeg_file_column}")
    
    if eeg_file_column not in eeg_list_df.columns:
        print(f"可用列名: {list(eeg_list_df.columns)}")
        raise ValueError(f"列 '{eeg_file_column}' 不存在于重排序文件中")
    
    # 获取重排序的EEG文件名列表
    reordered_eeg_files = eeg_list_df[eeg_file_column].astype(str).tolist()
    print(f"读取到 {len(reordered_eeg_files)} 个重排序的EEG文件名")
    
    # 2. 读取subject信息表
    print(f"读取subject信息表: {subject_info_path}")
    try:
        subject_df = read_data_file(subject_info_path)
    except Exception as e:
        raise ValueError(f"读取subject信息文件失败: {e}")
    
    # 检查必要的列是否存在
    required_columns = [eeg_column, age_column, gender_column]
    optional_columns = [tau_status_column, abeta_status_column]
    
    missing_columns = [col for col in required_columns if col not in subject_df.columns]
    if missing_columns:
        print(f"subject信息表中可用列名: {list(subject_df.columns)}")
        raise ValueError(f"subject信息表缺少以下必需列: {missing_columns}")
    
    # 检查可选列是否存在
    optional_missing = [col for col in optional_columns if col not in subject_df.columns]
    if optional_missing:
        print(f"警告: subject信息表缺少以下可选列: {optional_missing}")
        print(f"这些列将用None值填充")
    
    print(f"读取到 {len(subject_df)} 个subject信息")
    
    # 3. 提取subject信息表中的EEG文件名（去除扩展名）
    print("提取subject信息表中的EEG文件名（去除扩展名）...")
    subject_df["eeg_basename"] = subject_df[eeg_column].astype(str).apply(extract_basename)
    
    # 4. 创建从EEG文件名到subject信息的映射
    print("创建EEG文件名到subject信息的映射...")
    eeg_to_subject = {}
    for idx, row in subject_df.iterrows():
        eeg_name = row["eeg_basename"]
        subject_info = {
            "index": idx,
            "age": row[age_column],
            "education":row[education_column],
            "gender": row[gender_column],
            "original_eeg": row[eeg_column]
        }
        
        # 添加tau_status（如果存在）
        if tau_status_column in subject_df.columns:
            subject_info["tau_status"] = row[tau_status_column]
        else:
            subject_info["tau_status"] = None
            
        # 添加Abeta_status（如果存在）
        if abeta_status_column in subject_df.columns:
            subject_info["abeta_status"] = row[abeta_status_column]
        else:
            subject_info["abeta_status"] = None
        
        eeg_to_subject[eeg_name] = subject_info
    
    print(f"创建了 {len(eeg_to_subject)} 个EEG到subject的映射")
    
    # 5. 匹配重排序的EEG文件名
    print("匹配重排序的EEG文件名...")
    age_list = []
    gender_list = []
    tau_status_list = []
    abeta_status_list = []
    matched_indices = []
    unmatched_files = []
    
    for i, eeg_file in enumerate(reordered_eeg_files):
        # 提取基本文件名（不带扩展名）
        eeg_basename = extract_basename(eeg_file)
        
        if eeg_basename in eeg_to_subject:
            subject_info = eeg_to_subject[eeg_basename]
            age_list.append(subject_info["age"])
            gender_list.append(subject_info["gender"])
            tau_status_list.append(subject_info["tau_status"])
            abeta_status_list.append(subject_info["abeta_status"])
            matched_indices.append(i)
        else:
            # 尝试模糊匹配：去掉常见的文件后缀
            eeg_clean = re.sub(r'_[0-9]+$', '', eeg_basename)  # 去掉末尾的数字
            eeg_clean = re.sub(r'[_-]eeg$', '', eeg_clean, flags=re.IGNORECASE)  # 去掉_eeg或-EEG
            eeg_clean = re.sub(r'[_-]task.*$', '', eeg_clean, flags=re.IGNORECASE)  # 去掉_task-rest等
            
            if eeg_clean in eeg_to_subject:
                subject_info = eeg_to_subject[eeg_clean]
                age_list.append(subject_info["age"])
                gender_list.append(subject_info["gender"])
                education_list.append(subject_info['education'])
                tau_status_list.append(subject_info["tau_status"])
                abeta_status_list.append(subject_info["abeta_status"])
                matched_indices.append(i)
                print(f"模糊匹配成功: {eeg_file} -> {eeg_clean}")
            else:
                # 在subject信息表中搜索包含该文件名的记录
                matching_subjects = []
                for subj_eeg, subj_info in eeg_to_subject.items():
                    if eeg_basename in subj_eeg or subj_eeg in eeg_basename:
                        matching_subjects.append((subj_eeg, subj_info))
                
                if matching_subjects:
                    # 如果有多个匹配，取第一个
                    matched_eeg, subject_info = matching_subjects[0]
                    age_list.append(subject_info["age"])
                    gender_list.append(subject_info["gender"])
                    education_list.append(subject_info['education'])
                    tau_status_list.append(subject_info["tau_status"])
                    abeta_status_list.append(subject_info["abeta_status"])
                    matched_indices.append(i)
                    print(f"部分匹配成功: {eeg_file} -> {matched_eeg}")
                else:
                    age_list.append(None)
                    gender_list.append(None)
                    education_list.append(None)
                    tau_status_list.append(None)
                    abeta_status_list.append(None)
                    unmatched_files.append(eeg_file)
                    print(f"警告: 无法匹配文件: {eeg_file}")
    
    # 6. 统计匹配结果
    matched_count = len([x for x in age_list if x is not None])
    unmatched_count = len(unmatched_files)
    
    print(f"\n匹配结果统计:")
    print(f"  总文件数: {len(reordered_eeg_files)}")
    print(f"  成功匹配: {matched_count}")
    print(f"  未匹配: {unmatched_count}")
    print(f"  匹配率: {matched_count/len(reordered_eeg_files)*100:.2f}%")
    
    if unmatched_files:
        print(f"\n未匹配的文件名:")
        for f in unmatched_files[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(unmatched_files) > 10:
            print(f"  ... 还有 {len(unmatched_files)-10} 个未显示")
    
    # 7. 获取匹配的subject信息DataFrame
    matched_subject_indices = []
    for i, eeg_file in enumerate(reordered_eeg_files):
        eeg_basename = extract_basename(eeg_file)
        if eeg_basename in eeg_to_subject:
            matched_subject_indices.append(eeg_to_subject[eeg_basename]["index"])
    
    matched_subjects_df = subject_df.iloc[matched_subject_indices].copy() if matched_subject_indices else pd.DataFrame()
    
    return {
        "age_list": age_list,
        "gender_list": gender_list,
        "eudctaion_list": education_list,
        "tau_status_list": tau_status_list,
        "abeta_status_list": abeta_status_list,
        "matched_indices": matched_indices,
        "unmatched_files": unmatched_files,
        "matched_subjects": matched_subjects_df,
        "match_rate": matched_count/len(reordered_eeg_files) if reordered_eeg_files else 0
    }


def save_matched_results(
    result_dict: Dict[str, Any],
    output_dir: str = "matched_results",
    prefix: str = "matched"
) -> Dict[str, str]:
    """
    保存匹配结果
    
    参数:
        result_dict: match_eeg_files_to_subjects函数返回的结果字典
        output_dir: 输出目录
        prefix: 输出文件前缀
    
    返回:
        保存的文件路径字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # 1. 保存age列表
    age_list = result_dict["age_list"]
    age_path = output_dir / f"{prefix}_age_list.py"
    with open(age_path, 'w', encoding='utf-8') as f:
        f.write(f"# 匹配的年龄列表 (共{len(age_list)}个)\n")
        f.write(f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 匹配率: {result_dict.get('match_rate', 0)*100:.2f}%\n")
        f.write("age_list = [\n")
        for i, age in enumerate(age_list):
            if age is not None:
                f.write(f"    {age},  # 索引 {i}\n")
            else:
                f.write(f"    None,  # 索引 {i} (未匹配)\n")
        f.write("]\n")
    saved_files["age_list"] = str(age_path)
    print(f"年龄列表已保存: {age_path}")
    
    # 2. 保存gender列表
    gender_list = result_dict["gender_list"]
    gender_path = output_dir / f"{prefix}_gender_list.py"
    with open(gender_path, 'w', encoding='utf-8') as f:
        f.write(f"# 匹配的性别列表 (共{len(gender_list)}个)\n")
        f.write(f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 匹配率: {result_dict.get('match_rate', 0)*100:.2f}%\n")
        f.write("gender_list = [\n")
        for i, gender in enumerate(gender_list):
            if gender is not None:
                f.write(f"    '{gender}',  # 索引 {i}\n")
            else:
                f.write(f"    None,  # 索引 {i} (未匹配)\n")
        f.write("]\n")
    saved_files["gender_list"] = str(gender_path)
    print(f"性别列表已保存: {gender_path}")
    
    # 3. 保存为npy文件
    age_array = np.array(age_list, dtype=object)
    gender_array = np.array(gender_list, dtype=object)
    
    age_npy_path = output_dir / f"{prefix}_age.npy"
    gender_npy_path = output_dir / f"{prefix}_gender.npy"
    
    np.save(age_npy_path, age_array)
    np.save(gender_npy_path, gender_array)
    
    saved_files["age_npy"] = str(age_npy_path)
    saved_files["gender_npy"] = str(gender_npy_path)
    print(f"年龄数组已保存为npy: {age_npy_path}")
    print(f"性别数组已保存为npy: {gender_npy_path}")
    
    # 4. 保存为CSV文件
    matched_data = []
    for i, (age, gender) in enumerate(zip(age_list, gender_list)):
        matched_data.append({
            "index": i,
            "age": age if age is not None else "未匹配",
            "gender": gender if gender is not None else "未匹配"
        })
    
    matched_df = pd.DataFrame(matched_data)
    csv_path = output_dir / f"{prefix}_subjects.csv"
    matched_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    saved_files["csv"] = str(csv_path)
    print(f"匹配结果已保存为CSV: {csv_path}")
    
    # 5. 保存未匹配文件列表
    if result_dict["unmatched_files"]:
        unmatched_path = output_dir / f"{prefix}_unmatched_files.txt"
        with open(unmatched_path, 'w', encoding='utf-8') as f:
            f.write(f"# 未匹配的EEG文件名 (共{len(result_dict['unmatched_files'])}个)\n")
            f.write(f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for file in result_dict["unmatched_files"]:
                f.write(f"{file}\n")
        saved_files["unmatched"] = str(unmatched_path)
        print(f"未匹配文件列表已保存: {unmatched_path}")
    
    # 6. 保存匹配的subject信息
    if not result_dict["matched_subjects"].empty:
        subjects_path = output_dir / f"{prefix}_subject_info.csv"
        result_dict["matched_subjects"].to_csv(subjects_path, index=False, encoding='utf-8-sig')
        saved_files["subjects_csv"] = str(subjects_path)
        print(f"匹配的subject信息已保存: {subjects_path}")
    
    # 7. 保存匹配摘要
    summary = {
        "total_files": len(age_list),
        "matched_count": len([x for x in age_list if x is not None]),
        "unmatched_count": len(result_dict["unmatched_files"]),
        "match_rate": result_dict.get("match_rate", 0),
        "age_list_path": saved_files.get("age_list", ""),
        "gender_list_path": saved_files.get("gender_list", ""),
        "age_npy_path": saved_files.get("age_npy", ""),
        "gender_npy_path": saved_files.get("gender_npy", ""),
        "csv_path": saved_files.get("csv", ""),
        "unmatched_path": saved_files.get("unmatched", ""),
        "subjects_csv_path": saved_files.get("subjects_csv", "")
    }
    
    summary_path = output_dir / f"{prefix}_summary.json"
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    saved_files["summary"] = str(summary_path)
    print(f"匹配摘要已保存: {summary_path}")
    
    return saved_files

def load_matched_lists(file_path: str) -> Tuple[List, List]:
    """
    从保存的Python文件中加载匹配的列表
    
    参数:
        file_path: Python文件路径（可以是age_list或gender_list文件）
    
    返回:
        (age_list, gender_list) 元组
    """
    import importlib.util
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 根据文件名确定加载哪个列表
    if "age" in file_path.name.lower():
        # 加载age_list
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 执行文件内容，获取age_list变量
        local_vars = {}
        exec(content, {}, local_vars)
        
        age_list = local_vars.get("age_list", [])
        
        # 尝试加载对应的gender_list
        gender_file = file_path.parent / file_path.name.replace("age", "gender")
        gender_list = []
        if gender_file.exists():
            with open(gender_file, 'r', encoding='utf-8') as f:
                content = f.read()
            exec(content, {}, local_vars)
            gender_list = local_vars.get("gender_list", [])
        
        return age_list, gender_list
    
    elif "gender" in file_path.name.lower():
        # 加载gender_list
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        local_vars = {}
        exec(content, {}, local_vars)
        
        gender_list = local_vars.get("gender_list", [])
        
        # 尝试加载对应的age_list
        age_file = file_path.parent / file_path.name.replace("gender", "age")
        age_list = []
        if age_file.exists():
            with open(age_file, 'r', encoding='utf-8') as f:
                content = f.read()
            exec(content, {}, local_vars)
            age_list = local_vars.get("age_list", [])
        
        return age_list, gender_list
    
    else:
        raise ValueError("文件名应包含'age'或'gender'关键字")

def process_eeg_subject_matching(
    eeg_file_list_path: str,
    subject_info_path: str,
    output_dir: str = "matched_results",
    eeg_column: str = "EEG",
    age_column: str = "age",
    gender_column: str = "gender",
    eeg_file_column: str = None
) -> Dict[str, Any]:
    """
    完整的EEG文件名与subject信息匹配处理流程
    
    参数:
        eeg_file_list_path: 重排序EEG文件名Excel文件路径
        subject_info_path: subject信息Excel文件路径
        output_dir: 输出目录
        eeg_column: subject信息表中EEG列名
        age_column: subject信息表中年龄列名
        gender_column: subject信息表中性别列名
        eeg_file_column: 重排序文件中EEG文件名列名
    
    返回:
        包含所有结果的字典
    """
    print("=" * 60)
    print("EEG文件名与subject信息匹配处理")
    print("=" * 60)
    
    # 1. 匹配EEG文件名与subject信息
    result = match_eeg_files_to_subjects(
        eeg_file_list_path=eeg_file_list_path,
        subject_info_path=subject_info_path,
        eeg_column=eeg_column,
        age_column=age_column,
        gender_column=gender_column,
        eeg_file_column=eeg_file_column
    )
    
    # 2. 保存匹配结果
    saved_files = save_matched_results(result, output_dir)
    
    # 3. 打印摘要
    print("\n" + "=" * 60)
    print("匹配处理完成!")
    print("=" * 60)
    print(f"总文件数: {len(result['age_list'])}")
    print(f"成功匹配: {len([x for x in result['age_list'] if x is not None])}")
    print(f"匹配率: {result.get('match_rate', 0)*100:.2f}%")
    print(f"\n生成的文件:")
    for key, path in saved_files.items():
        print(f"  {key}: {Path(path).name}")
    
    # 4. 显示示例数据
    print(f"\n年龄列表示例 (前10个):")
    for i, age in enumerate(result['age_list'][:10]):
        print(f"  索引 {i}: {age}")
    
    print(f"\n性别列表示例 (前10个):")
    for i, gender in enumerate(result['gender_list'][:10]):
        print(f"  索引 {i}: {gender}")
    
    return {
        "match_result": result,
        "saved_files": saved_files
    }

if __name__ == "__main__":
    """
    使用示例
    """
    print("=" * 60)
    print("EEG文件名与subject信息匹配工具")
    print("=" * 60)
    
    # 1. 基本匹配
    result = match_eeg_files_to_subjects(
        eeg_file_list_path=" ",  # 重排序的EEG文件名CSV文件
        subject_info_path=" ",         # subject信息Excel文件
        eeg_column=" ",                             # subject信息表中EEG文件名列
        age_column=" ",                             # subject信息表中年龄列
        eduction_column=" ",                       #subject信息表中教育年限列
        gender_column=" ",                       # subject信息表中性别列
        tau_status_column=" ",               # subject信息表中tau状态列
        abeta_status_column=" ",           # subject信息表中Abeta状态列
        eeg_file_column=" "               # 重排序文件中EEG文件名列
    )
    
    # 获取结果
    age_list = result["age_list"]                  # 年龄列表
    gender_list = result["gender_list"]           # 性别列表
    eductaion_list = result["gender_list"]       # 教育年限列表
    tau_status_list = result["tau_status_list"]    # tau状态列表
    abeta_status_list = result["abeta_status_list"]  # Abeta状态列表

def run_lme_analysis(age, gender, tau_status, suvr_values, features_dict, output_dir=None):
    """
    AD病理与EEG功能关联的线性混合效应模型分析（含可视化）
    
    参数:
    age: 年龄列表 (n_subjects,)
    gender: 性别列表 ('男'或'女') (n_subjects,)
    tau_status: tau-PET状态 (0=阴性, 1=阳性) (n_subjects,)
    suvr_values: Aβ-PET SUVR值 (n_subjects,)
    features_dict: 字典，包含以下键值对:
        'source_pt': (n_subjects,68) theta功率谱密度
        'source_pb': (n_subjects,68) beta功率谱密度
    output_dir: 结果保存目录 (可选)
    
    返回:
    results: 包含所有分析结果的字典
    roi_results: ROI级别的回归结果
    """
    n_subjects = len(age)
    n_rois = 68
    
    # =====================
    # 1. 数据预处理
    # =====================
    # 转换性别为数值
    gender_numeric = [0 if g == '男' else 1 for g in gender]
    
    # Aβ标准化
    abeta_z = (suvr_values - np.mean(suvr_values)) / np.std(suvr_values)

    # 创建长格式数据框
    data_list = []
    for subj_idx in range(n_subjects):
        for roi_idx in range(n_rois):
            row = {
                'SubjectID': f'Subj_{subj_idx}',
                'ROI': roi_idx,
                'Abeta_SUVR': abeta_z[subj_idx],
                'Tau_status': tau_status[subj_idx],
                'Age': age[subj_idx],
                'Gender': gender_numeric[subj_idx],
                'theta_power': features_dict['source_pt'][subj_idx, roi_idx],
                'beta_power': features_dict['source_pb'][subj_idx, roi_idx],
                # 'alpha_power': features_dict['source_pa'][subj_idx, roi_idx],
                # 'delta_power': features_dict['source_pd'][subj_idx, roi_idx],
                # 'theta_connectivity': features_dict['theta_conn_ave'][subj_idx,roi_idx]
            }
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # 转换分类变量
    df['SubjectID'] = df['SubjectID'].astype('category')
    df['ROI'] = df['ROI'].astype('category')
    df['Tau_status'] = df['Tau_status'].astype('category')
    df['Gender'] = df['Gender'].astype('category')
    
    # =====================
    # 2. LME模型构建与分析
    # =====================
    # results = {}
    eeg_features = {
        'rTheta': 'theta_power',
        'rBeta': 'beta_power',
        # 'alpha band relative power': 'alpha_power',
        # 'delta band relative power': 'delta_power',
        # 'theta band functional connectivity': 'theta_connectivity'
    }

    # ROI级别结果存储
    roi_results = {
        'rTheta': {roi: None for roi in range(n_rois)},
        'rBeta': {roi: None for roi in range(n_rois)},
        # 'alpha band relative power': {roi: None for roi in range(n_rois)},
        # 'delta band relative power': {roi: None for roi in range(n_rois)},
        # 'theta band functional connectivity': {roi: None for roi in range(n_rois)}
    }

    # 为跨特征校正收集数据
    cross_feature_abeta_pvals = []
    cross_feature_tau_pvals = []
    cross_feature_interaction_pvals = []
    cross_feature_info = []  # 存储(特征名, roi编号)

    for feat_name, feat_col in eeg_features.items():
        # 初始化列表用于存储p值，以便进行跨特征FDR校正
        all_abeta_pvals = []
        all_tau_pvals = []
        all_interaction_pvals = []
        # feature_names = []
        
        for roi in range(n_rois):
            roi_df = df[df['ROI'] == roi].copy()
            
            # 1. 创建公式 - 包含随机截距（每个被试）
            formula = f"{feat_col} ~ Abeta_SUVR * Tau_status + Age + C(Gender)"
            
            # 2. 使用混合效应模型（而非普通OLS）
            model = smf.mixedlm(formula, roi_df, groups=roi_df['SubjectID'])
            
            # 3. 尝试多种优化方法（解决可能的收敛问题）
            result = None
            methods = ['bfgs', 'lbfgs', 'cg', 'nm']  # 多种优化方法
            
            for method in methods:
                try:
                    result = model.fit(method=method)
                    if result.converged:
                        print(f"ROI {roi} - {feat_name}: 使用 {method} 方法收敛成功")
                        break
                except Exception as e:
                    print(f"ROI {roi} - {feat_name}: 方法 {method} 失败: {str(e)}")
                    continue
            
            # 4. 如果主要方法都失败，尝试REML
            if result is None or not result.converged:
                try:
                    print(f"ROI {roi} - {feat_name}: 尝试REML方法")
                    result = model.fit(reml=True)
                except:
                    # 最终回退方案
                    result = model.fit()
            
            # 5. 存储结果
            roi_result = {
                'model': result,
                'summary': result.summary(),
                'params': result.fe_params,
                'pvalues': result.pvalues,
                # 'rsquared': result.rsquared,
                'converged': result.converged
            }
            
            roi_results[feat_name][roi] = roi_result
            
            # 提取相关p值用于FDR校正
            abeta_pval = result.pvalues.get('Abeta_SUVR', 1)
            tau_pval = result.pvalues.get('Tau_status',1)
            interaction_pval = result.pvalues.get('Abeta_SUVR:Tau_status', 1)
            
            # 特征内校正的收集
            all_abeta_pvals.append(abeta_pval)
            all_tau_pvals.append(tau_pval)
            all_interaction_pvals.append(interaction_pval)
            
            # 跨特征校正的收集
            cross_feature_abeta_pvals.append(abeta_pval)
            cross_feature_tau_pvals.append(tau_pval)
            cross_feature_interaction_pvals.append(interaction_pval)
            cross_feature_info.append((feat_name, roi))
        
        # 进行ROI级别（特征内部）FDR校正
        if all_abeta_pvals:
            _, abeta_qvals, _, _ = multipletests(all_abeta_pvals, method='fdr_bh')
            _, tau_qvals, _, _ = multipletests(all_tau_pvals,method='fdr_bh')
            _, interaction_qvals, _, _ = multipletests(all_interaction_pvals, method='fdr_bh')
            
            for i, roi in enumerate(range(n_rois)):
                if roi_results[feat_name][roi] is not None:
                    roi_results[feat_name][roi]['qvalues_within'] = {
                        'Abeta_SUVR': abeta_qvals[i],
                        'Tau_status': tau_qvals[i],
                        'Abeta_SUVR:Tau_status': interaction_qvals[i]
                    }

   # =====================
    # 3. 跨特征FDR校正
    # =====================
    # 对Aβ主效应进行跨特征FDR校正
    if cross_feature_abeta_pvals:
        _, cross_abeta_qvals, _, _ = multipletests(cross_feature_abeta_pvals, method='fdr_bh')

    if cross_feature_tau_pvals:
        _, cross_tau_qvals, _, _ = multipletests(cross_feature_tau_pvals, method='fdr_bh')
        
        # 将结果存回每个ROI
        for i, (feat_name, roi) in enumerate(cross_feature_info):
            if roi_results[feat_name][roi] is not None:
                if 'qvalues_cross' not in roi_results[feat_name][roi]:
                    roi_results[feat_name][roi]['qvalues_cross'] = {}
                roi_results[feat_name][roi]['qvalues_cross']['Abeta_SUVR'] = cross_abeta_qvals[i]
                roi_results[feat_name][roi]['qvalues_cross']['Tau_status'] = cross_tau_qvals[i]
    
    # 对交互效应进行跨特征FDR校正
    if cross_feature_interaction_pvals:
        _, cross_interaction_qvals, _, _ = multipletests(cross_feature_interaction_pvals, method='fdr_bh')
        
    if cross_feature_interaction_pvals:
        _, cross_interaction_qvals, _, _ = multipletests(cross_feature_interaction_pvals, method='fdr_bh')        
        # 将结果存回每个ROI
        for i, (feat_name, roi) in enumerate(cross_feature_info):
            if roi_results[feat_name][roi] is not None:
                if 'qvalues_cross' not in roi_results[feat_name][roi]:
                    roi_results[feat_name][roi]['qvalues_cross'] = {}
                roi_results[feat_name][roi]['qvalues_cross']['Abeta_SUVR:Tau_status'] = cross_interaction_qvals[i]
                

    # =====================
    # 3. 结果可视化
    # =====================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        overall_stats = compute_overall_stats(roi_results, tau_status)
        
        # 新增ROI回归线可视化
        #plot_combined_roi_regression(df, roi_results, overall_stats, output_dir)

        #plot_abeta_main_effect(df, roi_results, overall_stats, output_dir)

        # plot_tau_main_effect(df, roi_results, output_dir)
        
    
    return roi_results

def compute_overall_stats(roi_results, tau_status):
    """
    计算整体的SE和FDR校正的p值，包括Tau阴性和阳性组的整体统计
    
    参数:
    roi_results: ROI级别分析结果
    
    返回:
    overall_stats: 包含整体统计信息的字典，包括分组统计
    """
    # 初始化统计数据结构
    overall_stats = {
        'rTheta': {
            'Abeta_main': {},
            'Interaction': {},
            'Tau_negative': {},
            'Tau_positive': {}
        },
        'rBeta': {
            'Abeta_main': {},
            'Interaction': {},
            'Tau_negative': {},
            'Tau_positive': {}
        }
    }
    
    # 定义效应类型
    effect_types = ['Abeta_main', 'Interaction', 'Tau_negative', 'Tau_positive']
    
    for feat_name in ['rTheta', 'rBeta']:
        # 初始化列表用于收集β值和p值
        tau_negative_betas = []
        tau_positive_betas = []
        abeta_betas = []
        interaction_betas = []
        
        tau_negative_pvals = []
        tau_positive_pvals = []
        abeta_pvals = []
        interaction_pvals = []
        
        # 遍历所有ROI
        for roi in range(68):
            result = roi_results[feat_name].get(roi)
            if not result:
                continue
            
            # 收集Aβ主效应和交互效应统计
            abeta_beta = result['params'].get('Abeta_SUVR', 0)
            abeta_pval = result['pvalues'].get('Abeta_SUVR', 1)
            interaction_beta = result['params'].get('Abeta_SUVR:Tau_status', 0)
            interaction_pval = result['pvalues'].get('Abeta_SUVR:Tau_status', 1)
            
            abeta_betas.append(abeta_beta)
            abeta_pvals.append(abeta_pval)
            interaction_betas.append(interaction_beta)
            interaction_pvals.append(interaction_pval)
            
            # 根据实际tau_status值分类          
            if tau_status[roi] == 0:  # Tau阴性组
                tau_negative_betas.append(abeta_beta)
                tau_negative_pvals.append(abeta_pval)
            elif tau_status[roi] == 1:  # Tau阳性组
                tau_positive_betas.append(abeta_beta)
                tau_positive_pvals.append(abeta_pval)
    
        # 计算整体统计量
        def calculate_group_stats(betas, pvals, group_name):
            """计算分组统计量"""
            # 计算整体β值（平均值）
            overall_beta = np.mean(betas) if betas else 0
            
            # 计算整体SE（标准误）
            overall_se = np.std(betas, ddof=1) / np.sqrt(len(betas)) if betas else 0
            
            # 计算整体p值（使用Fisher方法合并p值）
            if pvals:
                _, combined_pval = stats.combine_pvalues(pvals, method='fisher')
            else:
                combined_pval = 1
                
            return overall_beta, overall_se, combined_pval
        
        # Aβ主效应整体统计
        abeta_beta, abeta_se, abeta_pval = calculate_group_stats(abeta_betas, abeta_pvals, 'Abeta_main')
        overall_stats[feat_name]['Abeta_main'] = {
            'beta': abeta_beta,
            'se': abeta_se,
            'pval': abeta_pval
        }
        
        # 交互效应整体统计
        interaction_beta, interaction_se, interaction_pval = calculate_group_stats(interaction_betas, interaction_pvals, 'Interaction')
        overall_stats[feat_name]['Interaction'] = {
            'beta': interaction_beta,
            'se': interaction_se,
            'pval': interaction_pval
        }
        
        # Tau阴性组整体统计
        tau_neg_beta, tau_neg_se, tau_neg_pval = calculate_group_stats(tau_negative_betas, tau_negative_pvals, 'Tau_negative')
        overall_stats[feat_name]['Tau_negative'] = {
            'beta': tau_neg_beta,
            'se': tau_neg_se,
            'pval': tau_neg_pval
        }
        
        # Tau阳性组整体统计
        tau_pos_beta, tau_pos_se, tau_pos_pval = calculate_group_stats(tau_positive_betas, tau_positive_pvals, 'Tau_positive')
        overall_stats[feat_name]['Tau_positive'] = {
            'beta': tau_pos_beta,
            'se': tau_pos_se,
            'pval': tau_pos_pval
        }
    
    # 对整体p值进行FDR校正（跨特征和效应类型）
    all_pvals = []
    effect_info = []
    
    for feat_name in overall_stats:
        for effect_type in effect_types:
            pval = overall_stats[feat_name][effect_type]['pval']
            all_pvals.append(pval)
            effect_info.append((feat_name, effect_type))
    
    if all_pvals:
        _, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')
        
        # 将q值存回
        for i, (feat_name, effect_type) in enumerate(effect_info):
            overall_stats[feat_name][effect_type]['qval'] = qvals[i]
    
    return overall_stats
        

if __name__ == "__main__":
  
    # 运行分析
    output_dir = ' '
    roi_results = run_lme_analysis(
        age=age,
        gender=gender,
        tau_status=tau_status,
        suvr_values=suvr_values,
        features_dict=features_dict,
        output_dir=output_dir
    )
    # 打印分析完成信息
    print("\n" + "="*80)
    print(f"分析完成! 结果保存在: {output_dir}")
    print(f"共分析ROI数量: {len(roi_results['rTheta'])}个theta功率 ROI")
    print(f"共分析ROI数量: {len(roi_results['rBeta'])}个beta功率 ROI")
    print("="*80)
    
    # 打印详细的统计结果
    print("\n" + "="*80)
    print("详细统计结果 (每个特征选取前5个ROI作为示例):")
    print("="*80)
    
    # 格式化函数 - 处理极小的p值
    def format_pvalue(pval):
        if pval < 0.001:
            return "<0.001"
        return f"{pval:.6f}"
    
    # 统计结果表头
    print("\n{:<6} {:<20} {:<10} {:<12} {:<12} {:<12} {:<12}".format(
        "ROI", "特征", "效应类型", "β值", "原始p值", "特征内q值", "跨特征q值"
    ))
    print("-" * 95)
    
    # 选取每个特征的前5个ROI打印结果
    for feat_name in ['rTheta', 'rBeta']:
    # 'alpha band relative power', 'delta band relative power','theta band functional connectivity']:
        # 打印Aβ主效应
        for roi in range(68): 
            result = roi_results[feat_name].get(roi)
            if result:
                # Aβ主效应
                print("{:<6} {:<20} {:<10} {:<12.4f} {:<12} {:<12} {:<12}".format(
                    roi, feat_name, "Aβ主效应",
                    result['params'].get('Abeta_SUVR', 0),
                    format_pvalue(result['pvalues'].get('Abeta_SUVR', 1)),
                    format_pvalue(result.get('qvalues_within', {}).get('Abeta_SUVR', 1)),
                    format_pvalue(result.get('qvalues_cross', {}).get('Abeta_SUVR', 1))
                ))

                            # Tau主效应
                tau_coef = result['params'].get('Tau_status', 0)
                tau_pval = result['pvalues'].get('Tau_status', 1)
                tau_q_within = result.get('qvalues_within', {}).get('Tau_status', 1)
                tau_q_cross = result.get('qvalues_cross', {}).get('Tau_status', 1)
                
                print("{:<6} {:<15} {:<12.4f} {:<12} {:<12} {:<12}".format(
                    roi, "Tau主效应",
                    tau_coef,
                    format_pvalue(tau_pval),
                    format_pvalue(tau_q_within),
                    format_pvalue(tau_q_cross)
                ))
            
                # Aβ×Tau交互效应
                print("{:<6} {:<20} {:<10} {:<12.4f} {:<12} {:<12} {:<12}".format(
                    roi, feat_name, "交互效应",
                    result['params'].get('Abeta_SUVR:Tau_status', 0),
                    format_pvalue(result['pvalues'].get('Abeta_SUVR:Tau_status', 1)),
                    format_pvalue(result.get('qvalues_within', {}).get('Abeta_SUVR:Tau_status', 1)),
                    format_pvalue(result.get('qvalues_cross', {}).get('Abeta_SUVR:Tau_status', 1))
                ))
                print("-" * 95)
    
    print("\n完整结果可通过查看保存的图像获取")    
    print("分析完成，结果保存在:", output_dir)

def run_lme_7network_analysis(age, gender, tau_status, suvr_values, features_dict, output_dir=None):
    """
    AD病理与EEG功能关联的线性混合效应模型分析（含可视化）
    
    参数:
    age: 年龄列表 (n_subjects,)
    gender: 性别列表 ('男'或'女') (n_subjects,)
    tau_status: tau-PET状态 (0=阴性, 1=阳性) (n_subjects,)
    suvr_values: Aβ-PET SUVR值 (n_subjects,)
    features_dict: 字典，包含以下键值对:
        'source_pt': (n_subjects,68) theta功率谱密度
        'source_pb': (n_subjects,68) beta功率谱密度
    output_dir: 结果保存目录 (可选)
    
    返回:
    results: 包含所有分析结果的字典
    roi_results: ROI级别的回归结果
    """
    n_subjects = len(age)
    n_networks = 14
    
    # =====================
    # 1. 数据预处理
    # =====================
    # 转换性别为数值
    gender_numeric = [0 if g == '男' else 1 for g in gender]
    
    # Aβ标准化
    abeta_z = (suvr_values - np.mean(suvr_values)) / np.std(suvr_values)

    # 创建长格式数据框
    data_list = []
    for subj_idx in range(n_subjects):
        for nw_idx in range(n_networks):
            row = {
                'SubjectID': f'Subj_{subj_idx}',
                'Network': nw_idx,
                'Abeta_SUVR': abeta_z[subj_idx],
                'Tau_status': tau_status[subj_idx],
                'Age': age[subj_idx],
                'Gender': gender_numeric[subj_idx],
                'theta_power': features_dict['net7_rpt'][subj_idx, nw_idx],
                'beta_power': features_dict['net7_rpb'][subj_idx, nw_idx],
                # 'alpha_power': features_dict['net7_rpa'][subj_idx, nw_idx],
                # 'delta_power': features_dict['net7_rpd'][subj_idx, nw_idx],
            }
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # 转换分类变量
    df['SubjectID'] = df['SubjectID'].astype('category')
    df['Network'] = df['Network'].astype('category')
    df['Tau_status'] = df['Tau_status'].astype('category')
    df['Gender'] = df['Gender'].astype('category')
    
    # =====================
    # 2. LME模型构建与分析
    # =====================
    # results = {}
    eeg_features = {
        'theta band relative power': 'theta_power',
        'beta band relative power': 'beta_power',
        # 'alpha band relative power': 'alpha_power',
        # 'delta band relative power': 'delta_power'
    }

    # Network级别结果存储
    network_results = {
        'theta band relative power': {nw: None for nw in range(n_networks)},
        'beta band relative power': {nw: None for nw in range(n_networks)},
        # 'alpha band relative power': {nw: None for nw in range(n_networks)},
        # 'delta band relative power': {nw: None for nw in range(n_networks)}
    }

    # 为跨特征校正收集数据
    cross_feature_abeta_pvals = []
    cross_feature_interaction_pvals = []
    cross_feature_info = []  # 存储(特征名, roi编号)

    for feat_name, feat_col in eeg_features.items():
        # 初始化列表用于存储p值，以便进行跨特征FDR校正
        all_abeta_pvals = []
        all_interaction_pvals = []
        # feature_names = []
        
        for network in range(n_networks):
            network_df = df[df['Network'] == network].copy()
            
            # 1. 创建公式 - 包含随机截距（每个被试）
            formula = f"{feat_col} ~ Abeta_SUVR * Tau_status + Age + C(Gender)"
            
            # 2. 使用混合效应模型（而非普通OLS）
            model = smf.mixedlm(formula, network_df, groups=network_df['SubjectID'])
            
            # 3. 尝试多种优化方法（解决可能的收敛问题）
            result = None
            methods = ['bfgs', 'lbfgs', 'cg', 'nm']  # 多种优化方法
            
            for method in methods:
                try:
                    result = model.fit(method=method)
                    if result.converged:
                        print(f"Network {network} - {feat_name}: 使用 {method} 方法收敛成功")
                        break
                except Exception as e:
                    print(f"Network {network} - {feat_name}: 方法 {method} 失败: {str(e)}")
                    continue
            
            # 4. 如果主要方法都失败，尝试REML
            if result is None or not result.converged:
                try:
                    print(f"network {network} - {feat_name}: 尝试REML方法")
                    result = model.fit(reml=True)
                except:
                    # 最终回退方案
                    result = model.fit()
            
            # 5. 存储结果
            network_result = {
                'model': result,
                'summary': result.summary(),
                'params': result.fe_params,
                'pvalues': result.pvalues,
                # 'rsquared': result.rsquared,
                'converged': result.converged
            }
            
            network_results[feat_name][network] = network_result
            
            # 提取相关p值用于FDR校正
            abeta_pval = result.pvalues.get('Abeta_SUVR', 1)
            interaction_pval = result.pvalues.get('Abeta_SUVR:Tau_status', 1)
            
            # 特征内校正的收集
            all_abeta_pvals.append(abeta_pval)
            all_interaction_pvals.append(interaction_pval)
            
            # 跨特征校正的收集
            cross_feature_abeta_pvals.append(abeta_pval)
            cross_feature_interaction_pvals.append(interaction_pval)
            cross_feature_info.append((feat_name, network))
        
        # 进行ROI级别（特征内部）FDR校正
        if all_abeta_pvals:
            _, abeta_qvals, _, _ = multipletests(all_abeta_pvals, method='fdr_bh')
            _, interaction_qvals, _, _ = multipletests(all_interaction_pvals, method='fdr_bh')
            
            for i, network in enumerate(range(n_networks)):
                if network_results[feat_name][network] is not None:
                    network_results[feat_name][network]['qvalues_within'] = {
                        'Abeta_SUVR': abeta_qvals[i],
                        'Abeta_SUVR:Tau_status': interaction_qvals[i]
                    }

   # =====================
    # 3. 跨特征FDR校正
    # =====================
    # 对Aβ主效应进行跨特征FDR校正
    if cross_feature_abeta_pvals:
        _, cross_abeta_qvals, _, _ = multipletests(cross_feature_abeta_pvals, method='fdr_bh')
        
        # 将结果存回每个ROI
        for i, (feat_name, network) in enumerate(cross_feature_info):
            if network_results[feat_name][network] is not None:
                if 'qvalues_cross' not in network_results[feat_name][network]:
                    network_results[feat_name][network]['qvalues_cross'] = {}
                network_results[feat_name][network]['qvalues_cross']['Abeta_SUVR'] = cross_abeta_qvals[i]
    
    # 对交互效应进行跨特征FDR校正
    if cross_feature_interaction_pvals:
        _, cross_interaction_qvals, _, _ = multipletests(cross_feature_interaction_pvals, method='fdr_bh')
        
        # 将结果存回每个ROI
        for i, (feat_name, network) in enumerate(cross_feature_info):
            if network_results[feat_name][network] is not None:
                if 'qvalues_cross' not in network_results[feat_name][network]:
                    network_results[feat_name][network]['qvalues_cross'] = {}
                network_results[feat_name][network]['qvalues_cross']['Abeta_SUVR:Tau_status'] = cross_interaction_qvals[i]
                

    # =====================
    # 3. 结果可视化
    # =====================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # overall_stats = compute_overall_stats(roi_results, tau_status)
        
        # # 新增ROI回归线可视化
        # plot_combined_roi_regression(df, roi_results, overall_stats, output_dir)

        # plot_abeta_main_effect(df, roi_results, overall_stats, output_dir)
    
    return network_results
    
if __name__ == "__main__":
    
    """测试LME分析和Network回归可视化"""
  
    # 运行分析
    output_dir = 'lme_results/network_results'
    network_results = run_lme_7network_analysis(
        age=age,
        gender=gender,
        tau_status=tau_status,
        suvr_values=suvr_values,
        features_dict=feat_7network_dict,
        output_dir=output_dir
    )
    # 打印分析完成信息
    print("\n" + "="*80)
    print(f"分析完成! 结果保存在: {output_dir}")
    print(f"共分析network数量: {len(network_results['theta band relative power'])}个theta功率 network")
    print(f"共分析network数量: {len(network_results['beta band relative power'])}个beta功率 network")
    # print(f"共分析network数量: {len(network_results['alpha band relative power'])}个beta功率 network")
    # print(f"共分析network数量: {len(network_results['delta band relative power'])}个beta功率 network")
    print("="*80)
    
    # 打印详细的统计结果
    print("\n" + "="*80)
    print("详细统计结果 (每个特征选取前5个ROI作为示例):")
    print("="*80)
    
    # 格式化函数 - 处理极小的p值
    def format_pvalue(pval):
        if pval < 0.001:
            return "<0.001"
        return f"{pval:.6f}"
    
    # 统计结果表头
    print("\n{:<6} {:<20} {:<10} {:<12} {:<12} {:<12} {:<12}".format(
        "ROI", "特征", "效应类型", "β值", "原始p值", "特征内q值", "跨特征q值"
    ))
    print("-" * 95)
    
    # 选取每个特征的前5个ROI打印结果
    for feat_name in ['theta band relative power', 'beta band relative power']: #, 'alpha band relative power', 'delta band relative power']:
        # 打印Aβ主效应
        for network in range(14):
            result = network_results[feat_name].get(network)
            if result:
                # Aβ主效应
                print("{:<6} {:<20} {:<10} {:<12.4f} {:<12} {:<12} {:<12}".format(
                    network, feat_name, "Aβ主效应",
                    result['params'].get('Abeta_SUVR', 0),
                    format_pvalue(result['pvalues'].get('Abeta_SUVR', 1)),
                    format_pvalue(result.get('qvalues_within', {}).get('Abeta_SUVR', 1)),
                    format_pvalue(result.get('qvalues_cross', {}).get('Abeta_SUVR', 1))
                ))
                
                # Aβ×Tau交互效应
                print("{:<6} {:<20} {:<10} {:<12.4f} {:<12} {:<12} {:<12}".format(
                    network, feat_name, "交互效应",
                    result['params'].get('Abeta_SUVR:Tau_status', 0),
                    format_pvalue(result['pvalues'].get('Abeta_SUVR:Tau_status', 1)),
                    format_pvalue(result.get('qvalues_within', {}).get('Abeta_SUVR:Tau_status', 1)),
                    format_pvalue(result.get('qvalues_cross', {}).get('Abeta_SUVR:Tau_status', 1))
                ))
                print("-" * 95)
    
    print("\n完整结果可通过查看保存的图像获取")    
    print("分析完成，结果保存在:", output_dir)
