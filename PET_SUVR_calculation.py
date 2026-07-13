import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 目录扫描和subject ID自动获取函数
# ============================================================================

def scan_subject_ids_from_root(pet_data_root, t1_data_root, dkt_root, 
                              min_pet_files=1, min_t1_files=1):
    """
    从根目录自动扫描可用的subject ID
    
    参数:
        pet_data_root: PET数据根目录
        t1_data_root: T1数据根目录
        dkt_root: DKT图谱根目录
        min_pet_files: PET目录最小DICOM文件数
        min_t1_files: T1目录最小DICOM文件数
    
    返回:
        available_subjects: 可用的subject ID列表
        subject_details: 每个subject的详细信息字典
    """
    print("=" * 80)
    print("自动扫描目录结构获取subject ID")
    print("=" * 80)
    
    pet_data_root = Path(pet_data_root)
    t1_data_root = Path(t1_data_root)
    dkt_root = Path(dkt_root)
    
    # 检查根目录是否存在
    roots_exist = []
    if pet_data_root.exists():
        print(f"✓ PET根目录存在: {pet_data_root}")
        roots_exist.append(True)
    else:
        print(f"✗ PET根目录不存在: {pet_data_root}")
        roots_exist.append(False)
    
    if t1_data_root.exists():
        print(f"✓ T1根目录存在: {t1_data_root}")
        roots_exist.append(True)
    else:
        print(f"✗ T1根目录存在: {t1_data_root}")
        roots_exist.append(False)
    
    if dkt_root.exists():
        print(f"✓ DKT根目录存在: {dkt_root}")
        roots_exist.append(True)
    else:
        print(f"✗ DKT根目录不存在: {dkt_root}")
        roots_exist.append(False)
    
    if not all(roots_exist):
        print("错误: 有根目录不存在，无法继续")
        return [], {}
    
    # 获取所有潜在的subject目录
    print(f"\n扫描目录结构...")
    
    # 从DKT根目录获取所有潜在的subject ID
    dkt_subjects = {}
    for item in dkt_root.iterdir():
        if item.is_dir():
            subject_id = item.name
            dkt_path = item / "mri" / "aparc.DKTatlas+aseg.mapped.mgz"
            if dkt_path.exists():
                dkt_subjects[subject_id] = {
                    'subject_id': subject_id,
                    'dkt_path': str(dkt_path),
                    'has_dkt': True
                }
            else:
                # 检查是否有其他DKT文件
                dkt_files = list((item / "mri").glob("*DKT*"))
                if dkt_files:
                    dkt_subjects[subject_id] = {
                        'subject_id': subject_id,
                        'dkt_path': str(dkt_files[0]),
                        'has_dkt': True
                    }
    
    print(f"在DKT根目录中找到 {len(dkt_subjects)} 个潜在subject")
    
    # 检查每个subject是否在PET和T1目录中存在
    available_subjects = []
    subject_details = {}
    
    for subject_id, dkt_info in dkt_subjects.items():
        pet_dir = pet_data_root / subject_id / "PET_Head_OSEM"
        t1_dir = t1_data_root / subject_id / "t1_gre_fsp_3d_sag_iso_ACS"
        
        details = {
            'subject_id': subject_id,
            'dkt_path': dkt_info['dkt_path'],
            'has_dkt': dkt_info['has_dkt'],
            'pet_dir': str(pet_dir),
            't1_dir': str(t1_dir),
            'has_pet_dir': pet_dir.exists(),
            'has_t1_dir': t1_dir.exists(),
            'pet_dicom_count': 0,
            't1_dicom_count': 0,
            'is_complete': False
        }
        
        # 检查PET目录
        if pet_dir.exists():
            pet_dicom_files = list(pet_dir.glob("*.dcm"))
            details['pet_dicom_count'] = len(pet_dicom_files)
            details['has_pet_dicom'] = len(pet_dicom_files) >= min_pet_files
        else:
            details['has_pet_dicom'] = False
        
        # 检查T1目录
        if t1_dir.exists():
            t1_dicom_files = list(t1_dir.glob("*.dcm"))
            details['t1_dicom_count'] = len(t1_dicom_files)
            details['has_t1_dicom'] = len(t1_dicom_files) >= min_t1_files
        else:
            details['has_t1_dicom'] = False
        
        # 判断是否完整
        details['is_complete'] = (
            details['has_dkt'] and 
            details['has_pet_dir'] and 
            details['has_t1_dir'] and
            details['has_pet_dicom'] and
            details['has_t1_dicom']
        )
        
        subject_details[subject_id] = details
        
        if details['is_complete']:
            available_subjects.append(subject_id)
    
    # 打印统计信息
    print(f"\n扫描完成!")
    print(f"总潜在subject数: {len(dkt_subjects)}")
    print(f"完整subject数: {len(available_subjects)}")
    
    if available_subjects:
        print(f"\n完整subject列表:")
        for i, subj_id in enumerate(available_subjects[:10]):  # 只显示前10个
            details = subject_details[subj_id]
            print(f"  {i+1:3d}. {subj_id} (PET: {details['pet_dicom_count']}文件, "
                  f"T1: {details['t1_dicom_count']}文件)")
        
        if len(available_subjects) > 10:
            print(f"  ... 和另外 {len(available_subjects) - 10} 个完整subject")
    
    # 检查不完整的subject
    incomplete_subjects = []
    for subj_id, details in subject_details.items():
        if not details['is_complete']:
            incomplete_subjects.append(subj_id)
    
    if incomplete_subjects:
        print(f"\n不完整的subject ({len(incomplete_subjects)}个):")
        for i, subj_id in enumerate(incomplete_subjects[:5]):  # 只显示前5个
            details = subject_details[subj_id]
            issues = []
            if not details['has_dkt']: issues.append("缺少DKT")
            if not details['has_pet_dir']: issues.append("缺少PET目录")
            elif not details['has_pet_dicom']: issues.append(f"PET文件不足({details['pet_dicom_count']})")
            if not details['has_t1_dir']: issues.append("缺少T1目录")
            elif not details['has_t1_dicom']: issues.append(f"T1文件不足({details['t1_dicom_count']})")
            
            print(f"  {subj_id}: {', '.join(issues)}")
        
        if len(incomplete_subjects) > 5:
            print(f"  ... 和另外 {len(incomplete_subjects) - 5} 个不完整subject")
    
    return available_subjects, subject_details

def scan_single_root_for_subjects(root_dir, data_type="PET"):
    """
    从单个根目录扫描subject ID
    
    参数:
        root_dir: 根目录路径
        data_type: 数据类型 ("PET", "T1", 或 "DKT")
    
    返回:
        subject_ids: 找到的subject ID列表
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        print(f"警告: 根目录不存在 {root_dir}")
        return []
    
    print(f"扫描{data_type}根目录: {root_dir}")
    
    subject_ids = []
    
    for item in root_dir.iterdir():
        if item.is_dir():
            subject_id = item.name
            
            # 根据数据类型检查目录结构
            if data_type == "PET":
                pet_dir = item / "PET_Head_OSEM"
                if pet_dir.exists():
                    dicom_files = list(pet_dir.glob("*.dcm"))
                    if dicom_files:
                        subject_ids.append(subject_id)
            
            elif data_type == "T1":
                t1_dir = item / "t1_gre_fsp_3d_sag_iso_ACS"
                if t1_dir.exists():
                    dicom_files = list(t1_dir.glob("*.dcm"))
                    if dicom_files:
                        subject_ids.append(subject_id)
            
            elif data_type == "DKT":
                dkt_path = item / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
                if dkt_path.exists():
                    subject_ids.append(subject_id)
                else:
                    # 检查是否有其他DKT文件
                    dkt_files = list((item / "mri").glob("*DKT*"))
                    if dkt_files:
                        subject_ids.append(subject_id)
    
    print(f"  在{data_type}根目录中找到 {len(subject_ids)} 个subject")
    
    return subject_ids

# ============================================================================
# 2. 图像加载和配准函数
# ============================================================================

def load_dicom_series(dicom_dir):
    """加载DICOM系列并转换为SimpleITK图像"""
    if not Path(dicom_dir).exists():
        raise FileNotFoundError(f"DICOM目录不存在: {dicom_dir}")
    
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        
        if len(dicom_files) == 0:
            raise FileNotFoundError(f"在{dicom_dir}中未找到DICOM文件")
        
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        print(f"  成功加载DICOM图像，尺寸: {image.GetSize()}")
        return image
    except Exception as e:
        raise Exception(f"加载DICOM失败: {e}")

def check_and_convert_image_types(pet_image, t1_image):
    """检查并转换图像数据类型以确保匹配"""
    # 获取图像数据类型
    pet_pixel_id = pet_image.GetPixelID()
    t1_pixel_id = t1_image.GetPixelID()
    
    pet_type = pet_image.GetPixelIDTypeAsString()
    t1_type = t1_image.GetPixelIDTypeAsString()
    
    print(f"  PET图像数据类型: {pet_type} (ID: {pet_pixel_id})")
    print(f"  T1图像数据类型: {t1_type} (ID: {t1_pixel_id})")
    
    # 检查数据类型是否匹配
    if pet_pixel_id == t1_pixel_id:
        print("  ✓ 图像数据类型匹配，无需转换")
        return pet_image, t1_image, False
    else:
        print("  ⚠ 图像数据类型不匹配，需要进行转换")
        
        # 选择目标数据类型（通常选择浮点型以获得最佳精度）
        target_type = sitk.sitkFloat32
        target_type_name = sitk.GetPixelIDValueAsString(target_type)
        
        print(f"  将图像转换为统一数据类型: {target_type_name}")
        
        # 转换图像数据类型
        pet_converted = sitk.Cast(pet_image, target_type)
        t1_converted = sitk.Cast(t1_image, target_type)
        
        print("  ✓ 图像数据类型转换完成")
        return pet_converted, t1_converted, True

def normalize_image_intensities(image, method="zscore"):
    """标准化图像强度值，提高配准稳定性"""
    print(f"  应用图像强度标准化: {method}")
    
    image_array = sitk.GetArrayFromImage(image)
    
    if method == "zscore":
        # Z-score标准化
        mean_val = np.mean(image_array)
        std_val = np.std(image_array)
        if std_val > 0:
            normalized_array = (image_array - mean_val) / std_val
        else:
            normalized_array = image_array - mean_val
    elif method == "minmax":
        # 最小-最大标准化到[0,1]范围
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        if max_val > min_val:
            normalized_array = (image_array - min_val) / (max_val - min_val)
        else:
            normalized_array = image_array
    else:
        # 不进行标准化
        normalized_array = image_array
    
    # 创建新图像
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)  # 复制空间信息
    
    return normalized_image

def check_image_compatibility(pet_image, t1_image):
    """检查PET和T1图像的兼容性"""
    pet_size = pet_image.GetSize()
    pet_spacing = pet_image.GetSpacing()
    t1_size = t1_image.GetSize() 
    t1_spacing = t1_image.GetSpacing()
    
    print(f"  PET图像 - 尺寸: {pet_size}, 间距: {pet_spacing}")
    print(f"  T1图像 - 尺寸: {t1_size}, 间距: {t1_spacing}")
    
    # 检查物理尺寸是否相似（考虑间距）
    pet_physical = [pet_size[i] * pet_spacing[i] for i in range(3)]
    t1_physical = [t1_size[i] * t1_spacing[i] for i in range(3)]
    
    print(f"  PET物理尺寸: {[f'{x:.1f}mm' for x in pet_physical]}")
    print(f"  T1物理尺寸: {[f'{x:.1f}mm' for x in t1_physical]}")
    
    # 检查是否需要重采样
    spacing_ratio = [t1_spacing[i] / pet_spacing[i] for i in range(3)]
    print(f"  间距比例(T1/PET): {spacing_ratio}")
    
    return pet_spacing, t1_spacing, spacing_ratio

def pet_to_t1_registration(pet_image, t1_image):
    """改进的PET到T1配准函数，包含多分辨率参数和进度监控"""
    print("  开始PET到T1配准（多分辨率策略）...")

    # 1. 检查并转换图像数据类型
    pet_converted, t1_converted, was_converted = check_and_convert_image_types(pet_image, t1_image)
    
    # 2. 可选：标准化图像强度（提高配准稳定性）
    use_intensity_normalization = True
    if use_intensity_normalization:
        print("  应用图像强度标准化...")
        pet_normalized = normalize_image_intensities(pet_converted, "zscore")
        t1_normalized = normalize_image_intensities(t1_converted, "zscore")
    else:
        pet_normalized = pet_converted
        t1_normalized = t1_converted
    
    # 创建配准器
    registration_method = sitk.ImageRegistrationMethod()
    
    # 1. 设置多分辨率参数（金字塔策略）
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # 2. 设置配准度量参数
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # 3. 设置优化器参数
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # 4. 简化进度监控
    iteration_count = [0]
    def command_iteration(method):
        iteration_count[0] += 1
        if iteration_count[0] == 1 or iteration_count[0] % 30 == 0:
            current_iteration = method.GetOptimizerIteration()
            current_value = method.GetMetricValue()
            print(f"    迭代 {current_iteration:3d}: 度量值 = {current_value:.6f}")
    
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    
    # 5. 执行配准
    print("  执行多分辨率配准...")
    transform = registration_method.Execute(t1_converted, pet_converted)
    
    # 输出最终结果
    final_iteration = registration_method.GetOptimizerIteration()
    final_value = registration_method.GetMetricValue()
    print(f"  配准完成! 最终迭代: {final_iteration}, 最终度量值: {final_value:.6f}")
    
    # 6. 应用变换
    print("  应用配准变换...")
    registered_pet = sitk.Resample(
        pet_image, 
        t1_image, 
        transform, 
        sitk.sitkLinear, 
        0.0, 
        pet_image.GetPixelID()
    )
    
    print("  ✓ 配准完成!")
    return registered_pet, transform

def improved_pet_to_t1_registration(pet_image, t1_image):
    """改进的PET到T1配准，处理分辨率不一致问题"""
    print("  === 开始改进的PET到T1配准 ===")
    
    # 1. 检查图像兼容性
    pet_spacing, t1_spacing, spacing_ratio = check_image_compatibility(pet_image, t1_image)
    
    # 2. 如果分辨率差异太大，先进行重采样
    working_pet = pet_image
    if max(spacing_ratio) > 2.0 or min(spacing_ratio) < 0.5:
        print("  分辨率差异较大，先进行重采样...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t1_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        working_pet = resampler.Execute(pet_image)
        print(f"  重采样后PET尺寸: {working_pet.GetSize()}")
    else:
        print("  分辨率差异在可接受范围内，直接进行配准")
    
    # 3. 执行改进的配准（包含多分辨率和进度监控）
    registered_pet, transform = pet_to_t1_registration(working_pet, t1_image)
    
    return registered_pet, transform

# ============================================================================
# 2. SUVR计算函数
# ============================================================================

def get_fastsurfer_label_mapping():
    """
    FreeSurfer 原版 Desikan-Killiany 68 ROI 完整ID→名称映射（每侧34，双侧68）
    """
    dk68_labels = {
        # 扣带回系列
        1001: "ctx-lh-bankssts",
        2001: "ctx-rh-bankssts",
        1003: "ctx-lh-caudalanteriorcingulate",
        2003: "ctx-rh-caudalanteriorcingulate",
        1004: "ctx-lh-caudalmiddlefrontal",  # DK原版用1004，DKT改为1005
        2004: "ctx-rh-caudalmiddlefrontal",
        1024: "ctx-lh-posteriorcingulate",
        2024: "ctx-rh-posteriorcingulate",
        1027: "ctx-lh-rostralanteriorcingulate",
        2027: "ctx-rh-rostralanteriorcingulate",
        1011: "ctx-lh-isthmuscingulate",
        2011: "ctx-rh-isthmuscingulate",

        # 枕叶
        1006: "ctx-lh-cuneus",
        2006: "ctx-rh-cuneus",
        1012: "ctx-lh-lateraloccipital",
        2012: "ctx-rh-lateraloccipital",
        1014: "ctx-lh-lingual",
        2014: "ctx-rh-lingual",
        1022: "ctx-lh-pericalcarine",
        2022: "ctx-rh-pericalcarine",

        # 颞叶
        1007: "ctx-lh-entorhinal",
        2007: "ctx-rh-entorhinal",
        1008: "ctx-lh-fusiform",
        2008: "ctx-rh-fusiform",
        1010: "ctx-lh-inferiortemporal",
        2010: "ctx-rh-inferiortemporal",
        1016: "ctx-lh-middletemporal",
        2016: "ctx-rh-middletemporal",
        1017: "ctx-lh-parahippocampal",
        2017: "ctx-rh-parahippocampal",
        1031: "ctx-lh-superiortemporal",
        2031: "ctx-rh-superiortemporal",
        1034: "ctx-lh-transversetemporal",  # DK原版1034，DKT改为1035
        2034: "ctx-rh-transversetemporal",

        # 顶叶
        1009: "ctx-lh-inferiorparietal",
        2009: "ctx-rh-inferiorparietal",
        1018: "ctx-lh-paracentral",
        2018: "ctx-rh-paracentral",
        1023: "ctx-lh-postcentral",
        2023: "ctx-rh-postcentral",
        1026: "ctx-lh-precuneus",
        2026: "ctx-rh-precuneus",
        1030: "ctx-lh-superiorparietal",
        2030: "ctx-rh-superiorparietal",
        1032: "ctx-lh-supramarginal",
        2032: "ctx-rh-supramarginal",

        # 额叶
        1013: "ctx-lh-lateralorbitofrontal",
        2013: "ctx-rh-lateralorbitofrontal",
        1015: "ctx-lh-medialorbitofrontal",
        2015: "ctx-rh-medialorbitofrontal",
        1019: "ctx-lh-parsopercularis",
        2019: "ctx-rh-parsopercularis",
        1020: "ctx-lh-parsorbitalis",
        2020: "ctx-rh-parsorbitalis",
        1021: "ctx-lh-parstriangularis",
        2021: "ctx-rh-parstriangularis",
        1025: "ctx-lh-precentral",
        2025: "ctx-rh-precentral",
        1028: "ctx-lh-rostralmiddlefrontal",
        2028: "ctx-rh-rostralmiddlefrontal",
        1029: "ctx-lh-superiorfrontal",
        2029: "ctx-rh-superiorfrontal",

        # 脑岛（DK/DKT共用ID）
        1036: "ctx-lh-insula",
        2036: "ctx-rh-insula",
    }
    
    # 反转字典：从标签名到标签编号
    name_to_label = {v: k for k, v in dkt_standard_labels.items()}
    
    # 转换fastsurfer标签名到标准DKT标签名
    label_mapping = {}
    missing_labels = []
    
    for fastsurfer_label in FASTSURFER_LABELS:
        if "-lh" in fastsurfer_label:
            base_name = fastsurfer_label.replace("-lh", "")
            standard_name = f"ctx-lh-{base_name}"
        elif "-rh" in fastsurfer_label:
            base_name = fastsurfer_label.replace("-rh", "")
            standard_name = f"ctx-rh-{base_name}"
        else:
            missing_labels.append(fastsurfer_label)
            continue
        
        if standard_name in name_to_label:
            label_mapping[fastsurfer_label] = name_to_label[standard_name]
        else:
            missing_labels.append(fastsurfer_label)
    
    if missing_labels:
        print(f"  警告: 以下fastsurfer标签在标准DKT中未找到: {missing_labels}")
    
    return label_mapping, FASTSURFER_LABELS

def calculate_suvr_for_subject(registered_pet_image, dkt_path, subject_id, output_dir=None):
    """
    为单个subject计算SUVR
    
    参数:
        registered_pet_image: 已配准的PET图像（SimpleITK Image对象）
        dkt_path: DKT图谱文件路径
        subject_id: 受试者ID
        output_dir: 输出目录路径
    
    返回:
        suvr_array: 68个ROI的SUVR值数组
        results_df: 包含详细结果的DataFrame
        reference_mean: 参考区域平均值
    """
    print(f"=== 为{subject_id}计算SUVR ===")
    
    if not isinstance(registered_pet_image, sitk.Image):
        raise TypeError("registered_pet_image必须是SimpleITK Image对象")
    
    print(f"  PET图像尺寸: {registered_pet_image.GetSize()}")
    
    # 加载DKT图谱
    if not os.path.exists(dkt_path):
        raise FileNotFoundError(f"DKT图谱文件不存在: {dkt_path}")
    
    atlas_img = nib.load(dkt_path)
    atlas_data = atlas_img.get_fdata()
    
    print(f"  DKT图谱尺寸: {atlas_data.shape}")
    
    # 获取标签映射
    label_mapping, label_order = get_fastsurfer_label_mapping()
    
    # 将SimpleITK PET图像转换为numpy数组
    pet_data = sitk.GetArrayFromImage(registered_pet_image)
    pet_data = np.transpose(pet_data, (2, 1, 0))  # 调整顺序以匹配nibabel
    
    print(f"  PET图像尺寸: {pet_data.shape}")
    print(f"  PET图像值范围: [{np.min(pet_data):.2f}, {np.max(pet_data):.2f}]")
    
    # 检查图像尺寸是否匹配
    if pet_data.shape != atlas_data.shape:
        print("  警告: PET图像和DKT图谱尺寸不匹配")
        print(f"  PET尺寸: {pet_data.shape}, DKT尺寸: {atlas_data.shape}")
        print("  将PET图像重采样到DKT图谱空间...")
        
        min_shape = (min(pet_data.shape[0], atlas_data.shape[0]),
                     min(pet_data.shape[1], atlas_data.shape[1]),
                     min(pet_data.shape[2], atlas_data.shape[2]))
        
        pet_data_resized = np.zeros(atlas_data.shape, dtype=pet_data.dtype)
        slices = tuple(slice(min(s1, s2)) for s1, s2 in zip(pet_data.shape, atlas_data.shape))
        pet_data_resized[slices] = pet_data[slices]
        pet_data = pet_data_resized
        
        print(f"  调整后尺寸: {pet_data.shape}")
    
    # 计算参考区域统计（左右小脑灰质）
    print(f"  计算参考区域（标签{REFERENCE_LABELS[0]}, {REFERENCE_LABELS[1]} 左右小脑灰质）...")
    
    left_cerebellum_mask = (atlas_data == REFERENCE_LABELS[0])
    right_cerebellum_mask = (atlas_data == REFERENCE_LABELS[1])
    combined_cerebellum_mask = left_cerebellum_mask | right_cerebellum_mask
    
    cerebellum_values = pet_data[combined_cerebellum_mask]
    cerebellum_values_nonzero = cerebellum_values[cerebellum_values > 0]
    
    if len(cerebellum_values_nonzero) == 0:
        raise ValueError(f"参考区域（左右小脑 标签{REFERENCE_LABELS}）没有有效的像素值")
    
    reference_mean = np.mean(cerebellum_values_nonzero)
    
    print(f"  参考区域平均值: {reference_mean:.4f}")
    print(f"  参考区域有效体素数: {len(cerebellum_values_nonzero)}")
    
    # 按照fastsurfer标签顺序计算SUVR
    print(f"  按fastsurfer顺序计算ROI SUVR (共{len(label_order)}个ROI)")
    roi_results = []
    suvr_values = []
    
    for fastsurfer_name in label_order:
        if fastsurfer_name in label_mapping:
            label = label_mapping[fastsurfer_name]
            roi_name = fastsurfer_name
            
            # 创建ROI掩码
            roi_mask = (atlas_data == label)
            roi_voxel_count = np.sum(roi_mask)
            
            if roi_voxel_count == 0:
                roi_mean = 0.0
                suvr = 0.0
                valid_voxel_count = 0
            else:
                roi_pixels = pet_data[roi_mask]
                roi_valid_pixels = roi_pixels[roi_pixels > 0]
                valid_voxel_count = len(roi_valid_pixels)
                
                if valid_voxel_count > 0:
                    roi_mean = np.mean(roi_valid_pixels)
                    suvr = roi_mean / reference_mean
                else:
                    roi_mean = 0.0
                    suvr = 0.0
        else:
            roi_name = fastsurfer_name
            label = 0
            roi_voxel_count = 0
            valid_voxel_count = 0
            roi_mean = 0.0
            suvr = 0.0
        
        roi_results.append({
            'Index': len(roi_results) + 1,
            'ROI_Name': roi_name,
            'Label': label,
            'Voxel_Count': int(roi_voxel_count),
            'Valid_Voxels': int(valid_voxel_count),
            'Mean_SUV': float(roi_mean),
            'SUVR': float(suvr)
        })
        
        suvr_values.append(float(suvr))
    
    # 转换为DataFrame
    results_df = pd.DataFrame(roi_results)
    
    # 计算总体统计
    valid_results = results_df[results_df['Valid_Voxels'] > 0]
    total_rois = len(results_df)
    valid_rois = len(valid_results)
    
    if valid_rois > 0:
        mean_suvr = valid_results['SUVR'].mean()
        print(f"  平均SUVR: {mean_suvr:.4f}")
    
    print(f"  有效ROI数量: {valid_rois}/{total_rois}")
    
    # 保存结果
    if output_dir is not None:
        save_subject_results(results_df, reference_mean, suvr_values, subject_id, output_dir)
    
    return np.array(suvr_values, dtype=np.float32), results_df, reference_mean

def save_subject_results(results_df, reference_mean, suvr_values, subject_id, output_dir):
    """保存单个subject的结果"""
    output_dir = Path(output_dir) / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV
    csv_path = output_dir / f"{subject_id}_suvr_results.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    CSV结果已保存: {csv_path}")
    
    # 保存NPY
    npy_path = output_dir / f"{subject_id}_suvr_values.npy"
    np.save(npy_path, np.array(suvr_values, dtype=np.float32))
    print(f"    NPY结果已保存: {npy_path}")
    
    # 保存简化版本
    simple_csv_path = output_dir / f"{subject_id}_suvr_simple.csv"
    simple_df = pd.DataFrame({
        'ROI': results_df['ROI_Name'],
        'SUVR': results_df['SUVR']
    })
    simple_df.to_csv(simple_csv_path, index=False, encoding='utf-8-sig')
    
    return csv_path, npy_path

# ============================================================================
# 4. 多subject处理流水线
# ============================================================================
def find_data_directory(base_path: Union[str, Path], subject_id: str, 
                       dir_patterns: List[str], description: str = "数据") -> Path:
    """
    查找数据目录，支持多种可能的目录名
    
    参数:
        base_path: 基础路径
        subject_id: 受试者ID
        dir_patterns: 可能的目录名模式列表
        description: 数据描述，用于错误信息
    
    返回:
        找到的目录路径
    """
    base_dir = Path(base_path) / subject_id
    
    if not base_dir.exists():
        raise FileNotFoundError(f"{description}基础目录不存在: {base_dir}")
    
    # 尝试所有可能的目录模式
    for dir_pattern in dir_patterns:
        potential_dir = base_dir / dir_pattern
        if potential_dir.exists() and potential_dir.is_dir():
            return potential_dir
    
    # 如果都没有找到，抛出异常
    raise FileNotFoundError(
        f"在{base_dir}中未找到{description}目录。"
        f"尝试了以下目录模式: {dir_patterns}"
    )


def get_pet_directory(pet_data_dir: Union[str, Path], subject_id: str) -> Path:
    """
    获取PET目录路径
    
    参数:
        pet_data_dir: PET数据根目录
        subject_id: 受试者ID
    
    返回:
        PET目录路径
    """
    # 可能的PET文件夹名
    pet_dir_patterns = [
        "PET_Head_OSEM",
        "PET_Head_OSEM_1",
        "PET_Head",  # 通用模式
        "PET_OSEM",  # 通用模式
    ]
    
    # 查找PET目录
    pet_dir = find_data_directory(pet_data_dir, subject_id, pet_dir_patterns)
    return pet_dir

def get_t1_file_path(t1_data_dir: Union[str, Path], subject_id: str) -> Path:
    """
    获取T1图像文件路径
    
    参数:
        t1_data_dir: T1数据根目录
        subject_id: 受试者ID
    
    返回:
        T1文件路径
    """
    # 可能的T1文件夹名
    t1_dir_patterns = [
        "t1_gre_fsp_3d_sag_iso_ACS",
        "t1_gre_fsp_3d_sag_iso_ACS_1",
        "t1_gre_fsp_3d_sag_iso_ACS_401",
        "t1_gre_fsp_3d_sag_iso",  # 通用模式
        "t1_gre_fsp",  # 通用模式
    ]
    
    # 查找T1目录
    t1_dir = t1_dir = find_data_directory(t1_data_dir, subject_id, t1_dir_patterns)
    
    return t1_dir

def process_single_subject(subject_id, pet_data_dir, t1_data_dir, dkt_root_dir, output_base_dir):
    """
    处理单个subject的完整流程
    
    参数:
        subject_id: 受试者ID
        pet_data_dir: PET数据根目录
        t1_data_dir: T1数据根目录
        dkt_root_dir: fastsurfer DKT图谱根目录
        output_base_dir: 输出根目录
    
    返回:
        suvr_array: 68个ROI的SUVR值数组
        success: 是否成功处理
        error_message: 错误信息（如果失败）
    """
    print(f"\n{'='*60}")
    print(f"开始处理subject: {subject_id}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建数据路径
        # pet_dir = Path(pet_data_dir) / subject_id / "PET_Head_OSEM"
        # t1_dir = Path(t1_data_dir) / subject_id / "t1_gre_fsp_3d_sag_iso_ACS"
        # 1. 获取PET目录路径
        #print("  查找PET目录...")
        pet_dir = get_pet_directory(pet_data_dir, subject_id)
        #print(f"  PET目录: {pet_dir}")
        
        # 2. 获取T1文件路径
        #print("  查找T1文件...")
        t1_dir = get_t1_file_path(t1_data_dir, subject_id)
        #print(f"  T1目录: {t1_dir.name}")
        dkt_path = Path(dkt_root_dir) / subject_id / "mri" / "aparc.DKTatlas+aseg.mapped.mgz"
        
        print(f"PET数据目录: {pet_dir}")
        print(f"T1数据目录: {t1_dir}")
        print(f"DKT图谱路径: {dkt_path}")
        
        # 检查文件是否存在
        if not pet_dir.exists():
            raise FileNotFoundError(f"PET目录不存在: {pet_dir}")
        if not t1_dir.exists():
            raise FileNotFoundError(f"T1目录不存在: {t1_dir}")
        if not dkt_path.exists():
            raise FileNotFoundError(f"DKT图谱不存在: {dkt_path}")
        
        # 检查DICOM文件数量
        pet_files = list(pet_dir.glob("*.dcm"))
        t1_files = list(t1_dir.glob("*.dcm"))
        print(f"PET DICOM文件数: {len(pet_files)}")
        print(f"T1 DICOM文件数: {len(t1_files)}")
        
        if len(pet_files) == 0:
            raise FileNotFoundError(f"PET目录中没有DICOM文件: {pet_dir}")
        if len(t1_files) == 0:
            raise FileNotFoundError(f"T1目录中没有DICOM文件: {t1_dir}")
        
        # 2. 加载PET和T1图像
        print("正在加载PET图像...")
        pet_image = load_dicom_series(pet_dir)
        
        print("正在加载T1图像...")
        t1_image = load_dicom_series(t1_dir)
        
        print(f"PET图像尺寸: {pet_image.GetSize()}, 间距: {pet_image.GetSpacing()}")
        print(f"T1图像尺寸: {t1_image.GetSize()}, 间距: {t1_image.GetSpacing()}")
        
        # 3. PET到T1配准
        print("开始PET到T1配准...")
        registered_pet, transform = improved_pet_to_t1_registration(pet_image, t1_image)
        
        print(f"配准后PET图像尺寸: {registered_pet.GetSize()}")
        print(f"配准后PET图像间距: {registered_pet.GetSpacing()}")
        
        # 4. 计算SUVR
        suvr_array, results_df, reference_mean = calculate_suvr_for_subject(
            registered_pet_image=registered_pet,
            dkt_path=str(dkt_path),
            subject_id=subject_id,
            output_dir=output_base_dir
        )
        
        # 5. 返回结果
        print(f"✓ {subject_id} 处理完成")
        return suvr_array, True, None
        
    except Exception as e:
        error_msg = f"处理{subject_id}失败: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return None, False, error_msg

def process_multiple_subjects(subject_ids, pet_data_dir, t1_data_dir, dkt_root_dir, output_base_dir):
    """
    处理多个subject
    
    参数:
        subject_ids: 受试者ID列表
        pet_data_dir: PET数据根目录
        t1_data_dir: T1数据根目录
        dkt_root_dir: fastsurfer DKT图谱根目录
        output_base_dir: 输出根目录
    
    返回:
        suvr_matrix: 形状为(n, 68)的SUVR值矩阵
        success_subjects: 成功处理的subject列表
        failed_subjects: 处理失败的subject列表
    """
    global global_suvr_matrix, global_subject_ids, global_statistics
    
    print("=" * 80)
    print("开始多subject处理流水线")
    print("=" * 80)
    print(f"总subject数量: {len(subject_ids)}")
    
    # 重置全局变量
    global_suvr_matrix = []
    global_subject_ids = []
    global_statistics = []
    
    success_subjects = []
    failed_subjects = []
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n[进度: {i}/{len(subject_ids)}]")
        
        suvr_array, success, error_msg = process_single_subject(
            subject_id=subject_id,
            pet_data_dir=pet_data_dir,
            t1_data_dir=t1_data_dir,
            dkt_root_dir=dkt_root_dir,
            output_base_dir=output_base_dir
        )
        
        if success and suvr_array is not None:
            # 存储结果
            global_suvr_matrix.append(suvr_array)
            global_subject_ids.append(subject_id)
            success_subjects.append(subject_id)
            
            # 计算统计信息
            valid_suvr = suvr_array[suvr_array > 0]
            if len(valid_suvr) > 0:
                stats = {
                    'subject_id': subject_id,
                    'mean_suvr': np.mean(valid_suvr),
                    'median_suvr': np.median(valid_suvr),
                    'std_suvr': np.std(valid_suvr),
                    'min_suvr': np.min(valid_suvr),
                    'max_suvr': np.max(valid_suvr),
                    'valid_rois': len(valid_suvr)
                }
            else:
                stats = {
                    'subject_id': subject_id,
                    'mean_suvr': 0.0,
                    'median_suvr': 0.0,
                    'std_suvr': 0.0,
                    'min_suvr': 0.0,
                    'max_suvr': 0.0,
                    'valid_rois': 0
                }
            global_statistics.append(stats)
        else:
            failed_subjects.append({
                'subject_id': subject_id,
                'error': error_msg
            })
    
    # 转换为numpy数组
    if len(global_suvr_matrix) > 0:
        suvr_matrix = np.array(global_suvr_matrix, dtype=np.float32)
        print(f"\n✓ 成功处理 {len(success_subjects)} 个subject")
        print(f"  SUVR矩阵形状: {suvr_matrix.shape}")
    else:
        suvr_matrix = np.array([], dtype=np.float32)
        print("\n⚠ 没有成功处理的subject")
    
    return suvr_matrix, success_subjects, failed_subjects

def save_global_results(suvr_matrix, subject_ids, output_dir, failed_subjects=None):
    """
    保存全局结果
    
    参数:
        suvr_matrix: SUVR值矩阵 (n, 68)
        subject_ids: 对应的subject ID列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存NPY格式 (n, 62)
    npy_path = output_dir / f"all_subjects_suvr_{timestamp}.npy"
    np.save(npy_path, suvr_matrix)
    print(f"全局NPY结果已保存: {npy_path}")
    
    # 2. 保存CSV格式
    csv_path = output_dir / f"all_subjects_suvr_{timestamp}.csv"
    suvr_df = pd.DataFrame(suvr_matrix, index=subject_ids, columns=FASTSURFER_LABELS)
    suvr_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"全局CSV结果已保存: {csv_path}")
    
    # 3. 保存统计信息
    if global_statistics:
        stats_df = pd.DataFrame(global_statistics)
        stats_csv_path = output_dir / f"all_subjects_statistics_{timestamp}.csv"
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
        print(f"统计信息已保存: {stats_csv_path}")
    
    # 4. 保存处理报告
    report_path = output_dir / f"processing_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("多subject处理报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总subject数量: {len(subject_ids) + (len(failed_subjects) if failed_subjects else 0)}\n")
        f.write(f"成功处理: {len(subject_ids)}\n")
        f.write(f"处理失败: {len(failed_subjects) if failed_subjects else 0}\n\n")
        
        f.write("成功处理的subject:\n")
        for subj in subject_ids:
            f.write(f"  {subj}\n")
        
        if failed_subjects and len(failed_subjects) > 0:
            f.write(f"\n失败的subject:\n")
            for failed in failed_subjects:
                f.write(f"  {failed['subject_id']}: {failed['error']}\n")
        
        f.write(f"\nSUVR矩阵形状: {suvr_matrix.shape}\n")
        
        if len(suvr_matrix) > 0:
            f.write(f"平均SUVR (所有subject和ROI): {np.nanmean(suvr_matrix):.4f}\n")
            f.write(f"SUVR标准差: {np.nanstd(suvr_matrix):.4f}\n")
    
    print(f"处理报告已保存: {report_path}")
    
    return npy_path, csv_path

# ============================================================================
# 5. 主运行函数
# ============================================================================

def run_auto_scan_pipeline():
    """
    自动扫描目录并运行完整流水线
    """
    print("=" * 80)
    print("自动扫描目录的多subject PET-T1配准和SUVR计算流水线")
    print("=" * 80)
    
    # ============================================================
    # 请根据您的实际情况修改以下参数
    # ============================================================
    
    # 1. 定义数据目录
    # PET数据根目录（包含每个subject的PET_Head_OSEM子目录）
    pet_data_root = "path/to/abeta_pet"
    
    # T1数据根目录（包含每个subject的t1_gre_fsp_3d_sag_iso_ACS子目录）
    t1_data_root = "path/to/abeta_pet_t1" 
    
    # fastsurfer DKT图谱根目录
    dkt_root = "path/to/abeta_aligned_mri"
    
    # 2. 输出目录
    output_base_dir = "suvr_results"
    
    # 3. 处理参数
    min_pet_files = 1  # PET目录最小DICOM文件数
    min_t1_files = 1   # T1目录最小DICOM文件数
    test_mode = False   # 是否测试模式（只处理前几个subject）
    max_subjects_test = 2  # 测试模式下最多处理的subject数量
    
    # ============================================================
    # 开始处理
    # ============================================================
    
    print(f"PET数据根目录: {pet_data_root}")
    print(f"T1数据根目录: {t1_data_root}")
    print(f"DKT图谱根目录: {dkt_root}")
    print(f"输出目录: {output_base_dir}")
    
    # 1. 自动扫描目录获取可用的subject ID
    available_subjects, subject_details = scan_subject_ids_from_root(
        pet_data_root=pet_data_root,
        t1_data_root=t1_data_root,
        dkt_root=dkt_root,
        min_pet_files=min_pet_files,
        min_t1_files=min_t1_files
    )
    
    if not available_subjects:
        print("错误: 没有找到可用的完整subject")
        return None, [], []
    
    print(f"\n找到 {len(available_subjects)} 个可用的完整subject")
    
    # 2. 测试模式：只处理前几个subject
    if test_mode and len(available_subjects) > max_subjects_test:
        print(f"\n测试模式: 只处理前{max_subjects_test}个subject")
        subjects_to_process = available_subjects[:max_subjects_test]
    else:
        subjects_to_process = available_subjects
    
    print(f"将要处理的subject数量: {len(subjects_to_process)}")
    
    # 3. 处理所有subject
    suvr_matrix, success_subjects, failed_subjects = process_multiple_subjects(
        subject_ids=subjects_to_process,
        pet_data_dir=pet_data_root,
        t1_data_dir=t1_data_root,
        dkt_root_dir=dkt_root,
        output_base_dir=output_base_dir
    )
    
    # 4. 保存全局结果
    if len(success_subjects) > 0:
        npy_path, csv_path = save_global_results(
            suvr_matrix=suvr_matrix,
            subject_ids=success_subjects,
            output_dir=output_base_dir,
            failed_subjects=failed_subjects
        )
        
        print(f"\n{'='*80}")
        print("处理完成!")
        print(f"{'='*80}")
        print(f"成功处理的subject: {len(success_subjects)}个")
        print(f"处理失败的subject: {len(failed_subjects)}个")
        print(f"SUVR矩阵形状: {suvr_matrix.shape} (n={len(success_subjects)}, 62个ROI)")
        print(f"主要输出文件:")
        print(f"  - NPY格式: {npy_path}")
        print(f"  - CSV格式: {csv_path}")
        
        if failed_subjects:
            print(f"\n失败的subject:")
            for failed in failed_subjects:
                print(f"  {failed['subject_id']}: {failed['error']}")
    else:
        print("\n⚠ 没有成功处理的subject")
    
    return suvr_matrix, success_subjects, failed_subjects

if __name__ == "__main__":
    
    # 2. 运行完整
    suvr_matrix, success_subjects, failed_subjects = run_auto_scan_pipeline()
