def extract_eeg_data(excel_path, dir_eeg):
    """
    从Excel中提取EEG目录下的所有文件对应的SUVR、MMSE和MOCA数据
    
    参数:
    excel_path: Excel文件路径
    dir_eeg: EEG文件所在目录路径
    
    返回:
    tuple: (suvr_list, mmse_list, moca_list) 三个列表
    """
    # 读取Excel数据
    df = pd.read_excel(excel_path)
    
    # 验证必要列是否存在
    required_columns = ['EEG编号', 'SUVR', 'MMSE', 'MOCA']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"false: don't find the necessary colume'{col}'")
    
    # 创建文件名标准化函数
    def normalize_filename(file_path):
        """提取基础文件名（不含路径、扩展名和特定后缀）"""
        filename = os.path.basename(file_path)  # 获取文件名
        
        # 使用正则表达式去除扩展名和特定后缀
        # 1. 首先去除任何文件扩展名（如 .mat, .set 等）
        name_without_ext = re.sub(r'\.[^.]*$', '', filename)
        
        # 2. 然后去除特定后缀（_EC 或 .set_EC）
        # 匹配以下两种格式：
        #   a) 以 "_EC" 结尾的名称
        #   b) 以 ".set_EC" 结尾的名称
        normalized_name = re.sub(r'(_EC|\.set_EC)$', '', name_without_ext)
        
        # 3. 处理特殊情况：如果文件名以数字结尾但没有下划线分隔符
        #   在最后的数字序列前添加分隔符（如果缺失）
        normalized_name = re.sub(r'(\D)(\d+)$', r'\1_\2', normalized_name)
        
        # 返回小写并去除前后空格
        return normalized_name.lower().strip()

    
    # 获取EEG目录中的所有文件
    eeg_file_list = [os.path.join(dir_eeg, f) for f in os.listdir(dir_eeg)
                 if os.path.isfile(os.path.join(dir_eeg, f))]
    
    
    # 构建文件名到数据的映射字典
    file_mapping = {}
    for _, row in df.iterrows():
        # 提取并标准化EEG文件名
        eeg_name = normalize_filename(str(row['EEG编号']))
        # 处理SUVR、MMSE、MOCA中的NaN值（'#N/A'已自动转换为NaN）
        suvr = row['SUVR']
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else np.nan
        moca = row['MOCA'] if pd.notna(row['MOCA']) else np.nan
        
        # 存储对应的数据
        file_mapping[eeg_name] = {
            'SUVR': suvr,
            'MMSE': mmse,
            'MOCA': moca
        }
    
    # 按EEG文件列表顺序提取数据
    suvr_list, mmse_list, moca_list = [], [], []
    unmatched_files = []  # 记录未匹配的文件
    
    for eeg_file in eeg_file_list:
        # 标准化当前EEG文件名
        normalized_name = normalize_filename(eeg_file)
        
        # 检查文件是否在Excel中有记录
        if normalized_name in file_mapping:
            entry = file_mapping[normalized_name]
            suvr_list.append(entry['SUVR'])
            mmse_list.append(entry['MMSE'])
            moca_list.append(entry['MOCA'])
        else:
            unmatched_files.append(os.path.basename(eeg_file))
    
    # 如果有未匹配文件，发出警告但不停止处理
    if unmatched_files:
        print(f"警告: {len(unmatched_files)}个EEG文件在Excel中没有对应记录")
        print("未匹配的文件:", ", ".join(unmatched_files[:10]) + ("..." if len(unmatched_files) > 10 else ""))
    
    return suvr_list, mmse_list, moca_list, eeg_file_list

# 测试函数
if __name__ == "__main__":
    # 示例文件路径（请替换为实际路径）
    excel_path = " "
    eeg_directory = " "  # EEG文件所在目录
    
    # 提取数据
    try:
        suvr_values, mmse_scores, moca_scores,eeg_file_list = extract_eeg_data(excel_path, eeg_directory)
        
        print(f"成功处理 {len(suvr_values)} 个EEG文件数据:")
        print("SUVR值:", suvr_values)
        print("MMSE分数:", mmse_scores)
        print("MOCA分数:", moca_scores)
        print('EEG数据:',eeg_file_list)

        np.save('suvr_values.npy',suvr_values)
    except Exception as e:
        print("处理出错:", str(e))

def get_source_en_power(file_list):
    raw_vhdr = mne.io.read_raw_brainvision("F:/EEG_raw/Shenzhen_Neurology_6169_ZhengMeiHe.vhdr", preload = False)
    ch_names = raw_vhdr.ch_names[:-2]
    info = mne.create_info(ch_names= ch_names,sfreq = 250, ch_types = 'eeg')
    # subject
    from mne.datasets import fetch_fsaverage
    
    fs_dir = fetch_fsaverage(verbose=True)
    subject_dir = os.path.dirname(fs_dir)
    
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    n = len(file_list)
    source_pd = np.zeros((n,68))
    source_pt = np.zeros((n,68))
    source_pa = np.zeros((n,68))
    source_pb = np.zeros((n,68))
    entropy_all = np.zeros((n,68))    
    
    for j in range(n):
    
        d = scio.loadmat(file_list[j])
        set = d['EEG'][0,0]['data']
        raw = mne.io.RawArray(set, info)
        raw.set_montage(bp_cl, on_missing='ignore')
        epochs = mne.make_fixed_length_epochs(raw, duration=10., overlap=0, preload=True, reject_by_annotation=True)
        epochs.set_eeg_reference(projection=True)
        
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)
        
        # 噪声协方差
        noise_cov = mne.compute_raw_covariance(raw)
        
        # 逆向因子
        
        inverse_operator = make_inverse_operator(
            epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
        )
        del fwd
        
        # 逆向因子应用
        method = "dSPM" 
        snr = 1.0    # use lower SNR for single epochs
        lambda2 = 1.0 / snr**2
        # stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori='normal',return_generator=True)
        labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subject_dir)
        labels.remove(labels[-1])
        #fmin = 8.0
        #fmax = 12.0
        #sfreq = raw.info["sfreq"]  # the sampling frequency
        
        for k in range(len(labels)):
            stcs = compute_source_psd_epochs(
                epochs,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                fmin=0.0,  #(1.5,4.,8.,12.,),  
                fmax=50.0,  #(4.,8.,12.,30.,), #12.0,
                bandwidth=4.0,  # bandwidth of the windows in Hz,
                label=labels[k],
                verbose=True,
            )
            psd_avg = 0.0
            for i, stc in enumerate(stcs):
                psd_avg += stc.data
            psd_avg /= len(epochs)
            freqs = stc.times
            stc.data = psd_avg
    
            m = np.sum(stc.data, axis=-1)
            psd_norm = np.divide(stc.data[:,1:], m[:,None])
            entropy = -np.sum(np.multiply(psd_norm, np.log2(psd_norm)), axis=-1)
            entropy_mean = entropy.mean()
            entropy_all[j,k] = entropy_mean
            
            psd = stc.data.T.mean(axis=1)
            f = stc.times
            f_res = f[1]-f[0]
    
            delta_band = [1.5, 4]
            idx_delta_band = np.logical_and(f >= delta_band[0], f <= delta_band[1])
            dp = simpson(psd[idx_delta_band], dx=f_res)
            
            theta_band = [4, 8]
            idx_theta_band = np.logical_and(f >= theta_band[0], f <= theta_band[1])
            tp = simpson(psd[idx_theta_band], dx=f_res)
            
            alpha_band = [8, 12]
            idx_alpha_band = np.logical_and(f >= alpha_band[0], f <= alpha_band[1])
            ap = simpson(psd[idx_alpha_band], dx=f_res)
    
            beta_band = [12, 30]
            idx_beta_band = np.logical_and(f >= beta_band[0], f <= beta_band[1])
            bp = simpson(psd[idx_beta_band], dx=f_res)
            
            gamma_band = [30, 50]
            idx_gamma_band = np.logical_and(f >= gamma_band[0], f <= gamma_band[1])
            gp = simpson(psd[idx_gamma_band], dx=f_res)
            
            rel_dp = dp/(dp+tp+ap+bp+gp)
            rel_tp = tp/(dp+tp+ap+bp+gp)
            rel_ap = ap/(dp+tp+ap+bp+gp)
            rel_bp = bp/(dp+tp+ap+bp+gp)
            source_pd[j,k] = rel_dp
            source_pt[j,k] = rel_tp
            source_pa[j,k] = rel_ap
            source_pb[j,k] = rel_bp
    return entropy_all, source_pd, source_pt, source_pa, source_pb

def get_7network_en_power(file_list):
    raw_vhdr = mne.io.read_raw_brainvision("F:/EEG_raw/Shenzhen_Neurology_6169_ZhengMeiHe.vhdr", preload = False)
    ch_names = raw_vhdr.ch_names[:-2]
    info = mne.create_info(ch_names= ch_names,sfreq = 250, ch_types = 'eeg')
    # subject
    from mne.datasets import fetch_fsaverage
    
    fs_dir = fetch_fsaverage(verbose=True)
    subject_dir = os.path.dirname(fs_dir)
    
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    n = len(file_list)
    source_pd = np.zeros((n,14))
    source_pt = np.zeros((n,14))
    source_pa = np.zeros((n,14))
    source_pb = np.zeros((n,14))
    entropy_all = np.zeros((n,14))    
    
    for j in range(n):
    
        d = scio.loadmat(file_list[j])
        set = d['EEG'][0,0]['data']
        raw = mne.io.RawArray(set, info)
        raw.set_montage(bp_cl, on_missing='ignore')
        epochs = mne.make_fixed_length_epochs(raw, duration=10., overlap=0, preload=True, reject_by_annotation=True)
        epochs.set_eeg_reference(projection=True)
        
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)
        
        # 噪声协方差
        noise_cov = mne.compute_raw_covariance(raw)
        
        # 逆向因子
        
        inverse_operator = make_inverse_operator(
            epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
        )
        del fwd
        
        # 逆向因子应用
        method = "dSPM" 
        snr = 1.0    # use lower SNR for single epochs
        lambda2 = 1.0 / snr**2
        # stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori='normal',return_generator=True)
        yeo_7_labels = mne.read_labels_from_annot("fsaverage", parc="Yeo2011_7Networks_N1000", subjects_dir=subject_dir)
        yeo_7_labels = yeo_7_labels[:-2]
        #fmin = 8.0
        #fmax = 12.0
        #sfreq = raw.info["sfreq"]  # the sampling frequency
        
        for k in range(len(yeo_7_labels)):
            stcs = compute_source_psd_epochs(
                epochs,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                fmin=0.0,  #(1.5,4.,8.,12.,),  
                fmax=50.0,  #(4.,8.,12.,30.,), #12.0,
                bandwidth=4.0,  # bandwidth of the windows in Hz,
                label=yeo_7_labels[k],
                verbose=True,
            )
            psd_avg = 0.0
            for i, stc in enumerate(stcs):
                psd_avg += stc.data
            psd_avg /= len(epochs)
            freqs = stc.times
            stc.data = psd_avg
    
            m = np.sum(stc.data, axis=-1)
            psd_norm = np.divide(stc.data[:,1:], m[:,None])
            entropy = -np.sum(np.multiply(psd_norm, np.log2(psd_norm)), axis=-1)
            entropy_mean = entropy.mean()
            entropy_all[j,k] = entropy_mean
            
            psd = stc.data.T.mean(axis=1)
            f = stc.times
            f_res = f[1]-f[0]
    
            delta_band = [1.5, 4]
            idx_delta_band = np.logical_and(f >= delta_band[0], f <= delta_band[1])
            dp = simpson(psd[idx_delta_band], dx=f_res)
            
            theta_band = [4, 8]
            idx_theta_band = np.logical_and(f >= theta_band[0], f <= theta_band[1])
            tp = simpson(psd[idx_theta_band], dx=f_res)
            
            alpha_band = [8, 12]
            idx_alpha_band = np.logical_and(f >= alpha_band[0], f <= alpha_band[1])
            ap = simpson(psd[idx_alpha_band], dx=f_res)
    
            beta_band = [12, 30]
            idx_beta_band = np.logical_and(f >= beta_band[0], f <= beta_band[1])
            bp = simpson(psd[idx_beta_band], dx=f_res)
            
            gamma_band = [30, 50]
            idx_gamma_band = np.logical_and(f >= gamma_band[0], f <= gamma_band[1])
            gp = simpson(psd[idx_gamma_band], dx=f_res)
            
            rel_dp = dp/(dp+tp+ap+bp+gp)
            rel_tp = tp/(dp+tp+ap+bp+gp)
            rel_ap = ap/(dp+tp+ap+bp+gp)
            rel_bp = bp/(dp+tp+ap+bp+gp)
            source_pd[j,k] = rel_dp
            source_pt[j,k] = rel_tp
            source_pa[j,k] = rel_ap
            source_pb[j,k] = rel_bp
    return entropy_all, source_pd, source_pt, source_pa, source_pb

def get_source_conn(file_list):
    
    delta_conn = np.empty((0,68,68))
    theta_conn = np.empty((0,68,68))
    alpha_conn = np.empty((0,68,68))
    beta_conn = np.empty((0,68,68))
    gamma_conn = np.empty((0,68,68))
    
    raw_vhdr = mne.io.read_raw_brainvision("F:/EEG_raw/Shenzhen_Neurology_6169_ZhengMeiHe.vhdr", preload = False)
    ch_names = raw_vhdr.ch_names[:-2]
    info = mne.create_info(ch_names= ch_names,sfreq = 250, ch_types = 'eeg')
    # subject
    from mne.datasets import fetch_fsaverage
    
    fs_dir = fetch_fsaverage(verbose=True)
    subject_dir = os.path.dirname(fs_dir)
    
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    
    for i in range(len(file_list)):
        d = scio.loadmat(file_list[i])
        set = d['EEG'][0,0]['data']
        raw = mne.io.RawArray(set, info)
        raw.set_montage(bp_cl, on_missing='ignore')
        epochs = mne.make_fixed_length_epochs(raw, duration=10., overlap=0, preload=True, reject_by_annotation=True)
        epochs.set_eeg_reference(projection=True)
        
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)
    
        # 噪声协方差
        noise_cov = mne.compute_raw_covariance(raw)
    
        # 逆向因子
    
        inverse_operator = make_inverse_operator(
            epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
        )
        del fwd
    
        # 逆向因子应用
    
        method = "dSPM" 
        snr = 1.0    # use lower SNR for single epochs
        lambda2 = 1.0 / snr**2
        stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori='normal',return_generator=True)
        labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subject_dir)
        labels.remove(labels[-1])
        label_colors = [label.color for label in labels]
        
        src = inverse_operator["src"]
        label_ts = mne.extract_label_time_course(
            stcs, labels, src, mode="mean_flip", allow_empty='ignore',return_generator=True
        )
        #fmin = 8.0
        #fmax = 12.0
        sfreq = raw.info["sfreq"]  # the sampling frequency
        con_methods = ["ciplv"]     #["pli", "wpli2_debiased", "ciplv"]
        
        con = spectral_connectivity_epochs(
            label_ts,
            method=con_methods,
            mode="multitaper",
            sfreq=sfreq,
            fmin=(1.5,4.,8.,12.,),  #8.0,
            fmax=(4.,8.,12.,30.,), #12.0,
            faverage=True,
            mt_adaptive=True,
            n_jobs=1,
        )
    
        de = con.get_data(output="dense")[:, :, 0]
        # de = np.delete(de,68,0)
        # de = np.delete(de,68,1)
        delta_conn = np.concatenate((delta_conn, np.expand_dims(de, axis=0)),axis = 0)
        th = con.get_data(output="dense")[:, :, 1]
        # th = np.delete(th,68,0)
        # th = np.delete(th,68,1)
        theta_conn = np.concatenate((theta_conn, np.expand_dims(th, axis=0)),axis = 0)
        al = con.get_data(output="dense")[:, :, 2]
        # al = np.delete(al,68,0)
        # al = np.delete(al,68,1)
        alpha_conn = np.concatenate((alpha_conn, np.expand_dims(al, axis=0)),axis = 0)
        be = con.get_data(output="dense")[:, :, 3]
        # be = np.delete(be,68,0)
        # be = np.delete(be,68,1)
        beta_conn = np.concatenate((beta_conn, np.expand_dims(be, axis=0)),axis = 0)
    return labels, delta_conn, theta_conn, alpha_conn, beta_conn