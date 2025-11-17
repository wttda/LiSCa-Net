import os
import torch
import random
import numpy as np
import scipy.io as sio



def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_img(img):
    """Normalize image to [0, 1] range using min-max normalization."""
    max_val = np.max(img)
    min_val = np.min(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)

def get_clean_HSI(cfg):
    if cfg.datasets.scene_name == 'pavia':
        test_HSI_path = os.path.join(cfg.datasets.HSI_dir, 'img_clean_pavia_withoutNormalization.mat')
    elif cfg.datasets.scene_name == 'WashingtonDC':
        test_HSI_path = os.path.join(cfg.datasets.HSI_dir, 'img_clean_dc_withoutNormalization.mat')
    else:
        raise ValueError('Invalid scene name')
    c_hsi = sio.loadmat(test_HSI_path)['img_clean']
    return c_hsi.astype(np.float32)

def save_mat(mat_path, mat_data):
    if isinstance(mat_data, dict):
        sio.savemat(mat_path, mat_data)
    else:
        sio.savemat(mat_path, {'data': mat_data})
















