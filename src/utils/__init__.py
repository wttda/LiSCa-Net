import csv
import yaml
from .metrics import *
from ..utils.util import *
from easydict import EasyDict
from .add_noise import NoiseSimulator


class ConfigSimulated:
    def __init__(self, cfg_file='src/basic_config/base_conf_simu.yml'):
        with open(cfg_file, 'r', encoding='utf-8') as f:
            self.cfg = EasyDict(yaml.safe_load(f))
        self._setup_dirs()

    # noinspection PyUnresolvedReferences
    def _setup_dirs(self, log_dir='log'):
        mkdir(self.cfg.save_dir)

        scene_dir = os.path.join(self.cfg.save_dir, self.cfg.datasets.scene_name)
        mkdir(scene_dir)
        self.cfg.scene_dir = scene_dir

        case_dir = os.path.join(str(scene_dir), self.cfg.noise.case)
        mkdir(case_dir)
        self.cfg.case_dir = case_dir

        log_dir = os.path.join(str(case_dir), log_dir)
        mkdir(log_dir)
        self.cfg.log_dir = log_dir

    def __getattr__(self, name):
        return getattr(self.cfg, name)

def merge_cfg(a, b):
    """Recursively merge two configuration dictionaries."""
    for key in b:
        if key in a and isinstance(a[key], EasyDict) and isinstance(b[key], EasyDict):
            merge_cfg(a[key], b[key])
        else:
            a[key] = b[key]

def human_format(num):
    """Convert number to human-readable format."""
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 5:
        magnitude += 1
        num /= 1000.0
    return f"{num:.1f}{' KMGTPE'[magnitude]}"

def human_format_k(num):
    """Convert number to human-readable format with k as unit."""
    if num == 0:
        return "0.0k"
    else:
        return f"{num / 1000.0:.1f}k"

def summary(module: dict, cfg=None):
    summary_ = ''
    summary_ += '-' * 100 + '\n'
    total_params = sum(sum(p.numel() for p in v.parameters()) for k, v in module.items())
    summary_ += f'Total parameters: {human_format_k(total_params)}\n\n'

    # model
    for k, v in module.items():
        # get parameter number
        param_num = sum(p.numel() for p in v.parameters())
        # get information about architecture and parameter number
        summary_ += '[%s] parameters: %s -->' % (k, human_format(param_num)) + '\n'
        summary_ += str(v) + '\n\n'
    summary_ += '-' * 100 + '\n'

    if cfg is not None:
        save_path = os.path.join(cfg.save_dir, f'num_params_{cfg.datasets.scene_name}.csv')
        name = f'LiSCa_Net_stage_{cfg.stage}'
        row = [name, f"{total_params / 1000.0:.1f}k"]
        rows = []
        file_exists = os.path.exists(save_path)
        if file_exists:
            with open(save_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        method_exists = False
        for i, r in enumerate(rows):
            if r and r[0] == name:
                rows[i] = row
                method_exists = True
                break
        if not method_exists:
            if not rows:
                rows.append(['Method', 'Parameters (k)'])
            rows.append(row)
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return summary_








