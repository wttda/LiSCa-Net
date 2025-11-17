from src.utils import *


class Config:
    def __init__(self, cfg_file='src/LiSCa_Net/conf/config.yml'):
        with open(cfg_file, 'r') as f:
            self.cfg = EasyDict(yaml.safe_load(f))
        self._setup_dirs()

    # noinspection PyUnresolvedReferences
    def _setup_dirs(self, ckpt_dir='checkpoint'):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(parent_dir, ckpt_dir)
        mkdir(ckpt_dir)
        self.cfg.ckpt_dir = ckpt_dir

    def __getattr__(self, name):
        return getattr(self.cfg, name)

























