# -*- coding: utf-8 -*-
from src.utils import *
from importlib import import_module
from collections import defaultdict
from src.utils.base_main import BaseMain


class MixNoiseRemoval(BaseMain):
    def __init__(self, config):
        BaseMain.__init__(self, config)
        self._mk_dirs()
        self._paper_cache = defaultdict(lambda: defaultdict(dict))

    def _mk_dirs(self):
        self.data_dir = os.path.join(self.cfg.case_dir, 'Data')
        mkdir(self.data_dir)

    def main(self):
        self._before_exp()
        self._run_exp()
        self._after_exp()

    def _before_exp(self):
        set_seed(self.cfg.manual_seed)

        # Device settings
        if self.cfg.device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg.gpu_ids)

        # Load clean HSI
        self.c_hsi = get_clean_HSI(self.cfg)
        self.rows, self.cols, self.nbd = self.c_hsi.shape
        self.npx = self.rows * self.cols

        # Simulate noisy HSI
        noise_simulator = NoiseSimulator(self.cfg)
        self.c_hsi, self.n_hsi = noise_simulator.simulate(self.c_hsi)
        self._metrics_noisy()

    def _run_exp(self):
        self._run_mtd(self.cfg.methods)
        self._after_mtd(self.cfg.methods)

    def _after_exp(self):
        self.flush_metrics_to_csv()
        self.logger.highlight("All HSI denoising methods completed.")

    def _run_mtd(self, mtd_name):
        self.cfg.cfg.name = mtd_name
        self._run_LiSCa_Net()
        self.logger.val(f'| Test dataset: {self.cfg.datasets.scene_name} | Noise case: {self.cfg.noise.case} |')

    def _after_mtd(self, mtd_name):
        # Calculate metrics
        metric_dict = self.get_metrics(self.d_hsi2d, self.c_hsi.reshape(self.npx, self.nbd, order='F').T)

        mpsnr = metric_dict['MPSNR']
        mssim = metric_dict['MSSIM']
        ergas = metric_dict['ERGAS']

        # Log metrics
        self.logger.val(f'Metrics for {mtd_name}: - MPSNR: {mpsnr:.4f}, MSSIM: {mssim:.4f}, ERGAS: {ergas:.4f}.')

        if self.cfg.save_result:
            self._metrics_data(mtd_name, mpsnr, mssim, self.run_time, ergas)

    def _run_LiSCa_Net(self):
        ################## Stage1 ###################
        # Train
        Trainer = getattr(import_module(f'src.LiSCa_Net.trainer.trainer_stage1'), 'TrainerStage1')
        trainer = Trainer(self.cfg, self.n_hsi, self.logger, self.c_hsi)
        trainer.train()
        # Test
        d_hsi2d_stage1, time_train_stage1, time_test_stage1 = trainer.test()

        ################## Stage2 ###################
        n_hsi = d_hsi2d_stage1.T.reshape(self.rows, self.cols, self.nbd, order='F')
        if self.cfg.save_denoised_hsi:
            save_mat(f'{self.data_dir}/{trainer.cfg.name}_stage{trainer.cfg.stage}.mat', n_hsi)

        # Train
        Trainer = getattr(import_module(f'src.LiSCa_Net.trainer.trainer_stage2'), 'TrainerStage2')
        trainer = Trainer(self.cfg, n_hsi, self.logger, self.c_hsi)
        trainer.train()
        # Test
        self.d_hsi2d, time_train_stage2, time_test_stage2 = trainer.test()
        if self.cfg.save_denoised_hsi:
            d_hsi = self.d_hsi2d.T.reshape(self.rows, self.cols, self.nbd, order='F')
            save_mat(f'{self.data_dir}/{trainer.cfg.name}_stage{trainer.cfg.stage}.mat', d_hsi)

        time_stage1 = time_train_stage1 + time_test_stage1
        time_stage2 = time_train_stage2 + time_test_stage2
        self.run_time = [time_stage1, time_stage2]


if __name__ == '__main__':
    cfg = ConfigSimulated()
    mixed_noise_removal = MixNoiseRemoval(cfg)
    mixed_noise_removal.main()
