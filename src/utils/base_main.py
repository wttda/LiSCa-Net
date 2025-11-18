import pandas as pd
from src.utils import *
from src.utils.plot_figs import *
from src.utils.logger import Logger


class BaseMain:
    def main(self):
        raise NotImplementedError

    def __init__(self, config):
        self.cfg = config
        self.logger = Logger(None, log_dir=self.cfg.log_dir, log_file_option='w', write_log=self.cfg.write_log)

        self.n_hsi = self.c_hsi = self._paper_cache = self.rows = self.cols = self.nbd = self.npx = None
        self.scene_name = self.cfg.datasets.scene_name

    def get_metrics(self, hsi2d, c_hsi2d):
        """Calculate all evaluation metrics between two HSI datasets.
        Args:
            hsi2d: HSI data (noisy or denoised) - shape (bands, pixels)
            c_hsi2d: Clean HSI data - shape (bands, pixels)

        Returns: Dictionary containing all metrics
        """
        metric_dict = {
            'MPSNR': MPSNR(hsi2d, c_hsi2d),
            'MSSIM': MSSIM(hsi2d, c_hsi2d, self.rows, self.cols),
            'ERGAS': ERGAS(hsi2d, c_hsi2d, self.rows, self.cols),
        }
        return metric_dict

    def _metrics_data(self, method_name, mpsnr, mssim, runtime, ergas):
        """Save metrics data."""
        if not self.cfg.save_result:
            return
        self._paper_cache[(self.scene_name, self.cfg.noise.case)][method_name] = {
            'MPSNR': f'{mpsnr:.2f}',
            'MSSIM': f'{mssim:.3f}',
            'ERGAS': f'{ergas:.1f}',
            'time': format_runtime(runtime)
        }

    def _metrics_noisy(self):
        c_hsi2d = self.c_hsi.reshape(self.npx, self.nbd, order='F').T
        n_hsi2d = self.n_hsi.reshape(self.npx, self.nbd, order='F').T
        metric_dict = self.get_metrics(n_hsi2d, c_hsi2d)

        mpsnr = metric_dict['MPSNR']
        mssim = metric_dict['MSSIM']
        ergas = metric_dict['ERGAS']
        self.logger.val(f'Metrics for noisy HSI - MPSNR: {mpsnr:.4f}, MSSIM: {mssim:.4f}, ERGAS: {ergas:.4f}.')

        self.noisy_psnr = mpsnr
        self.noisy_mssim = mssim
        self.noisy_ergas = ergas

    def flush_metrics_to_csv(self):
        """Write metrics data to CSV file"""
        if not self.cfg.save_result or not self._paper_cache:
            return

        _metrics = ['MPSNR', 'MSSIM', 'ERGAS', 'time']
        for (scene, case), algo_dict in self._paper_cache.items():
            save_path = os.path.join(self.cfg.save_dir, f"{scene}_{case}.csv")
            noisy_vals = {
                'MPSNR': f'{self.noisy_psnr:.2f}',
                'MSSIM': f'{self.noisy_mssim:.3f}',
                'ERGAS': f'{self.noisy_ergas:.1f}',
                'time': '0.0'
            }
            rows_data = [{'Method': 'Noisy', **noisy_vals}]
            for method, val_dict in algo_dict.items():
                method_vals = {}
                for m in _metrics:
                    method_vals[m] = val_dict[m]
                rows_data.append({'Method': method, **method_vals})
            final_df = pd.DataFrame(rows_data)
            if os.path.isfile(save_path):
                old_df = pd.read_csv(save_path)
                combined_df = pd.concat([old_df, final_df]).drop_duplicates(subset=['Method'], keep='last')
                combined_df.to_csv(save_path, index=False)
            else:
                final_df.to_csv(save_path, index=False)
            self.logger.info(f'Paper-style metrics saved to {save_path}')


