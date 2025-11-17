import math
from ..utils import *
from time import time
from .base import Base
from scipy.linalg import sqrtm, inv
from torch.utils.data import TensorDataset, DataLoader


class TrainerStage2(Base):
    def __init__(self, base_cfg, n_hsi, logger, c_hsi=None):
        cfg = Config(cfg_file='src/LiSCa_Net/conf/config_stage2.yml')
        merge_cfg(cfg.cfg, base_cfg.cfg)
        super().__init__(cfg, n_hsi, logger, c_hsi)

        self.time_train = self.time_test = self.iter = None
        self.epoch = self.iter = 1

        [self.rows, self.cols, self.nbd] = self.n_hsi.shape
        self.npx = self.rows * self.cols
        self.n_hsi = self._pad(self.n_hsi)
        self.train_cfg.dl.batch_size = math.ceil(self.nbd / self.train_cfg.dl.batch_num)

        # transform HSI
        if self.cfg.transform_hsi:
            self.n_hsi, self.Rw = self._transform_hsi(self.n_hsi)

    def train(self):
        # initialize
        self._before_train()

        t1 = time()
        # training
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()
        t2 = time()
        self.time_train = t2 - t1
        self.logger.val(f"Model training time: {self.time_train:.1f}s")

        self._after_train()

    def _run_step(self):
        batch = next(self.train_dl_iter)
        inputs = batch[0].unsqueeze(1).to(self.device)

        # forward, cal losses
        losses = self._forward_fn(self.model, self.loss, inputs)

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        for opt in self.opt.values():
            opt.step()

        # zero grad
        for opt in self.opt.values():
            opt.zero_grad(set_to_none=True)

        # save losses
        for key, value in losses.items():
            self.loss_dict[key] = value.item()

    def _set_dl(self, ds_cfg, shuffle=False):
        """Set train/validation/test dataloader."""
        train_data = np.transpose(self.n_hsi, (2, 0, 1))
        dataloader = DataLoader(
            TensorDataset(torch.from_numpy(train_data).float()),
            batch_size=ds_cfg.batch_size,
            shuffle=shuffle,
            num_workers=ds_cfg.num_workers,
            pin_memory=ds_cfg.pin_memory
        )
        return dataloader

    @torch.no_grad()
    def test(self):
        self._before_test()
        t1 = time()
        denoised = self._test_denoise(self.test_dl, self.denoiser)
        t2 = time()
        self.time_test = t2 - t1
        self.logger.val(f'Testing time: {self.time_test: .2f}s')

        # finish message
        self.logger.highlight(self.logger.get_finish_msg())
        return denoised, self.time_train, self.time_test

    @torch.no_grad()
    def validation(self):
        denoiser = self.model['denoiser'].module
        d_hsi2d = self._test_denoise(self.val_dl, denoiser)
        if self.c_hsi is not None:
            c_hsi2d = self.c_hsi.reshape(self.npx, self.nbd, order='F').T
            mpsnr = MPSNR(d_hsi2d, c_hsi2d)
            self.logger.info(f'{self.status} validation... MPSNR: {mpsnr:.10f}dB')

    @torch.no_grad()
    def _test_denoise(self, dl, denoiser):
        dnd = []
        for batch in dl:
            pred_batch = denoiser(batch[0].unsqueeze(1).to(self.device)).cpu()
            dnd.append(batch[0].unsqueeze(1) - pred_batch)
        dnd = torch.cat(dnd, dim=0).squeeze(1).numpy()
        if self.cfg.transform_hsi:
            dnd_trans = self._retransform_data(dnd.reshape(dnd.shape[0], -1, order='F'))
            dnd = dnd_trans.reshape(dnd.shape, order='F')
        if self.pad_info[2] > 0 or self.pad_info[3] > 0:
            dnd = dnd[:, :self.pad_info[0], :self.pad_info[1]]
        dnd = dnd.reshape(dnd.shape[0], -1, order='F')
        return dnd

    def _pad(self, hsi):
        """Pad HSI to make it square and ensure even dimensions."""
        orig_rows, orig_cols, nbd = hsi.shape

        # Make the HSI square
        max_dim = max(orig_rows, orig_cols)
        pad_rows, pad_cols = max_dim - orig_rows, max_dim - orig_cols
        if pad_rows > 0 or pad_cols > 0:
            hsi = np.pad(hsi, ((0, pad_rows), (0, pad_cols), (0, 0)), mode='reflect')

        # Ensure even dimensions
        row_pad = hsi.shape[0] % 2  # 1 if odd, 0 if even
        col_pad = hsi.shape[1] % 2  # 1 if odd, 0 if even
        if row_pad or col_pad:
            hsi = np.pad(hsi, ((0, row_pad), (0, col_pad), (0, 0)), mode='reflect')
            pad_rows += row_pad
            pad_cols += col_pad
        self.pad_info = (orig_rows, orig_cols, pad_rows, pad_cols)
        return hsi

    def _transform_hsi(self, n_hsi):
        """Transform the noisy HSI to Gaussian iid noise if it is Gaussian non-iid noise."""
        rows, cols, nbd = n_hsi.shape
        n_hsi = n_hsi.reshape(rows * cols, nbd, order='F').T
        _, Rw = self.estNoise(n_hsi)
        Rw_inv = np.linalg.inv(Rw + 1e-8 * np.eye(Rw.shape[0])) if np.linalg.cond(Rw) > 1e12 else inv(Rw)
        n_hsi = np.dot(sqrtm(Rw_inv), n_hsi)
        return n_hsi.T.reshape(rows, cols, nbd, order='F'), Rw

    def estNoise(self, n_hsi, noise_type='additive'):
        """estNoise : Hyperspectral noise estimation.
        This function infers the noise in a hyperspectral data set, by assuming that the reflectance
        at a given band is well modeled by a linear regression on the remaining bands.
        """
        if not isinstance(n_hsi, np.ndarray) or n_hsi.dtype.kind not in 'fc':
            raise ValueError('The data set must be an L x N matrix')
        if noise_type.lower() not in ['additive', 'poisson']:
            raise ValueError('Unknown noise type')
        if self.nbd < 2:
            raise ValueError('Too few bands to estimate the noise.')
        if noise_type.lower() == 'poisson':
            sqy = np.sqrt(np.clip(n_hsi, 0, None))
            u, Ru = self.estAdditiveNoise(sqy)
            x = (sqy - u) ** 2
            w = np.sqrt(x) * u * 2
            Rw = np.dot(w, w.T) / n_hsi.shape[1]
        else:  # Additive noise
            w, Rw = self.estAdditiveNoise(n_hsi)
        return w, Rw

    @staticmethod
    def estAdditiveNoise(n_hsi, small=1e-6):
        """Estimate the additive noise in a hyperspectral data set.
        This function estimates the noise by performing linear regression on the remaining bands for each band.
        """
        nbd, npx = n_hsi.shape
        w = np.zeros((nbd, npx))
        RR = np.dot(n_hsi, n_hsi.T)
        RRi = np.linalg.inv(RR + small * np.eye(nbd))
        for i in range(nbd):
            XX = RRi - np.outer(RRi[:, i], RRi[i, :]) / RRi[i, i]
            RRa = RR[:, i].copy()
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[i] = 0
            w[i, :] = n_hsi[i, :] - beta @ n_hsi
        Rw = np.diag(np.diag(np.dot(w, w.T) / npx))
        return w, Rw

    def _retransform_data(self, d_hsi):
        """Retransform the denoised HSI to the original noise case."""
        return np.dot(sqrtm(self.Rw), d_hsi)
