import math
from ..utils import *
from time import time
from .base import Base
from src.utils.metrics import MPSNR
from torch.utils.data import TensorDataset, DataLoader


class TrainerStage1(Base):
    def __init__(self, base_cfg, n_hsi, logger, c_hsi=None):
        cfg = Config(cfg_file='src/LiSCa_Net/conf/config_stage1.yml')
        merge_cfg(cfg.cfg, base_cfg.cfg)
        super().__init__(cfg, n_hsi, logger, c_hsi)
        [self.rows, self.cols, self.nbd] = self.n_hsi.shape
        self.npx = self.rows * self.cols

        self.time_train = self.time_test = self.iter = None
        self.epoch = 1
        self.train_cfg.dl.batch_size = math.ceil(self.npx / self.train_cfg.dl.batch_num)

    def train(self):
        # initializing
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
        inputs = batch[0].to(self.device)
        weights = batch[1].to(self.device)

        # forward, cal losses
        losses = self._forward_fn(self.model, self.loss, inputs, weights)

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
        train_data = self.n_hsi.reshape(self.npx, self.nbd, order='F')
        weight = np.ones_like(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(weight).float())

        dataloader = DataLoader(
            dataset,
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
            output = denoiser(batch[0].to(self.device))
            dnd.append(output.cpu())
        dnd = torch.cat(dnd, dim=0).squeeze(1).numpy().T
        return dnd






