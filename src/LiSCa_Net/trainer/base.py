from torch import nn
from ..utils import *
from torch import optim
from ..loss import Loss
from src.utils.logger import Logger
from ..model import get_model_class

status_len = 15


class Base:
    def train(self):
        raise NotImplementedError('define this function for each trainer')

    def test(self):
        raise NotImplementedError('define this function for each trainer')

    def validation(self):
        raise NotImplementedError('define this function for each trainer')

    def _set_dl(self, ds_cfg, shuffle=False):
        raise NotImplementedError('define this function for each trainer')

    def _run_step(self):
        raise NotImplementedError('define this function for each trainer')

    def __init__(self, cfg, n_hsi, logger, c_hsi):
        self.cfg, self.n_hsi, self.logger, self.c_hsi = cfg, n_hsi, logger, c_hsi
        self.train_cfg, self.test_cfg, self.val_cfg = cfg.training, cfg.testing, cfg.validation
        self.ckpt_cfg, self.device = cfg.checkpoint, cfg.device

        self.module = self.model = self.denoiser = self.opt = self.max_epoch = None
        self.iter = self.max_iter = self.loss_dict = self.loss_log = self.epoch = None

    def _set_module(self):
        model_type = self.cfg.model.type
        model_cfg = self.cfg.model.get(model_type, None)
        module = {'denoiser': get_model_class(model_type)(**(model_cfg or {}))}
        return module

    def _set_opt(self):
        """Set optimizer."""
        opt = {}
        for key in self.module:
            opt[key] = self._set_one_opt(self.train_cfg.opt, self.module[key].parameters())
        return opt

    @staticmethod
    def _forward_fn(module, loss, loader_train, weight=None):
        return loss(loader_train, module, weight)

    def _before_train(self):
        # initialing
        self.module = self._set_module()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # training dataset dataloader
        self.train_dl = self._set_dl(self.train_cfg.dl, shuffle=self.train_cfg.dl.shuffle)

        # validation dataset dataloader
        if self.val_cfg.val:
            self.val_dl = self._set_dl(self.val_cfg.dl, shuffle=False)

        # other config
        self.max_epoch = self.train_cfg.max_epoch
        self.loss_dict = {}
        self.loss_log = []
        self.loss = Loss(self.train_cfg.loss)
        self.max_iter = self.train_dl.__len__()

        # set optimizer
        self.opt = self._set_opt()
        for opt in self.opt.values():
            opt.zero_grad(set_to_none=True)

        # wrapping and device setting
        self.model = {key: nn.DataParallel(self.module[key]).to(self.device) for key in self.module}
        for _optim in self.opt.values():
            for state in _optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        # start message
        self.logger = Logger((self.max_epoch, self.max_iter),
                             log_dir=self.cfg.log_dir,
                             log_file_option='a',
                             write_log=self.cfg.write_log
                             )
        self.logger.val(f"{'-' * 20} {self.cfg.name} stage{self.cfg.stage} {'-' * 20}")
        self.logger.val(f"Configuration: {self.cfg.__dict__}")
        self.logger.info(summary(self.module, self.cfg))
        self.logger.start((self.epoch - 1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self._set_status(f"epoch {self.epoch:03d}/{self.max_epoch:03d}")
        self.train_dl_iter = iter(self.train_dl)

        # model training mode
        self._train_mode()

    def _run_epoch(self):
        for self.iter in range(1, self.max_iter + 1):
            self._run_step()
            self._after_step()

    def _after_step(self):
        # print loss
        self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch - 1, self.iter - 1))

    def _after_epoch(self):
        # adjust learning rate
        if self.train_cfg.scheduler.enable:
            self._adjust_lr()

        # save checkpoint
        if self._should_save_ckpt():
            self.save_ckpt()

        # validation
        if self._should_run_val():
            self._eval_mode()
            self._set_status(f'[val {self.epoch:03d}]')
            self.validation()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_status(self, status: str):
        assert len(status) <= status_len, f"Status string exceeds {status_len} characters (now {len(status)})."
        if len(status.split(' ')) == 2:
            s0, s1 = status.split(' ')
            self.status = f"{s0.rjust(status_len // 2)} {s1.ljust(status_len // 2)}"
        else:
            sp = status_len - len(status)
            self.status = ''.ljust(sp // 2) + status + ''.ljust((sp + 1) // 2)

    def _set_one_opt(self, opt_cfg, parameters):
        lr = float(self.train_cfg.init_lr)
        if opt_cfg.type == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt_cfg.SGD.momentum),
                             weight_decay=float(opt_cfg.SGD.weight_decay))
        elif opt_cfg['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt_cfg.Adam.betas)
        elif opt_cfg['type'] == 'AdamW':
            return optim.Adam(parameters, lr=lr, betas=opt_cfg.AdamW.betas,
                              weight_decay=float(opt_cfg.AdamW.weight_decay))
        else:
            raise RuntimeError(f'ambiguous optimizer type: {opt_cfg.type}')

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()

    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _adjust_lr(self):
        sched = self.train_cfg.scheduler
        if sched.type == 'step':
            '''step decreasing scheduler
            Args:
                step_size: step size(epoch) to decay the learning rate
                gamma: decay rate
            '''
            args = sched.step
            if self.epoch % args.step_size == 0:
                for optimizer in self.opt.values():
                    lr_before = optimizer.param_groups[0]['lr']
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_before * float(args.gamma)
        elif sched.type == 'linear':
            '''linear decreasing scheduler
            Args:
                step_size: step size(epoch) to decrease the learning rate
                gamma: decay rate for reset learning rate
            '''
            args = sched.linear
            if not hasattr(self, 'reset_lr'):
                self.reset_lr = float(self.train_cfg.init_lr) * float(args.gamma) ** (
                        (self.epoch - 1) // args.step_size)

            # reset lr to initial value
            if self.epoch % args.tep_size == 0:
                self.reset_lr = float(self.train_cfg.init_lr) * float(args.gamma) ** (self.epoch // args.step_size)
                for optimizer in self.opt.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.reset_lr
            # linear decaying
            else:
                ratio = (self.epoch % args.step_size) / args.step_size
                curr_lr = (1 - ratio) * self.reset_lr
                for optimizer in self.opt.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
        else:
            raise RuntimeError(f"Ambiguous scheduler type: {sched['type']}")

    def _should_save_ckpt(self):
        """Check if checkpoint should be saved"""
        if self.epoch == self.max_epoch:
            return True
        if self.epoch < self.ckpt_cfg.start_epoch:
            return False
        return (self.epoch - self.ckpt_cfg.start_epoch) % self.ckpt_cfg.interval_epoch == 0

    def save_ckpt(self):
        torch.save({'epoch': self.epoch,
                    'model_weight': {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key: self.opt[key].state_dict() for key in self.opt}},
                   os.path.join(self.cfg.ckpt_dir, self._ckpt_name())
                   )

    def _ckpt_name(self):
        return f"{self.cfg.name}_stage{self.cfg.stage}_{self.cfg.datasets.scene_name}_{self.cfg.noise.case}.pth"

    def _should_run_val(self):
        """Check if validation should be run"""
        if not self.val_cfg.val or self.epoch < self.val_cfg.start_epoch:
            return False
        if self.epoch == self.max_epoch:
            return True
        return (self.epoch - self.val_cfg.start_epoch) % self.val_cfg.interval_epoch == 0






    def load_ckpt(self, name=None):
        file_name = (os.path.join(self.cfg.ckpt_dir, name) if name is not None
                     else os.path.join(self.cfg.ckpt_dir, self._ckpt_name()))
        assert os.path.isfile(file_name), f"There is no checkpoint: {file_name}"

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_ckpt = torch.load(str(file_name))
        self.epoch = saved_ckpt['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_ckpt['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.opt:
                self.opt[key].load_state_dict(saved_ckpt['optimizer_weight'][key])

        # print message
        self.logger.note(f"[{self.status}] model loaded: {file_name}")

    def print_loss(self):
        total_loss = sum(self.loss_dict.values())
        self.loss_log.append(total_loss)
        if len(self.loss_log) > 100:
            self.loss_log.pop(0)
        if self._should_print_loss():
            # print status and lr
            loss_str = f"[{self.status}] {self.iter:04d}/{self.max_iter:04d}, lr:{self._get_current_lr():.1e} | "
            # print losses
            avg_loss = np.mean(self.loss_log)
            loss_str += f"avg_100 : {avg_loss:.6f} | "
            for key in self.loss_dict:
                loss_str += f"{key} : {self.loss_dict[key]:.5f} | "
                self.loss_dict[key] = 0.0
            self.logger.info(loss_str)

    def _should_print_loss(self):
        """Check if loss should be printed based on iteration and epoch conditions."""
        # Check iteration conditions
        iter_condition = (self.iter % self.cfg.log.interval_iter == 0 and self.iter != 0) or (
                    self.iter == self.max_iter)
        # Check epoch conditions
        epoch_condition = ((self.epoch - 1) % self.cfg.log.interval_epoch == 0 and self.epoch != 0) or (
                    self.epoch == self.max_epoch)
        return iter_condition and epoch_condition

    def _get_current_lr(self):
        for first_optim in self.opt.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']
        return None

    def _before_test(self):
        # initializing
        self.module = self._set_module()
        self._set_status(f"test")

        # load checkpoint file
        ckpt_name = self.test_cfg.pretrained
        self.load_ckpt(name=ckpt_name)
        self.epoch = self.test_cfg.ckpt_epoch

        # load test dataset and model
        self.test_dl = self._set_dl(self.test_cfg.dl, shuffle=False)
        self.model = {key: nn.DataParallel(self.module[key]).to(self.device) for key in self.module}

        # evaluation mode
        self._eval_mode()

        # start message
        self.logger.highlight(self.logger.get_start_msg())

        # set denoiser
        self.denoiser = self.model['denoiser'].module




