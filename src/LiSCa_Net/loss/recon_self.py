import torch
from . import register_loss
import torch.nn.functional as F
operation_seed_counter = 0


@register_loss
class self_L1:
    def __call__(self, target_noisy, model):
        model_output = model['denoiser'](target_noisy)
        return F.l1_loss(target_noisy, model_output, reduction='none')


@register_loss
class self_L2:
    def __call__(self, target_noisy, model):
        model_output = model['denoiser'](target_noisy)
        return F.mse_loss(target_noisy, model_output, reduction='none')


@register_loss
class Loss_Res:
    def __call__(self, noisy_img, model):
        noisy1, noisy2 = pair_downsampler(noisy_img)
        pred1 = noisy1 - model['denoiser'](noisy1)
        pred2 = noisy2 - model['denoiser'](noisy2)
        loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))
        return loss_res


@register_loss
class Loss_Cons:
    def __call__(self, noisy_img, model):
        noisy1, noisy2 = pair_downsampler(noisy_img)
        pred1 = noisy1 - model['denoiser'](noisy1)
        pred2 = noisy2 - model['denoiser'](noisy2)
        noisy_denoised = noisy_img - model['denoiser'](noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
        return loss_cons


def pair_downsampler(x):
    """Downsampling method in ZS-N2N."""
    c = x.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(x.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(x.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(x, filter1, stride=2, groups=c)
    output2 = F.conv2d(x, filter2, stride=2, groups=c)
    return output1, output2

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)
