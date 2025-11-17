import torch.nn as nn
from . import register_model


@register_model
class N2N_2D(nn.Module):
    def __init__(self, nch_in: int = 1, nch_out: int = 1, nch_ker: int = 32):
        super(N2N_2D, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(nch_in, nch_ker, 3, padding=1)
        self.conv2 = nn.Conv2d(nch_ker, nch_ker, 3, padding=1)
        self.conv3 = nn.Conv2d(nch_ker, nch_out, 1)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
