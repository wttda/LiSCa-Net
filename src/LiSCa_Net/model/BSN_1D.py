import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from . import register_model
import torch.nn.functional as F


@register_model
class BSN_1D(nn.Module):
    def __init__(
        self,
        nch_in: int = 1,
        nch_out: int = 1,
        blindspot: bool = True,
        zero_output_weights: bool = False,
        depth: int = 6,
        nch_ker: int = 10,
    ):
        super(BSN_1D, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv1d = ShiftConv1d if self.blindspot else nn.Conv1d
        self.depth = depth
        self.nch_ker = nch_ker

        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            if blindspot:
                return nn.Sequential(Shift1d(1), max_pool)
            return max_pool

        # First encoder block
        self.encode_first = nn.Sequential(
            self.Conv1d(nch_in, self.nch_ker, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv1d(self.nch_ker, self.nch_ker, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            _max_pool_block(nn.MaxPool1d(2)),
        )

        # Middle encoder blocks
        self.encode_middle = nn.ModuleList()
        for i in range(self.depth - 1):
            block = nn.Sequential(
                self.Conv1d(self.nch_ker, self.nch_ker, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                _max_pool_block(nn.MaxPool1d(2)),
            )
            self.encode_middle.append(block)

        # Last encoder block
        self.encode_last = nn.Sequential(
            self.Conv1d(self.nch_ker, self.nch_ker, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        self.decode_last = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        self.decode_middle = nn.ModuleList()
        for i in range(self.depth - 1):
            in_ch = self.nch_ker * 2 if i == 0 else self.nch_ker * 3
            out_ch = self.nch_ker * 2
            block = nn.Sequential(
                self.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            self.decode_middle.append(block)

        # First decoder block
        self.decode_first = nn.Sequential(
            self.Conv1d(self.nch_ker * 2 + nch_in, self.nch_ker * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            self.shift = Shift1d(1)
            nin_a_io = self.nch_ker * 4
        else:
            nin_a_io = self.nch_ker * 2

        self.output_conv = self.Conv1d(self.nch_ker * 2, nch_out, 1)
        self.output_block = nn.Sequential(
            self.Conv1d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv1d(nin_a_io, self.nch_ker * 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv
        )

        # Initialize weights
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        if self.blindspot:
            x = torch.cat([rotate1d(x, rot) for rot in (0, 180)], dim=0)

        # Encoder
        skip_lengths = [x.size(-1)]
        pool = self.encode_first(x)
        pools = [pool]
        skip_lengths.append(pool.size(-1))
        for encode_block in self.encode_middle:
            pool = encode_block(pool)
            pools.append(pool)
            skip_lengths.append(pool.size(-1))
        encoded = self.encode_last(pool)

        # Decoder
        upsamples = self.decode_last(encoded)
        pools.pop()
        for i, decode_block in enumerate(self.decode_middle):
            if upsamples.size(-1) < skip_lengths[-(i + 2)]:
                diff = skip_lengths[-(i + 2)] - upsamples.size(-1)
                upsamples = F.pad(upsamples, (diff // 2 + 1, diff // 2))
            concat = torch.cat((upsamples, pools.pop()), dim=1)
            upsamples = decode_block(concat)

        if upsamples.size(-1) < skip_lengths[0]:
            diff = skip_lengths[0] - upsamples.size(-1)
            upsamples = F.pad(upsamples, (diff // 2 + 1, diff // 2))
        concat = torch.cat((upsamples, x), dim=1)
        decoded = self.decode_first(concat)

        # Output
        if self.blindspot:
            rotated_batch = torch.chunk(self.shift(decoded), 2, dim=0)
            x = torch.cat([rotate1d(rotated, rot) for rotated, rot in zip(rotated_batch, (0, 180))], dim=1)
        output = self.output_block(x)

        return output.squeeze()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")


class ShiftConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shift = Shift1d(self.kernel_size[0] // 2)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x


class Crop1d(nn.Module):
    """Crop input using slicing."""
    def __init__(self, crop: Tuple[int, int]):
        """Args:
            crop (Tuple[int, int]): (left, right) amount to crop from the start and end.
        """
        super().__init__()
        self.crop = crop
        assert len(crop) == 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x (Tensor): Input tensor of shape (batch, channel, length)

        Returns (Tensor): Cropped tensor
        """
        left, right = self.crop
        start, end = left, x.shape[-1] - right
        return x[..., start:end]


class Shift1d(nn.Module):
    """Shift a 1D signal left or right."""
    def __init__(self, shift: int):
        super().__init__()
        self.shift = shift
        a, b = abs(shift), 0
        if shift < 0:
            a, b = b, a

        self.pad = nn.ConstantPad1d((a, b), 0.0)
        self.crop = Crop1d((b, a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)


def rotate1d(x: torch.Tensor, angle: int) -> torch.Tensor:
    """Rotate 1D signal by angle."""
    if angle == 0:
        return x
    elif angle == 180:
        return x.flip(-1)
    else:
        raise NotImplementedError("Only 0 or 180 degree rotation supported")
