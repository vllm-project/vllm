# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import torch
from torch import nn, pow, sin
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from vllm.transformers_utils.qwen2_code2wav_dit.model.dit import DiT
from vllm.transformers_utils.qwen2_code2wav_dit.model.t2w_cfm import CodecCFM
from vllm.transformers_utils.qwen2_code2wav_dit.model.utils import (
    load_checkpoint)


class CausalConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]),
                                  self.weight, self.bias)


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self,
                 in_features,
                 alpha=1.0,
                 alpha_trainable=True,
                 alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(
            -1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self,
                 in_features,
                 alpha=1.0,
                 alpha_trainable=True,
                 alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(
            -1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width,
                         kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):

    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x,
                       self.filter.expand(C, -1, -1),
                       stride=self.stride,
                       groups=C)

        return out


class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (int(6 * ratio // 2) *
                            2 if kernel_size is None else kernel_size)
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size -
                                                  self.stride) // 2
        self.pad_right = (self.pad * self.stride +
                          (self.kernel_size - self.stride + 1) // 2)
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(x,
                       self.filter.expand(C, -1, -1),
                       stride=self.stride,
                       groups=C)

        return out


class TorchActivation1d(nn.Module):

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(torch.nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
        snake_logscale=True,
        frequency='50hz',
        causal_type='1',
    ):
        super(AMPBlock1, self).__init__()

        self.frequency = frequency
        if self.frequency == '50hz':
            self.convs1 = nn.ModuleList([
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )),
            ])
        else:
            self.convs1 = nn.ModuleList([
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        # padding=get_padding(kernel_size, dilation[0]),
                    )),
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        # padding=get_padding(kernel_size, dilation[1]),
                    )),
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        # padding=get_padding(kernel_size, dilation[2]),
                    )),
            ])
        self.convs1.apply(init_weights)

        if causal_type == '1':
            self.convs2 = nn.ModuleList([
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )),
            ])
        else:
            self.convs2 = nn.ModuleList([
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        # padding=get_padding(kernel_size, 1),
                    )),
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        # padding=get_padding(kernel_size, 1),
                    )),
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        # padding=get_padding(kernel_size, 1),
                    )),
            ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2)  # total number of conv layers

        Activation1d = TorchActivation1d

        if (activation == "snake"
            ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif (
                activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=SnakeBeta(
                    channels, alpha_logscale=snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        if causal_type == '1':
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()
        else:
            self.pre_conv = weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    padding=get_padding(kernel_size, 1),
                ))
            self.pre_conv.apply(init_weights)
            if activation == "snake":
                self.pre_act = Activation1d(
                    activation=Snake(channels, alpha_logscale=snake_logscale))
            elif activation == "snakebeta":
                self.pre_act = Activation1d(activation=SnakeBeta(
                    channels, alpha_logscale=snake_logscale))
            else:
                raise NotImplementedError(
                    "activation incorrectly specified. check the config file and look for 'activation'."
                )

    def forward(self, x):
        if self.frequency == '50hz':
            return self.forward_50hz(x)
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

    def forward_50hz(self, x):
        x = self.pre_conv(x)
        x = self.pre_act(x)
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):

    def __init__(
            self,
            channels,
            kernel_size=3,
            dilation=(1, 3),
            activation=None,
            snake_logscale=True,
    ):
        super(AMPBlock2, self).__init__()

        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )),
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # total number of conv layers

        Activation1d = TorchActivation1d

        if (activation == "snake"
            ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif (
                activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=SnakeBeta(
                    channels, alpha_logscale=snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(
        self,
        frequency: str = '50hz',  # 50hz or 25 hz
        num_mels=80,
        initial_kernel=5,
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[5, 3, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 7, 4, 4, 4, 4],
        resblock_type="1",
        snake_logscale=True,
        activation="snakebeta",
        use_tanh_at_final=False,
        use_bias_at_final=False,
    ):
        super(BigVGAN, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(
            Conv1d(num_mels,
                   upsample_initial_channel,
                   initial_kernel,
                   1,
                   padding=initial_kernel // 2))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if resblock_type == "1" else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList([
                    weight_norm(
                        ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2**(i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        ))
                ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            if frequency == '50hz':
                causal_type = '1'
            else:
                if i > 1:
                    causal_type = '1'
                else:
                    causal_type = '2'
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    resblock(
                        ch,
                        k,
                        d,
                        activation=activation,
                        snake_logscale=snake_logscale,
                        frequency=frequency,
                        causal_type=causal_type,
                    ))

        Activation1d = TorchActivation1d

        # post conv
        if (activation == "snake"
            ):  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif (
                activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        # whether to use bias for the final conv_post. Defaults to True for backward compatibility
        self.use_bias_at_final = use_bias_at_final
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = use_tanh_at_final

    def _normalize(self, S, max_abs_value, min_db):
        return torch.clamp(
            (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value,
            -max_abs_value, max_abs_value)

    def _amp_to_db(self, x, min_level_db):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        min_level = torch.ones_like(x) * min_level
        return 20 * torch.log10(torch.maximum(min_level, x))

    def apm_to_db(self, apm_mel):
        mel_spec = torch.exp(apm_mel)

        mel_spec = self._amp_to_db(mel_spec, -115) - 20
        mel_spec = self._normalize(mel_spec, 1, -115)

        return mel_spec

    def forward(self, x, is_db=False):
        if not is_db:
            x = self.apm_to_db(x)
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0,
                            max=1.0)  # bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Qwen2Code2wavBigvgan(torch.nn.Module):

    def __init__(
        self,
        ckpt,
        frequency: str = '50hz',  # 50hz or 25 hz
        device='cpu',
        with_weight_norm: bool = True,
    ):
        super().__init__()
        self.frequency = frequency
        initial_kernel = 7 if frequency == '50hz' else 5
        resblock_kernel_sizes = [3, 7, 11
                                 ] if frequency == '50hz' else [3, 5, 9, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [
            1, 3, 5
        ]] if frequency == '50hz' else [[1, 3, 5], [1, 3, 5], [1, 3, 5],
                                        [1, 3, 5]]
        self.vocoder = BigVGAN(
            num_mels=80,
            frequency=frequency,
            initial_kernel=initial_kernel,
            upsample_initial_channel=1536,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=[5, 3, 2, 2, 2, 2],
            upsample_kernel_sizes=[11, 7, 4, 4, 4, 4],
            resblock_type="1",
            snake_logscale=True,
            activation="snakebeta",
            use_tanh_at_final=False,
            use_bias_at_final=False,
        )
        if isinstance(ckpt, str):
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        if with_weight_norm:
            loaded_keys = self.vocoder.load_state_dict(state_dict['generator'],
                                                       strict=False)
            self.vocoder.remove_weight_norm()
        else:
            self.vocoder.remove_weight_norm()
            loaded_keys = self.vocoder.load_state_dict(state_dict['generator'],
                                                       strict=False)
        unexpected_keys = [
            k for k in loaded_keys.unexpected_keys
            if 'downsample' not in k and 'upsample' not in k
        ]
        assert unexpected_keys == [], f"Unexpected keys (except downsample/upsample): {loaded_keys.unexpected_keys}"
        missing_keys = [
            k for k in loaded_keys.missing_keys
            if 'downsample' not in k and 'upsample' not in k
        ]
        assert missing_keys == [], f"Missing keys (except downsample/upsample): {missing_keys}"
        self.vocoder.eval()
        self.use_f0 = False
        self.mel_bin = 80
        self.device = device
        self.vocoder = self.vocoder.to(device)

    @torch.no_grad()
    def forward(self, mel, wav=None):
        if len(mel.shape) != 3:
            mel = mel.unsqueeze(0)

        if mel.shape[-1] == self.mel_bin:
            mel = mel.transpose(1, 2)

        mel = mel.to(self.device)
        y_g_hat = self.vocoder(mel)
        audio = y_g_hat.squeeze().cpu()
        return audio

    def cache_forward(self, mel, future_cache_size, past_cache_size):
        if len(mel.shape) != 3:
            mel = mel.unsqueeze(0)

        if mel.shape[-1] == self.mel_bin:
            mel = mel.transpose(1, 2)

        mel = mel.to(self.device)
        y_g_hat = self.vocoder(mel,
                               past_cache_size=past_cache_size,
                               future_cache_size=future_cache_size)
        audio = y_g_hat.squeeze().detach().cpu()
        return audio


class Qwen2Code2wavDit(torch.nn.Module):

    def __init__(
        self,
        ckpt,
        frequency: str = '50hz',  # 50hz or 25 hz
        device='cpu',
    ):
        super().__init__()
        self.freqnecy = frequency
        self.device = device
        self.dit = DiT(
            dim=1024,
            depth=22 if frequency == '50hz' else 32,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            use_codec=True,
            repeats=2 if frequency == '50hz' else 4,
            attn_processor='stream_block_sr'
            if frequency == '50hz' else 'stream_block_8_L_4',
            text_num_embeds=8193 if frequency == '50hz' else 32769,
            mel_dim=80,
        )
        self.mel_spec_kwargs = dict(
            target_sample_rate=16000,
            n_mel_channels=80,
            hop_length=160,
        )
        self.odeint_kwargs = dict(
            method="rk4" if frequency == '50hz' else "euler", )
        self.cfm_model = CodecCFM(
            transformer=self.dit,
            mel_spec_kwargs=self.mel_spec_kwargs,
            odeint_kwargs=self.odeint_kwargs,
        ).to(device)
        self.cfm_model = load_checkpoint(self.cfm_model,
                                         ckpt,
                                         device,
                                         use_ema=True)

    def sample(self,
               cond,
               ref_mel,
               codec,
               steps=10,
               cfg_strength=0.5,
               sway_sampling_coef=-1.0):
        y_all = torch.randn([1, 30000, 80],
                            device=self.device,
                            dtype=ref_mel.dtype)
        expect_y_len = codec.shape[1] * (2 if self.freqnecy == '50hz' else 4)
        y0 = y_all[:, :expect_y_len]
        with torch.inference_mode():
            generated, _ = self.cfm_model.sample(
                cond=cond,
                ref_mel=ref_mel,
                codec=codec,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                y0=y0)
        generated = generated.to(torch.float32)
        generated_mel_spec = generated.permute(0, 2, 1)
        return generated_mel_spec

    def fast_block_sample(
        self,
        cond,
        codec,
        ref_mel,
        y0,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
    ):
        return self.cfm_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )


class Qwen2Code2wav(torch.nn.Module):

    def __init__(
            self,
            dit_ckpt,
            bigvgan_ckpt,
            device='cpu',
            with_weight_norm: bool = True,
            frequency: str = '50hz',  # 50hz or 25 hz
    ):
        super().__init__()
        self.freqnecy = frequency
        self.code2wav_dit_model = Qwen2Code2wavDit(ckpt=dit_ckpt,
                                                   frequency=frequency,
                                                   device=device)
        self.code2wav_bigvgan_model = Qwen2Code2wavBigvgan(
            ckpt=bigvgan_ckpt,
            frequency=frequency,
            device=device,
            with_weight_norm=with_weight_norm)
        self.device = device

    def forward(self, cond, ref_mel, codec):
        generated_mel = self.code2wav_dit_model.sample(cond, ref_mel, codec)
        generated_mel = generated_mel.permute(0, 2, 1)
        waveform = self.code2wav_bigvgan_model(generated_mel)
        return waveform

    def init_variables(self, cond, ref_mel, codec_all, bs_mel):
        self.bs_codec = bs_mel // (2 if self.freqnecy == '50hz' else 4)
        self.past_cache_size = bs_mel * (2 if self.freqnecy == '50hz' else 4)
        self.future_cache_size = bs_mel * 1
        self.chunk_size = bs_mel * (3 if self.freqnecy == '50hz' else 1)
        self.gt_codec_len = codec_all.shape[1]
        self.gt_mel_len = (2 if self.frequency == "50hz" else
                           4) * self.gt_codec_len
        if 0 < self.gt_mel_len <= bs_mel * 4:
            self.n_iter = 1
        else:
            self.n_iter = math.ceil(
                (self.gt_mel_len - self.future_cache_size) / self.chunk_size)
        self.future_size = 20 if self.freqnecy == '50hz' else 13
        self.past_size = 20 if self.freqnecy == '50hz' else 51
        self.generated_list = []
        self.audio_list3 = []
        self.y_all = torch.randn([1, 30000, 80],
                                 device=self.device,
                                 dtype=ref_mel.dtype)

    def process_initial_chunk(self, cond, ref_mel, codec_all, y_all, steps):
        factor = 2 if self.freqnecy == '50hz' else 4
        y0 = y_all[:, :self.chunk_size + self.future_cache_size]
        codec = codec_all[:, :(self.chunk_size + self.future_cache_size) //
                          factor]
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )
        self.generated_list.append(
            generated.to(torch.float32)[:, :self.chunk_size, :])
        mel = self.generated_list[0]
        audio = self.code2wav_bigvgan_model(mel)
        audio_output = audio[:-self.future_size * 240]
        self.audio_list3.append(audio_output)

    def process_little_chunk(self, cond, ref_mel, codec_all, y_all, steps):
        y0 = y_all[:, :self.gt_mel_len]
        codec = codec_all
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )
        self.generated_list.append(generated.to(torch.float32)[:, :, :])
        mel = self.generated_list[0]
        audio = self.code2wav_bigvgan_model(mel)
        audio_output = audio
        self.audio_list3.append(audio_output)

    def process_subsequent_chunks(self, cond, ref_mel, codec_all, y_all, i,
                                  steps):
        factor = 2 if self.freqnecy == '50hz' else 4
        start_index = max(i * self.chunk_size - self.past_cache_size, 0)
        end_index = min((i + 1) * self.chunk_size + self.future_cache_size,
                        self.gt_mel_len)
        y0 = y_all[:, start_index:end_index]
        codec = codec_all[:, start_index // factor:end_index // factor]
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )

        if self.freqnecy == '50hz':
            if start_index == 0:
                mel = self.generated_list[0]
                self.generated_list.append(
                    generated.to(torch.float32)
                    [:, i * self.chunk_size:-self.future_cache_size, :])
            else:
                self.generated_list.append(
                    generated.to(torch.float32)
                    [:, self.past_cache_size:-self.future_cache_size, :])
                mel = torch.cat([
                    self.generated_list[i - 1][:, -self.future_size * 2:, :],
                    self.generated_list[i]
                ],
                                dim=1)
        else:
            if start_index == 0:
                mel = self.generated_list[0]
                self.generated_list.append(
                    generated.to(torch.float32)
                    [:, i * self.chunk_size:-self.future_cache_size, :])
            else:
                self.generated_list.append(
                    generated.to(torch.float32)
                    [:, self.past_cache_size:-self.future_cache_size, :])
                if len(self.generated_list) <= 2:
                    mel = torch.cat(self.generated_list, dim=1)
                else:  # all past mel length >= self.past_size + self.future_size
                    mel = torch.cat([
                        self.generated_list[i - 2], self.generated_list[i - 1],
                        self.generated_list[i]
                    ],
                                    dim=1)

        audio = self.code2wav_bigvgan_model(mel)

        if self.freqnecy == '50hz':
            audio_output = audio[self.future_size * 240:-self.future_size *
                                 240]
        else:
            if len(self.generated_list) <= 2:
                audio_output = audio[(self.past_size - self.chunk_size) *
                                     240:-self.future_size * 240]
            else:  # all past mel length >= self.past_size + self.future_size
                audio_output = audio[self.past_size * 240:-self.future_size *
                                     240]
        self.audio_list3.append(audio_output)

    def process_final_chunk(self, cond, ref_mel, codec_all, y_all, steps):
        factor = 2 if self.freqnecy == '50hz' else 4
        start_index = max(
            (self.n_iter - 1) * self.chunk_size - self.past_cache_size, 0)
        end_index = self.gt_codec_len * factor
        y0 = y_all[:, start_index:end_index]
        codec = codec_all[:, start_index // factor:self.gt_codec_len]
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )
        self.generated_list.append(
            generated.to(torch.float32)[:, self.past_cache_size:, :])
        if self.freqnecy == '50hz':
            mel = torch.cat([
                self.generated_list[-2][:, -self.future_size * 2:, :],
                self.generated_list[-1]
            ],
                            dim=1)
        else:
            if len(self.generated_list) <= 2:
                mel = torch.cat(self.generated_list, dim=1)
            else:
                mel = torch.cat([
                    self.generated_list[-3], self.generated_list[-2],
                    self.generated_list[-1]
                ],
                                dim=1)
        audio = self.code2wav_bigvgan_model(mel)
        if self.freqnecy == '50hz':
            audio_output = audio[self.future_size * 240:]
        else:
            if len(self.generated_list) <= 2:
                audio_output = audio[(self.past_size - self.chunk_size) * 240:]
            else:
                audio_output = audio[self.past_size * 240:]
        self.audio_list3.append(audio_output)

    def get_full_audio(self):
        audio = torch.cat(self.audio_list3, dim=0)
        return audio

    def fast_forward(self, cond, ref_mel, codec, steps=10, bs_mel=24):
        if self.freqnecy == "50hz":
            assert self.bs_mel == 24
        else:
            assert self.bs_mel == 32
        self.init_variables(cond, ref_mel, codec, bs_mel)
        with torch.inference_mode():
            if self.n_iter <= 0:
                return
            if self.n_iter == 1:
                self.process_little_chunk(cond, ref_mel, codec, self.y_all,
                                          steps)
            else:
                self.process_initial_chunk(cond, ref_mel, codec, self.y_all,
                                           steps)
                for i in range(1, self.n_iter - 1):
                    self.process_subsequent_chunks(cond, ref_mel, codec,
                                                   self.y_all, i, steps)

                self.process_final_chunk(cond, ref_mel, codec, self.y_all,
                                         steps)
            return self.get_full_audio()
