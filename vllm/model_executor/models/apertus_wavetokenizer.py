# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus WavTokenizer implementation.

This file provides a clean, self-contained, and statically-instantiated
version of WavTokenizer40 to avoid dynamic imports or external path injection.
"""

import math
import os
import typing as tp
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from einops import rearrange, repeat

from torch.nn.utils import weight_norm

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_distributed() -> bool:
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0) -> None:
    if not is_distributed():
        return
    tensors_list = [
        t for t in tensors if torch.is_floating_point(t) or torch.is_complex(t)
    ]
    if not tensors_list:
        return
    device = tensors_list[0].device
    local_len = len(tensors_list)
    tensor = torch.tensor([local_len], device=device, dtype=torch.long)
    torch.distributed.all_reduce(tensor)
    if tensor.item() != local_len * torch.distributed.get_world_size():
        raise RuntimeError(
            f"Mismatch in number of params: ours is {local_len}, "
            "at least one worker has a different one."
        )

    handles = []
    for t in tensors_list:
        handle = torch.distributed.broadcast(t.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


class SLSTM(nn.Module):
    """LSTM wrapper for SEANet encoder/decoder."""

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.permute(2, 0, 1)
        y, _ = self.lstm(x1)
        y = y.permute(1, 2, 0)
        if self.skip:
            y = y + x
        return y


class ConvLayerNorm(nn.LayerNorm):
    """Convolution-friendly LayerNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = rearrange(x, "b t ... -> b ... t")
        return x


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return nn.utils.spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs: tp.Any
) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0
) -> torch.Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]) -> torch.Tensor:
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    def __init__(
        self,
        *args: tp.Any,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        *args: tp.Any,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: dict[str, tp.Any] | None = None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).",
                stacklevel=2,
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: dict[str, tp.Any] | None = None,
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.Sequence[int] = (3, 1),
        dilations: tp.Sequence[int] = (1, 1),
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict[str, tp.Any] | None = None,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        if norm_params is None:
            norm_params = {}
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        kernel_sizes = list(kernel_sizes)
        dilations = list(dilations)
        assert len(kernel_sizes) == len(dilations), (
            "Kernel sizes count must match dilations count"
        )
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.Sequence[int] = (8, 5, 4, 2),
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict[str, tp.Any] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
    ):
        super().__init__()
        if norm_params is None:
            norm_params = {}
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = 1
        model: list[nn.Module] = [
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            model += [
                act(**activation_params),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.Sequence[int] = (8, 5, 4, 2),
        activation: str = "ELU",
        activation_params: dict | None = None,
        final_activation: str | None = None,
        final_activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict[str, tp.Any] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        if norm_params is None:
            norm_params = {}
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: list[nn.Module] = [
            SConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        for i, ratio in enumerate(self.ratios):
            model += [
                act(**activation_params),
                SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            mult //= 2

        model += [
            act(**activation_params),
            SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(
    x: torch.Tensor, n_categories: int, epsilon: float = 1e-5
) -> torch.Tensor:
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int) -> torch.Tensor:
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(
    samples: torch.Tensor, num_clusters: int, num_iters: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    dim, dtype = samples.shape[-1], samples.dtype
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor) -> None:
        if self.inited:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        broadcast_tensors(self.buffers())

    def replace_(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples: torch.Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... d -> (...) d")

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(
        self, embed_ind: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_ind, self.embed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        return self.postprocess_emb(embed_ind, shape)

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self.dequantize(embed_ind)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        _codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training and self.commitment_weight > 0:
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss = loss + commit_loss * self.commitment_weight
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class LanguageVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs: tp.Any):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(
        self, x: torch.Tensor, n_q: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantized_out, indices, loss = layer(residual)
            all_indices.append(indices)
            all_losses.append(loss)
        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices)
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor
    penalty: torch.Tensor | None = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.vq = LanguageVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(
        self, x: torch.Tensor, frame_rate: int, bandwidth: float | None = None
    ) -> QuantizedResult:
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        nq_choice = [4, 6, 8]
        if self.training:
            choice = int(torch.randint(0, 3, (1,)).item())
            n_q = nq_choice[choice]
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def infer(
        self, x: torch.Tensor, frame_rate: int, bandwidth: float | None = None
    ) -> QuantizedResult:
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = 1
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def get_num_quantizers_for_bandwidth(
        self, frame_rate: int, bandwidth: float | None = None
    ) -> int:
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.0:
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, frame_rate: int) -> float:
        return math.log2(self.bins) * frame_rate

    def encode(
        self, x: torch.Tensor, frame_rate: int, bandwidth: float | None = None
    ) -> torch.Tensor:
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vq.encode(x, n_q=n_q)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.vq.decode(codes)
        return quantized


class EncodecModel(nn.Module):
    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: ResidualVectorQuantizer,
        target_bandwidths: list[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: float | None = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.bandwidth: float | None = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2**self.bits_per_codebook == self.quantizer.bins, (
            "quantizer bins must be a power of 2."
        )

    @property
    def segment_length(self) -> int | None:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> int | None:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def _encode_frame(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    def encode(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride
            assert stride is not None

        encoded_frames = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _decode_frame(
        self, encoded_frame: tuple[torch.Tensor, torch.Tensor | None]
    ) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def decode(
        self, encoded_frames: list[tuple[torch.Tensor, torch.Tensor | None]]
    ) -> torch.Tensor:
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, : x.shape[-1]]


def _linear_overlap_add(frames: list[torch.Tensor], stride: int) -> torch.Tensor:
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset : offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset : offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def convert_audio(
    wav: torch.Tensor, sr: int, target_sr: int, target_channels: int
) -> torch.Tensor:
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(
            f"Impossible to convert from {channels} to {target_channels}"
        )
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


class FeatureExtractor(nn.Module):
    def forward(self, audio: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: tp.Sequence[float] = (1.5, 3.0, 6.0, 12.0),
        train_codebooks: bool = False,
        num_quantizers: int = 1,
        dowmsamples: tp.Sequence[int] = (6, 5, 5, 4),
        vq_bins: int = 16384,
        vq_kmeans: int = 800,
    ):
        super().__init__()
        self.frame_rate = 25
        n_q = num_quantizers
        encoder = SEANetEncoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=list(dowmsamples),
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        decoder = SEANetDecoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=[8, 5, 4, 2],
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        quantizer = ResidualVectorQuantizer(
            dimension=512,
            n_q=n_q,
            bins=vq_bins,
            kmeans_iters=vq_kmeans,
            decay=0.99,
            kmeans_init=True,
        )

        if encodec_model == "encodec_24khz":
            self.encodec = EncodecModel(
                encoder=encoder,
                decoder=decoder,
                quantizer=quantizer,
                target_bandwidths=list(bandwidths),
                sample_rate=24000,
                channels=1,
            )
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. "
                "Supported options are 'encodec_24khz'."
            )
        for param in self.encodec.parameters():
            param.requires_grad = True
        self.bandwidths = list(bandwidths)

    def forward(
        self, audio: torch.Tensor, bandwidth_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            self.encodec.train()
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer(
            emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
        )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty
        return quantized, codes, commit_loss

    def infer(
        self, audio: torch.Tensor, bandwidth_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            self.encodec.train()
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer.infer(
            emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
        )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty
        return quantized, codes, commit_loss


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv1d(
            in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(
                    in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv1d(
                    in_channels, self.out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(temb * torch.sigmoid(temb))[:, :, None, None]
        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h = q.shape
        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = self.proj_out(h_)
        return x + h_


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class AdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.shift = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        return x * scale + shift


class Backbone(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

        self.temb_ch = 0
        block_in = dim
        dropout = 0.1

        pos_net: list[nn.Module] = [
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            AttnBlock(block_in),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True),
        ]
        self.pos_net = nn.Sequential(*pos_net)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, bandwidth_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_net(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class ISTFT(nn.Module):
    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        assert (window_envelope > 1e-11).all()
        y = y / window_envelope
        return y


class FourierHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x_cos = torch.cos(p)
        y_sin = torch.sin(p)
        S = mag * (x_cos + 1j * y_sin)
        audio = self.istft(S)
        return audio


DEFAULT_CONFIG = {
    "model": {
        "init_args": {
            "feature_extractor": {
                "class_path": "EncodecFeatures",
                "init_args": {
                    "encodec_model": "encodec_24khz",
                    "bandwidths": [6.6, 6.6, 6.6, 6.6],
                    "train_codebooks": True,
                    "num_quantizers": 1,
                    "dowmsamples": [6, 5, 5, 4],
                    "vq_bins": 4096,
                    "vq_kmeans": 200,
                },
            },
            "backbone": {
                "class_path": "VocosBackbone",
                "init_args": {
                    "input_channels": 512,
                    "dim": 768,
                    "intermediate_dim": 2304,
                    "num_layers": 12,
                    "adanorm_num_embeddings": 4,
                },
            },
            "head": {
                "class_path": "ISTFTHead",
                "init_args": {
                    "dim": 768,
                    "n_fft": 2400,
                    "hop_length": 600,
                    "padding": "same",
                },
            },
        }
    }
}


def instantiate_class(args: tuple[tp.Any, ...], init: dict[str, tp.Any]) -> tp.Any:
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_path = init["class_path"]
    class_name = class_path.split(".")[-1]

    mapping = {
        "EncodecFeatures": EncodecFeatures,
        "VocosBackbone": VocosBackbone,
        "ISTFTHead": ISTFTHead,
    }

    if class_name in mapping:
        return mapping[class_name](*args, **kwargs)

    raise ValueError(f"Unsupported class_path in configuration: {class_path}")


class OriginalWavTokenizer(nn.Module):
    def __init__(
        self, feature_extractor: FeatureExtractor, backbone: Backbone, head: FourierHead
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> "OriginalWavTokenizer":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_hparams0802(cls, config_path: str | None = None) -> "OriginalWavTokenizer":
        if config_path is not None and os.path.exists(config_path):
            if config_path.endswith(".json"):
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                config = full_config.get("audio_config", DEFAULT_CONFIG)
            else:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
        else:
            config = DEFAULT_CONFIG
        init_args = config["model"]["init_args"]
        feature_extractor = instantiate_class(
            args=(), init=init_args["feature_extractor"]
        )
        backbone = instantiate_class(args=(), init=init_args["backbone"])
        head = instantiate_class(args=(), init=init_args["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained0802(
        cls, config_path: str | None, model_path: str
    ) -> "OriginalWavTokenizer":
        model = cls.from_hparams0802(config_path)
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict_raw = load_file(model_path, device="cpu")
        else:
            raw_state = torch.load(model_path, map_location="cpu")
            if isinstance(raw_state, dict) and "state_dict" in raw_state:
                state_dict_raw = raw_state["state_dict"]
            else:
                state_dict_raw = raw_state
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        features, _, _ = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def encode(
        self, audio_input: torch.Tensor, **kwargs: tp.Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, discrete_codes, _ = self.feature_extractor(audio_input, **kwargs)
        return features, discrete_codes

    @torch.inference_mode()
    def encode_infer(
        self, audio_input: torch.Tensor, **kwargs: tp.Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, discrete_codes, _ = self.feature_extractor.infer(
            audio_input, **kwargs
        )
        return features, discrete_codes

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    @torch.inference_mode()
    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.feature_extractor, EncodecFeatures), (
            "Feature extractor should be an instance of EncodecFeatures"
        )

        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        layers = self.feature_extractor.encodec.quantizer.vq.layers
        tmp = torch.cat([vq.codebook for vq in layers], dim=0)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)
        return features


class WavTokenizerBase(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str | None = None,
        torch_compile: bool = True,
    ):
        super().__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.torch_compile = torch_compile
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def output_sample_rate(self) -> int:
        return 24000

    @property
    def codebook_size(self) -> int:
        return 4096

    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate

    def tokens_from_waveform_samples(self, num_waveform_samples: int) -> int:
        if num_waveform_samples < 0:
            raise ValueError("num_waveform_samples must be non-negative")
        return (num_waveform_samples + self.downsample_rate - 1) // self.downsample_rate

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = audio.to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)
        with torch.no_grad():
            _, discrete_codes = self.model.encode_infer(
                audio, bandwidth_id=bandwidth_id
            )
        if discrete_codes.dim() == 3:
            discrete_codes = discrete_codes.squeeze(0)
        return discrete_codes

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device)
        tokens = tokens.long()
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            features = self.model.codes_to_features(tokens)
            bandwidth_id = torch.tensor([0]).to(self.device)
            audio = self.model.decode(features, bandwidth_id=bandwidth_id)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def forward(self, *args: tp.Any, **kwargs: tp.Any) -> torch.Tensor:
        return self.model(*args, **kwargs)


class WavTokenizer40(WavTokenizerBase):
    name = "wavtokenizer-40"
    tokens_per_second = 40
    model_variant = "large-600"
    _downsample_rate = 600

    def _load_model(self) -> None:
        try:
            ckpt_name = "wavtokenizer_large_unify_600_24k.ckpt"
            sf_name = "wavtokenizer_large_unify_600_24k.safetensors"
            if self.checkpoint:
                if os.path.isdir(self.checkpoint):
                    sf_path = os.path.join(self.checkpoint, sf_name)
                    ckpt_path = os.path.join(self.checkpoint, ckpt_name)
                    if os.path.exists(sf_path):
                        model_path = sf_path
                    elif os.path.exists(ckpt_path):
                        model_path = ckpt_path
                    else:
                        raise FileNotFoundError(
                            f"Neither '{sf_name}' nor '{ckpt_name}' was found in directory '{self.checkpoint}'."
                        )
                elif os.path.isfile(self.checkpoint):
                    model_path = self.checkpoint
                else:
                    raise FileNotFoundError(
                        f"WavTokenizer checkpoint path '{self.checkpoint}' does not exist or is not a file/directory."
                    )
            else:
                raise ValueError(
                    "WavTokenizer checkpoint path must be specified. "
                    "Dynamic downloading on the fly is disabled."
                )

            logger.info(
                "Loading WavTokenizer large-600 (40 tokens/s) from %s", model_path
            )

            config_path = None
            if self.checkpoint:
                if os.path.isdir(self.checkpoint):
                    possible_yaml = os.path.join(self.checkpoint, "config.yaml")
                    possible_json = os.path.join(self.checkpoint, "config.json")
                    if os.path.exists(possible_yaml):
                        config_path = possible_yaml
                    elif os.path.exists(possible_json):
                        config_path = possible_json
                elif os.path.isfile(self.checkpoint):
                    possible_yaml = os.path.join(
                        os.path.dirname(self.checkpoint), "config.yaml"
                    )
                    possible_json = os.path.join(
                        os.path.dirname(self.checkpoint), "config.json"
                    )
                    if os.path.exists(possible_yaml):
                        config_path = possible_yaml
                    elif os.path.exists(possible_json):
                        config_path = possible_json

            self.model = OriginalWavTokenizer.from_pretrained0802(
                config_path, model_path
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            if self.torch_compile and dynamo is not None:
                try:
                    dynamo.config.recompile_limit = int(
                        os.environ.get("WAVTOKENIZER_DYNAMO_RECOMPILE_LIMIT", "64")
                    )
                except Exception:
                    pass
                self.model.feature_extractor.encodec.encoder = torch.compile(
                    self.model.feature_extractor.encodec.encoder,
                    dynamic=False,
                )
                logger.info(
                    "WavTokenizer large-600 loaded successfully (40 tokens/s, compiled)"
                )
            else:
                logger.info(
                    "WavTokenizer large-600 loaded successfully "
                    "(40 tokens/s, eager mode)"
                )

        except Exception as e:
            logger.error("Error loading WavTokenizer: %s", e)
            raise
