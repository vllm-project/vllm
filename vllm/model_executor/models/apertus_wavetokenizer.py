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
from einops import rearrange
from torch.nn.utils import weight_norm

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_CONFIG = {
    "model": {
        "init_args": {
            "feature_extractor": {
                "class_path": "EncodecFeatures",
                "init_args":  {
                    "encodec_model":   "encodec_24khz",
                    "bandwidths":      [6.6, 6.6, 6.6, 6.6],
                    "train_codebooks": True,
                    "num_quantizers":  1,
                    "dowmsamples":     [6, 5, 5, 4],
                    "vq_bins":         4096,
                    "vq_kmeans":       200,
                    },
                },
            "backbone":          {
                "class_path": "VocosBackbone",
                "init_args":  {
                    "input_channels":         512,
                    "dim":                    768,
                    "intermediate_dim":       2304,
                    "num_layers":             12,
                    "adanorm_num_embeddings": 4,
                    },
                },
            "head":              {
                "class_path": "ISTFTHead",
                "init_args":  {
                    "dim":        768,
                    "n_fft":      2400,
                    "hop_length": 600,
                    "padding":    "same",
                    },
                },
            },
        },
    }





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


CONV_NORMALIZATIONS = frozenset(
        [
            "none",
            "weight_norm",
            "spectral_norm",
            "time_layer_norm",
            "layer_norm",
            "time_group_norm",
            ],
        )


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return nn.utils.spectral_norm(module)
    return module


def get_norm_module(
        module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs: tp.Any,
        ) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    return nn.Identity()


def get_extra_padding_for_conv1d(
        x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0,
        ) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
        x: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0,
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
                x, kernel_size, stride, padding_total,
                )
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode,
                    )
        return self.conv(x)


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
                    ),
            ]
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                            mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base ** j, 1],
                            norm=norm,
                            norm_params=norm_params,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            compress=compress,
                            true_skip=true_skip,
                            ),
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
        if kmeans_init:
            embed = torch.zeros(codebook_size, dim)
        else:
            embed = torch.empty(codebook_size, dim)
            nn.init.kaiming_uniform_(embed)
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def init_embed_(self, data: torch.Tensor) -> None:
        return

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
            self, embed_ind: torch.Tensor, shape: torch.Size,
            ) -> torch.Tensor:
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_ind, self.embed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        x = self.preprocess(x)
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)
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

    def forward(
            self, x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        loss = torch.tensor([0.0], device=device)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class LanguageVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs: tp.Any):
        super().__init__()
        self.layers = nn.ModuleList(
                [VectorQuantization(**kwargs) for _ in range(num_quantizers)],
                )

    def forward(
            self, x: torch.Tensor, n_q: int | None = None,
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

    def infer(
            self, x: torch.Tensor, frame_rate: int, bandwidth: float | None = None,
            ) -> QuantizedResult:
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = 1
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def get_bandwidth_per_quantizer(self, frame_rate: int) -> float:
        return math.log2(self.bins) * frame_rate


class EncodecModel(nn.Module):
    def __init__(
            self,
            encoder: SEANetEncoder,
            quantizer: ResidualVectorQuantizer,
            target_bandwidths: list[float],
            sample_rate: int,
            channels: int,
            normalize: bool = False,
            segment: float | None = None,
            overlap: float = 0.01,
            name: str = "unset",
            decoder: nn.Module | None = None,
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
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, (
            "quantizer bins must be a power of 2."
        )


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
                    quantizer=quantizer,
                    target_bandwidths=list(bandwidths),
                    sample_rate=24000,
                    channels=1,
                    )
        else:
            raise ValueError(
                    f"Unsupported encodec_model: {encodec_model}. "
                    "Supported options are 'encodec_24khz'.",
                    )
        for param in self.encodec.parameters():
            param.requires_grad = True
        self.bandwidths = list(bandwidths)

    def infer(
            self, audio: torch.Tensor, bandwidth_id: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            self.encodec.train()
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer.infer(
                emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id],
                )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty
        return quantized, codes, commit_loss


def instantiate_class(args: tuple[tp.Any, ...], init: dict[str, tp.Any]) -> tp.Any:
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_path = init["class_path"]
    class_name = class_path.split(".")[-1]

    if class_name == "EncodecFeatures":
        return EncodecFeatures(*args, **kwargs)

    raise ValueError(f"Unsupported class_path in configuration: {class_path}")


class OriginalWavTokenizer(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    @torch.inference_mode()
    def encode_infer(
            self, audio_input: torch.Tensor, **kwargs: tp.Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        features, discrete_codes, _ = self.feature_extractor.infer(
                audio_input, **kwargs,
                )
        return features, discrete_codes


class WavTokenizerBase(nn.Module):
    def __init__(
            self,
            device: str = "cuda",
            checkpoint: str | None = None,
            torch_compile: bool = True,
            audio_config: dict | None = None,
            ):
        super().__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.torch_compile = torch_compile
        self.audio_config = audio_config
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

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = audio.to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)
        with torch.no_grad():
            _, discrete_codes = self.model.encode_infer(
                    audio, bandwidth_id=bandwidth_id,
                    )
        if discrete_codes.dim() == 3:
            discrete_codes = discrete_codes.squeeze(0)
        return discrete_codes


class WavTokenizer40(WavTokenizerBase):
    name = "wavtokenizer-40"
    tokens_per_second = 40
    model_variant = "large-600"
    _downsample_rate = 600

    def _load_model(self) -> None:
        assert self.checkpoint is not None, "checkpoint path must be provided"

        # Check for safetensors or ckpt file
        safetensors_path = os.path.join(self.checkpoint, "wavtokenizer_large_unify_600_24k.safetensors")
        ckpt_path = os.path.join(self.checkpoint, "wavtokenizer_large_unify_600_24k.ckpt")

        # Initialize self.model using config
        config = self.audio_config if self.audio_config is not None else DEFAULT_CONFIG
        init_args = config["model"]["init_args"]
        feature_extractor = instantiate_class(args=(), init=init_args["feature_extractor"])
        self.model = OriginalWavTokenizer(feature_extractor=feature_extractor)

        # Load weights
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict_raw = load_file(safetensors_path, device="cpu")
        elif os.path.exists(ckpt_path):
            raw_state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw_state, dict) and "state_dict" in raw_state:
                state_dict_raw = raw_state["state_dict"]
            else:
                state_dict_raw = raw_state
        else:
            raise FileNotFoundError(
                    f"WavTokenizer checkpoint not found under {self.checkpoint}. "
                    f"Expected either 'wavtokenizer_large_unify_600_24k.safetensors' or "
                    f"'wavtokenizer_large_unify_600_24k.ckpt'.",
                    )

        state_dict = {}
        for k, v in state_dict_raw.items():
            if k.startswith("feature_extractor."):
                if not k.startswith("feature_extractor.encodec.decoder"):
                    state_dict[k] = v

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        if self.torch_compile and dynamo is not None:
            try:
                dynamo.config.recompile_limit = int(
                        os.environ.get("WAVTOKENIZER_DYNAMO_RECOMPILE_LIMIT", "64"),
                        )
            except Exception:
                pass
            self.model.feature_extractor.encodec.encoder = torch.compile(
                    self.model.feature_extractor.encodec.encoder,
                    dynamic=False,
                    )
            logger.info("WavTokenizer large-600 loaded successfully (compiled)")
        else:
            logger.info("WavTokenizer large-600 loaded successfully (eager)")


def build_audio_tokenizer(
        type: str,
        model_path: str,
        device: str = "cuda:0",
        audio_config: dict | None = None,
        ) -> nn.Module:
    if type != "wavtokenizer":
        raise NotImplementedError(f"Unsupported audio tokenizer type: {type}")

    assert audio_config is not None, "audio_config must be provided to build_audio_tokenizer"

    tokenizer_compile = audio_config.get("apertus_audio_tokenizer_compile", False)

    tokenizer = WavTokenizer40(
            device=device,
            checkpoint=model_path,
            torch_compile=tokenizer_compile,
            audio_config=audio_config,
            )
    return tokenizer
