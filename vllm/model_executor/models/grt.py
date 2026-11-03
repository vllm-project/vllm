# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GRT (Geometric Rotational Transformer) model.

Every projection is a chain of butterfly-permuted Givens rotations, the token
embedding stores phase angles, and the LM head is a cosine similarity against
the embedding table times one learned temperature. There are no dense weight
matrices to shard, so tensor and pipeline parallelism are not supported.

The checkpoints are fp32 and the rotation chains lose too much precision in
bf16, so parameters and math stay fp32 whatever the model dtype; only q/k/v
are cast to the attention dtype, which is also the KV cache dtype.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.grt import GRTConfig

from .utils import AutoWeightsLoader, maybe_prefix

_GATE_MODES = ("free", "bounded", "bounded_depth", "none")


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False)


def _check_config(config: GRTConfig) -> None:
    hidden, heads = config.hidden_size, config.num_attention_heads
    if hidden % 2 or hidden % heads or (hidden // heads) % 2:
        raise ValueError(
            f"GRT needs an even hidden_size and head_dim; got hidden_size={hidden}, "
            f"num_attention_heads={heads}"
        )
    if config.residual_gate_mode not in _GATE_MODES:
        raise ValueError(
            f"Unknown GRT residual_gate_mode {config.residual_gate_mode!r}"
        )
    if config.residual_norm_mode not in ("none", "pearlnorm"):
        raise ValueError(
            f"Unknown GRT residual_norm_mode {config.residual_norm_mode!r}"
        )


def _butterfly_perms(n_stages: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage `k` pairs dims at stride `2 ** (k % log2(dim // 2))`.

    Checkpoints store these; building them keeps dummy-weight runs in range.
    """
    n_levels = max(1, int(math.log2(dim // 2))) if dim > 2 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        perms.append(
            [
                start + i + offset
                for start in range(0, dim, 2 * stride)
                for i in range(stride)
                for offset in (0, stride)
            ]
        )
    perms_t = torch.tensor(perms)
    return perms_t, perms_t.argsort(dim=-1)


def _compose_gathers(perms: torch.Tensor, inv_perms: torch.Tensor) -> torch.Tensor:
    """One gather per stage boundary: each stage's inverse permutation composed
    with the next stage's permutation, `[K + 1, dim]`."""
    return torch.cat([perms[:1], inv_perms[:-1].gather(1, perms[1:]), inv_perms[-1:]])


def _rotate(
    x0: torch.Tensor, x1: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


def _clifford_features(
    x0: torch.Tensor, x1: torch.Tensor, mixer_bivector: bool
) -> torch.Tensor:
    """`[S0, V1, V2, B12]`: pair energy, both vector parts and the bivector
    with the next pair."""
    x0_next, x1_next = x0.roll(-1, dims=-1), x1.roll(-1, dims=-1)
    # The mixer was trained with the other bivector contraction.
    if mixer_bivector:
        bivector = x0 * x0_next - x1 * x1_next
    else:
        bivector = x0 * x1_next - x1 * x0_next
    return torch.cat([x0 * x0 + x1 * x1, x0, x1, bivector], dim=-1)


def _quantize_symmetric(
    tensor: torch.Tensor, bits: int = 8, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel/per-row weight quantization (`bits` in `{8, 4}`).

    For `bits == 8`, stores signed `int8` values in `[-127, 127]`.
    For `bits == 4`, packs two signed 4-bit values in `[-7, 7]` per `uint8` byte
    along `dim` (which must have even size).
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")
    qmax = 127.0 if bits == 8 else 7.0
    t_f32 = tensor.to(torch.float32)
    scale = t_f32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(t_f32 / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return q, scale
    if q.shape[dim] % 2 != 0:
        raise ValueError(
            f"INT4 packing requires even size along dim={dim}, got {q.shape[dim]}"
        )
    q_u8 = (q + 8).to(torch.uint8)
    low = q_u8.narrow(dim, 0, q_u8.shape[dim] // 2) if False else None
    idx0 = torch.arange(0, q_u8.shape[dim], 2, device=q.device)
    idx1 = idx0 + 1
    low = q_u8.index_select(dim, idx0)
    high = q_u8.index_select(dim, idx1)
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed, scale


def _dequantize_symmetric(
    qweight: torch.Tensor, scale: torch.Tensor, bits: int = 8, dim: int = -1
) -> torch.Tensor:
    """Dequantize a weight tensor produced by `_quantize_symmetric`."""
    if bits == 8:
        return qweight.to(torch.float32) * scale.to(torch.float32)
    if bits == 4:
        low = (qweight & 0x0F).to(torch.int8) - 8
        high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
        d = dim if dim >= 0 else qweight.ndim + dim
        unpacked = torch.stack([low, high], dim=d + 1).flatten(d, d + 1)
        return unpacked.to(torch.float32) * scale.to(torch.float32)
    raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")


def _resolve_quant_bits(
    config: GRTConfig, quant_config: QuantizationConfig | None
) -> int | None:
    bits = getattr(config, "quantize_bits", None)
    if bits in (4, 8):
        return int(bits)
    qcfg = getattr(config, "quantization_config", None)
    if isinstance(qcfg, dict):
        b = qcfg.get("bits") or qcfg.get("num_bits")
        if b in (4, 8):
            return int(b)
        if qcfg.get("load_in_4bit"):
            return 4
        if qcfg.get("load_in_8bit"):
            return 8
    if quant_config is not None:
        name = str(getattr(quant_config, "get_name", lambda: "")()).lower()
        if any(k in name for k in ("int4", "fp4", "awq", "gptq", "4bit")):
            return 4
        if any(k in name for k in ("int8", "fp8", "8bit", "online")):
            return 8
        wbits = getattr(quant_config, "weight_bits", None)
        if wbits in (4, 8):
            return int(wbits)
    return None


def _coupled_chain(
    x: torch.Tensor,
    groups: int,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    gathers: torch.Tensor | None,
    feature_eps: float,
    mixer_bivector: bool = False,
    coupling_scale: torch.Tensor | None = None,
    coupling_bits: int | None = None,
) -> torch.Tensor:
    """Content-modulated rotation chain over `x` of shape `[T, H, D]`.

    `gathers` (`[K + 1, D]`) permute `D`; the pair math runs on `groups` groups
    of `H * D / groups` values with `angles` `[K, groups, P]`, `coupling`
    `[K, groups, 4P, P]` and positive `scales` `[K, groups, 2P]`.
    """
    t, h, d = x.shape
    for k in range(angles.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        g = x.reshape(t, groups, -1)
        x0, x1 = g[..., 0::2], g[..., 1::2]
        features = _clifford_features(x0, x1, mixer_bivector)
        features = features / (features.norm(dim=-1, keepdim=True) + feature_eps)
        ck = (
            _dequantize_symmetric(
                coupling[k], coupling_scale[k], bits=coupling_bits, dim=-2
            )
            if coupling_scale is not None and coupling_bits is not None
            else coupling[k]
        )
        modulation = torch.einsum("tgj,gji->tgi", features, ck)
        theta = angles[k] + torch.tanh(modulation)
        x = (_rotate(x0, x1, theta.cos(), theta.sin()) * scales[k]).view(t, h, d)
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


def _plain_chain(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scales: torch.Tensor,
    gathers: torch.Tensor | None,
) -> torch.Tensor:
    for k in range(cos.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        x = _rotate(x[..., 0::2], x[..., 1::2], cos[k], sin[k]) * scales[k]
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


class GRTPearlNorm(nn.Module):
    """RMS over pair radii: rescales each pair's magnitude, keeps its phase."""

    def __init__(self, size: int):
        super().__init__()
        self.scale = _param(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius_sq = x[..., 0::2].square() + x[..., 1::2].square()
        return x * torch.rsqrt(radius_sq.mean(-1, keepdim=True) + 1e-8) * self.scale


class GRTEmbedding(nn.Module):
    """`hidden_size // 2` phase angles per token plus a log magnitude."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.phase_angles = _param(vocab_size, hidden_size // 2)
        self.log_magnitude = _param(vocab_size)
        self.register_buffer("_embedding_table_q", None, persistent=False)
        self.register_buffer("_embedding_table_scale", None, persistent=False)
        self._quant_bits: int | None = None

    @staticmethod
    def _vectors(angles: torch.Tensor, log_magnitude: torch.Tensor) -> torch.Tensor:
        pairs = torch.stack([angles.cos(), angles.sin()], dim=-1).flatten(-2)
        return pairs * log_magnitude.clamp(-5.0, 5.0).exp().unsqueeze(-1)

    def table(self) -> torch.Tensor:
        if (
            self._embedding_table_q is not None
            and self._embedding_table_scale is not None
            and self._quant_bits is not None
        ):
            return _dequantize_symmetric(
                self._embedding_table_q,
                self._embedding_table_scale,
                bits=self._quant_bits,
                dim=-1,
            )
        return self._vectors(self.phase_angles, self.log_magnitude)

    def quantize_weights(self, bits: int = 8) -> None:
        full_table = self.table()
        qweight, scale = _quantize_symmetric(full_table, bits=bits, dim=-1)
        self.register_buffer("_embedding_table_q", qweight, persistent=False)
        self.register_buffer("_embedding_table_scale", scale, persistent=False)
        self._quant_bits = bits
        self.phase_angles = nn.Parameter(
            torch.empty(0, dtype=torch.float32, device=qweight.device),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            self._embedding_table_q is not None
            and self._embedding_table_scale is not None
            and self._quant_bits is not None
        ):
            q_rows = self._embedding_table_q[input_ids]
            s_rows = self._embedding_table_scale[input_ids]
            return _dequantize_symmetric(q_rows, s_rows, bits=self._quant_bits, dim=-1)
        return self._vectors(
            self.phase_angles[input_ids], self.log_magnitude[input_ids]
        )


class GRTPositionalRotation(nn.Module):
    """Rotates every pair of the stream by `position * exp(log_freqs)`, once,
    at the input."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        n_pairs = config.hidden_size // 2
        log_freqs = torch.arange(n_pairs, dtype=torch.float32) / n_pairs
        log_freqs = -log_freqs * math.log(config.pos_freq_base)
        if config.learnable_rope:
            self.log_freqs = nn.Parameter(log_freqs, requires_grad=False)
        else:
            self.register_buffer("log_freqs", log_freqs, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # fp32 angles: bf16 cannot represent every position above 256.
        angles = positions.to(torch.float32).unsqueeze(-1) * self.log_freqs.exp()
        return _rotate(x[..., 0::2], x[..., 1::2], angles.cos(), angles.sin())


class GRTAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections are rotation chains."""

    def __init__(
        self,
        config: GRTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.n_heads = h = config.num_attention_heads
        self.head_dim = d = config.hidden_size // h
        k, p = config.n_qkv_rotations, d // 2
        for name in ("q", "k", "v", "o"):
            setattr(self, f"{name}_angles", _param(k, h, p))
            setattr(self, f"{name}_scales", _param(k, h, d))
            setattr(self, f"{name}_coupling", _param(k, h, 4 * p, p))

        # Filled after loading. q, k and v read the same input and share the
        # permutations, so they run as one chain of 3H heads.
        f32 = torch.float32
        self.register_buffer(
            "qkv_angles", torch.empty(k, 3 * h, p, dtype=f32), persistent=False
        )
        self.register_buffer(
            "qkv_coupling", torch.empty(k, 3 * h, 4 * p, p, dtype=f32), persistent=False
        )
        self.register_buffer(
            "qkv_gains", torch.empty(k, 3 * h, d, dtype=f32), persistent=False
        )
        self.register_buffer(
            "o_gains", torch.empty(k, h, d, dtype=f32), persistent=False
        )
        self.register_buffer("qkv_coupling_scale", None, persistent=False)
        self.register_buffer("o_coupling_scale", None, persistent=False)
        self._quant_bits: int | None = None
        # A single-stage chain has no butterfly and no stored permutations.
        gathers = None
        if k > 1:
            perms, inv_perms = _butterfly_perms(k, d)
            self.register_buffer("_attn_perms", perms)
            self.register_buffer("_attn_inv_perms", inv_perms)
            gathers = torch.empty(k + 1, d, dtype=torch.long)
        self.register_buffer("gathers", gathers, persistent=False)

        self.attn = Attention(
            h,
            d,
            d**-0.5,
            num_kv_heads=h,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.attn_dtype = torch.get_default_dtype()

    def fold_constants(self) -> None:
        qkv = ("q", "k", "v")
        self.qkv_angles.copy_(torch.cat([getattr(self, f"{n}_angles") for n in qkv], 1))
        if self._quant_bits is None:
            self.qkv_coupling.copy_(
                torch.cat([getattr(self, f"{n}_coupling") for n in qkv], 1)
            )
        self.qkv_gains.copy_(
            F.softplus(torch.cat([getattr(self, f"{n}_scales") for n in qkv], 1))
        )
        self.o_gains.copy_(F.softplus(self.o_scales))
        if self.gathers is not None:
            self.gathers.copy_(_compose_gathers(self._attn_perms, self._attn_inv_perms))

    def quantize_weights(self, bits: int = 8) -> None:
        if self._quant_bits is not None:
            qkv_full = _dequantize_symmetric(
                self.qkv_coupling,
                self.qkv_coupling_scale,
                bits=self._quant_bits,
                dim=-2,
            )
            o_full = _dequantize_symmetric(
                self.o_coupling, self.o_coupling_scale, bits=self._quant_bits, dim=-2
            )
        else:
            qkv = ("q", "k", "v")
            qkv_full = (
                torch.cat([getattr(self, f"{n}_coupling") for n in qkv], 1)
                if self.q_coupling.numel() > 0
                else self.qkv_coupling
            )
            o_full = self.o_coupling
        qkv_q, qkv_s = _quantize_symmetric(qkv_full, bits=bits, dim=-2)
        o_q, o_s = _quantize_symmetric(o_full, bits=bits, dim=-2)
        self.register_buffer("qkv_coupling", qkv_q, persistent=False)
        self.register_buffer("qkv_coupling_scale", qkv_s, persistent=False)
        self.o_coupling = nn.Parameter(o_q, requires_grad=False)
        self.register_buffer("o_coupling_scale", o_s, persistent=False)
        self._quant_bits = bits
        for name in ("q", "k", "v"):
            setattr(
                self,
                f"{name}_coupling",
                nn.Parameter(
                    torch.empty(0, dtype=torch.float32, device=qkv_q.device),
                    requires_grad=False,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h, d = x.shape[0], self.n_heads, self.head_dim
        qkv = _coupled_chain(
            x.view(t, h, d).repeat(1, 3, 1),
            3 * h,
            self.qkv_angles,
            self.qkv_coupling,
            self.qkv_gains,
            self.gathers,
            1e-6,
            coupling_scale=self.qkv_coupling_scale,
            coupling_bits=self._quant_bits,
        )
        q, k, v = (
            part.reshape(t, h * d).to(self.attn_dtype) for part in qkv.split(h, dim=1)
        )
        out = self.attn(q, k, v).view(t, h, d).to(torch.float32)
        out = _coupled_chain(
            out,
            h,
            self.o_angles,
            self.o_coupling,
            self.o_gains,
            self.gathers,
            1e-6,
            coupling_scale=self.o_coupling_scale,
            coupling_bits=self._quant_bits,
        )
        return out.view(t, h * d)


class GRTMixer(nn.Module):
    """Mixes across heads at full width; the coupling is block-diagonal over
    groups of pairs."""

    def __init__(
        self, config: GRTConfig, quant_config: QuantizationConfig | None = None
    ):
        super().__init__()
        d = config.hidden_size
        n_pairs, k = d // 2, config.n_mixer_rotations
        self.group = next((g for g in (16, 8, 4, 2) if n_pairs % g == 0), 1)
        self.n_groups = n_pairs // self.group
        self.angles = _param(k, n_pairs)
        self.scales = _param(k, d)
        self.coupling = _param(k, self.n_groups, 4 * self.group, self.group)
        self.register_buffer("coupling_scale", None, persistent=False)
        self._quant_bits: int | None = None
        perms, inv_perms = _butterfly_perms(k, d)
        self.register_buffer("_perms_stack", perms)
        self.register_buffer("_inv_perms_stack", inv_perms)
        self.register_buffer(
            "gains", torch.empty(k, d, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gathers", torch.empty(k + 1, d, dtype=torch.long), persistent=False
        )

    def fold_constants(self) -> None:
        self.gains.copy_(F.softplus(self.scales))
        self.gathers.copy_(_compose_gathers(self._perms_stack, self._inv_perms_stack))

    def quantize_weights(self, bits: int = 8) -> None:
        full_c = (
            _dequantize_symmetric(
                self.coupling, self.coupling_scale, bits=self._quant_bits, dim=-2
            )
            if self._quant_bits is not None and self.coupling_scale is not None
            else self.coupling
        )
        q, s = _quantize_symmetric(full_c, bits=bits, dim=-2)
        self.coupling = nn.Parameter(q, requires_grad=False)
        self.register_buffer("coupling_scale", s, persistent=False)
        self._quant_bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, d = x.shape
        k = self.angles.shape[0]
        y = _coupled_chain(
            x.view(t, 1, d),
            self.n_groups,
            self.angles.view(k, self.n_groups, self.group),
            self.coupling,
            self.gains.view(k, self.n_groups, 2 * self.group),
            self.gathers,
            # 1e-2, not the projections' 1e-6: the trained mixer depends on it.
            1e-2,
            mixer_bivector=True,
            coupling_scale=self.coupling_scale,
            coupling_bits=self._quant_bits,
        )
        return y.view(t, d)


class GRTFeedForward(nn.Module):
    """Zero-pad to `intermediate_size`, rotate, radial GELU on pair magnitudes,
    rotate, truncate back to `hidden_size`."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = width = config.intermediate_size
        k = config.ffn_n_rotations
        self.expand_angles = _param(k, width // 2)
        self.expand_scales = _param(k, width)
        self.contract_angles = _param(k, width // 2)
        self.contract_scales = _param(k, width)
        # Filled after loading: [cos, sin] of the angles and the positive scales.
        f32 = torch.float32
        self.register_buffer(
            "rotations", torch.empty(2, 2, k, width // 2, dtype=f32), persistent=False
        )
        self.register_buffer(
            "gains", torch.empty(2, k, width, dtype=f32), persistent=False
        )
        gathers = None
        if k > 1:
            perms, inv_perms = _butterfly_perms(k, width)
            self.register_buffer("_ffn_perms", perms)
            self.register_buffer("_ffn_inv_perms", inv_perms)
            gathers = torch.empty(k + 1, width, dtype=torch.long)
        self.register_buffer("gathers", gathers, persistent=False)

    def fold_constants(self) -> None:
        angles = torch.stack([self.expand_angles, self.contract_angles])
        self.rotations.copy_(torch.stack([angles.cos(), angles.sin()], dim=1))
        self.gains.copy_(
            F.softplus(torch.stack([self.expand_scales, self.contract_scales]))
        )
        if self.gathers is not None:
            self.gathers.copy_(_compose_gathers(self._ffn_perms, self._ffn_inv_perms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (expand_cos, expand_sin), (contract_cos, contract_sin) = self.rotations
        x = F.pad(x, (0, self.intermediate_size - self.hidden_size))
        x = _plain_chain(x, expand_cos, expand_sin, self.gains[0], self.gathers)
        pairs = x.view(*x.shape[:-1], -1, 2)
        radius = pairs.norm(dim=-1, keepdim=True)
        x = (pairs * (F.gelu(radius) / (radius + 1e-8))).flatten(-2)
        x = _plain_chain(x, contract_cos, contract_sin, self.gains[1], self.gathers)
        return x[..., : self.hidden_size]


class GRTDecoderLayer(nn.Module):
    """norm -> attention -> norm -> mixer -> norm -> feed-forward, each written
    into the residual stream through a learned gate."""

    def __init__(
        self,
        config: GRTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        d = config.hidden_size
        self.gate_mode = (
            config.residual_gate_mode if config.learnable_residual_gates else "none"
        )
        if self.gate_mode != "none":
            # The mixer gate exists whenever gates do, with or without a mixer.
            self.res_gate_attn = _param()
            self.res_gate_mixer = _param()
            self.res_gate_ffn = _param()
        self.register_buffer(
            "gates", torch.ones(3, dtype=torch.float32), persistent=False
        )

        self.norm1 = GRTPearlNorm(d)
        self.attention = GRTAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.attention"
        )
        self.mixer = (
            GRTMixer(config, quant_config=quant_config)
            if config.n_mixer_rotations > 0
            else None
        )
        self.norm_mixer = GRTPearlNorm(d) if self.mixer is not None else None
        self.norm_ffn = GRTPearlNorm(d)
        self.ffn = GRTFeedForward(config)
        self.post_norm_attn = GRTPearlNorm(d) if config.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(d) if config.peri_ln else None
        self.res_norm_block = (
            GRTPearlNorm(d) if config.residual_norm_mode == "pearlnorm" else None
        )

    def fold_constants(self) -> None:
        if self.gate_mode != "none":
            raw = torch.stack(
                [self.res_gate_attn, self.res_gate_mixer, self.res_gate_ffn]
            )
            if self.gate_mode == "bounded":
                raw = 1.0 + 0.5 * raw.tanh()
            elif self.gate_mode == "bounded_depth":
                raw = 0.55 + 0.45 * raw.tanh()
            self.gates.copy_(raw)
        self.attention.fold_constants()
        if self.mixer is not None:
            self.mixer.fold_constants()
        self.ffn.fold_constants()

    def quantize_weights(self, bits: int = 8) -> None:
        self.attention.quantize_weights(bits)
        if self.mixer is not None:
            self.mixer.quantize_weights(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention(self.norm1(x))
        if self.post_norm_attn is not None:
            y = self.post_norm_attn(y)
        x = x + self.gates[0] * y
        if self.mixer is not None and self.norm_mixer is not None:
            x = x + self.gates[1] * self.mixer(self.norm_mixer(x))
        y = self.ffn(self.norm_ffn(x))
        if self.post_norm_ffn is not None:
            y = self.post_norm_ffn(y)
        x = x + self.gates[2] * y
        if self.res_norm_block is not None:
            x = self.res_norm_block(x)
        return x


class GRTCosineHead(nn.Module):
    """The head's learned temperature; its table is the tied input embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.output_scale = _param()


@support_torch_compile
class GRTModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embedding = GRTEmbedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = GRTPositionalRotation(config)
        self.blocks = nn.ModuleList(
            GRTDecoderLayer(
                config,
                vllm_config.cache_config,
                vllm_config.quant_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def quantize_weights(self, bits: int = 8) -> None:
        self.embedding.quantize_weights(bits)
        for block in self.blocks:
            block.quantize_weights(bits)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        x = self.pos_encoding(inputs_embeds.to(torch.float32), positions)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class GRTForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        _check_config(config)
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(
                "GRT has no dense weights to shard; use tensor_parallel_size=1"
            )
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = GRTModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = GRTCosineHead()
        # Normalised embedding table times the temperature, filled after loading.
        self.register_buffer(
            "head",
            torch.empty(config.vocab_size, config.hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_head_scale", None, persistent=False)
        self._quant_bits: int | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states / (
            hidden_states.norm(dim=-1, keepdim=True) + 1e-6
        )
        head = (
            _dequantize_symmetric(
                self.head, self._head_scale, bits=self._quant_bits, dim=-1
            )
            if self._quant_bits is not None and self._head_scale is not None
            else self.head
        )
        return hidden_states @ head.T

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        table = self.model.embedding.table()
        unit_table = table / (table.norm(dim=-1, keepdim=True) + 1e-6)
        full_head = unit_table * F.softplus(self.lm_head.output_scale)
        self.model.quantize_weights(bits)
        head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
        self.register_buffer("head", head_q, persistent=False)
        self.register_buffer("_head_scale", head_s, persistent=False)
        self._quant_bits = bits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        for block in self.model.blocks:
            block.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.quantize_weights(bits=bits)
        else:
            table = self.model.embedding.table()
            table = table / (table.norm(dim=-1, keepdim=True) + 1e-6)
            self.head.copy_(table * F.softplus(self.lm_head.output_scale))
