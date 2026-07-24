# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch

from vllm.model_executor.hw_agnostic.custom_op import CustomOp


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# YaRN math.


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@CustomOp.register("deepseek_v4_scaling_rotary_embedding")
class DeepseekV4ScalingRotaryEmbedding(CustomOp):
    """YaRN rotary embedding — V4 variant.

    Differs from V3: RoPE on the LAST ``rotary_dim`` (not first), fp32
    ``cos_sin_cache``, query-only RoPE (``key=None``) supported, and an
    ``inverse`` flag that flips ``sin``.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = float(
            _yarn_get_mscale(scaling_factor, float(mscale))
            / _yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            * attn_factor
        )

        cache_fp32 = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache_fp32, persistent=False)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        return (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            int(self.max_position_embeddings * self.scaling_factor),
            device=inv_freq.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        return torch.cat((cos, sin), dim=-1)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        head_size = query.size(-1)
        query_rot = query[..., -self.rotary_dim :]
        key_rot = key[..., -self.rotary_dim :] if key is not None else None

        if self.rotary_dim < head_size:
            query_pass = query[..., : -self.rotary_dim]
            key_pass = key[..., : -self.rotary_dim] if key is not None else None

        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        if inverse:
            sin = -sin
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        orig_dtype = query.dtype
        query_rot = (query_rot * cos + rotate_fn(query_rot) * sin).to(orig_dtype)
        if key_rot is not None:
            key_rot = (key_rot * cos + rotate_fn(key_rot) * sin).to(orig_dtype)

        if self.rotary_dim < head_size:
            query = torch.cat((query_pass, query_rot), dim=-1)
            key = torch.cat((key_pass, key_rot), dim=-1) if key is not None else None
        else:
            query = query_rot
            key = key_rot
        return query, key


# Reused across layers (DSv4 builds two embeddings per layer).
_ROPE_CACHE: dict[tuple, DeepseekV4ScalingRotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    rope_parameters: dict | None = None,
    is_neox_style: bool = False,
    dtype: torch.dtype | None = None,
) -> DeepseekV4ScalingRotaryEmbedding:
    """DSv4 rope factory. Only ``deepseek_yarn`` / ``deepseek_llama_scaling``
    with ``is_deepseek_v4=True`` are supported."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    if rotary_dim := rope_parameters.get("rope_dim", None):
        rotary_dim = int(rotary_dim)
    else:
        partial = rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_size * partial)

    if not rope_parameters.get("is_deepseek_v4", False):
        raise ValueError(
            "hw-agnostic get_rope only supports DSv4 (is_deepseek_v4=True)."
        )
    if scaling_type not in ("deepseek_yarn", "deepseek_llama_scaling"):
        raise ValueError(
            f"hw-agnostic get_rope only supports rope_type in "
            f"('deepseek_yarn', 'deepseek_llama_scaling'); got {scaling_type!r}."
        )

    rope_parameters_args = tuple(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in rope_parameters.items()
    )
    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dtype,
    )
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]

    scaling_factor = rope_parameters["factor"]
    original_max_position = rope_parameters["original_max_position_embeddings"]
    extra = {
        k: v
        for k, v in rope_parameters.items()
        if k
        in (
            "extrapolation_factor",
            "attn_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
        )
    }
    rope = DeepseekV4ScalingRotaryEmbedding(
        head_size,
        rotary_dim,
        original_max_position,
        base,
        is_neox_style,
        scaling_factor,
        dtype,
        **extra,
    )
    _ROPE_CACHE[key] = rope
    return rope
