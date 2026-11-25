# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

from .base import Conv1dPrefillKernel, M

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

# What FlashInfer's conv1d is built for, on compute capability 10.0 and 10.3.
FLASHINFER_CONV1D_WIDTH = 4
FLASHINFER_CONV1D_ACTIVATION = "silu"
FLASHINFER_CONV1D_DTYPE = torch.bfloat16
# FlashInfer inlines the prefill chunk map up to this many sequences.
_MAX_INLINE_PREFILLS = 32


class FlashInferConv1dPrefillKernel(Conv1dPrefillKernel):
    backend = MambaBackendEnum.FLASHINFER

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CUDA

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability | None) -> bool:
        return capability is not None and (capability.major, capability.minor) in (
            (10, 0),
            (10, 3),
        )

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype == FLASHINFER_CONV1D_DTYPE

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = []
        if cfg.conv_width != FLASHINFER_CONV1D_WIDTH:
            reasons.append(f"width {cfg.conv_width} not supported")
        if cfg.conv_activation != FLASHINFER_CONV1D_ACTIVATION:
            reasons.append(f"activation {cfg.conv_activation} not supported")
        if isinstance(cfg.kv_cache_spec, MambaSpec):
            state_dtype = cfg.kv_cache_spec.dtypes[0]
            if state_dtype != FLASHINFER_CONV1D_DTYPE:
                reasons.append(f"conv state dtype {state_dtype} not supported")
        return reasons

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.mamba import causal_conv1d_fn  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def register_warmup(
        cls,
        *,
        dim: int,
        state_len: int,
        dim_first: bool,
        x_row_stride: int,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype | None,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        _WARMUP.register_warmup(
            _Geometry(
                dim, state_len, dim_first, x_row_stride, weight_dtype, bias_dtype
            ),
            max_num_seqs,
            max_num_batched_tokens,
        )

    @classmethod
    def prepare_metadata(cls, metadata: M, query_start_loc_p_cpu: torch.Tensor) -> M:
        # With speculative blocks the slots are a strided block-table column,
        # and FlashInfer reads contiguous ones. A separate copy, as the SSD
        # kernels are specialized on the strided column. Contiguous slots get
        # no copy, so forward reads the ones update_block_table replaces; it
        # runs only without speculative decoding, where they are contiguous.
        indices = metadata.state_indices_tensor_p
        assert indices is not None
        if indices.is_contiguous():
            return metadata
        return replace(metadata, conv_state_indices_p=indices.contiguous())

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
    ) -> torch.Tensor:
        assert metadata.query_start_loc_p is not None
        indices = metadata.conv_state_indices_p
        if indices is None:
            indices = metadata.state_indices_tensor_p
        assert indices is not None
        return cls._fn()(
            x.T,
            weight,
            conv_states,
            metadata.query_start_loc_p,
            indices,
            metadata.has_initial_states_p,
            bias=bias,
        ).T

    @staticmethod
    @cache
    def _fn():
        from flashinfer.mamba import causal_conv1d_fn

        return causal_conv1d_fn


@dataclass(frozen=True, slots=True)
class _Geometry:
    """A layer's conv, per TP rank, as FlashInfer specializes on it."""

    dim: int
    state_len: int
    dim_first: bool
    x_row_stride: int
    weight_dtype: torch.dtype
    bias_dtype: torch.dtype | None


@dataclass(frozen=True, slots=True)
class _WarmupCase:
    geometry: _Geometry
    num_seqs: int
    num_tokens: int


class _PrefillWarmup(VllmJitKernel[_WarmupCase]):
    """Compiles FlashInfer's conv1d prefill variants for a geometry, which
    FlashInfer otherwise compiles on first use."""

    def get_warmup_keys(
        self, geometry: _Geometry, max_num_seqs: int, max_num_batched_tokens: int
    ) -> list[_WarmupCase]:
        """One prefill per variant a batch can reach: one sequence, a few
        (chunk map inlined) or many, each with FlashInfer's short and long
        token tile, which it picks by tokens per sequence."""
        cases = []
        for num_seqs in (1, 2, _MAX_INLINE_PREFILLS + 1):
            long_tile = 64 if num_seqs > _MAX_INLINE_PREFILLS else 128
            for num_tokens in (num_seqs, long_tile * num_seqs):
                if num_seqs <= max_num_seqs and num_tokens <= max_num_batched_tokens:
                    cases.append(_WarmupCase(geometry, num_seqs, num_tokens))
        return cases

    @torch.inference_mode()
    def compile(self, compile_key: _WarmupCase) -> None:
        from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

        g, num_seqs = compile_key.geometry, compile_key.num_seqs
        device = torch.device(current_platform.device_type)
        dtype = FLASHINFER_CONV1D_DTYPE
        # Token-major rows of a wider projection and a conv state pool in the
        # layer's layout, as the layer passes them.
        rows = torch.randn(
            compile_key.num_tokens, g.x_row_stride, device=device, dtype=dtype
        )
        if g.dim_first:
            state = torch.zeros(
                num_seqs, g.dim, g.state_len, device=device, dtype=dtype
            )
        else:
            state = torch.zeros(
                num_seqs, g.state_len, g.dim, device=device, dtype=dtype
            ).transpose(1, 2)
        weight = torch.randn(
            g.dim, FLASHINFER_CONV1D_WIDTH, device=device, dtype=g.weight_dtype
        )
        bias = (
            None
            if g.bias_dtype is None
            else torch.randn(g.dim, device=device, dtype=g.bias_dtype)
        )
        lengths = torch.full(
            (num_seqs,), compile_key.num_tokens // num_seqs, dtype=torch.int32
        )
        lengths[-1] += compile_key.num_tokens % num_seqs
        query_start_loc = torch.zeros(num_seqs + 1, dtype=torch.int32)
        query_start_loc[1:] = lengths.cumsum(0)
        kernel = FlashInferConv1dPrefillKernel
        metadata = kernel.prepare_metadata(
            BaseMambaAttentionMetadata(
                num_prefills=num_seqs,
                num_prefill_tokens=compile_key.num_tokens,
                num_decodes=0,
                num_decode_tokens=0,
                num_reqs=num_seqs,
                # vLLM always passes the flags, which selects compiled variants.
                has_initial_states_p=torch.arange(num_seqs, device=device) % 2 == 1,
                query_start_loc_p=query_start_loc.to(device),
                state_indices_tensor_p=torch.arange(
                    num_seqs, dtype=torch.int32, device=device
                ),
                state_indices_tensor_d=None,
                query_start_loc_d=None,
                num_accepted_tokens=None,
                seq_lens=lengths.to(device),
            ),
            query_start_loc,
        )
        kernel.forward(
            rows[:, : g.dim].T,
            state,
            weight,
            bias,
            FLASHINFER_CONV1D_ACTIVATION,
            metadata,
        )


_WARMUP = _PrefillWarmup()
