# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from .base import Conv1dDecodeKernel, M

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

# What FlashInfer's conv1d is built for, on compute capability 10.0 and 10.3.
FLASHINFER_CONV1D_WIDTH = 4
FLASHINFER_CONV1D_ACTIVATION = "silu"
FLASHINFER_CONV1D_DTYPE = torch.bfloat16
# FlashInfer skips rows with this slot.
_PAD_SLOT = -1


class FlashInferConv1dDecodeKernel(Conv1dDecodeKernel):
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
            from flashinfer.mamba import causal_conv1d_update  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @staticmethod
    @cache
    def _fn():
        from flashinfer.mamba import causal_conv1d_update

        return causal_conv1d_update

    @classmethod
    def allocate_buffers(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: MambaSpec,
        max_cudagraph_decodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty((max_cudagraph_decodes,), dtype=torch.int32, device=device)

    @classmethod
    def prepare_metadata(
        cls, buffers: torch.Tensor | None, metadata: M, num_rows: int, padded: bool
    ) -> M:
        # FlashInfer reads one contiguous slot per row, and skips rows whose
        # slot is -1 rather than vLLM's null block.
        assert buffers is not None
        assert metadata.state_indices_tensor_d is not None
        slots = metadata.state_indices_tensor_d[:num_rows, 0]
        indices = buffers[:num_rows] if padded else torch.empty_like(slots)
        indices.copy_(slots, non_blocking=True)
        indices.masked_fill_(indices == NULL_BLOCK_ID, _PAD_SLOT)
        return replace(metadata, conv_state_indices_d=indices)

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
        *,
        max_query_len: int,
    ) -> torch.Tensor:
        assert metadata.conv_state_indices_d is not None
        accepted = metadata.num_accepted_tokens
        return cls._fn()(
            x,
            weight,
            conv_state,
            metadata.conv_state_indices_d,
            cu_seqlens=None if accepted is None else metadata.query_start_loc_d,
            num_accepted_tokens=accepted,
            bias=bias,
        )
