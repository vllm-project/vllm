# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

from .base import MambaSSDKernel, SSDLayerParams, SSDScanWarmup, SSDWarmupKey

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

# Shapes and activation dtype FlashInfer's SSD scan is built for.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128
FLASHINFER_SSD_DTYPE = torch.bfloat16


class FlashInferSSDKernel(MambaSSDKernel):
    backend = MambaBackendEnum.FLASHINFER

    @classmethod
    def _register_warmups(cls, key: SSDWarmupKey) -> None:
        super()._register_warmups(key)
        _SCAN_WARMUP.register_warmup(key)

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CUDA

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability | None) -> bool:
        return capability is not None and capability.major == 10

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype == FLASHINFER_SSD_DTYPE

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.mamba import ssd_combined_fwd  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = []
        if isinstance(cfg.kv_cache_spec, MambaSpec):
            _, head_dim, state_size = cfg.kv_cache_spec.shapes[1]
            if head_dim != FLASHINFER_SSD_HEAD_DIM:
                reasons.append(
                    f"FlashInfer SSD requires head_dim {FLASHINFER_SSD_HEAD_DIM}, "
                    f"got {head_dim}"
                )
            if state_size != FLASHINFER_SSD_STATE_SIZE:
                reasons.append(
                    "FlashInfer SSD requires ssm_state_size "
                    f"{FLASHINFER_SSD_STATE_SIZE}, got {state_size}"
                )
        return reasons

    @classmethod
    def prepare_metadata(
        cls,
        metadata: "Mamba2AttentionMetadata",
        kv_cache_spec: MambaSpec,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> "Mamba2AttentionMetadata":
        query_start_loc_p = metadata.query_start_loc_p
        assert query_start_loc_p is not None
        device = query_start_loc_p.device
        num_prefills = query_start_loc_p_cpu.numel() - 1
        states_shape = (num_prefills, *kv_cache_spec.shapes[1])
        state_dtype = kv_cache_spec.dtypes[1]
        if not metadata.prep_initial_states:
            metadata = replace(
                metadata,
                zero_initial_states_p=torch.zeros(
                    states_shape, dtype=state_dtype, device=device
                ),
            )
        checkpoint = metadata.checkpoint_meta
        if checkpoint is None:
            return metadata
        offsets = checkpoint.checkpoint_offsets
        # Each row's checkpoint goes to its own row of the output; rows without
        # one have no token to capture.
        return replace(
            metadata,
            checkpoint_chunk_idx=torch.arange(
                num_prefills, dtype=torch.int32, device=device
            ),
            checkpoint_token_indices_p=torch.where(
                offsets > 0, query_start_loc_p[:-1] + offsets, -1
            ),
            checkpoint_states_p=torch.empty(
                states_shape, dtype=state_dtype, device=device
            ),
        )

    @classmethod
    def _scan(
        cls,
        layer: SSDLayerParams,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from flashinfer.mamba import ssd_combined_fwd

        # FlashInfer takes the state dtype from the initial states, so batches
        # without cached state pass zeros.
        _, final_states = ssd_combined_fwd(
            x.unsqueeze(0),
            dt.unsqueeze(0),
            layer.A,
            B.unsqueeze(0),
            C.unsqueeze(0),
            D=layer.D,
            dt_bias=layer.dt_bias,
            dt_softplus=True,
            initial_states=initial_states,
            cu_seqlens=metadata.query_start_loc_p,
            checkpoint_token_indices=metadata.checkpoint_token_indices_p,
            checkpoint_state_slots=metadata.checkpoint_chunk_idx,
            checkpoint_states=metadata.checkpoint_states_p,
            out=out.unsqueeze(0),
        )
        return final_states, metadata.checkpoint_states_p


_SCAN_WARMUP = SSDScanWarmup(FlashInferSSDKernel)
