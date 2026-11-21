# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    selective_state_update_replayssm_flashinfer,
)
from vllm.platforms.interface import PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

from .base import M, SSUBuffers, SSULayer
from .replayssm_base import ReplaySSMLayer
from .replayssm_tracker import TrackerBuffers, TrackerReplaySSMKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

# FlashInfer's checkpointing_ssu replays at most this many cached tokens.
MAX_BUFFER_LEN = 16


@dataclass
class FlashInferReplaySSMBuffers(TrackerBuffers):
    # checkpointing_ssu's two-kernel scratch, for up to max_num_seqs rows.
    cb_scaled: torch.Tensor
    cumAdt_vec: torch.Tensor
    cb_old: torch.Tensor


class FlashInferReplaySSMKernel(TrackerReplaySSMKernel):
    """FlashInfer's `checkpointing_ssu`, for standard and speculative decoding.
    Selected only with `--mamba-backend flashinfer`."""

    backend = MambaBackendEnum.FLASHINFER
    name = "FlashInfer ReplaySSM"
    # FlashInfer loads state and ring caches with 16-byte cp.async instructions.
    state_alignment = 16

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CUDA

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return True

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
                checkpointing_ssu,
            )
        except ImportError:
            return False
        return True

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = super().validate_kernel(cfg)
        if cfg.replayssm_buffer_len > MAX_BUFFER_LEN:
            reasons.append(
                f"{cls.name} requires --replayssm-buffer-len <= {MAX_BUFFER_LEN}"
            )
        spec = cfg.kv_cache_spec
        if (
            cfg.use_stochastic_rounding
            and isinstance(spec, MambaSpec)
            and spec.dtypes[1] != torch.float16
        ):
            reasons.append(
                f"stochastic rounding requires a float16 SSM cache, got "
                f"{spec.dtypes[1]}"
            )
        if cfg.use_prefix_caching and cfg.dtype not in (None, torch.bfloat16):
            reasons.append(
                f"{cls.name} with prefix caching requires bfloat16 model inputs "
                "for its replay rings; SSM state cache dtype is independent"
            )
        return reasons

    @classmethod
    def uses_rand_seed_slot(cls, mamba_config: MambaConfig) -> bool:
        return mamba_config.enable_stochastic_rounding

    @classmethod
    def allocate_buffers(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: MambaSpec,
        max_cudagraph_decodes: int,
        device: torch.device,
    ) -> FlashInferReplaySSMBuffers:
        from flashinfer.mamba.checkpointing_ssu import (
            allocate_checkpointing_ssu_scratch,
        )

        tracker = super().allocate_buffers(
            vllm_config, kv_cache_spec, max_cudagraph_decodes, device
        )
        cb_scaled, cumAdt_vec, cb_old = allocate_checkpointing_ssu_scratch(
            batch_size=vllm_config.scheduler_config.max_num_seqs,
            # x_cache is (nheads, ring_len, head_dim).
            num_heads=kv_cache_spec.shapes[2][0],
            num_predicted_tokens=1 + vllm_config.num_speculative_tokens,
            max_window=vllm_config.cache_config.replayssm_buffer_len,
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        return FlashInferReplaySSMBuffers(
            state_indices_d=tracker.state_indices_d,
            cb_scaled=cb_scaled,
            cumAdt_vec=cumAdt_vec,
            cb_old=cb_old,
        )

    @classmethod
    def prepare_metadata(
        cls, buffers: SSUBuffers | None, metadata: M, num_rows: int, padded: bool
    ) -> M:
        assert isinstance(buffers, FlashInferReplaySSMBuffers)
        metadata = super().prepare_metadata(buffers, metadata, num_rows, padded)
        return replace(
            metadata,
            replayssm_scratch=(
                buffers.cb_scaled[:num_rows],
                buffers.cumAdt_vec[:num_rows],
                buffers.cb_old[:num_rows],
            ),
        )

    @classmethod
    def forward(
        cls,
        layer: SSULayer,
        metadata: "BaseMambaAttentionMetadata",
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> None:
        layer = cast(ReplaySSMLayer, layer)
        indices = metadata.replayssm_state_indices_d
        assert indices is not None and metadata.replayssm_scratch is not None
        cu_seqlens = metadata.query_start_loc_d
        max_seqlen = None
        if layer.ssu_state.num_spec > 0:
            query_len = 1 + layer.ssu_state.num_spec
            max_seqlen = query_len
            batch = indices.size(0)
            if x.size(0) == batch * query_len:
                # Full verify windows: dense (batch, query_len, ...) rows.
                x, dt, B, C, out = (
                    t.view(batch, query_len, *t.shape[1:]) for t in (x, dt, B, C, out)
                )
                cu_seqlens = max_seqlen = None
        mamba_config = layer.mamba_config
        if layer.ssu_state.rand_seeds is not None:
            # The first layer refreshes every layer's seed slot.
            layer.ssu_state.rand_seeds.random_(0, 2**32)
        x_cache, dt_cache, B_cache = layer.kv_cache[2:5]
        selective_state_update_replayssm_flashinfer(
            state,
            x,
            dt,
            A,
            B,
            C,
            out,
            x_cache,
            B_cache,
            dt_cache,
            layer.ssu_state.ring_start,
            layer.ssu_state.prev_num_accepted,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=indices,
            scratch=metadata.replayssm_scratch,
            enable_stochastic_rounding=mamba_config.enable_stochastic_rounding,
            stochastic_rounding_philox_rounds=(
                mamba_config.stochastic_rounding_philox_rounds
            ),
            rand_seed=layer.ssu_state.rand_seed,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            enable_pdl=False,
        )
