# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Protocol, cast

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.platforms.interface import PlatformEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from .base import M, SSUBuffers, SSULayer
from .replayssm_base import ReplaySSMLayer
from .replayssm_tracker import TrackerBuffers, TrackerReplaySSMKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

logger = init_logger(__name__)

# FlashInfer's checkpointing_ssu replays at most this many cached tokens.
MAX_BUFFER_LEN = 16


@cache
def flashinfer_replayssm_autotune_supported() -> bool:
    """Return True when FlashInfer exposes ReplaySSM autotuning."""
    try:
        from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
            CheckpointingSSURunner,
        )
    except ImportError:
        return False
    return True


class TunableReplaySSMLayer(ReplaySSMLayer, Protocol):
    def warmup_ssu_inputs(self, num_tokens: int) -> tuple[torch.Tensor, ...]:
        """Random `(x, dt, A, B, C, D, dt_bias)` shaped like `num_tokens`
        decode tokens."""
        ...


@dataclass
class FlashInferReplaySSMBuffers(TrackerBuffers):
    # checkpointing_ssu's two-kernel scratch, for up to max_num_seqs rows.
    cb_scaled: torch.Tensor
    cumAdt_vec: torch.Tensor
    cb_old: torch.Tensor


class FlashInferReplaySSMKernel(TrackerReplaySSMKernel):
    """FlashInfer's `checkpointing_ssu`, for standard and speculative decoding.
    Selected automatically only when the Triton ReplaySSM kernels can't run."""

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
    def needs_autotune(cls) -> bool:
        if flashinfer_replayssm_autotune_supported():
            return True
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotuning because "
            "flashinfer.mamba.checkpointing_ssu.CheckpointingSSURunner "
            "is unavailable."
        )
        return False

    @classmethod
    def autotune(cls, layer: SSULayer, num_reqs: int) -> None:
        # FlashInfer tunes only dense calls with scratch, and profiles every
        # smaller batch bucket from one call at the largest.
        from flashinfer.mamba.checkpointing_ssu import (
            allocate_checkpointing_ssu_scratch,
        )

        layer = cast(TunableReplaySSMLayer, layer)
        query_len = 1 + layer.ssu_state.num_spec
        x, dt, A, B, C, D, dt_bias = layer.warmup_ssu_inputs(num_reqs * query_len)
        scratch = allocate_checkpointing_ssu_scratch(
            batch_size=num_reqs,
            num_heads=x.size(1),
            num_predicted_tokens=query_len,
            max_window=layer.ssu_state.buffer_len,
            dtype=x.dtype,
            device=x.device,
        )
        indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=x.device)
        cls._run(
            layer,
            layer.kv_cache[1],
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            torch.empty_like(x),
            indices,
            scratch,
            cu_seqlens=None,
        )

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

    @staticmethod
    @cache
    def _fn():
        from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

        return checkpointing_ssu

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
        indices = metadata.replayssm_state_indices_d
        assert indices is not None and metadata.replayssm_scratch is not None
        cls._run(
            cast(ReplaySSMLayer, layer),
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            out,
            indices,
            metadata.replayssm_scratch,
            metadata.query_start_loc_d,
        )

    @classmethod
    def _run(
        cls,
        layer: ReplaySSMLayer,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        indices: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None,
    ) -> None:
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
        if x.dim() == 3:
            # One token per row, or packed rows when cu_seqlens is set.
            dim = 0 if cu_seqlens is not None else 1
            x, dt, B, C, out = (t.unsqueeze(dim) for t in (x, dt, B, C, out))

        mamba_config = layer.mamba_config
        rand_seed = None
        if mamba_config.enable_stochastic_rounding:
            if layer.ssu_state.rand_seeds is not None:
                # The first layer refreshes every layer's seed slot.
                layer.ssu_state.rand_seeds.random_(0, 2**32)
            rand_seed = layer.ssu_state.rand_seed
            if rand_seed is None:
                rand_seed = torch.randint(
                    0, 2**32, (1,), device=state.device, dtype=torch.int64
                )
        x_cache, dt_cache, B_cache = layer.kv_cache[2:5]
        cb_scaled, cumAdt_vec, cb_old = scratch
        cls._fn()(
            state,
            x_cache,
            B_cache,
            dt_cache,
            layer.ssu_state.ring_start,
            layer.ssu_state.prev_num_accepted,
            x,
            dt,
            A,
            B,
            C,
            out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=indices,
            pad_slot_id=NULL_BLOCK_ID,
            rand_seed=rand_seed,
            philox_rounds=mamba_config.stochastic_rounding_philox_rounds or 10,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            enable_pdl=False,
            cb_scaled=cb_scaled,
            cumAdt_vec=cumAdt_vec,
            cb_old=cb_old,
        )
