# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, cast

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import plan_replayssm_decode
from vllm.platforms.interface import PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

from .base import M, SSUBuffers, SSULayer
from .replayssm_base import MambaReplaySSMKernel, ReplaySSMLayer
from .triton import triton_stochastic_rounding_reasons

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


@dataclass
class TritonReplaySSMBuffers(SSUBuffers):
    # Filled on the GPU each step by `plan_replayssm_decode`.
    write_pos: torch.Tensor
    is_flush: torch.Tensor
    # Precomputed B·C products, (rows, ngroups, buffer_len).
    bc_pre: torch.Tensor
    # The mamba block size with prefix caching, else 0.
    align_block_size: int


class TritonReplaySSMKernel(MambaReplaySSMKernel):
    """Standard (non-speculative) decode. The group's first layer plans each
    step's ring positions on the GPU, so the model runner keeps no ring
    state."""

    backend = MambaBackendEnum.TRITON
    name = "Triton ReplaySSM"

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return True

    @classmethod
    def supports_ubatching(cls) -> bool:
        return True

    @classmethod
    def supports_pipeline_parallel(cls) -> bool:
        return True

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        return super().validate_kernel(cfg) + triton_stochastic_rounding_reasons(cfg)

    @classmethod
    def allocate_buffers(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: MambaSpec,
        max_cudagraph_decodes: int,
        device: torch.device,
    ) -> TritonReplaySSMBuffers:
        rows = max(max_cudagraph_decodes, vllm_config.scheduler_config.max_num_seqs)
        # The page holds (conv, ssm, x_cache, dt_cache, B_cache); B_cache is
        # (ngroups, ring_len, dstate).
        ngroups = kv_cache_spec.shapes[4][0]
        cache_config = vllm_config.cache_config
        return TritonReplaySSMBuffers(
            write_pos=torch.zeros((rows,), dtype=torch.int32, device=device),
            is_flush=torch.zeros((rows,), dtype=torch.int8, device=device),
            bc_pre=torch.empty(
                (rows, ngroups, cache_config.replayssm_buffer_len),
                dtype=torch.float32,
                device=device,
            ),
            align_block_size=(
                kv_cache_spec.block_size if cache_config.enable_prefix_caching else 0
            ),
        )

    @classmethod
    def prepare_metadata(
        cls, buffers: SSUBuffers | None, metadata: M, num_rows: int, padded: bool
    ) -> M:
        assert isinstance(buffers, TritonReplaySSMBuffers)
        # Padded rows have NULL_BLOCK_ID slots and are planned as no-ops.
        return replace(
            metadata,
            write_pos_d=buffers.write_pos[:num_rows],
            is_flush_d=buffers.is_flush[:num_rows],
            bc_pre_scratch=buffers.bc_pre[:num_rows],
            # All rows, so a padded full-CUDA-graph batch can reuse it.
            replayssm_seq_lens_d=(
                metadata.seq_lens[: metadata.num_reqs]
                if buffers.align_block_size
                else None
            ),
            replayssm_align_block_size=buffers.align_block_size,
        )

    @classmethod
    def on_prefill_states_written(
        cls, layer: SSULayer, metadata: "BaseMambaAttentionMetadata"
    ) -> None:
        # Decode appends to the ring prefill leaves empty; the group's
        # planning layer resets the group's shared positions.
        layer = cast(ReplaySSMLayer, layer)
        if layer.ssu_state.plans_decode:
            assert metadata.state_indices_tensor_p is not None
            layer.ssu_state.ring_start.index_fill_(
                0, metadata.state_indices_tensor_p.long(), 0
            )

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
            selective_state_update_replayssm_output_only,
        )

        return selective_state_update_replayssm_output_only

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
        assert metadata.state_indices_tensor_d is not None
        assert metadata.write_pos_d is not None and metadata.is_flush_d is not None
        if layer.ssu_state.plans_decode:
            # Every layer of the cache group reads the same planned rows.
            plan_replayssm_decode(
                layer.ssu_state.ring_start,
                metadata.state_indices_tensor_d,
                metadata.write_pos_d,
                metadata.is_flush_d,
                buffer_len=layer.ssu_state.buffer_len,
                seq_lens=metadata.replayssm_seq_lens_d,
                align_block_size=metadata.replayssm_align_block_size,
            )
        x_cache, dt_cache, B_cache = layer.kv_cache[2:5]
        mamba_config = layer.mamba_config
        cls._fn()(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            bc_pre=metadata.bc_pre_scratch,
            write_pos=metadata.write_pos_d,
            is_flush=metadata.is_flush_d,
            max_cache_len=layer.ssu_state.buffer_len,
            state_batch_indices=metadata.state_indices_tensor_d,
            out=out,
            enable_stochastic_rounding=mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=mamba_config.stochastic_rounding_philox_rounds,
        )
