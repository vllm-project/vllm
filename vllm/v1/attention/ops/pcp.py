# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.v1.attention.ops.cp_common import (
    DirectCPWorkspace,
    direct_cp_multicast_enabled,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)


class DirectPCPFusedNormRopeWorkspace(DirectCPWorkspace):
    """Persistent NVLS workspace for sparse-MLA PCP cache dispatch.

    Each rank contributes the same number of local tokens. Decode rows are
    replicated, while prefill rows are sequence-sharded. The dispatch kernel
    writes cache-ready payloads into the symmetric window, and the combine kernel
    scatters the unique rows into local paged caches.
    """

    PACKED_ROW_BYTES = {
        "fp8": 720,
        "fp8_e4m3": 720,
        "fp8_ds_mla": 800,
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_local_tokens: int,
        kv_cache_dtype: str,
        index_rope_interleave: bool,
        num_ubatches: int = 1,
    ) -> None:
        if group.size() <= 1:
            raise ValueError("fused_norm_rope_pcp requires at least two ranks")
        if max_local_tokens < 1:
            raise ValueError("max_local_tokens must be positive")
        if num_ubatches < 1:
            raise ValueError("num_ubatches must be positive")
        super().__init__(group, device, num_ubatches)
        self.max_local_tokens = max_local_tokens
        self.fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"
        self.index_rope_interleave = index_rope_interleave

        payload_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_local_tokens,
            self.PACKED_ROW_BYTES[kv_cache_dtype],
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_payload, _ = self._allocate(payload_shape, torch.uint8)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        payload_multicast_ptrs = self._multicast_ptrs(self.received_payload)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(payload_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(
            payload_ptr and signal_ptr
            for payload_ptr, signal_ptr in self.multicast_ptrs
        ):
            raise RuntimeError(
                "fused_norm_rope_pcp requires NVLS symmetric-memory multicast"
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 2))
        self.phase = self.received_signal.new_zeros((num_ubatches, 1))
        torch.accelerator.synchronize()

    def forward(
        self,
        *,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        q_weight: torch.Tensor,
        q_eps: float,
        kv_c: torch.Tensor,
        kv_weight: torch.Tensor,
        mla_k_scale: torch.Tensor,
        kv_eps: float,
        k_pe: torch.Tensor,
        k_pe_cos_sin: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weight: torch.Tensor | None,
        index_bias: torch.Tensor | None,
        index_eps: float,
        index_cos_sin: torch.Tensor | None,
        topk_indices: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor | None,
        mla_cache: torch.Tensor,
        index_cache: torch.Tensor | None,
        num_decode_tokens: int,
    ) -> torch.Tensor:
        local_tokens = q_c.shape[0]
        if local_tokens > self.max_local_tokens:
            raise ValueError(
                f"local tokens {local_tokens} exceed fused PCP capacity "
                f"{self.max_local_tokens}"
            )
        if not 0 <= num_decode_tokens <= local_tokens:
            raise ValueError(
                f"invalid decode token count {num_decode_tokens} for "
                f"{local_tokens} local tokens"
            )
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(f"PCP ubatch {ubatch} exceeds {self.num_ubatches} slots")
        payload_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        q_out = q_c.new_empty(q_c.shape)
        torch.ops._C.fused_norm_rope_pcp(
            positions,
            q_c,
            q_weight,
            q_eps,
            q_out,
            kv_c,
            kv_weight,
            mla_k_scale,
            kv_eps,
            k_pe,
            k_pe_cos_sin,
            index_k,
            index_weight,
            index_bias,
            index_eps,
            index_cos_sin,
            topk_indices,
            self.received_payload[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            self.phase[ubatch],
            mla_slot_mapping,
            index_slot_mapping,
            mla_cache,
            index_cache,
            num_decode_tokens,
            self.rank,
            self.fp8_ds_mla,
            self.index_rope_interleave,
            payload_multicast_ptr,
            signal_multicast_ptr,
        )
        return q_out


@functools.cache
def _get_fused_pcp_norm_rope_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_local_tokens: int,
    kv_cache_dtype: str,
    index_rope_interleave: bool,
    num_ubatches: int,
) -> DirectPCPFusedNormRopeWorkspace | None:
    if group.world_size <= 1 or not direct_cp_multicast_enabled(
        group,
        torch.uint8,
        envs.VLLM_USE_FUSED_PCP_NORM_ROPE,
    ):
        return None
    workspace = DirectPCPFusedNormRopeWorkspace(
        group.device_group,
        device,
        max_local_tokens,
        kv_cache_dtype,
        index_rope_interleave,
        num_ubatches,
    )
    logger.info_once("Using fused symmetric-memory PCP norm/RoPE cache dispatch.")
    return workspace


def get_fused_pcp_norm_rope_workspace(
    vllm_config: VllmConfig,
    model_config: Any,
    kv_cache_dtype: str,
    device: torch.device,
) -> DirectPCPFusedNormRopeWorkspace | None:
    parallel_config = vllm_config.parallel_config
    if not (
        kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_ds_mla")
        and model_config.q_lora_rank in (1536, 2048)
        and model_config.kv_lora_rank == 512
        and model_config.qk_rope_head_dim == 64
        and getattr(model_config, "index_head_dim", None) == 128
    ):
        return None
    return _get_fused_pcp_norm_rope_workspace(
        get_pcp_group(),
        device,
        vllm_config.scheduler_config.max_num_batched_tokens,
        kv_cache_dtype,
        getattr(model_config, "indexer_rope_interleave", False),
        max(parallel_config.num_ubatches, 1),
    )


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    gathered_prefills = tuple(
        pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
