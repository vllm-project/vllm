# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelBlockTransferPolicy: model-specific transfer intelligence.

This module defines the ``ModelBlockTransferPolicy`` ABC and its concrete
implementations for Dense and Mamba models.  The policy encapsulates all
model-specific logic that was previously scattered across ``worker.py``
and ``TransferTopology``, making both of those model-agnostic.

The policy is an *immutable config holder*: its state is set once at
``__init__`` and never mutated.  It computes results but stores no
per-engine state (that lives on ``TransferTopology._engines``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
    MambaEngineTransferInfo,
    _physical_head_range,
    _range_overlap,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class ModelBlockTransferPolicy(ABC):
    """Abstract base for model-specific block transfer logic.

    Concrete subclasses encapsulate:
    - Model identity (is_mamba, per-group flags)
    - Mamba state sizes and conv decomposition
    - Per-engine transfer info computation (``build_engine_transfer_info``)
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_mamba(self) -> bool:
        """Whether this policy handles a hybrid Mamba+Attention model."""

    @property
    @abstractmethod
    def mamba_group_flags(self) -> list[bool]:
        """Per-group flag: True if the group is a Mamba (SSM) group."""

    def is_mamba_group(self, group_idx: int) -> bool:
        return self.mamba_group_flags[group_idx]

    @property
    @abstractmethod
    def ssm_sizes(self) -> tuple[int, int]:
        """(conv_state_bytes, ssm_state_bytes) per logical block.

        Returns (0, 0) for dense models.
        """

    @property
    @abstractmethod
    def conv_decomp(self) -> MambaConvSplitInfo | None:
        """Conv-state sub-projection decomposition, or None for dense."""

    # ------------------------------------------------------------------
    # Per-engine transfer info (data operations)
    # ------------------------------------------------------------------

    # TODO (ZhanqiuHu): Revisit data packing for local facts and remote facts.
    @abstractmethod
    def build_engine_transfer_info(
        self,
        *,
        # Local facts (from TransferTopology).
        tp_rank: int,
        tp_size: int,
        is_mla: bool,
        total_num_kv_heads: int,
        is_kv_layout_blocks_first: bool,
        local_block_len: int,
        # Remote facts (from NixlAgentMetadata handshake).
        remote_tp_size: int,
        remote_block_size: int,
        remote_block_len: int,
        remote_physical_blocks_per_logical: int,
    ) -> EngineTransferInfo:
        """Compute transfer info for a remote engine.

        Dense models return ``EngineTransferInfo``.
        Mamba models return ``MambaEngineTransferInfo``.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        kv_cache_config: KVCacheConfig, tp_size: int
    ) -> ModelBlockTransferPolicy:
        """Create the appropriate policy based on model architecture."""
        is_mamba_group = [
            isinstance(group.kv_cache_spec, MambaSpec)
            for group in kv_cache_config.kv_cache_groups
        ]
        if any(is_mamba_group):
            return MambaModelBlockTransferPolicy(
                kv_cache_config=kv_cache_config,
                is_mamba_group=is_mamba_group,
                tp_size=tp_size,
            )
        return DenseModelBlockTransferPolicy(kv_cache_config)


# ======================================================================
# Dense (pure-attention) policy
# ======================================================================


class DenseModelBlockTransferPolicy(ModelBlockTransferPolicy):
    def __init__(self, kv_cache_config: KVCacheConfig):
        self._num_groups = len(kv_cache_config.kv_cache_groups)

    @property
    def is_mamba(self) -> bool:
        return False

    @property
    def mamba_group_flags(self) -> list[bool]:
        return [False] * self._num_groups

    @property
    def ssm_sizes(self) -> tuple[int, int]:
        return (0, 0)

    @property
    def conv_decomp(self) -> MambaConvSplitInfo | None:
        return None

    def build_engine_transfer_info(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        is_mla: bool,
        total_num_kv_heads: int,
        is_kv_layout_blocks_first: bool,
        local_block_len: int,
        remote_tp_size: int,
        remote_block_size: int,
        remote_block_len: int,
        remote_physical_blocks_per_logical: int,
    ) -> EngineTransferInfo:
        return EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_len=remote_block_len,
            remote_block_size=remote_block_size,
            remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
        )


# ======================================================================
# Mamba (hybrid SSM+Attention) policy
# ======================================================================


class MambaModelBlockTransferPolicy(ModelBlockTransferPolicy):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        is_mamba_group: list[bool],
        tp_size: int,
    ):
        self._is_mamba_group = is_mamba_group

        mamba_spec = next(
            group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        conv_nbytes = torch.tensor(
            [],
            dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
        ).element_size()
        ssm_nbytes = torch.tensor(
            [],
            dtype=mamba_spec.dtypes[1],  # type: ignore[misc]
        ).element_size()
        conv_shape = torch.Size(mamba_spec.shapes[0])
        ssm_shape = torch.Size(mamba_spec.shapes[1])
        self._ssm_sizes = (
            conv_shape.numel() * conv_nbytes,
            ssm_shape.numel() * ssm_nbytes,
        )

        assert is_conv_state_dim_first(), (
            "3-read Mamba conv transfer requires DS conv state layout. "
            "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
        )
        self._conv_decomp = derive_mamba_conv_split(mamba_spec, tp_size)

    @property
    def is_mamba(self) -> bool:
        return True

    @property
    def mamba_group_flags(self) -> list[bool]:
        return self._is_mamba_group

    @property
    def ssm_sizes(self) -> tuple[int, int]:
        return self._ssm_sizes

    @property
    def conv_decomp(self) -> MambaConvSplitInfo | None:
        return self._conv_decomp

    def build_engine_transfer_info(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        is_mla: bool,
        total_num_kv_heads: int,
        is_kv_layout_blocks_first: bool,
        local_block_len: int,
        remote_tp_size: int,
        remote_block_size: int,
        remote_block_len: int,
        remote_physical_blocks_per_logical: int,
    ) -> MambaEngineTransferInfo:
        K = total_num_kv_heads
        local_tp = tp_size
        local_rank = tp_rank

        is_remote_replicated = remote_tp_size > K
        remote_physical_heads = max(1, K // remote_tp_size)

        if local_tp >= remote_tp_size:
            assert local_tp % remote_tp_size == 0
            tp_ratio = local_tp // remote_tp_size
        else:
            assert remote_tp_size % local_tp == 0
            tp_ratio = -(remote_tp_size // local_tp)

        abs_tp = -tp_ratio if tp_ratio < 0 else 1

        mamba_range: range | None = None
        if tp_ratio < 0:
            mamba_range = range(local_rank * abs_tp, (local_rank + 1) * abs_tp)

        # ---- FA read targets ----
        if is_mla or tp_ratio >= 0:
            num_fa_reads = 1
            fa_source_ranks: list[int] = (
                [0]
                if is_mla
                else [local_rank // tp_ratio if tp_ratio > 0 else local_rank]
            )
        else:
            local_needs = _physical_head_range(local_tp, K, local_rank)
            search_range = (
                mamba_range if mamba_range is not None else range(remote_tp_size)
            )
            seen: set[tuple[int, int]] = set()
            fa_source_ranks = []
            for p in search_range:
                p_has = _physical_head_range(remote_tp_size, K, p)
                ov = _range_overlap(local_needs, p_has)
                if len(ov) > 0:
                    key = (ov.start, ov.stop)
                    if key not in seen:
                        seen.add(key)
                        fa_source_ranks.append(p)
            if not fa_source_ranks:
                for p in range(remote_tp_size):
                    p_has = _physical_head_range(remote_tp_size, K, p)
                    ov = _range_overlap(local_needs, p_has)
                    if len(ov) > 0:
                        key = (ov.start, ov.stop)
                        if key not in seen:
                            seen.add(key)
                            fa_source_ranks.append(p)
            num_fa_reads = len(fa_source_ranks)

        # ---- All source ranks (mamba + FA) ----
        if mamba_range is not None and abs_tp > num_fa_reads:
            num_mamba_reads = abs_tp
            all_source_ranks = list(mamba_range)
        else:
            num_mamba_reads = num_fa_reads
            all_source_ranks = list(fa_source_ranks)

        # ---- FA descriptor bytes ----
        effective_block_len = min(local_block_len, remote_block_len)
        if is_kv_layout_blocks_first:
            fa_descriptor_bytes = effective_block_len // 2
        else:
            fa_descriptor_bytes = effective_block_len

        # ---- Validation ----
        is_local_replicated = local_tp > K
        if is_local_replicated and is_remote_replicated and tp_ratio > 0:
            logger.info(
                "Both-replicated hetero-TP: local_tp=%d > remote_tp=%d > K=%d.",
                local_tp,
                remote_tp_size,
                K,
            )
        tt_set = set(all_source_ranks)
        for t in fa_source_ranks:
            if t not in tt_set:
                logger.error(
                    "FA source rank %d NOT in all_source_ranks %s.",
                    t,
                    all_source_ranks,
                )
        if is_kv_layout_blocks_first and tp_ratio < 0 and num_fa_reads > 0:
            local_k_half = local_block_len // 2
            remote_k_half = remote_block_len // 2
            expected = local_k_half // num_fa_reads
            if expected != remote_k_half:
                logger.warning(
                    "FA size mismatch: local_k_half=%d / reads=%d = %d, "
                    "but remote_k_half=%d.",
                    local_k_half,
                    num_fa_reads,
                    expected,
                    remote_k_half,
                )

        return MambaEngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_len=remote_block_len,
            remote_block_size=remote_block_size,
            remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
            remote_fa_source_ranks=tuple(fa_source_ranks),
            remote_all_source_ranks=tuple(all_source_ranks),
            remote_num_fa_reads=num_fa_reads,
            remote_num_mamba_reads=num_mamba_reads,
            remote_fa_descriptor_bytes=fa_descriptor_bytes,
            is_remote_replicated=is_remote_replicated,
            remote_physical_heads=remote_physical_heads,
        )

    # ------------------------------------------------------------------
    # Orchestration methods
    # ------------------------------------------------------------------

    def should_skip_fa(self, info: MambaEngineTransferInfo, remote_rank: int) -> bool:
        """Whether to skip FA groups for this remote rank."""
        return remote_rank not in info.fa_source_set

    def fa_head_slot(
        self,
        info: MambaEngineTransferInfo,
        remote_rank: int,
        total_num_kv_heads: int,
    ) -> int:
        """Index into local FA block for this remote rank's head data.

        For remote ranks in ``fa_source_ranks``, returns 0, 1, …, reads-1.
        For ranks NOT in ``fa_source_ranks`` (replicated duplicates),
        returns the slot of the matching source rank with the same head.
        """
        fa_index = info.fa_source_indices
        if remote_rank in fa_index:
            return fa_index[remote_rank]
        K = total_num_kv_heads
        remote_tp = info.remote_tp_size
        r_head = _physical_head_range(remote_tp, K, remote_rank)
        for target in info.remote_fa_source_ranks:
            t_head = _physical_head_range(remote_tp, K, target)
            if _range_overlap(r_head, t_head):
                return fa_index[target]
        return 0

    def fa_rank_offset(
        self,
        info: MambaEngineTransferInfo,
        remote_kv_block_len: int,
        tp_rank: int,
        tp_size: int,
        is_mla: bool,
        total_num_kv_heads: int,
    ) -> int:
        """Byte offset into remote FA block for this local rank.

        When local TP is replicated (local_tp > K), multiple local ranks
        share a head.  Computes offset *relative to the target remote
        rank's first head* so it works regardless of how many heads the
        remote has.  Returns 0 when local does not index into remote.
        """
        tp_ratio = (
            tp_size // info.remote_tp_size
            if tp_size >= info.remote_tp_size
            else -(info.remote_tp_size // tp_size)
        )  # noqa: E501
        if is_mla or tp_ratio <= 0:
            return 0
        K = total_num_kv_heads
        is_local_replicated = tp_size > K
        if is_local_replicated:
            local_head = tp_rank * K // tp_size
            p_rank = info.remote_fa_source_ranks[0]
            p_start = p_rank * K // info.remote_tp_size
            return (local_head - p_start) * remote_kv_block_len
        return tp_rank % tp_ratio * remote_kv_block_len

    def needs_split_handles(
        self,
        info: MambaEngineTransferInfo,
        tp_size: int,
        is_mla: bool,
    ) -> bool:
        """Whether per-remote-rank split handles are needed.

        True when FA and mamba have different read counts, requiring
        different splitting factors in the local handle.
        """
        tp_ratio = (
            tp_size // info.remote_tp_size
            if tp_size >= info.remote_tp_size
            else -(info.remote_tp_size // tp_size)
        )  # noqa: E501
        return tp_ratio < 0 and not is_mla and len(info.remote_all_source_ranks) > 1

    def compute_split_handle_data(
        self,
        info: MambaEngineTransferInfo,
        src_blocks_data: list[tuple[int, int, int]],
        num_fa_descs: int,
        abs_tp: int,
        total_num_kv_heads: int,
    ) -> list[list[tuple[int, int, int]]]:
        """Per-remote-rank (addr, len, dev) triples for split handles.

        FA descriptors (indices < num_fa_descs) are sliced by
        ``remote_num_fa_reads``; mamba descriptors are sliced uniformly
        by ``abs_tp``.
        """
        all_handle_data: list[list[tuple[int, int, int]]] = []
        for p_idx, p_rank in enumerate(info.remote_all_source_ranks):
            handle_data: list[tuple[int, int, int]] = []
            skip_fa = self.should_skip_fa(info, p_rank)
            fa_slot = (
                self.fa_head_slot(info, p_rank, total_num_kv_heads)
                if not skip_fa
                else 0
            )
            for j, (addr, local_len, dev) in enumerate(src_blocks_data):
                if j < num_fa_descs:
                    assert info.remote_num_fa_reads >= 1
                    fa_chunk = local_len // info.remote_num_fa_reads
                    handle_data.append((addr + fa_slot * fa_chunk, fa_chunk, dev))
                else:
                    mamba_chunk = local_len // abs_tp
                    handle_data.append((addr + p_idx * mamba_chunk, mamba_chunk, dev))
            all_handle_data.append(handle_data)
        return all_handle_data

    def filter_block_ids_for_rank(
        self,
        info: MambaEngineTransferInfo,
        remote_rank: int,
        local_ids: BlockIds,
        remote_ids: BlockIds,
    ) -> tuple[BlockIds, BlockIds]:
        """Zero out FA groups for remote ranks outside ``fa_source_ranks``.

        Returns (filtered_local_ids, filtered_remote_ids).  When the
        remote rank carries FA data for this local rank, returns the
        inputs unchanged.
        """
        if not self.should_skip_fa(info, remote_rank):
            return local_ids, remote_ids
        num_groups = len(local_ids)
        filtered_local: list[list[int]] = [
            [] if not self._is_mamba_group[g] else local_ids[g]
            for g in range(num_groups)
        ]
        filtered_remote: list[list[int]] = [
            [] if not self._is_mamba_group[g] else remote_ids[g]
            for g in range(num_groups)
        ]
        return filtered_local, filtered_remote

    def describe_mamba(
        self,
        info: MambaEngineTransferInfo,
        tp_rank: int,
        tp_size: int,
        total_num_kv_heads: int,
    ) -> str:
        """One-line summary of Mamba transfer config for logging."""
        tp_ratio = (
            tp_size // info.remote_tp_size
            if tp_size >= info.remote_tp_size
            else -(info.remote_tp_size // tp_size)
        )  # noqa: E501
        return (
            f"MambaTransferPolicy("
            f"tp_ratio={tp_ratio}, "
            f"K={total_num_kv_heads}, "
            f"local_tp={tp_size}, "
            f"remote_tp={info.remote_tp_size}, "
            f"local_rank={tp_rank}, "
            f"fa_reads={info.remote_num_fa_reads}, "
            f"mamba_reads={info.remote_num_mamba_reads}, "
            f"fa_sources={list(info.remote_fa_source_ranks)}, "
            f"all_sources={list(info.remote_all_source_ranks)}, "
            f"fa_desc_bytes={info.remote_fa_descriptor_bytes}, "
            f"remote_block_len={info.remote_block_len})"
        )
