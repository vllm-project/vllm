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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
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
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation.

    Computed upfront by ``compute_read_specs`` so that the worker's read
    loop is a simple iteration with no model-specific branching.
    """

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


class ModelBlockTransferPolicy(ABC):
    """Abstract base for model-specific block transfer logic.

    Concrete subclasses encapsulate:
    - Model identity (is_mamba, per-group flags)
    - Mamba state sizes and conv decomposition
    - Per-engine transfer info computation (``build_engine_transfer_info``)
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        physical_blocks_per_logical: int,
    ):
        self._kv_cache_config = kv_cache_config
        self._physical_blocks_per_logical = physical_blocks_per_logical

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
    # Registration helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_page_size(
        self,
        layer_spec: KVCacheSpec,
        physical_ratio: int,
    ) -> int:
        """Physical page size in bytes for one layer."""
        ...

    @abstractmethod
    def get_num_blocks(
        self,
        layer_spec: KVCacheSpec,
        num_blocks: int,
        logical_num_blocks: int,
    ) -> int:
        """Number of blocks to register for this layer spec."""
        ...

    @abstractmethod
    def compute_layer_block_bytes(
        self,
        layer_spec: KVCacheSpec,
        physical_page_size: int,
        physical_ratio: int,
    ) -> int:
        """Block byte size for one layer (entry for ``block_len_per_layer``)."""
        ...

    @abstractmethod
    def get_tensor_shape(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> torch.Size | None:
        """Tensor shape for ``TpKVTopology`` (None for Mamba)."""
        ...

    @abstractmethod
    def get_block_len(
        self,
        layer_idx: int,
        first_split: bool,
        block_len_per_layer: list[int],
        is_blocks_first: bool,
        mamba_view: bool = False,
    ) -> int:
        """Block length for one K/V (or conv/ssm) element."""
        ...

    # ------------------------------------------------------------------
    # Descriptor ID computation + block ID mapping
    # ------------------------------------------------------------------

    @abstractmethod
    def get_block_descs_ids(
        self,
        block_ids: BlockIds,
        num_regions: int,
        dst_num_blocks: int,
        block_len_per_layer: list[int],
        block_size_ratio: float | None = None,
        physical_blocks_per_logical: int = 1,
    ) -> np.ndarray:
        """Compute NIXL descriptor IDs for a set of block IDs."""
        ...

    @abstractmethod
    def logical_to_kernel_block_ids(
        self,
        block_ids: BlockIds,
    ) -> BlockIds:
        """Convert logical block IDs to kernel physical block IDs."""
        ...

    @abstractmethod
    def logical_to_remote_kernel_block_ids(
        self,
        block_ids: BlockIds,
        remote_ratio: int,
    ) -> BlockIds:
        """Map logical block IDs to physical kernel block IDs on remote."""
        ...

    # ------------------------------------------------------------------
    # Local descriptor building
    # ------------------------------------------------------------------

    @abstractmethod
    def build_local_descs(
        self,
        base_addresses: list[int],
        block_len_per_layer: list[int],
        num_blocks: int,
        logical_num_blocks: int,
        block_size_ratio: int,
        device_id: int,
        is_blocks_first: bool,
    ) -> list[tuple[int, int, int]]:
        """Build local (src) descriptor tuples for NIXL registration."""
        ...

    def _build_fa_local_descs(
        self,
        base_addresses: list[int],
        block_len_per_layer: list[int],
        num_blocks: int,
        block_size_ratio: int,
        device_id: int,
        is_blocks_first: bool,
    ) -> list[tuple[int, int, int]]:
        """Build FA local descriptors (shared by Dense and Mamba)."""
        result: list[tuple[int, int, int]] = []
        n_blocks = num_blocks * block_size_ratio
        for i, base_addr in enumerate(base_addresses):
            # The new block_len is using prefill block_len;
            # and num_blocks is multiple with N
            kv_block_len = (
                self.get_block_len(
                    i,
                    True,
                    block_len_per_layer,
                    is_blocks_first,
                )
                // block_size_ratio
            )
            # Jump one page_size, but ssm page_size may be bigger when kernel
            # locks block size to a specific value.
            page_stride = block_len_per_layer[i] // block_size_ratio
            for block_id in range(n_blocks):
                # (addr, len, device id)
                result.append(
                    (
                        base_addr + block_id * page_stride,
                        kv_block_len,
                        device_id,
                    )
                )
            if is_blocks_first:
                second_split = self.get_block_len(
                    i,
                    False,
                    block_len_per_layer,
                    is_blocks_first,
                )
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks.
                for block_id in range(n_blocks):
                    # Register addresses for V cache (K registered first).
                    v_addr = base_addr + block_id * page_stride + kv_block_len
                    result.append(
                        (
                            v_addr,
                            second_split,
                            device_id,
                        )
                    )
        return result

    # ------------------------------------------------------------------
    # Remote descriptor building
    # ------------------------------------------------------------------

    @abstractmethod
    def build_remote_descs(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
        tp_ratio: int,
        tp_rank: int,
        use_mla: bool,
        block_len_per_layer: list[int],
        is_blocks_first: bool,
        indexes_into_remote: bool,
        transfer_config: Any | None = None,
        physical_blocks_per_logical: int = 1,
        tp_size: int = 1,
        total_num_kv_heads: int = 1,
    ) -> list[tuple[int, int, int]]:
        """Build remote (dst) descriptor tuples."""
        ...

    @abstractmethod
    def build_src_split_handles(
        self,
        src_blocks_data: list[tuple[int, int, int]],
        num_descs: int,
        abs_tp: int,
        transfer_config: Any | None = None,
        tp_size: int = 1,
        is_mla: bool = False,
        total_num_kv_heads: int = 1,
    ) -> list[list[tuple[int, int, int]]]:
        """Build split handle data for P_TP > D_TP scenario."""
        ...

    def compute_read_specs(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        remote_ranks: list[int],
        physical_blocks_per_logical: int = 1,
        transfer_config: Any | None = None,
    ) -> list[ReadSpec]:
        """Compute the full set of read operations needed for a request.

        Returns one ``ReadSpec`` per remote rank that requires a read.
        The worker iterates the result without model-specific branching.
        MLA trimming (keeping only the first spec) is handled by the worker.
        """
        _ = (physical_blocks_per_logical, transfer_config)
        return [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
            )
            for rank in remote_ranks
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        kv_cache_config: KVCacheConfig,
        layer_specs: dict[str, KVCacheSpec],
        physical_blocks_per_logical: int,
        tp_size: int,
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
                layer_specs=layer_specs,
                physical_blocks_per_logical=physical_blocks_per_logical,
            )
        return DenseModelBlockTransferPolicy(
            kv_cache_config,
            physical_blocks_per_logical,
        )


# ======================================================================
# Dense (pure-attention) policy
# ======================================================================


class DenseModelBlockTransferPolicy(ModelBlockTransferPolicy):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        physical_blocks_per_logical: int,
    ):
        super().__init__(kv_cache_config, physical_blocks_per_logical)
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

    def compute_page_size(self, layer_spec, physical_ratio):
        return layer_spec.page_size_bytes // physical_ratio

    def get_num_blocks(self, layer_spec, num_blocks, logical_num_blocks):
        return num_blocks

    def compute_layer_block_bytes(self, layer_spec, physical_page_size, physical_ratio):
        return physical_page_size

    def get_tensor_shape(self, kv_caches):
        return next(iter(kv_caches.values())).shape

    def get_block_len(
        self,
        layer_idx,
        first_split,
        block_len_per_layer,
        is_blocks_first,
        mamba_view=False,
    ):
        if is_blocks_first:
            return block_len_per_layer[layer_idx] // 2
        return block_len_per_layer[layer_idx]

    def get_block_descs_ids(
        self,
        block_ids,
        num_regions,
        dst_num_blocks,
        block_len_per_layer,
        block_size_ratio=None,
        physical_blocks_per_logical=1,
    ):
        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        region_ids = np.arange(num_regions)[:, None]
        block_ids_arr = np.concatenate(block_ids)[None, :]
        return (region_ids * num_blocks + block_ids_arr).flatten()

    def logical_to_kernel_block_ids(self, block_ids):
        if self._physical_blocks_per_logical == 1:
            return block_ids
        block_arange = np.arange(
            0,
            self._physical_blocks_per_logical,
        ).reshape(1, -1)
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical,
                block_arange,
            ).tolist()
            for group in block_ids
        ]

    def logical_to_remote_kernel_block_ids(
        self,
        block_ids,
        remote_ratio,
    ):
        if remote_ratio == 1:
            return block_ids
        local_arange = np.arange(
            self._physical_blocks_per_logical,
        ).reshape(1, -1)
        return [
            (np.array(group).reshape(-1, 1) * remote_ratio + local_arange)
            .flatten()
            .tolist()
            for group in block_ids
        ]

    def compute_read_specs(
        self,
        local_block_ids,
        remote_block_ids,
        remote_ranks,
        physical_blocks_per_logical=1,
        transfer_config=None,
    ):
        _ = (physical_blocks_per_logical, transfer_config)
        expanded = self.logical_to_kernel_block_ids(remote_block_ids)
        return [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=local_block_ids,
                remote_block_ids=expanded,
            )
            for rank in remote_ranks
        ]

    def build_local_descs(
        self,
        base_addresses,
        block_len_per_layer,
        num_blocks,
        logical_num_blocks,
        block_size_ratio,
        device_id,
        is_blocks_first,
    ):
        return self._build_fa_local_descs(
            base_addresses,
            block_len_per_layer,
            num_blocks,
            block_size_ratio,
            device_id,
            is_blocks_first,
        )

    def build_remote_descs(
        self,
        nixl_agent_meta,
        block_size_ratio,
        tp_ratio,
        tp_rank,
        use_mla,
        block_len_per_layer,
        is_blocks_first,
        indexes_into_remote,
        transfer_config=None,
        physical_blocks_per_logical=1,
        tp_size: int = 1,
        total_num_kv_heads: int = 1,
    ):
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
        # Register all remote blocks, but only the corresponding kv heads.
        _ = (tp_size, total_num_kv_heads, physical_blocks_per_logical, transfer_config)
        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(
            nixl_agent_meta.kv_caches_base_addr,
        ):
            # Read our whole local region size from remote.
            local_block_len = self.get_block_len(
                i,
                True,
                block_len_per_layer,
                is_blocks_first,
            )
            # using remote kv_block_len as transfer unit
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                local_block_len = remote_kv_block_len
            if tp_ratio < 0 and not use_mla:
                # Remote tp is bigger: read a chunk of local region from remote
                local_block_len = local_block_len // (-tp_ratio)
            rank_offset = (
                tp_rank % tp_ratio * remote_kv_block_len if indexes_into_remote else 0
            )
            num_blocks = nixl_agent_meta.num_blocks
            page_size = nixl_agent_meta.block_lens[i]
            dev_id = nixl_agent_meta.device_id
            for blk in range(num_blocks):
                addr = base_addr + blk * page_size + rank_offset
                result.append((addr, local_block_len, dev_id))
            if is_blocks_first:
                second_split = self.get_block_len(
                    i,
                    False,
                    block_len_per_layer,
                    is_blocks_first,
                )
                if tp_ratio < 0 and not use_mla:
                    second_split = second_split // (-tp_ratio)
                for blk in range(num_blocks):
                    addr = base_addr + blk * page_size + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    result.append((v_addr, second_split, dev_id))
        return result

    def build_src_split_handles(
        self,
        src_blocks_data,
        num_descs,
        abs_tp,
        transfer_config=None,
        tp_size: int = 1,
        is_mla: bool = False,
        total_num_kv_heads: int = 1,
    ):
        _ = (num_descs, transfer_config, tp_size, is_mla, total_num_kv_heads)
        result: list[list[tuple[int, int, int]]] = []
        for i in range(abs_tp):
            blocks_data: list[tuple[int, int, int]] = []
            for addr, local_block_len, own_rank in src_blocks_data:
                remote_block_len = local_block_len // abs_tp
                blocks_data.append(
                    (
                        addr + i * remote_block_len,
                        remote_block_len,
                        own_rank,
                    )
                )
            result.append(blocks_data)
        return result

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
        layer_specs: dict[str, KVCacheSpec],
        physical_blocks_per_logical: int,
    ):
        super().__init__(kv_cache_config, physical_blocks_per_logical)
        self._is_mamba_group = is_mamba_group

        mamba_spec = next(
            spec for spec in layer_specs.values() if isinstance(spec, MambaSpec)
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

    def compute_page_size(self, layer_spec, physical_ratio):
        if isinstance(layer_spec, MambaSpec):
            return layer_spec.page_size_bytes
        return layer_spec.page_size_bytes // physical_ratio

    def get_num_blocks(self, layer_spec, num_blocks, logical_num_blocks):
        if isinstance(layer_spec, MambaSpec):
            return logical_num_blocks
        return num_blocks

    def compute_layer_block_bytes(self, layer_spec, physical_page_size, physical_ratio):
        if isinstance(layer_spec, MambaSpec):
            return physical_page_size // physical_ratio
        return physical_page_size

    def get_tensor_shape(self, kv_caches):
        return None

    def get_block_len(
        self,
        layer_idx,
        first_split,
        block_len_per_layer,
        is_blocks_first,
        mamba_view=False,
    ):
        # For indexing only half (either just the K or V part).
        if is_blocks_first:
            if mamba_view:
                # NOTE (NickLucche) Mamba Opt: this is already skipping the
                # padding so we're only transferring the minimum required bytes.
                return self._ssm_sizes[not first_split]
            return block_len_per_layer[layer_idx] // 2
        return block_len_per_layer[layer_idx]

    def get_block_descs_ids(
        self,
        block_ids,
        num_regions,
        dst_num_blocks,
        block_len_per_layer,
        block_size_ratio=None,
        physical_blocks_per_logical=1,
    ):
        # NOTE (NickLucche) With HMA, every kv group has the same number of
        # layers and layers from different groups share the same kv tensor.
        # eg block_ids=[[1,2],[3]]->blocks [1,2] need to be read across all
        # regions, same for [3], but group0-group1 blocks will always differ
        # (different areas).  Therefore we can just flatten the block_ids and
        # compute the descs ids for all groups at once.

        # Compute desc ids per group using the right stride: FA descs have
        # num_blocks entries per region (kernel granularity), SSM descs have
        # logical_blocks entries per region (no kernel splitting).
        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        region_ids = np.arange(num_regions)[:, None]
        ratio = physical_blocks_per_logical
        logical_blocks = num_blocks // ratio
        num_fa_descs = num_regions * num_blocks
        # NOTE (NickLucche) SSM and Attention blocks regions can be exchanged
        # arbitrarily by manager. Therefore, descs are duplicated for SSM and
        # Attention like so:
        # desc_handle->[descs_fa (all regions) | descs_ssm (all regions)].
        # This is like having two "low-level views" of the same storage.
        # `num_fa_descs` offset must be computed per-engine since P and D can
        # have different num_blocks (and thus different FA descs counts).
        # 3-read mamba: 4 regions per unique cache tensor (x, B, C, ssm).
        mamba_region_ids = np.arange(
            len(block_len_per_layer) * 4,
        )[:, None]
        all_descs: list[np.ndarray] = []
        for i, group in enumerate(block_ids):
            group_arr = np.asarray(group)[None, :]
            if self._is_mamba_group[i]:
                # Mamba blocks are 1:1 logical-to-physical (no expansion).
                all_descs.append(
                    (
                        mamba_region_ids * logical_blocks + group_arr + num_fa_descs
                    ).flatten()
                )
            else:
                all_descs.append((region_ids * num_blocks + group_arr).flatten())
        return np.concatenate(all_descs)

    def logical_to_kernel_block_ids(self, block_ids):
        if self._physical_blocks_per_logical == 1:
            return block_ids
        block_arange = np.arange(
            0,
            self._physical_blocks_per_logical,
        ).reshape(1, -1)
        group_specs = self._kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical,
                block_arange,
            ).tolist()
            if not isinstance(
                group_specs[i].kv_cache_spec,
                MambaSpec,
            )
            else group
            for i, group in enumerate(block_ids)
        ]

    def logical_to_remote_kernel_block_ids(
        self,
        block_ids,
        remote_ratio,
    ):
        if remote_ratio == 1:
            return block_ids
        local_arange = np.arange(
            self._physical_blocks_per_logical,
        ).reshape(1, -1)
        group_specs = self._kv_cache_config.kv_cache_groups
        result: list[list[int]] = []
        for i, group in enumerate(block_ids):
            if not isinstance(
                group_specs[i].kv_cache_spec,
                MambaSpec,
            ):
                arr = np.array(group).reshape(-1, 1)
                expanded = (arr * remote_ratio + local_arange).flatten()
                result.append(expanded.tolist())
            else:
                result.append(group)
        return result

    def build_local_descs(
        self,
        base_addresses,
        block_len_per_layer,
        num_blocks,
        logical_num_blocks,
        block_size_ratio,
        device_id,
        is_blocks_first,
    ):
        fa_descs = self._build_fa_local_descs(
            base_addresses,
            block_len_per_layer,
            num_blocks,
            block_size_ratio,
            device_id,
            is_blocks_first,
        )
        num_regions = len(base_addresses) * (2 if is_blocks_first else 1)
        assert len(fa_descs) == num_regions * num_blocks
        # TODO (ZhanqiuHu): For homogeneous TP (tp_ratio == 1), the 3-read
        # split is unnecessary — a single conv desc per block suffices.
        # Consider adding a fast path that falls back to the standard 2-region
        # registration when no hetero-TP remote has been seen.  Currently we
        # always register 4 regions because local descs are created before
        # knowing the remote TP.
        logger.debug("Registering local Mamba descriptors (4 regions/layer)")
        mamba_descs = self._build_mamba_local_descs(
            base_addresses,
            block_len_per_layer,
            logical_num_blocks,
            block_size_ratio,
            device_id,
        )
        return fa_descs + mamba_descs

    def _build_mamba_local_descs(
        self,
        base_addresses: list[int],
        block_len_per_layer: list[int],
        logical_num_blocks: int,
        block_size_ratio: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Build 4 desc regions (x, B, C, ssm) per layer for local
        mamba blocks, enabling the 3-read transfer with DS conv layout.

        Conv state sub-projection decomposition requires DS (dim, state_len)
        conv layout so that x/B/C sub-projections are contiguous in memory.
        """
        assert block_size_ratio == 1, (
            "Mamba 3-read transfer with block_size_ratio != 1 "
            f"is not tested. Got block_size_ratio={block_size_ratio}."
        )
        conv_offsets = self._conv_decomp.local_conv_offsets
        # SSM States come in tuples (conv_size, ssm_state_size)
        conv_size, ssm_size = self._ssm_sizes
        n_blocks = logical_num_blocks * block_size_ratio
        phys_ratio = self._physical_blocks_per_logical

        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(base_addresses):
            page_stride = block_len_per_layer[i] // block_size_ratio * phys_ratio
            # 3 conv sub-projection regions (x, B, C)
            for off, sz in conv_offsets:
                for blk in range(n_blocks):
                    result.append(
                        (
                            base_addr + blk * page_stride + off,
                            sz,
                            device_id,
                        )
                    )
            # SSM temporal state follows the conv state.
            for blk in range(n_blocks):
                result.append(
                    (
                        base_addr + blk * page_stride + conv_size,
                        ssm_size,
                        device_id,
                    )
                )
        return result

    def build_remote_descs(
        self,
        nixl_agent_meta,
        block_size_ratio,
        tp_ratio,
        tp_rank,
        use_mla,
        block_len_per_layer,
        is_blocks_first,
        indexes_into_remote,
        transfer_config=None,
        physical_blocks_per_logical=1,
        tp_size: int = 1,
        total_num_kv_heads: int = 1,
    ):
        # indexes_into_remote is not used for Mamba: FA offset is computed
        # via fa_rank_offset which accounts for GQA/HMA head mapping.
        _ = indexes_into_remote
        info = cast(MambaEngineTransferInfo, transfer_config)
        result: list[tuple[int, int, int]] = []
        result.extend(
            self._build_fa_remote_descs(
                nixl_agent_meta,
                info,
                tp_ratio,
                tp_rank,
                tp_size,
                total_num_kv_heads,
                block_size_ratio,
                is_blocks_first,
                use_mla,
                block_len_per_layer,
            )
        )
        result.extend(
            self._build_mamba_remote_descs(
                nixl_agent_meta,
                tp_ratio,
                tp_rank,
                physical_blocks_per_logical,
            )
        )
        return result

    def _build_fa_remote_descs(
        self,
        nixl_agent_meta,
        info: MambaEngineTransferInfo,
        tp_ratio: int,
        tp_rank: int,
        tp_size: int,
        total_num_kv_heads: int,
        block_size_ratio: int,
        is_blocks_first: bool,
        use_mla: bool,
        block_len_per_layer: list[int],
    ):
        """Build remote FA descriptors for mamba models using
        transfer_cfg for GQA-aware sizing."""
        assert block_size_ratio == 1, (
            "Mamba 3-read transfer with block_size_ratio != 1 "
            f"is not tested. Got {block_size_ratio=}."
        )
        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(
            nixl_agent_meta.kv_caches_base_addr,
        ):
            local_block_len = self.get_block_len(
                i,
                True,
                block_len_per_layer,
                is_blocks_first,
            )
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                local_block_len = remote_kv_block_len
            if tp_ratio < 0 and not use_mla:
                local_block_len = local_block_len // info.remote_num_fa_reads
            rank_offset = self.fa_rank_offset(
                info,
                remote_kv_block_len,
                tp_rank=tp_rank,
                tp_size=tp_size,
                is_mla=use_mla,
                total_num_kv_heads=total_num_kv_heads,
            )
            num_blocks = nixl_agent_meta.num_blocks
            page_size = nixl_agent_meta.block_lens[i]
            dev_id = nixl_agent_meta.device_id
            for blk in range(num_blocks):
                addr = base_addr + blk * page_size + rank_offset
                result.append((addr, local_block_len, dev_id))
            if is_blocks_first:
                second_split = self.get_block_len(
                    i,
                    False,
                    block_len_per_layer,
                    is_blocks_first,
                )
                if tp_ratio < 0 and not use_mla:
                    second_split = second_split // info.remote_num_fa_reads
                for blk in range(num_blocks):
                    addr = base_addr + blk * page_size + rank_offset
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    result.append(
                        (
                            v_addr,
                            second_split,
                            dev_id,
                        )
                    )
        return result

    def _build_mamba_remote_descs(
        self,
        nixl_agent_meta,
        tp_ratio,
        tp_rank,
        physical_blocks_per_logical,
    ):
        """Build 4 remote desc regions (x, B, C, ssm) per layer
        for the 3-read transfer.

        Mamba conv state is always TP-sharded, even when attention KV
        is replicated (num_kv_heads < tp_size).
        """
        effective_ratio = max(tp_ratio, 1)
        local_offset = tp_rank % effective_ratio
        conv_size_remote = nixl_agent_meta.ssm_sizes[0]

        if tp_ratio >= 1:
            # D_TP >= P_TP: P page is larger, D reads its slice.
            conv_offsets = self._conv_decomp.remote_conv_offsets(
                local_offset,
                effective_ratio,
            )
            # SSM temporal state is also TP-sharded on the heads dimension.
            ssm_read_size = self._ssm_sizes[1]
        else:
            # NOTE (ZhanqiuHu): tp_ratio < 0 means P_TP > D_TP, so P pages
            # are smaller than D's.  self._conv_decomp has D-sized dimensions,
            # but we need P-sized offsets.  Scale down by |tp_ratio|.
            abs_ratio = -tp_ratio
            xb_p = self._conv_decomp.x_bytes // abs_ratio
            bb_p = self._conv_decomp.b_bytes // abs_ratio
            conv_offsets = [
                (0, xb_p),
                (xb_p, bb_p),
                (xb_p + bb_p, bb_p),
            ]
            ssm_read_size = nixl_agent_meta.ssm_sizes[1]

        # Assume same num_blocks for mamba and fa
        remote_ratio = physical_blocks_per_logical
        num_blocks = nixl_agent_meta.num_blocks // remote_ratio
        dev_id = nixl_agent_meta.device_id

        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(
            nixl_agent_meta.kv_caches_base_addr,
        ):
            # NOTE (ZhanqiuHu): use per-layer block_lens[i], not [0], in case
            # block lengths vary across layers (e.g. MLA).
            page_stride = nixl_agent_meta.block_lens[i] * remote_ratio
            for off, sz in conv_offsets:
                for blk in range(num_blocks):
                    result.append(
                        (
                            base_addr + blk * page_stride + off,
                            sz,
                            dev_id,
                        )
                    )
            for blk in range(num_blocks):
                ssm_addr = (
                    base_addr
                    + blk * page_stride
                    + conv_size_remote
                    + local_offset * ssm_read_size
                )
                result.append((ssm_addr, ssm_read_size, dev_id))
        return result

    def build_src_split_handles(
        self,
        src_blocks_data,
        num_descs,
        abs_tp,
        transfer_config=None,
        tp_size: int = 1,
        is_mla: bool = False,
        total_num_kv_heads: int = 1,
    ):
        info = cast(MambaEngineTransferInfo, transfer_config)
        assert transfer_config is not None
        if self.needs_split_handles(
            info,
            tp_size=tp_size,
            is_mla=is_mla,
        ):
            result = list(
                self.compute_split_handle_data(
                    info,
                    src_blocks_data,
                    num_descs,
                    abs_tp,
                    total_num_kv_heads=total_num_kv_heads,
                )
            )
            logger.info(
                "Mamba-HMA split handles: targets=%s, fa_reads=%s, "
                "fa_entry=%s, mamba_reads=%s, num_descs=%s",
                info.remote_all_source_ranks,
                info.remote_num_fa_reads,
                info.remote_fa_descriptor_bytes,
                info.remote_num_mamba_reads,
                num_descs,
            )
            return result
        return []

    def compute_read_specs(
        self,
        local_block_ids,
        remote_block_ids,
        remote_ranks,
        physical_blocks_per_logical=1,
        transfer_config=None,
    ):
        expanded = self.logical_to_remote_kernel_block_ids(
            remote_block_ids,
            physical_blocks_per_logical,
        )
        info = cast(MambaEngineTransferInfo, transfer_config)
        assert transfer_config is not None
        specs: list[ReadSpec] = []
        for rank in remote_ranks:
            filtered_local, filtered_remote = self.filter_block_ids_for_rank(
                info,
                rank,
                local_block_ids,
                expanded,
            )
            specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=filtered_local,
                    remote_block_ids=filtered_remote,
                )
            )
        return specs

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
