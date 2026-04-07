# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-specific block transfer policies for NIXL connector.

ModelBlockTransferPolicy is a pure computation / data provider — no NIXL
side effects. Worker.py calls these methods for model-specific math and
feeds the results to NIXL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (  # noqa: E501
        MambaConvSplitInfo,
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
    """ABC for model-specific block transfer policies.

    Concrete subclasses encapsulate Dense (FA-only) vs Hybrid-SSM (Mamba)
    differences so that worker.py remains model-agnostic.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        physical_blocks_per_logical: int,
    ):
        self._kv_cache_config = kv_cache_config
        self._physical_blocks_per_logical = physical_blocks_per_logical

    @staticmethod
    def create(
        kv_cache_config: KVCacheConfig,
        layer_specs: dict[str, KVCacheSpec],
        physical_blocks_per_logical: int,
        tp_size: int,
    ) -> ModelBlockTransferPolicy:
        """Factory: returns Dense or Mamba policy based on kv_cache_config."""
        has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        if has_mamba:
            return MambaModelBlockTransferPolicy(
                kv_cache_config,
                layer_specs,
                physical_blocks_per_logical,
                tp_size,
            )
        return DenseModelBlockTransferPolicy(
            kv_cache_config,
            physical_blocks_per_logical,
        )

    # ---- Properties ----

    @property
    @abstractmethod
    def is_mamba(self) -> bool:
        """Whether this model has Mamba/SSM layers."""
        ...

    @property
    @abstractmethod
    def ssm_sizes(self) -> tuple[int, int]:
        """(conv_bytes, ssm_bytes) for handshake metadata."""
        ...

    def is_mamba_group(self, group_idx: int) -> bool:
        """Whether the given KV cache group index is a Mamba/SSM group."""
        return False

    # ---- Registration helpers ----

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

    # ---- Descriptor ID computation + block ID mapping ----

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

    # ---- Local descriptor building ----

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
            kv_block_len = (
                self.get_block_len(
                    i,
                    True,
                    block_len_per_layer,
                    is_blocks_first,
                )
                // block_size_ratio
            )
            page_stride = block_len_per_layer[i] // block_size_ratio
            for block_id in range(n_blocks):
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
                for block_id in range(n_blocks):
                    v_addr = base_addr + block_id * page_stride + kv_block_len
                    result.append(
                        (
                            v_addr,
                            second_split,
                            device_id,
                        )
                    )
        return result

    # ---- Remote descriptor building ----

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
        return [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
            )
            for rank in remote_ranks
        ]


# ---- Concrete implementations ----


class DenseModelBlockTransferPolicy(ModelBlockTransferPolicy):
    """Block transfer policy for dense (FA-only) models."""

    @property
    def is_mamba(self) -> bool:
        return False

    @property
    def ssm_sizes(self) -> tuple[int, int]:
        return (0, 0)

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
    ):
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
    ):
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


class MambaModelBlockTransferPolicy(ModelBlockTransferPolicy):
    """Block transfer policy for hybrid SSM (Mamba) models."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        layer_specs: dict[str, KVCacheSpec],
        physical_blocks_per_logical: int,
        tp_size: int,
    ):
        super().__init__(kv_cache_config, physical_blocks_per_logical)
        self._is_mamba_group = [
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        ]
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
        self._mamba_ssm_size = (
            conv_shape.numel() * conv_nbytes,
            ssm_shape.numel() * ssm_nbytes,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            ssm_conv_transfer_utils as sct,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        assert is_conv_state_dim_first(), (
            "3-read Mamba conv transfer requires DS conv state layout. "
            "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
        )
        self._conv_decomp: MambaConvSplitInfo = sct.derive_mamba_conv_split(
            mamba_spec, tp_size
        )

    @property
    def is_mamba(self) -> bool:
        return True

    @property
    def ssm_sizes(self) -> tuple[int, int]:
        return self._mamba_ssm_size

    def is_mamba_group(self, group_idx: int) -> bool:
        return self._is_mamba_group[group_idx]

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
        if is_blocks_first:
            if mamba_view:
                return self._mamba_ssm_size[not first_split]
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
        ratio = physical_blocks_per_logical
        logical_blocks = num_blocks // ratio
        num_fa_descs = num_regions * num_blocks
        mamba_region_ids = np.arange(
            len(block_len_per_layer) * 4,
        )[:, None]
        all_descs: list[np.ndarray] = []
        for i, group in enumerate(block_ids):
            group_arr = np.asarray(group)[None, :]
            if self._is_mamba_group[i]:
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
        """
        assert block_size_ratio == 1, (
            "Mamba 3-read transfer with block_size_ratio != 1 "
            f"is not tested. Got block_size_ratio={block_size_ratio}."
        )
        conv_offsets = self._conv_decomp.local_conv_offsets
        conv_size, ssm_size = self._mamba_ssm_size
        n_blocks = logical_num_blocks * block_size_ratio
        phys_ratio = self._physical_blocks_per_logical

        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(base_addresses):
            page_stride = block_len_per_layer[i] // block_size_ratio * phys_ratio
            for off, sz in conv_offsets:
                for blk in range(n_blocks):
                    result.append(
                        (
                            base_addr + blk * page_stride + off,
                            sz,
                            device_id,
                        )
                    )
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
    ):
        transfer_cfg = transfer_config
        result: list[tuple[int, int, int]] = []
        result.extend(
            self._build_fa_remote_descs(
                nixl_agent_meta,
                transfer_cfg,
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
        transfer_cfg,
        block_size_ratio,
        is_blocks_first,
        use_mla,
        block_len_per_layer,
    ):
        """Build remote FA descriptors for mamba models using
        transfer_cfg for GQA-aware sizing."""
        assert block_size_ratio == 1, (
            "Mamba 3-read transfer with block_size_ratio != 1 "
            f"is not tested. Got {block_size_ratio=}."
        )
        tp_ratio = transfer_cfg.tp_ratio
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
                local_block_len = local_block_len // transfer_cfg.physical_fa_num_reads
            rank_offset = transfer_cfg.fa_rank_offset(
                remote_kv_block_len,
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
                    second_split = second_split // transfer_cfg.physical_fa_num_reads
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
        for the 3-read transfer."""
        effective_ratio = max(tp_ratio, 1)
        local_offset = tp_rank % effective_ratio
        conv_size_remote = nixl_agent_meta.ssm_sizes[0]

        if tp_ratio >= 1:
            conv_offsets = self._conv_decomp.remote_conv_offsets(
                local_offset,
                effective_ratio,
            )
            ssm_read_size = self._mamba_ssm_size[1]
        else:
            abs_ratio = -tp_ratio
            xb_p = self._conv_decomp.x_bytes // abs_ratio
            bb_p = self._conv_decomp.b_bytes // abs_ratio
            conv_offsets = [
                (0, xb_p),
                (xb_p, bb_p),
                (xb_p + bb_p, bb_p),
            ]
            ssm_read_size = nixl_agent_meta.ssm_sizes[1]

        remote_ratio = physical_blocks_per_logical
        num_blocks = nixl_agent_meta.num_blocks // remote_ratio
        dev_id = nixl_agent_meta.device_id

        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(
            nixl_agent_meta.kv_caches_base_addr,
        ):
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
    ):
        transfer_cfg = transfer_config
        assert transfer_cfg is not None
        if transfer_cfg.needs_split_handles:
            result = list(
                transfer_cfg.compute_split_handle_data(
                    src_blocks_data,
                    num_descs,
                    abs_tp,
                )
            )
            logger.info(
                "Mamba-HMA split handles: targets=%s, fa_reads=%s, "
                "fa_entry=%s, mamba_reads=%s, num_descs=%s",
                transfer_cfg.transfer_targets,
                transfer_cfg.physical_fa_num_reads,
                transfer_cfg.fa_entry_size,
                transfer_cfg.mamba_num_reads,
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
        transfer_cfg = transfer_config
        specs: list[ReadSpec] = []
        for rank in remote_ranks:
            filtered_local, filtered_remote = transfer_cfg.filter_block_ids_for_rank(
                rank,
                local_block_ids,
                expanded,
                self._is_mamba_group,
            )
            specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=filtered_local,
                    remote_block_ids=filtered_remote,
                )
            )
        return specs
