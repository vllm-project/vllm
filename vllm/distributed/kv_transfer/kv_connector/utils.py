# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

EngineId = str
# block ids as returned by the hybrid KV cache manager. list[list[int]] are allow
# mutability and are for connector internal use only.
BlockIds = tuple[list[int], ...] | list[list[int]]


def get_kv_connector_cache_layout():
    # NOTE (NickLucche) When running disaggregated PD with NIXL, HND layout is
    # used for faster transfer.
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD."
        )
    return "NHD"


class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, expected_finished_count: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._expected_finished_count = expected_finished_count

    @classmethod
    def from_connector(cls, connector: "KVConnectorBase", world_size: int):
        return cls(connector.get_finished_count() or world_size)

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        if not outputs[output_rank]:
            return None

        # Aggregate kv_connector_output from all workers

        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(
                    req_id, self._expected_finished_count
                )
                remaining_count_dict[req_id] = remaining_count - 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        aggregated_kv_connector_worker_meta = None
        combined_kv_cache_events = None
        invalid_block_ids = set[int]()
        for model_runner_output in outputs:
            assert model_runner_output is not None
            kv_output = model_runner_output.kv_connector_output
            if not kv_output:
                continue
            # Allow the worker to dynamically update the expected number of
            # finished sending/recving for new requests.
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count

            update_finished_set(
                kv_output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                kv_output.finished_recving, self._recv_remaining_count, finished_recving
            )

            # Aggregate kv_connector_stats from all workers.
            if aggregated_kv_connector_stats is None:
                # Use the first worker's kv_connector_stats as accumulator.
                aggregated_kv_connector_stats = kv_output.kv_connector_stats
            elif kv_connector_stats := kv_output.kv_connector_stats:
                if aggregated_kv_connector_stats is None:
                    aggregated_kv_connector_stats = kv_connector_stats
                else:
                    assert isinstance(
                        aggregated_kv_connector_stats, type(kv_connector_stats)
                    )
                    aggregated_kv_connector_stats = (
                        aggregated_kv_connector_stats.aggregate(kv_connector_stats)
                    )

            # Aggregate kv_connector_worker_meta from all workers.
            if aggregated_kv_connector_worker_meta is None:
                # Use the first worker's kv_connector_worker_meta as accumulator.
                aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
            elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
                aggregated_kv_connector_worker_meta = (
                    aggregated_kv_connector_worker_meta.aggregate(
                        kv_connector_worker_meta
                    )
                )

            # Combine kv_cache_events from all workers.
            if combined_kv_cache_events is None:
                # Use the first worker's kv_cache events as start event list.
                combined_kv_cache_events = kv_output.kv_cache_events
            elif kv_cache_events := kv_output.kv_cache_events:
                assert isinstance(
                    combined_kv_cache_events,
                    type(kv_cache_events),
                )
                worker_kv_cache_events = kv_cache_events.get_all_events()
                combined_kv_cache_events.add_events(worker_kv_cache_events)
                combined_kv_cache_events.increment_workers(1)

            invalid_block_ids |= kv_output.invalid_block_ids

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        assert output is not None
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            kv_cache_events=combined_kv_cache_events or None,
            kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=self._expected_finished_count,
        )

        return output


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: torch.device | str,
    dst_device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_indices = torch.tensor(src_block_ids, device=src_device, dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids, device=dst_device, dtype=torch.int64)
    return src_indices, dst_indices


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    if (
        not src_kv_caches
        or not dst_kv_caches
        or not src_block_ids
        or not dst_block_ids
        or len(src_block_ids) != len(dst_block_ids)
    ):
        return

    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device,
    )

    if direction == "h2d":
        copy_fn = current_platform.insert_blocks_to_device
    else:
        copy_fn = current_platform.swap_out_blocks_to_host
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)


def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache blocks to the local block_size.
    (Only works for local blocksize > remote blocksize)

    example:
    local blocksize = 16 tokens, remote blocksize = 4 tokens
    local block[0] = remote block[0, 1, 2, 3]
    remote is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    local is  |h0-b0..................|h1-b0..................|...
    permute is to:
    1. view => view remote as n_blocks * remote_shape(H,remoteN,D)
    2. permute => (H, nblocks, remoteN, D)
    3. flatten => (H, localN, D)
    """
    blocks_to_update = cache.index_select(0, indices)
    # use physical order
    blocks_to_update = blocks_to_update.permute(0, 2, 1, 3)
    n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .flatten(2, 3)
    )
    permuted_blocks = permuted_blocks.permute(0, 2, 1, 3)
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_layout_on_receive(cache, indices):
    """Transforms the layout of received KV cache blocks to the local format.

    This method corrects layout mismatches from direct memory copies by
    permuting the tensor dimensions.

    - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
    - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`

    Implementation:
    - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
    - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
    - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

    """
    blocks_to_update = cache.index_select(0, indices)
    target_shape = list(blocks_to_update.shape)
    target_shape[0] = -1
    inv_order = [0, 2, 1, 3]
    src_shape = tuple(target_shape[i] for i in inv_order)
    blocks_to_update = cache.index_select(0, indices)
    permuted_blocks = blocks_to_update.reshape(src_shape).permute(*inv_order)
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache to the local block_size and HND.
    (Only works for local blocksize > remote blocksize)

    prefill is HND, smaller block_size
    decode(local) is NHD, larger block_size
    """
    blocks_to_update = cache.index_select(0, indices)

    block_size, n_kv_heads, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 1, 3, 2, 4)
        .flatten(1, 2)
    )
    cache.index_copy_(0, indices, permuted_blocks)


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...], bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


@dataclass
class TpKVTopology:
    """
    Helper class for tensor parallel and KV topology information for
    mapping between local and remote TP workers.
    """

    tp_rank: int
    dcp_rank: int
    remote_tp_size: dict[EngineId, int]
    remote_dcp_size: dict[EngineId, int]
    remote_pcp_size: dict[EngineId, int]
    is_mla: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    engine_id: EngineId
    remote_block_size: dict[EngineId, int]
    tensor_shape: torch.Size | None = None
    is_mamba: bool = False

    def __post_init__(self):
        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks. This is used to register the memory regions correctly.
        attn_backend = self.attn_backends[0]
        if not self.is_mamba:
            _MOCK_BLOCK_SIZE = 16
            kv_cache_shape: tuple[int, ...] = attn_backend.get_kv_cache_shape(
                num_blocks=1, block_size=_MOCK_BLOCK_SIZE, num_kv_heads=1, head_size=1
            )
            logger.debug("Test kv_cache_shape: %s", kv_cache_shape)
        # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
        # we just mock num_blocks to 1 for the dimension check below.
        # Hybrid SSM models assume a single blocks_first layout
        self._is_kv_layout_blocks_first = self.is_mamba or (
            len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1
        )

        self._cross_layers_blocks = False
        if self.tensor_shape is not None:
            self._cross_layers_blocks = (
                len(self.tensor_shape) == len(kv_cache_shape) + 1
            )
            self.tensor_shape: torch.Size

        if self._cross_layers_blocks:
            logger.debug("Using cross-layer KV cache")
            # prepend layers dimension
            _MOCK_NUM_LAYERS = 80
            kv_cache_shape = (_MOCK_NUM_LAYERS,) + kv_cache_shape
            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=self._cross_layers_blocks
                )
            except (AttributeError, NotImplementedError):
                assert self.tensor_shape is not None
                kv_cache_stride_order = tuple(range(len(self.tensor_shape)))

            # In case of cross layers permute kv_cache_shape according to
            # stride_order to retrieve physical position of block_size
            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        return self._is_kv_layout_blocks_first

    @property
    def split_k_and_v(self) -> bool:
        # Whether to register regions for K and V separately (when present).
        return not (
            self._cross_layers_blocks or self.is_mla or self.is_kv_layout_blocks_first
        )

    @property
    def tp_size(self) -> int:
        return self.remote_tp_size[self.engine_id]

    @property
    def dcp_size(self) -> int:
        return self.remote_dcp_size[self.engine_id]

    @property
    def pcp_size(self) -> int:
        return self.remote_pcp_size[self.engine_id]

    @property
    def block_size(self) -> int:
        return self.remote_block_size[self.engine_id]

    @property
    def cross_layers_blocks(self) -> bool:
        return self._cross_layers_blocks

    def tp_ratio(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Calculate the tensor parallel ratio between local and remote TP.
        We can think of it as the number of local TP workers-per-remote TP
        workers. Local workers will read from the same remote TP worker in
        groups of size `tp_ratio`.If remote tp_size > local tp_size, the
        ratio is flipped (remote_size/local_size) and the returned value is
        negative.
        """
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size

        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        # P TP > D TP case, return the ratio as negative
        return -remote_tp_size // self.tp_size

    def block_size_ratio(
        self,
        remote_block_size: int,
    ) -> int:
        """
        Calculate the block size ratio between local and remote TP.
        """
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def tp_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.tp_ratio(remote_tp_size)

    def block_size_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_block_size = self.remote_block_size[remote_engine_id]
        return self.block_size_ratio(remote_block_size)

    def is_kv_replicated(self, engine_id: EngineId) -> bool:
        """
        Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        When they are equal, each TP rank still owns one distinct KV head,
        so this is not considered replication.
        """
        tp_size = self.remote_tp_size[engine_id]
        return tp_size > self.total_num_kv_heads

    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    def get_target_remote_ranks(
        self,
        remote_tp_size: int,
        local_tp_rank: int | None = None,
    ) -> list[int]:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from. When remote tp_size > local tp_size, we
        read from multiple remote ranks.
        """
        if local_tp_rank is None:
            local_tp_rank = self.tp_rank
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [local_tp_rank // tp_ratio]

        # P TP > D TP case, D reads from |tp_ratio| remote workers.
        tp_ratio = -tp_ratio
        return [local_tp_rank * tp_ratio + i for i in range(tp_ratio)]

    def get_target_remote_ranks_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> list[int]:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.get_target_remote_ranks(remote_tp_size)

    @staticmethod
    def get_valid_worker_keys(
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
    ) -> list[tuple[int, int]]:
        # Return all valid remote worker keys based on PCP, TP and DCP layout.
        # DCP rank uses the grouping index after transpose (tp, pcp):
        # flat_idx = tp_rank * pcp_size + pcp_rank.
        valid_remote_keys = []
        for remote_pcp_rank in range(remote_pcp_size):
            for remote_tp_rank in range(remote_tp_size):
                flat_idx = remote_tp_rank * remote_pcp_size + remote_pcp_rank
                remote_dcp_rank = flat_idx % remote_dcp_size
                valid_remote_keys.append((remote_tp_rank, remote_dcp_rank))
        return valid_remote_keys

    def has_kv_cache_overlap(
        self,
        remote_tp_rank: int,
        remote_dcp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
    ) -> bool:
        """Check KV cache overlap between local worker and a remote worker.

        Overlap requires both:
        1) KV head-range overlap.
        2) DCP token-slice overlap.
        """

        return self._has_kv_cache_overlap_for_local_rank(
            local_tp_rank=self.tp_rank,
            local_dcp_rank=self.dcp_rank,
            remote_tp_rank=remote_tp_rank,
            remote_dcp_rank=remote_dcp_rank,
            remote_tp_size=remote_tp_size,
            remote_dcp_size=remote_dcp_size,
            remote_pcp_size=remote_pcp_size,
        )

    def _has_kv_cache_overlap_for_local_rank(
        self,
        local_tp_rank: int,
        local_dcp_rank: int,
        remote_tp_rank: int,
        remote_dcp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
    ) -> bool:
        """Check KV cache overlap between a specific local rank and remote."""

        # Condition H: head overlap.
        if self.is_mla and remote_pcp_size == remote_dcp_size:
            # For MLA models with aligned PCP and DCP, all workers in the same
            # TP group share KV cache. Restrict overlap to the TP rank mapping
            # so each local rank handshakes only with its mapped remote ranks.
            mapped_remote_ranks = self.get_target_remote_ranks(
                remote_tp_size=remote_tp_size,
                local_tp_rank=local_tp_rank,
            )
            head_overlap = remote_tp_rank in mapped_remote_ranks
        else:
            num_kv_heads = self.total_num_kv_heads
            local_eff_tp = min(self.tp_size, num_kv_heads)
            remote_eff_tp = min(remote_tp_size, num_kv_heads)

            local_kv_group = local_tp_rank * local_eff_tp // self.tp_size
            remote_kv_group = remote_tp_rank * remote_eff_tp // remote_tp_size

            head_overlap = (
                local_kv_group * remote_eff_tp < (remote_kv_group + 1) * local_eff_tp
                and remote_kv_group * local_eff_tp
                < (local_kv_group + 1) * remote_eff_tp
            )

        # Condition T: token overlap.
        common_dcp = np.gcd(self.dcp_size, remote_dcp_size)
        token_overlap = local_dcp_rank % common_dcp == remote_dcp_rank % common_dcp

        return head_overlap and token_overlap

    def get_target_remote_worker_keys(
        self,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
    ) -> list[tuple[int, int]]:
        """Select remote worker keys that have KV overlap with local worker."""

        remote_worker_keys: list[tuple[int, int]] = []
        for remote_tp_rank, remote_dcp_rank in self.get_valid_worker_keys(
            remote_tp_size, remote_dcp_size, remote_pcp_size
        ):
            if self.has_kv_cache_overlap(
                remote_tp_rank=remote_tp_rank,
                remote_dcp_rank=remote_dcp_rank,
                remote_tp_size=remote_tp_size,
                remote_dcp_size=remote_dcp_size,
                remote_pcp_size=remote_pcp_size,
            ):
                remote_worker_keys.append((remote_tp_rank, remote_dcp_rank))
        return remote_worker_keys

    def get_target_remote_worker_keys_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> list[tuple[int, int]]:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        remote_dcp_size = self.remote_dcp_size[remote_engine_id]
        remote_pcp_size = self.remote_pcp_size[remote_engine_id]
        return self.get_target_remote_worker_keys(
            remote_tp_size,
            remote_dcp_size,
            remote_pcp_size,
        )

    def calculate_local_consumer_count(
        self,
        remote_engine_id: EngineId,
        remote_worker_key: tuple[int, int],
    ) -> int:
        """Return the number of local workers that notify this remote worker.

        This enumerates all local (tp, dcp) workers and counts those that
        overlap with the given remote worker under the same overlap rules used
        by ``has_kv_cache_overlap``.
        """
        remote_tp_rank, remote_dcp_rank = remote_worker_key
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        remote_dcp_size = self.remote_dcp_size[remote_engine_id]
        remote_pcp_size = self.remote_pcp_size[remote_engine_id]

        consumer_count = 0
        for local_tp_rank, local_dcp_rank in self.get_valid_worker_keys(
            self.tp_size, self.dcp_size, self.pcp_size
        ):
            if self._has_kv_cache_overlap_for_local_rank(
                local_tp_rank=local_tp_rank,
                local_dcp_rank=local_dcp_rank,
                remote_tp_rank=remote_tp_rank,
                remote_dcp_rank=remote_dcp_rank,
                remote_tp_size=remote_tp_size,
                remote_dcp_size=remote_dcp_size,
                remote_pcp_size=remote_pcp_size,
            ):
                consumer_count += 1

        return consumer_count

    @staticmethod
    def get_block_positions(block_num: int, dcp_size: int, dcp_rank: int) -> np.ndarray:
        local_positions = np.arange(block_num)
        global_positions = local_positions * dcp_size + dcp_rank
        return global_positions

    def get_matched_blocks(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        remote_dcp_size: int,
        remote_dcp_rank: int,
        local_block_offset: int = 0,
    ) -> tuple[BlockIds, BlockIds]:
        remote_global_positions = self.get_block_positions(
            block_num=len(remote_block_ids[0]),
            dcp_size=remote_dcp_size,
            dcp_rank=remote_dcp_rank,
        )
        local_global_positions = (
            self.get_block_positions(
                block_num=len(local_block_ids[0]),
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_rank,
            )
            + local_block_offset
        )
        matched_positions = np.intersect1d(
            remote_global_positions, local_global_positions
        )
        # Subtract offset before dividing to map back to local block indices
        local_matched_indices = (
            matched_positions - local_block_offset
        ) // self.dcp_size
        remote_matched_indices = matched_positions // remote_dcp_size
        local_matched_block_ids = [
            [block_ids[i] for i in local_matched_indices if i < len(block_ids)]
            for block_ids in local_block_ids
        ]
        remote_matched_block_ids = [
            [block_ids[i] for i in remote_matched_indices if i < len(block_ids)]
            for block_ids in remote_block_ids
        ]
        return local_matched_block_ids, remote_matched_block_ids

    def get_transfer_cache_regions(
        self, cache: torch.Tensor, layer_spec: "KVCacheSpec"
    ) -> list[torch.Tensor] | torch.Tensor:
        """Return the cache tensor(s) to register as NIXL memory regions,
        also accounting for hybrid SSM models specificities.
        """
        if isinstance(layer_spec, MambaSpec):
            # Register the whole kv cache shared tensor, including SSM/Conv. This is
            # similar to FI with the difference that SSM/Conv have different sizes
            conv, ssm = cache
            return [conv]

        # Check may be hacky but it's matching `_update_hybrid_attention_mamba_layout`.
        if self.is_mamba and cache.shape[0] == 2:
            # When MAMBA is present, all backends are blocks first, so that blocks
            # can be shared between attention layers and mamba layers. Runner
            # `_update_hybrid_attention_mamba_layout` already adjusted strides
            # for FlashAttn-like backends so its num_blocks first.
            # Swap [2<>num_blocks] dims to get required layout for hybrid SSM.
            cache = cache.transpose(0, 1)

        # Regular case: backends like FA register K/V in separate regions
        return cache if self.split_k_and_v else [cache]


def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for the given layers.

    Args:
        vllm_config: The current vLLM configuration.
        layer_names: Optional list of layer names to scope the lookup.
            When None, all attention layers are considered.

    Returns:
        Deduplicated list of attention backend classes.
    """
    layer_type = cast(type[Any], AttentionLayerBase)
    layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
    if layers:
        seen: dict[str, type[AttentionBackend]] = {}
        for layer in layers.values():
            backend = layer.get_attn_backend()
            seen[backend.full_cls_name()] = backend
        return list(seen.values())

    # Fallback for tests, when static_forward_context is empty.
    logger.debug(
        "No layers found in the vLLM config. Falling back to default attention backend."
    )
    from vllm.v1.attention.selector import get_attn_backend

    return [
        get_attn_backend(
            head_size=vllm_config.model_config.get_head_size(),
            dtype=vllm_config.model_config.dtype,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            use_mla=vllm_config.model_config.use_mla,
        )
    ]


def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]:
    """Get the first attention backend for the given layers."""
    return get_current_attn_backends(vllm_config, layer_names)[0]
