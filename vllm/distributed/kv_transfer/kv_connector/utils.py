# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

EngineId = str


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

    from vllm.platforms import current_platform

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
    remote_tp_size: dict[EngineId, int]
    is_mla: bool
    total_num_kv_heads: int
    attn_backend: type[AttentionBackend]
    engine_id: EngineId
    remote_block_size: dict[EngineId, int]
    tensor_shape: torch.Size | None = None

    def __post_init__(self):
        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks. This is used to register the memory regions correctly.
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=4, head_size=1
        )
        # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
        # we just mock num_blocks to 1 for the dimension check below.
        self._is_kv_layout_blocks_first = (
            len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1
        )

        self._kv_heads_position: int | None = None
        self._cross_layers_blocks = False
        if self.tensor_shape is not None:
            self._cross_layers_blocks = (
                len(self.tensor_shape) == len(kv_cache_shape) + 1
            )

            if self._cross_layers_blocks:
                # prepend layers dimension
                kv_cache_shape = (80,) + kv_cache_shape
            try:
                kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=self._cross_layers_blocks
                )
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(self.tensor_shape)))

            logger.info("XXX shape: %s", kv_cache_shape)
            # permute kv_cache_shape according to stride_order
            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

            physical_block_size_position = kv_cache_shape.index(16)

            assert physical_block_size_position is not None
            self._physical_block_size_position = -(
                len(kv_cache_shape) - physical_block_size_position
            )
            logger.info(
                "XXX shape %s blk_size_pos %d",
                kv_cache_shape,
                self._physical_block_size_position,
            )

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
    def block_size(self) -> int:
        return self.remote_block_size[self.engine_id]

    @property
    def cross_layers_blocks(self) -> bool:
        return self._cross_layers_blocks

    @property
    def block_size_position(self) -> int:
        return self._physical_block_size_position

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
        """
        tp_size = self.remote_tp_size[engine_id]
        return tp_size // self.total_num_kv_heads >= 1

    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    def get_target_remote_ranks(
        self,
        remote_tp_size: int,
    ) -> list[int]:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from. When remote tp_size > local tp_size, we
        read from multiple remote ranks.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]

        # P TP > D TP case, D reads from |tp_ratio| remote workers.
        tp_ratio = -tp_ratio
        return [self.tp_rank * tp_ratio + i for i in range(tp_ratio)]

    def get_target_remote_ranks_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> list[int]:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.get_target_remote_ranks(remote_tp_size)


def get_current_attn_backend(vllm_config: VllmConfig):
    layer_type = cast(type[Any], AttentionLayerBase)
    layers = get_layers_from_vllm_config(vllm_config, layer_type, None)
    if layers:
        backend = next(iter(layers.values())).get_attn_backend()
    else:
        # Fallback for tests, when static_forward_context is empty.
        logger.debug(
            "No layers found in the vLLM config. "
            "Falling back to default attention backend."
        )
        from vllm.v1.attention.selector import get_attn_backend

        backend = get_attn_backend(
            head_size=vllm_config.model_config.get_head_size(),
            dtype=vllm_config.model_config.dtype,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            block_size=vllm_config.cache_config.block_size,
            use_mla=vllm_config.model_config.use_mla,
        )
    return backend
