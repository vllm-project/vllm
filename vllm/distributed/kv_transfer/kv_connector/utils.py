# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)


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


@dataclass
class TpKVTopology:
    """
    Helper class for tensor parallel and KV topology information for
    mapping between local and remote TP workers.
    """

    tp_rank: int
    remote_tp_size: dict[str, int]
    is_mla: bool
    total_num_kv_heads: int
    attn_backend: type[AttentionBackend]
    engine_id: str
    remote_block_size: dict[str, int]

    def __post_init__(self):
        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks. This is used to register the memory regions correctly.
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
        # we just mock num_blocks to 1 for the dimension check below.
        self._is_kv_layout_blocks_first = (
            len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1
        )

        attn_backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self._use_pallas = attn_backend == AttentionBackendEnum.PALLAS

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        return self._is_kv_layout_blocks_first

    @property
    def split_k_and_v(self) -> bool:
        # Whether to register regions for K and V separately (when present).
        return not (self.is_mla or self._use_pallas or self.is_kv_layout_blocks_first)

    @property
    def tp_size(self) -> int:
        return self.remote_tp_size[self.engine_id]

    @property
    def block_size(self) -> int:
        return self.remote_block_size[self.engine_id]

    def tp_ratio(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Calculate the tensor parallel ratio between local and remote TP.
        We can think of it as the number of local TP workers-per-remote TP
        workers. Local workers will read from the same remote TP worker in
        groups of size `tp_ratio`.
        """
        assert self.tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {self.tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return self.tp_size // remote_tp_size

    def block_size_ratio(
        self,
        remote_block_size: int,
    ) -> float:
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
        remote_engine_id: str,
    ) -> int:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.tp_ratio(remote_tp_size)

    def block_size_ratio_from_engine_id(
        self,
        remote_engine_id: str,
    ) -> float:
        remote_block_size = self.remote_block_size[remote_engine_id]
        return self.block_size_ratio(remote_block_size)

    def is_kv_replicated(self, engine_id: str) -> bool:
        """
        Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        tp_size = self.remote_tp_size[engine_id]
        return tp_size // self.total_num_kv_heads >= 1

    def replicates_kv_cache(self, remote_engine_id: str) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    def get_target_remote_rank(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        return self.tp_rank // tp_ratio

    def get_target_remote_rank_from_engine_id(
        self,
        remote_engine_id: str,
    ) -> int:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.get_target_remote_rank(remote_tp_size)
