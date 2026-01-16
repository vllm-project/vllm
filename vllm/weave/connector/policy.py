# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterable

from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.request import Request

from .metadata import ReqId, RequestPhase

logger = init_logger(__name__)


class WeavePolicy:
    """Pure policy for deciding per-step KV transfers.

    This layer is intentionally kept independent of the connector's mechanics
    (worker jobs, output aggregation, etc.). It is allowed to mutate the
    scheduler-owned state dicts that it is given.

    Today this policy is behavior-identical to vLLM's native offloading
    connector (i.e., it's a refactor only). Later steps will add phase-aware
    and budgeting logic here.
    """

    def __init__(
        self,
        *,
        offloaded_block_size: int,
        block_size_factor: int,
        manager: OffloadingManager,
    ):
        self.offloaded_block_size = offloaded_block_size
        self.block_size_factor = block_size_factor
        self.manager = manager

    def get_reqs_to_store(
        self,
        scheduler_output: SchedulerOutput,
        *,
        requests: dict[ReqId, Request],
        request_block_ids: dict[ReqId, list[int]],
        next_stored_block_idx: dict[ReqId, int],
        reqs_being_stored: dict[ReqId, set[BlockHash]],
        get_block_hashes: Callable[[Request, int, int | None], Iterable[BlockHash]],
        request_phases: dict[ReqId, RequestPhase],
    ) -> dict[ReqId, TransferSpec]: # 本轮step需要的store的计划集合
        reqs_to_store: dict[ReqId, TransferSpec] = {}

        # Iterate over both new and cached requests.
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                request_block_ids[req_id] += new_block_ids

            block_ids = request_block_ids[req_id]
            req = requests[req_id]
            _ = request_phases.get(req_id)

            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size

            start_block_idx = next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx
            if num_new_blocks <= 0:
                continue

            # NOTE: In async scheduling, placeholders may temporarily make
            # len(req.block_hashes) < num_blocks * self.block_size_factor.
            new_block_hashes = get_block_hashes(req, start_block_idx, num_blocks)

            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            next_stored_block_idx[req_id] = num_blocks

            if not store_output.block_hashes_to_store:
                continue

            block_hashes_to_store = set(store_output.block_hashes_to_store)

            # Mark all blocks in range as recently used.
            self.manager.touch(get_block_hashes(req, 0, num_blocks))

            dst_spec = store_output.store_spec

            # Compute GPU source block IDs for the subset that needs storing.
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(get_block_hashes(req, start_block_idx, num_blocks)):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])

            src_spec = GPULoadStoreSpec(src_block_ids)
            reqs_to_store[req_id] = (src_spec, dst_spec)

            reqs_being_stored.setdefault(req_id, set()).update(block_hashes_to_store)

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store
