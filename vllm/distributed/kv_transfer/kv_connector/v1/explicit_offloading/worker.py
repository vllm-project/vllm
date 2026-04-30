# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.common import (
    ExOffloadingConnectorMetadata,
    ExOffloadingRequestContext,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorageKVCacheConfig,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ExOffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None

        self._tp_rank = get_tensor_model_parallel_rank()
        self._tp_size = get_tensor_model_parallel_world_size()
        self._block_size = vllm_config.cache_config.block_size
        self._engine_id = vllm_config.kv_transfer_config.engine_id
        self._use_mla = vllm_config.model_config.use_mla
        self._total_num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()

        self._transfer_topo = TransferTopology(
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            block_size=self._block_size,
            engine_id=self._engine_id,
            is_mla=self._use_mla,
            is_mamba=False,
            total_num_kv_heads=self._total_num_kv_heads,
            attn_backends=[get_current_attn_backend(vllm_config)],
        )

        self._io_loop = asyncio.new_event_loop()
        self._io_thread = threading.Thread(
            target=_async_loop, args=(self._io_loop,), daemon=True
        )
        self._io_thread.start()

        self._finished_load_req_ids: set[str] = set()
        self._finished_save_req_ids: set[str] = set()

        self._invalid_block_ids: set[int] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kvcache_config = ExOffloadingStorageKVCacheConfig(
            kv_caches=kv_caches,
            split_k_and_v=self._transfer_topo.split_k_and_v,
            is_block_first=self._transfer_topo.is_kv_layout_blocks_first,
        )

    async def _start_load_task(self, ctx: ExOffloadingRequestContext):
        assert ctx.request_id not in self._finished_load_req_ids

        try:
            ctx.exkvcache.update_kv_layout(
                tp_rank=self._tp_rank,
                replicates_kv_cache=self._transfer_topo.local_replicates_kv_cache,
            )
            await ctx.exkvcache.prefetch(self._kvcache_config)
        except Exception as e:
            logger.error(
                "(%s:%s) Failed loading kv cache: %s", ctx.id, ctx.request_id, e
            )
            assert ctx.exkvcache.block_ids is not None
            self._invalid_block_ids.update(ctx.exkvcache.block_ids)

        self._finished_load_req_ids.add(ctx.request_id)

    def start_load_kv(self, metadata: ExOffloadingConnectorMetadata):
        asyncio.run_coroutine_threadsafe(
            run_tasks(self._start_load_task, metadata.load_req_ctx), self._io_loop
        )

    async def _start_save_task(self, ctx: ExOffloadingRequestContext):
        assert ctx.request_id not in self._finished_save_req_ids

        if self._transfer_topo.local_replicates_kv_cache and self._tp_rank == 0:
            try:
                ctx.exkvcache.update_kv_layout(
                    tp_rank=self._tp_rank,
                    replicates_kv_cache=self._transfer_topo.local_replicates_kv_cache,
                )
                await ctx.exkvcache.backup(self._kvcache_config)
            except Exception as e:
                logger.error(
                    "(%s:%s) Failed saving kv cache: %s", ctx.id, ctx.request_id, e
                )

        self._finished_save_req_ids.add(ctx.request_id)

    def start_save_kv(self, metadata: ExOffloadingConnectorMetadata):
        if not metadata.save_req_ctx:
            return

        asyncio.run_coroutine_threadsafe(
            run_tasks(self._start_save_task, metadata.save_req_ctx), self._io_loop
        )

    async def _fetch_finished_reqs(self) -> tuple[set[str], set[str]]:
        finished_save_reqs = self._finished_save_req_ids
        finished_load_reqs = self._finished_load_req_ids

        self._finished_save_req_ids = set()
        self._finished_load_req_ids = set()

        return finished_save_reqs, finished_load_reqs

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        res = asyncio.run_coroutine_threadsafe(
            self._fetch_finished_reqs(), self._io_loop
        )

        return res.result() if res else (set(), set())

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def shutdown(self) -> None:
        if self._io_loop.is_running():
            self._io_loop.call_soon_threadsafe(self._io_loop.stop)
            self._io_thread.join()


async def run_tasks(fn, ctx_list: list[ExOffloadingRequestContext]):
    await asyncio.gather(*[fn(ctx) for ctx in ctx_list])
