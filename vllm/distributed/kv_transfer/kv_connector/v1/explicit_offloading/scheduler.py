# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.common import (
    ExKVCacheContext,
    ExOffloadingConnectorMetadata,
    ExOffloadingRequestContext,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ExOffloadingParams:
    id: str | None = None
    stored_kvcache: list[dict[str, Any]] = field(default_factory=list)
    fresh_kvcache: list[dict[str, Any]] = field(default_factory=list)
    load_threshold: int = 0


def extract_kvcache_params(
    kv_transfer_params: dict[str, Any] | None,
) -> ExOffloadingParams | None:
    if kv_transfer_params is None:
        return None

    params = None
    for k, v in kv_transfer_params.items():
        if k == "kvcache_params":
            params = ExOffloadingParams(**v)

    return params


class ExOffloadingConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        self._block_size = vllm_config.cache_config.block_size
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size

        self._requests: dict[str, ExOffloadingRequestContext] = {}
        self._sessions: set[str] = set()

        self._loading_requests: set[str] = set()
        self._saving_requests: set[str] = set()

        self._block_bytes_per_layer = []
        for tensor in kv_cache_config.kv_cache_tensors:
            self._block_bytes_per_layer.append(
                tensor.size // kv_cache_config.num_blocks
            )

        self._kv_bytes_per_token = sum(self._block_bytes_per_layer) // self._block_size

    def _add_request(self, req_ctx: ExOffloadingRequestContext):
        self._requests[req_ctx.request_id] = req_ctx
        self._sessions.add(req_ctx.id)

    def _remove_request(self, request_id: str):
        if request_id not in self._requests:
            return

        req_ctx_id = self._requests[request_id].id
        self._sessions.remove(req_ctx_id)
        del self._requests[request_id]

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if request.request_id in self._requests:
            return 0, False

        params = extract_kvcache_params(request.kv_transfer_params)
        if params is None or params.id is None:
            return 0, False

        assert params.id not in self._sessions

        stored_exkvcache = ExKVCacheContext(
            params.stored_kvcache, block_size=self._block_size
        )
        fresh_exkvcache = ExKVCacheContext(
            params.fresh_kvcache, block_size=self._block_size
        )

        assert stored_exkvcache.token_end == fresh_exkvcache.token_start

        stored_exkvcache.truncate_prefix(num_computed_tokens)

        req_ctx = ExOffloadingRequestContext(
            id=params.id,
            request_id=request.request_id,
            stored_exkvcache=stored_exkvcache,
            fresh_exkvcache=fresh_exkvcache,
            num_computed_tokens=num_computed_tokens,
        )

        self._add_request(req_ctx)

        if stored_exkvcache.token_length <= params.load_threshold:
            return 0, False

        return stored_exkvcache.token_length, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return

        stored_exkvcache = self._requests[request.request_id].stored_exkvcache

        group_block_ids = blocks.get_block_ids()
        assert len(group_block_ids) == 1, "Not support HMA"
        stored_exkvcache.bind_block_ids(group_block_ids[0])
        stored_exkvcache.update_kv_layout(kv_length_per_token=self._kv_bytes_per_token)

        self._loading_requests.add(request.request_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        need_load_requests = [
            self._requests[request_id] for request_id in self._loading_requests
        ]
        need_save_requests = [
            self._requests[request_id] for request_id in self._saving_requests
        ]

        for request_id in self._saving_requests:
            self._remove_request(request_id)

        self._loading_requests.clear()
        self._saving_requests.clear()

        return ExOffloadingConnectorMetadata(
            load_req_ctx=need_load_requests,
            save_req_ctx=need_save_requests,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        pass

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if len(block_ids) == 0 or request.request_id not in self._requests:
            return False, None

        assert request.request_id not in self._saving_requests

        req_ctx = self._requests[request.request_id]
        fresh_exkvcache = req_ctx.fresh_exkvcache

        num_tokens_aligned = round_down(request.num_tokens, self._block_size)
        fresh_exkvcache.truncate_suffix(num_tokens_aligned)
        fresh_exkvcache.bind_block_ids(block_ids)
        fresh_exkvcache.update_kv_layout(kv_length_per_token=self._kv_bytes_per_token)

        if fresh_exkvcache.token_length == 0:
            self._remove_request(request.request_id)
            return False, dict(
                kvcache_params=ExOffloadingParams(
                    id=req_ctx.id, fresh_kvcache=fresh_exkvcache.result(self._tp_size)
                )
            )

        self._saving_requests.add(request.request_id)

        return True, dict(
            kvcache_params=ExOffloadingParams(
                id=req_ctx.id, fresh_kvcache=fresh_exkvcache.result(self._tp_size)
            )
        )

    def take_events(self) -> Iterable[KVCacheEvent]:
        return []

    def shutdown(self):
        pass
