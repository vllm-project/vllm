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
        self._use_mla = vllm_config.model_config.use_mla

        self._loading_requests: dict[str, ExOffloadingRequestContext] = {}
        self._saving_requests: dict[str, ExOffloadingRequestContext] = {}

        self._block_bytes_per_layer = []
        for tensor in kv_cache_config.kv_cache_tensors:
            self._block_bytes_per_layer.append(
                tensor.size // kv_cache_config.num_blocks
            )

        self._kv_bytes_per_token = sum(self._block_bytes_per_layer) // self._block_size

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = extract_kvcache_params(request.kv_transfer_params)
        if params is None or params.id is None:
            logger.warning(
                "Request %s has invalid kvcache_params: %s", request.request_id, params
            )
            return 0, False

        stored_exkvcache = ExKVCacheContext(
            params.stored_kvcache, block_size=self._block_size
        )

        if stored_exkvcache.token_length > request.num_tokens:
            logger.warning(
                "Request %s has invalid stored_kvcache.token_length(=%d) "
                "which is greater than request.num_tokens(=%d)",
                request.request_id,
                stored_exkvcache.token_length,
                request.num_tokens,
            )
            return 0, False

        matched = stored_exkvcache.token_length - num_computed_tokens
        if matched <= params.load_threshold:
            return 0, False

        return matched, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return

        assert request.request_id not in self._loading_requests, (
            f"Request {request.request_id} is already loading"
        )

        params = extract_kvcache_params(request.kv_transfer_params)
        if params is None or params.id is None:
            logger.warning(
                "Request %s has invalid kvcache_params: %s", request.request_id, params
            )
            return

        stored_exkvcache = ExKVCacheContext(
            params.stored_kvcache, block_size=self._block_size
        )
        fresh_exkvcache = ExKVCacheContext(
            params.fresh_kvcache, block_size=self._block_size
        )

        assert stored_exkvcache.token_end == fresh_exkvcache.token_start

        num_computed_tokens = stored_exkvcache.token_length - num_external_tokens
        stored_exkvcache.truncate_prefix(num_computed_tokens)

        group_block_ids = blocks.get_block_ids()
        assert len(group_block_ids) == 1, "Not support HMA"
        stored_exkvcache.bind_block_ids(group_block_ids[0])
        stored_exkvcache.update_kv_layout(kv_length_per_token=self._kv_bytes_per_token)

        self._loading_requests[request.request_id] = ExOffloadingRequestContext(
            id=params.id,
            request_id=request.request_id,
            exkvcache=stored_exkvcache,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        need_load_requests = [
            self._loading_requests[request_id] for request_id in self._loading_requests
        ]
        need_save_requests = [
            self._saving_requests[request_id] for request_id in self._saving_requests
        ]

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
        if len(block_ids) == 0:
            return False, None

        assert request.request_id not in self._saving_requests, (
            f"Request {request.request_id} is already saving"
        )

        params = extract_kvcache_params(request.kv_transfer_params)
        if params is None or params.id is None:
            logger.warning(
                "Request %s has invalid kvcache_params: %s", request.request_id, params
            )
            return False, None

        fresh_exkvcache = ExKVCacheContext(
            params.fresh_kvcache, block_size=self._block_size
        )

        num_tokens_aligned = round_down(request.num_tokens, self._block_size)
        fresh_exkvcache.truncate_suffix(num_tokens_aligned)
        fresh_exkvcache.bind_block_ids(block_ids)
        fresh_exkvcache.update_kv_layout(kv_length_per_token=self._kv_bytes_per_token)

        kv_xfer_params = dict(
            kvcache_params=ExOffloadingParams(
                id=params.id,
                fresh_kvcache=fresh_exkvcache.result(self._tp_size, self._use_mla),
            )
        )

        if fresh_exkvcache.token_length == 0:
            return False, kv_xfer_params

        self._saving_requests[request.request_id] = ExOffloadingRequestContext(
            id=params.id,
            request_id=request.request_id,
            exkvcache=fresh_exkvcache,
        )

        return True, kv_xfer_params

    def take_events(self) -> Iterable[KVCacheEvent]:
        return []

    def shutdown(self):
        pass
