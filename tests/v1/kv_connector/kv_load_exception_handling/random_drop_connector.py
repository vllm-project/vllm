# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@dataclass
class RandomDropConnectorMetadata(KVConnectorMetadata):
    req_meta: dict[str, list[int]]


class RandomDropConnector(KVConnectorBase_V1):
    """
    A connector designed for fault tolerance testing by randomly dropping 
    kv data during the process of loading or receiving KV cache.

    This class simulates real-world scenarios where requests or data 
    might be lost or timeout, allowing developers to test and validate the 
    system's ability to handle such failures.

    Attributes:
        finished_recving_kv_req_ids (set[str]): A set of request IDs that 
            have completed receiving KV cache data.
        finished_loading_dict (dict[str, int]): A dictionary that tracks 
            the actual number of tokens loaded from the remote KV store 
            for each completed request. The keys are request IDs, and 
            the values are the corresponding token counts.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self.failure_request: list[str] = []
        self._reqs_need_recv: dict[str, list[int]] = {}
        self._finish_load: dict[str, int] = {}

        self.chunk_size = 256

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.failure_request:
            self.failure_request.remove(request.request_id)
            return 0, False
        num_external_hit_tokens = request.num_prompt_tokens - 1
        logger.info(
            "request %s num_prompt_tokens %d num_external_hit_tokens %d",
            request.request_id, request.num_prompt_tokens,
            num_external_hit_tokens)
        return num_external_hit_tokens, True

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if num_external_tokens > 0:
            self._reqs_need_recv[
                request.
                request_id] = request.prompt_token_ids[:num_external_tokens]

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        req_meta = self._reqs_need_recv.copy()
        self._reqs_need_recv.clear()
        return RandomDropConnectorMetadata(req_meta)

    def add_failure_request(self, request: "Request"):
        self.failure_request.append(request.request_id)

    def start_load_kv(self, forward_context, **kwargs) -> None:
        for request_id, hit_tokens in self._get_connector_metadata(
        ).req_meta.items():
            num_actual_load_tokens = self.load_kv(request_id, hit_tokens)
            logger.info("request %s hit_tokens %d num_actual_load_tokens %d",
                        request_id, len(hit_tokens), num_actual_load_tokens)
            self._finish_load[request_id] = num_actual_load_tokens

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        pass

    def load_kv(self, request_id, hit_tokens):
        num_actual_load_tokens = random.randint(0, len(hit_tokens))
        return num_actual_load_tokens

    def get_finished_loading(self) -> dict[str, int]:
        if not self._finish_load:
            return {}
        finished_loading = self._finish_load.copy()
        self._finish_load.clear()

        return finished_loading
