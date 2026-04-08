
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnectorMetadata,
    ReqMeta,
    ExampleConnector,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)



class CrossDPExampleConnector(ExampleConnector):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        
        self._cross_requests_need_load: list[dict[str, Request]] = [
            {} for _ in range(vllm_config.parallel_config.dp_per_domain)
        ]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Hack for only decode instance.
        """
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Hack for only decode instance.
        """
        return None

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Hack for only decode instance.
        """
        return len(request.prompt_token_ids) - 1, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens > 0:
            for cp_rank in request.cp_ranks:
                self._cross_requests_need_load[cp_rank][request.request_id] = request
    
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ExampleConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            # if new_req.req_id in self._requests_need_load:
            if new_req.req_id in self._cross_requests_need_load[scheduler_output.cp_rank]:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_prompt(token_ids, mm_hashes):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._cross_requests_need_load[scheduler_output.cp_rank]:
                continue

            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            # NOTE(rob): cached_req_data does not have the full
            # list of token ids (only new tokens). So we look it
            # up in the actual request object.
            request = self._cross_requests_need_load[scheduler_output.cp_rank][req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]

            # NOTE(rob): For resumed req, new_block_ids is all
            # of the block_ids for the request.
            assert new_block_ids is not None
            block_ids = new_block_ids[0]

            meta.add_request(
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
            )
            total_need_load += 1

        assert total_need_load == len(self._cross_requests_need_load[scheduler_output.cp_rank])
        self._cross_requests_need_load[scheduler_output.cp_rank].clear()
        return meta
    def clear_reqs_need_recv(self):
        return 
