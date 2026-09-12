# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A minimal KVConnectorBase_V1 that promises a block-aligned prefix and then
rejects the load (serves zero bytes, reports the blocks as load errors).

Used by test_kv_load_reject_recompute.py to check that
kv_load_failure_policy="recompute" recovers to the no-connector baseline.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class RejectRecomputeMetadata(KVConnectorMetadata):
    # (req_id, block_ids) for each request promised-then-rejected this step.
    reqs: list[tuple[str, list[int]]] = field(default_factory=list)


class RejectRecomputeConnector(KVConnectorBase_V1):
    """Promises a block-aligned prefix synchronously, then rejects the load."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size
        self._need_load: dict[str, bool] = {}
        self._load_errors: set[int] = set()

    # ---- scheduler role ----
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        n_prompt = len(request.prompt_token_ids or [])
        # Largest block-aligned prefix strictly below the prompt (>= 1 tail token).
        matched = ((n_prompt - 1) // self._block_size) * self._block_size
        new = matched - num_computed_tokens
        if new <= 0:
            return 0, False
        return new, False  # load_async=False -> synchronous load

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._need_load[request.request_id] = True

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = RejectRecomputeMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._need_load:
                meta.reqs.append((new_req.req_id, list(new_req.block_ids[0])))
        self._need_load.clear()
        return meta

    # ---- worker role ----
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, RejectRecomputeMetadata)
        for _req_id, block_ids in meta.reqs:
            # Reject: load nothing, mark the promised blocks as failed loads.
            self._load_errors.update(block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        errs, self._load_errors = self._load_errors, set()
        return errs

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer, attn_metadata, **kwargs: Any
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return
