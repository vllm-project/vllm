# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@dataclass
class LoadRecoveryExampleConnectorMetadata(ExampleConnectorMetadata):
    req_to_block_ids: dict[str, set[int]] = field(default_factory=dict)

    @classmethod
    def from_base(cls, base: ExampleConnectorMetadata):
        return cls(requests=base.requests)


class LoadRecoveryExampleConnector(ExampleConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._async_load = vllm_config.kv_transfer_config.get_from_extra_config(
            "async_load", False
        )
        self._invalid_block_ids: set = None
        self._seen_requests: set = set()
        self._req_to_block_ids: dict[str, list[int]] = dict()

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, LoadRecoveryExampleConnectorMetadata)
        index, failed_request = next(
            (
                (i, x)
                for i, x in enumerate(connector_metadata.requests)
                if not x.is_store
            ),
            (None, None),
        )
        if index is not None:
            del connector_metadata.requests[index]
            self._invalid_block_ids = set(
                (
                    failed_request.slot_mapping[:: self._block_size] // self._block_size
                ).tolist()
            )
            logger.info(
                "Simulating failure to load all KV blocks for the "
                "first load request. Total blocks: %d",
                len(self._invalid_block_ids),
            )
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._invalid_block_ids = None
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        if self._async_load and forward_context.attn_metadata is None:
            # Bypass  sanity check in super().start_load_kv
            forward_context.attn_metadata = "None"

        super().start_load_kv(forward_context, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._async_load:
            meta = self._get_connector_metadata()
            assert isinstance(meta, LoadRecoveryExampleConnectorMetadata)
            if meta.req_to_block_ids:
                return None, set(meta.req_to_block_ids)

        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._invalid_block_ids

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if request.request_id in self._seen_requests:
            return 0, False

        self._seen_requests.add(request.request_id)

        num_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        return num_tokens, self._async_load and num_tokens > 0

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        super().update_state_after_alloc(request, blocks, num_external_tokens)

        if num_external_tokens > 0:
            self._req_to_block_ids[request.request_id] = blocks.get_block_ids()[0]

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        if not self._async_load:
            base = super().build_connector_meta(scheduler_output)
            meta = LoadRecoveryExampleConnectorMetadata.from_base(base)
        else:
            meta = LoadRecoveryExampleConnectorMetadata()
            if self._requests_need_load:
                for req_id, request in self._requests_need_load.items():
                    meta.add_request(
                        token_ids=request.prompt_token_ids,
                        block_ids=self._req_to_block_ids[req_id],
                        block_size=self._block_size,
                        is_store=False,
                        mm_hashes=[],
                    )
                # Clear state
                self._requests_need_load.clear()
        meta.req_to_block_ids = self._req_to_block_ids
        self._req_to_block_ids = dict()
        return meta
