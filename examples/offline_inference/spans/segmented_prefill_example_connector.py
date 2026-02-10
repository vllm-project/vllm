# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorMetadata,
    ReqId
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@dataclass
class SegmentedPrefillOffloadConnectorMetadata(OffloadingConnectorMetadata):
    _req_to_gaps: dict[ReqId, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_base(cls, base: OffloadingConnectorMetadata):
        return cls(reqs_to_load=base.reqs_to_load, reqs_to_store=base.reqs_to_store)


class SegmentedPrefillOffloadConnector(OffloadingConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._req_to_gaps: dict[str, list[tuple[int, int]]] = dict()
        self._block_size = 16
        self.current_num_computed_tokens = 0
        self.current_num_external_tokens = 0
        self.current_request: Request | None = None

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, SegmentedPrefillOffloadConnectorMetadata)
        super().bind_connector_metadata(connector_metadata)
        for req_id, _ in connector_metadata.reqs_to_load.items():
            if req_id in connector_metadata._req_to_gaps:
                gaps = connector_metadata._req_to_gaps[req_id]
                # self.override_slot_mapping_gaps(req_id, gaps)
                total_gap_tokens = sum(end - start for start, end in gaps)
                logger.info(
                    "Simulating gaps in KV token blocks for the "
                    "first load request. Total tokens: %d",
                    total_gap_tokens,
                )

    def _choose_gaps(
            self, num_computed_tokens: int, num_external_tokens: int, request: Request
        ) -> list[tuple[int, int]]:
            # Create gaps starting at positions where token id 10 appears, with length 32
            external_start = num_computed_tokens
            external_end = num_computed_tokens + num_external_tokens
            gap_length = 32
            print(f"Choosing gaps. external_end = {external_end}, external_start = {external_start}")
            if external_end - external_start < gap_length:
                return []
            
            gaps = []
            
            # Find all positions where token id 10 appears within the external tokens range
            for i, token_id in enumerate(request.prompt_token_ids):
                if token_id == 10 and external_start <= i < external_end:
                    print(f"Found SPAN start at index {i}. Starting gap...")
                    gap_start = i
                    gap_end = min(gap_start + gap_length, external_end)
                    
                    # Only add the gap if it has the full length or if it's at the boundary
                    if gap_end - gap_start == gap_length or gap_end == external_end:
                        gaps.append((gap_start, gap_end))
            print(f"Gaps: {gaps}")
            return gaps

    def _print_gaps_representation(
        self,
        gaps: list[tuple[int, int]],
        num_external_tokens: int,
        num_computed_tokens: int,
    ) -> None:
        """Print a human-readable representation of the tokens and gaps for debugging."""
        total_tokens = num_computed_tokens + num_external_tokens
        block_size = self._block_size
        representation = []
        for block_start in range(0, total_tokens, block_size):
            block_end = min(block_start + block_size, total_tokens)
            block_chars = []
            for i in range(block_start, block_end):
                if i < num_computed_tokens:
                    block_chars.append("C")  # Computed token
                else:
                    # Check if in gap
                    in_gap = any(start <= i < end for start, end in gaps)
                    block_chars.append("-" if in_gap else "E")  # Gap or External token
            # Determine the character for this block
            unique_chars = set(block_chars)
            # print 'X' if mixed token types in block
            char = unique_chars.pop() if len(unique_chars) == 1 else "X"
            representation.append(char)

        print("Cache status per block (C=computed, E=external, -=gap, X=mixed):")
        print("".join(representation))
        print("Gaps: ", gaps)
        print(
            "Total tokens: ",
            total_tokens,
            ", computed tokens: ",
            num_computed_tokens,
            ", external tokens: ",
            num_external_tokens,
        )

    @staticmethod
    def override_slot_mapping_gaps(
        slot_mapping: torch.Tensor, gaps: list[tuple[int, int]]
    ) -> None:
        """create gaps in slot_mapping by mapping them to an incorrect value"""
        if not gaps:
            return
        gap_value = slot_mapping[-1].item()  # use last value
        for start, end in gaps:
            slot_mapping[start:end] = gap_value

    def get_computed_token_gaps(
        self,
        request: "Request",
    ) -> list[tuple[int, int]] | None:
        logger.debug("In SegPrefill connector, get_computed_token_gaps:")
        logger.debug("_req_to_gaps len: %d.", len(self._req_to_gaps))
        print(f"self.current_num_external_tokens = {self.current_num_external_tokens}")
        if self.current_num_external_tokens >= self._block_size * 2:
            gaps = self._choose_gaps(self.current_num_computed_tokens, self.current_num_external_tokens, self.current_request)
            self._req_to_gaps[self.current_request.request_id] = gaps
            print(f"Added gaps {gaps} to request_id {self.current_request.request_id}")
            print(f"Now, req_to_gaps is: {self._req_to_gaps}")
            self._print_gaps_representation(
                gaps, self.current_num_external_tokens, self.current_num_computed_tokens
            )
        self.current_num_computed_tokens = 0
        self.current_num_external_tokens = 0
        self.current_request = None
        return self._req_to_gaps.get(request.request_id)
        # return None

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # return super().get_num_new_matched_tokens(
        #     request, num_computed_tokens
        # )
        num_external_tokens, flag = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        print(f"got external tokens: {num_external_tokens}")
        num_external_tokens = (
            0 if num_external_tokens is None else num_external_tokens
        )  # don't simulated async lookup for now

        self.current_num_computed_tokens = num_computed_tokens
        self.current_num_external_tokens = num_external_tokens
        self.current_request = request

        return num_external_tokens, flag

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        base = super().build_connector_meta(scheduler_output)
        assert isinstance(base, OffloadingConnectorMetadata)
        meta = SegmentedPrefillOffloadConnectorMetadata.from_base(base)

        meta._req_to_gaps = self._req_to_gaps.copy()

        self._req_to_gaps.clear()
        # print("Cleared _req_to_gaps!")
        return meta
