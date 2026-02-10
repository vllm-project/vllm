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
class SegmentedPrefillExampleConnectorMetadata(OffloadingConnectorMetadata):
    _req_to_gaps: dict[ReqId, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_base(cls, base: OffloadingConnectorMetadata):
        return cls(reqs_to_load=base.reqs_to_load, reqs_to_store=base.reqs_to_store)


class SegmentedPrefillExampleConnector(OffloadingConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig",):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._req_to_gaps: dict[ReqId, list[tuple[int, int]]] = dict()
        self._block_size: int = 16 # self.offloaded_block_size

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, SegmentedPrefillExampleConnectorMetadata)
        for req_id, _ in connector_metadata.reqs_to_load.items():
            if req_id in connector_metadata._req_to_gaps:
                gaps = connector_metadata._req_to_gaps[req_id]
                self.override_slot_mapping_gaps(req_id, gaps) # <====== OMER: I need to change this...
                total_gap_tokens = sum(end - start for start, end in gaps)
                logger.info(
                    "Simulating gaps in KV token blocks for the "
                    "first load request. Total tokens: %d",
                    total_gap_tokens,
                )
        super().bind_connector_metadata(connector_metadata)

    def _choose_gaps(
            self, num_computed_tokens: int, num_external_tokens: int, request: Request
        ) -> list[tuple[int, int]]:
            # Create gaps starting at positions where token id 10 appears, with length 32
            external_start = num_computed_tokens
            external_end = num_computed_tokens + num_external_tokens
            gap_length = 64
            
            if external_end - external_start < gap_length:
                return []
            
            gaps = []
            
            # Find all positions where token id 10 appears within the external tokens range
            for i, token_id in enumerate(request.prompt_token_ids):
                if token_id == 10 and external_start <= i < external_end:
                    gap_start = i
                    gap_end = min(gap_start + gap_length, external_end)
                    
                    # Only add the gap if it has the full length or if it's at the boundary
                    if gap_end - gap_start == gap_length or gap_end == external_end:
                        gaps.append((gap_start, gap_end))
            
            return gaps


    # def _choose_gaps(
    #     self, num_computed_tokens: int, num_external_tokens: int
    # ) -> list[tuple[int, int]]:
    #     # Simulate gaps in the external tokens, at block_size granularity.
    #     # Create gaps of growing size (1, 2, 3 blocks etc.) in the last num_external tokens,
    #     # with non-gap sections of the same growing size between them,
    #     # ensuring the last block is not a gap, and all aligned to block_size.
    #     block_size = self._block_size
    #     external_start = num_computed_tokens
    #     external_end = num_computed_tokens + num_external_tokens
    #     if external_end - external_start < block_size:
    #         return []
    #     gaps = []
    #     current_pos = external_start
    #     size = 32
    #     is_gap = True
    #     while current_pos < external_end:
    #         segment_size_tokens = size * block_size
    #         segment_end = min(current_pos + segment_size_tokens, external_end)
    #         if is_gap:
    #             if segment_end == external_end:
    #                 # If this gap would be the last segment, skip it to ensure last is non-gap
    #                 break
    #             gaps.append((current_pos, segment_end))
    #         current_pos = segment_end
    #         is_gap = not is_gap
    #         if is_gap:
    #             size += 1
    #     return gaps

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

    def override_slot_mapping_gaps(
        req_id, gaps: list[tuple[int, int]]
    ) -> None:
        """create gaps in request by mapping them to an incorrect value"""
        # print(f"Request {req_id}: {self.reqs_to_load[req_id]}")
        if not gaps:
            return
        gap_value = slot_mapping[-1].item()  # use last value
        for start, end in gaps:
            slot_mapping[start:end] = gap_value
            a = 1

    def get_computed_token_gaps(
        self,
        request: "Request",
    ) -> list[tuple[int, int]] | None:
        return self._req_to_gaps.get(request.request_id)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        num_external_tokens, _ = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        print("==============================GET_NUM_NEW_MATCHED_TOKENS=========================================")
        
        num_external_tokens = (
            0 if num_external_tokens is None else num_external_tokens
        )  # don't simulated async lookup for now

        # pick requests with at least 2*block_size external tokens to simulate gaps
        if num_external_tokens >= self._block_size * 2:
            gaps = self._choose_gaps(num_computed_tokens, num_external_tokens, request)
            self._req_to_gaps[request.request_id] = gaps
            self._print_gaps_representation(
                gaps, num_external_tokens, num_computed_tokens
            )

        return num_external_tokens, False

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        base = super().build_connector_meta(scheduler_output)
        print("==============================BUILD_CONNECTOR_META=========================================")
        
        assert isinstance(base, OffloadingConnectorMetadata)
        meta = SegmentedPrefillExampleConnectorMetadata.from_base(base)
        meta._req_to_gaps = self._req_to_gaps.copy()

        self._req_to_gaps.clear()
        return meta
