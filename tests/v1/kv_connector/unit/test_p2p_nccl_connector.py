# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test


def _make_kv_cache_config(block_size: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype="float32",
                ),
            )
        ],
    )


def _make_vllm_config(block_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        kv_transfer_config=KVTransferConfig(
            kv_connector="P2pNcclConnector",
            kv_role="kv_producer",
        ),
    )


def _make_scheduler_output(
    *,
    scheduled_new_reqs: list[NewRequestData] | None = None,
    scheduled_cached_reqs: CachedRequestData | None = None,
    num_scheduled_tokens: dict[str, int],
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs or [],
        scheduled_cached_reqs=scheduled_cached_reqs or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_producer_handles_chunked_prefill_without_new_blocks() -> None:
    block_size = 16
    req_id = "req-1"
    prompt_token_ids = list(range(20))
    block_ids = [0, 1]
    connector = P2pNcclConnector(
        _make_vllm_config(block_size),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(block_size),
    )

    first_chunk = _make_scheduler_output(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=(block_ids,),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        num_scheduled_tokens={req_id: 17},
    )

    connector.build_connector_meta(first_chunk)

    final_chunk_without_new_blocks = _make_scheduler_output(
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[17],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={req_id: 3},
    )

    metadata = connector.build_connector_meta(final_chunk_without_new_blocks)

    assert isinstance(metadata, P2pNcclConnectorMetadata)
    assert len(metadata.requests) == 1
    request_meta = metadata.requests[0]
    assert request_meta.request_id == req_id
    assert request_meta.block_ids.tolist() == block_ids
    assert request_meta.num_tokens == len(prompt_token_ids)
