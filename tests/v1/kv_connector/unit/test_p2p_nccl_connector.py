# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2pNcclConnector's kv_transfer_params flow.

Tests the scheduler-side contract without GPU, NCCL, or distributed init.
"""

import pytest

from vllm import SamplingParams
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector(kv_role: str) -> P2pNcclConnector:
    """Build a scheduler-side P2pNcclConnector."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=64,
            max_model_len=1024,
            enable_chunked_prefill=True,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="P2pNcclConnector",
            kv_role=kv_role,
            kv_port="14579",
        ),
        device_config=DeviceConfig("cpu"),
        attention_config=AttentionConfig(),
    )
    return P2pNcclConnector(config, KVConnectorRole.SCHEDULER)


def _make_request(
    request_id: str,
    num_tokens: int = 10,
    kv_transfer_params: dict | None = None,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        eos_token_id=50256,
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def _make_blocks(block_ids: list[int]) -> KVCacheBlocks:
    blocks = tuple([KVCacheBlock(block_id=bid) for bid in block_ids])
    return KVCacheBlocks(blocks=(blocks,))


def _make_scheduler_output(
    new_reqs: list[NewRequestData] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    cached_req_ids: list[str] | None = None,
    cached_num_computed: list[int] | None = None,
    cached_new_block_ids: list | None = None,
    resumed_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    cached = CachedRequestData(
        req_ids=cached_req_ids or [],
        resumed_req_ids=resumed_req_ids or set(),
        new_token_ids=[[] for _ in (cached_req_ids or [])],
        all_token_ids={},
        new_block_ids=cached_new_block_ids or [None] * len(cached_req_ids or []),
        num_computed_tokens=cached_num_computed or [],
        num_output_tokens=[0] * len(cached_req_ids or []),
    )
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens or {},
        total_num_scheduled_tokens=sum((num_scheduled_tokens or {}).values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _new_req(
    req_id: str,
    block_ids: list[int],
    prompt_token_ids: list[int],
    num_computed_tokens: int = 0,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_ids=(block_ids,),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_producer_consumer_handoff():
    """The params returned by the producer must produce correct consumer
    metadata -- this is the core contract of the kv_transfer_params design."""

    producer = _make_connector("kv_producer")
    consumer = _make_connector("kv_consumer")

    # -- Producer side --
    prod_req = _make_request(
        "prefill-42",
        num_tokens=10,
        kv_transfer_params={"remote_kv_addr": "10.0.1.3:22001"},
    )
    producer.update_state_after_alloc(
        prod_req, _make_blocks([0, 1]), num_external_tokens=0
    )

    tokens = list(range(10))
    prod_sched = _make_scheduler_output(
        new_reqs=[_new_req("prefill-42", [0, 1], tokens)],
        num_scheduled_tokens={"prefill-42": 10},
    )
    prod_meta = producer.build_connector_meta(prod_sched)
    assert len(prod_meta.requests) == 1
    assert prod_meta.requests[0].remote_kv_addr == "10.0.1.3:22001"

    # Producer finishes -> returns params (these go through the proxy)
    _, params = producer.request_finished(prod_req, block_ids=[0, 1])
    assert params is not None
    assert params["remote_request_id"] == "prefill-42"
    assert params["remote_kv_addr"] == producer._kv_addr

    # -- Consumer side (receives params via proxy) --
    cons_req = _make_request("decode-99", num_tokens=10, kv_transfer_params=params)
    consumer.update_state_after_alloc(
        cons_req, _make_blocks([5, 6]), num_external_tokens=9
    )

    cons_sched = _make_scheduler_output(
        new_reqs=[_new_req("decode-99", [5, 6], tokens)],
        num_scheduled_tokens={"decode-99": 10},
    )
    cons_meta = consumer.build_connector_meta(cons_sched)

    # Consumer metadata must reference producer's request_id
    assert len(cons_meta.requests) == 1
    assert cons_meta.requests[0].request_id == "prefill-42"
    assert cons_meta.requests[0].local_request_id == "decode-99"
    assert cons_meta.requests[0].remote_kv_addr == producer._kv_addr


def test_two_step_chunked_prefill():
    """A prompt too large for one scheduling step must be handled
    across two steps without losing the remote address."""

    producer = _make_connector("kv_producer")

    tokens = list(range(20))  # 20 tokens
    req = _make_request(
        "req-chunk",
        num_tokens=20,
        kv_transfer_params={"remote_kv_addr": "10.0.1.3:22001"},
    )
    producer.update_state_after_alloc(req, _make_blocks([0, 1]), num_external_tokens=0)

    # Step 1: only 8 of 20 tokens scheduled -> chunked
    step1 = _make_scheduler_output(
        new_reqs=[_new_req("req-chunk", [0, 1], tokens, num_computed_tokens=0)],
        num_scheduled_tokens={"req-chunk": 8},
    )
    meta1 = producer.build_connector_meta(step1)
    # Not ready yet -- should produce no metadata
    assert len(meta1.requests) == 0
    assert "req-chunk" in producer.chunked_prefill

    # Step 2: remaining 12 tokens scheduled as cached req
    step2 = _make_scheduler_output(
        cached_req_ids=["req-chunk"],
        cached_num_computed=[8],
        cached_new_block_ids=[([2, 3],)],
        num_scheduled_tokens={"req-chunk": 12},
    )
    meta2 = producer.build_connector_meta(step2)

    # Now the full prompt is prefilled -> metadata emitted
    assert len(meta2.requests) == 1
    assert meta2.requests[0].request_id == "req-chunk"
    assert meta2.requests[0].remote_kv_addr == "10.0.1.3:22001"
    # Block IDs accumulated across both steps
    assert list(meta2.requests[0].block_ids.numpy()) == [0, 1, 2, 3]
    # Chunked state cleaned up
    assert "req-chunk" not in producer.chunked_prefill


def test_non_disagg_request_skipped():
    """A request arriving at the producer without kv_transfer_params
    must be silently skipped, not crash in _get_remote_kv_addr."""

    producer = _make_connector("kv_producer")

    # Request has no kv_transfer_params -> not in _requests_need_save
    tokens = list(range(10))
    sched = _make_scheduler_output(
        new_reqs=[_new_req("plain-req", [0, 1], tokens)],
        num_scheduled_tokens={"plain-req": 10},
    )
    meta = producer.build_connector_meta(sched)
    assert len(meta.requests) == 0


def test_mixed_disagg_and_plain_requests():
    """Only disagg requests produce metadata; non-disagg are skipped."""

    producer = _make_connector("kv_producer")

    disagg_req = _make_request(
        "disagg-1",
        num_tokens=10,
        kv_transfer_params={"remote_kv_addr": "10.0.1.3:22001"},
    )
    producer.update_state_after_alloc(
        disagg_req, _make_blocks([0]), num_external_tokens=0
    )

    tokens = list(range(10))
    sched = _make_scheduler_output(
        new_reqs=[
            _new_req("plain-1", [1], tokens),
            _new_req("disagg-1", [0], tokens),
            _new_req("plain-2", [2], tokens),
        ],
        num_scheduled_tokens={
            "plain-1": 10,
            "disagg-1": 10,
            "plain-2": 10,
        },
    )
    meta = producer.build_connector_meta(sched)
    assert len(meta.requests) == 1
    assert meta.requests[0].request_id == "disagg-1"


def test_request_finished_cleanup():
    """request_finished must clean up internal state and return the right
    kv_transfer_params for the proxy to forward."""

    producer = _make_connector("kv_producer")

    req = _make_request(
        "req-1",
        num_tokens=10,
        kv_transfer_params={"remote_kv_addr": "10.0.1.3:22001"},
    )
    producer.update_state_after_alloc(req, _make_blocks([0, 1]), num_external_tokens=0)
    assert "req-1" in producer._requests_need_save

    delay, params = producer.request_finished(req, block_ids=[0, 1])

    # Producer returns False (no async send at scheduler level)
    assert delay is False
    # Params contain what the proxy needs
    assert params == {
        "remote_request_id": "req-1",
        "remote_kv_addr": producer._kv_addr,
    }
    # Internal state cleaned up
    assert "req-1" not in producer._requests_need_save


def test_consumer_request_finished_returns_no_params():
    """Consumer's request_finished should return no kv_transfer_params."""

    consumer = _make_connector("kv_consumer")

    req = _make_request(
        "decode-1",
        num_tokens=10,
        kv_transfer_params={
            "remote_request_id": "prefill-1",
            "remote_kv_addr": "10.0.0.1:14579",
        },
    )
    consumer.update_state_after_alloc(req, _make_blocks([0, 1]), num_external_tokens=9)

    delay, params = consumer.request_finished(req, block_ids=[0, 1])
    assert delay is False
    assert params is None
