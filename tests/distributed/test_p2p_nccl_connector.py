# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

_PROMPT = list(range(10))


def _make_connector(is_producer: bool) -> P2pNcclConnector:
    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = 16
    vllm_config.kv_transfer_config.is_kv_producer = is_producer
    return P2pNcclConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=MagicMock(),
    )


def _new_req(
    req_id: str,
    external_req_id: str | None,
    prompt_token_ids: list[int] | None = None,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else _PROMPT,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1, 2],),
        num_computed_tokens=0,
        lora_request=None,
        external_req_id=external_req_id,
    )


def _sched_out(
    new_reqs: list[NewRequestData],
    num_scheduled_tokens: dict[str, int] | None = None,
) -> MagicMock:
    out = MagicMock()
    out.scheduled_new_reqs = new_reqs
    out.scheduled_cached_reqs = CachedRequestData.make_empty()
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {r.req_id: len(r.prompt_token_ids or []) for r in new_reqs}
    out.num_scheduled_tokens = num_scheduled_tokens
    return out


def test_producer_and_consumer_use_same_stable_key():
    """Both engines produce the same NCCL tensor key when external_req_id matches."""
    external_req_id = "user-req___prefill_addr_1.2.3.4:9000___decode_addr_5.6.7.8:9001"
    producer_req_id = external_req_id + "-aabbccdd"
    consumer_req_id = external_req_id + "-11223344"

    producer = _make_connector(is_producer=True)
    consumer = _make_connector(is_producer=False)

    prod_meta = producer.build_connector_meta(
        _sched_out([_new_req(producer_req_id, external_req_id)])
    )
    consumer._requests_need_load[consumer_req_id] = (MagicMock(), [0, 1, 2])
    cons_meta = consumer.build_connector_meta(
        _sched_out([_new_req(consumer_req_id, external_req_id)])
    )

    assert isinstance(prod_meta, P2pNcclConnectorMetadata)
    assert isinstance(cons_meta, P2pNcclConnectorMetadata)
    assert len(prod_meta.requests) == 1
    assert len(cons_meta.requests) == 1
    assert prod_meta.requests[0].request_id == external_req_id
    assert cons_meta.requests[0].request_id == external_req_id
    assert prod_meta.requests[0].request_id == cons_meta.requests[0].request_id


def test_without_external_req_id_engines_disagree():
    """Without external_req_id the two engines produce different keys (the bug)."""
    external_req_id = "user-req-abc"
    producer_req_id = external_req_id + "-aabbccdd"
    consumer_req_id = external_req_id + "-11223344"

    producer = _make_connector(is_producer=True)
    consumer = _make_connector(is_producer=False)

    prod_meta = producer.build_connector_meta(
        _sched_out([_new_req(producer_req_id, external_req_id=None)])
    )
    consumer._requests_need_load[consumer_req_id] = (MagicMock(), [0, 1])
    cons_meta = consumer.build_connector_meta(
        _sched_out([_new_req(consumer_req_id, external_req_id=None)])
    )

    assert prod_meta.requests[0].request_id == producer_req_id
    assert cons_meta.requests[0].request_id == consumer_req_id
    assert prod_meta.requests[0].request_id != cons_meta.requests[0].request_id


@pytest.mark.parametrize("is_producer", [True, False])
def test_falls_back_to_req_id_when_no_external_req_id(is_producer: bool):
    """Non-disaggregated path: external_req_id=None falls back to req_id."""
    connector = _make_connector(is_producer=is_producer)
    if not is_producer:
        connector._requests_need_load["req-standalone"] = (MagicMock(), [0, 1])
    meta = connector.build_connector_meta(
        _sched_out([_new_req("req-standalone", external_req_id=None)])
    )
    assert len(meta.requests) == 1
    assert meta.requests[0].request_id == "req-standalone"


def test_chunked_prefill_final_chunk_uses_stable_id():
    """stable_id registered on partial chunk is used when final chunk is emitted."""
    connector = _make_connector(is_producer=True)
    req_id = "req-prefill-xxxx"
    external_req_id = "req-stable"
    prompt = list(range(20))

    step1 = MagicMock()
    step1.scheduled_new_reqs = [_new_req(req_id, external_req_id, prompt_token_ids=prompt)]
    step1.scheduled_cached_reqs = CachedRequestData.make_empty()
    step1.num_scheduled_tokens = {req_id: 8}  # 8 < 20 → partial

    meta1 = connector.build_connector_meta(step1)
    assert len(meta1.requests) == 0
    assert connector._req_stable_id[req_id] == external_req_id

    cached_reqs = MagicMock()
    cached_reqs.req_ids = [req_id]
    cached_reqs.num_computed_tokens = [8]
    cached_reqs.new_block_ids = [([2, 3],)]
    cached_reqs.resumed_req_ids = set()

    step2 = MagicMock()
    step2.scheduled_new_reqs = []
    step2.scheduled_cached_reqs = cached_reqs
    step2.num_scheduled_tokens = {req_id: 12}  # 8+12=20 → final

    meta2 = connector.build_connector_meta(step2)
    assert len(meta2.requests) == 1
    assert meta2.requests[0].request_id == external_req_id
