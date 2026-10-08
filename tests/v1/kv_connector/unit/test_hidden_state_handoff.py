# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler flow of the P/D hidden-state handoff."""

import copy

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

# opt-125m: a 768-wide fp16 hidden state, 1536 bytes. Each token slot holds
# (96 + 96) * 2 bytes per head per layer, 768 over the two layers.
TAIL_TOKENS = 2


def _config(block_size):
    return KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0", "layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=96,
                    dtype=torch.float16,
                ),
            ),
        ],
    )


def _make(kv_role="kv_consumer"):
    vllm_config = create_vllm_config(
        kv_connector_extra_config={"hidden_state_handoff": True},
        kv_role=kv_role,
        kv_load_failure_policy="recompute",
    )
    bs = vllm_config.cache_config.block_size
    scheduler = create_scheduler(vllm_config, kv_cache_config=_config(bs))
    assert scheduler.hidden_state_record_group_id == 0
    assert scheduler.hidden_state_record_tail_tokens == TAIL_TOKENS
    return vllm_config, scheduler


def _num_blocks(scheduler, request):
    return len(scheduler.kv_cache_manager.get_block_ids(request.request_id)[0])


def _recv(scheduler, request, failed=False):
    so = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    out = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    out.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id},
        failed_recving={request.request_id} if failed else set(),
    )
    scheduler.update_from_output(so, out)


def test_decode_samples_from_record():
    """D loads the prompt's blocks plus those of the record's slots past the
    prompt, then samples the last prompt token from the record."""
    vllm_config, scheduler = _make()
    bs = vllm_config.cache_config.block_size
    # The record fits the last block, starts a new one, or spills into one.
    for num_tokens in (bs * 2 + 5, bs * 2, bs * 2 + bs - 1):
        request = create_request(
            block_size=bs,
            num_tokens=num_tokens,
            do_remote_prefill=True,
            num_remote_blocks=3,
        )
        scheduler.add_request(request)
        _recv(scheduler, request)
        assert _num_blocks(scheduler, request) == cdiv(num_tokens + TAIL_TOKENS, bs)
        so = scheduler.schedule()
        rid = request.request_id
        assert so.hidden_state_record_req_ids == {rid}
        assert so.num_scheduled_tokens[rid] == 1
        (new_req,) = so.scheduled_new_reqs
        assert new_req.num_computed_tokens == num_tokens - 1
        scheduler.update_from_output(so, create_model_runner_output([request]))
        so = scheduler.schedule()
        assert not so.hidden_state_record_req_ids
        assert request.num_computed_tokens == num_tokens + 1
        scheduler.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
        scheduler.update_from_output(so, create_model_runner_output([]))


def test_record_blocks_not_zeroed_under_load():
    """Blocks the load writes the record into are not zeroed under it."""
    vllm_config, scheduler = _make()
    bs = vllm_config.cache_config.block_size
    scheduler.needs_kv_cache_zeroing = True
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        manager._record_new_block_ids = True
    # The record starts a block past the prompt's.
    request = create_request(
        block_size=bs, num_tokens=bs * 2, do_remote_prefill=True, num_remote_blocks=3
    )
    scheduler.add_request(request)
    so = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _num_blocks(scheduler, request) == 3
    assert not so.new_block_ids_to_zero


def test_failed_recv_recomputes():
    vllm_config, scheduler = _make()
    bs = vllm_config.cache_config.block_size
    request = create_request(
        block_size=bs, num_tokens=bs * 2 + 3, do_remote_prefill=True
    )
    scheduler.add_request(request)
    _recv(scheduler, request, failed=True)
    so = scheduler.schedule()
    assert not so.hidden_state_record_req_ids
    assert so.num_scheduled_tokens[request.request_id] > 1


def test_prefiller_hands_over_record_blocks():
    """P reserves the record's slots past the prompt and hands over their
    blocks with the prompt's."""
    vllm_config, scheduler = _make(kv_role="kv_producer")
    bs = vllm_config.cache_config.block_size
    for num_tokens in (bs * 2 + 3, bs * 2, bs * 2 + bs - 1):
        request = create_request(
            block_size=bs, num_tokens=num_tokens, do_remote_decode=True
        )
        scheduler.add_request(request)
        so = scheduler.schedule()
        assert so.num_scheduled_tokens[request.request_id] == num_tokens
        outs = scheduler.update_from_output(
            so, create_model_runner_output([request], use_eos=True)
        )
        params = outs[0].outputs[0].kv_transfer_params
        (remote_block_ids,) = params["remote_block_ids"]
        assert len(remote_block_ids) == cdiv(num_tokens + TAIL_TOKENS, bs)


def test_local_request_reserves_no_record():
    vllm_config, scheduler = _make(kv_role="kv_producer")
    bs = vllm_config.cache_config.block_size
    request = create_request(block_size=bs, num_tokens=bs * 2)
    scheduler.add_request(request)
    scheduler.schedule()
    assert _num_blocks(scheduler, request) == 2


@pytest.mark.parametrize("num_heads_read", [2, 1])
def test_record_round_trip_through_kv_slots(num_heads_read):
    """Records written past the prompt into every KV head read back whole,
    also from a head slice (a decoder at a different TP size)."""
    import numpy as np

    from vllm.v1.core.hidden_state_record import RecordLayout
    from vllm.v1.worker.gpu.hidden_state_handoff import HiddenStateHandoff

    torch.manual_seed(0)
    num_blocks, num_heads, block_size = 8, 2, 4
    layout = RecordLayout(hidden_size=24, window=1, num_aux=2)
    # Two layers' [blocks, heads, tokens, head_size (K + V)] fp16 caches.
    caches = [
        torch.zeros(num_blocks, num_heads, block_size, 2 * size, dtype=torch.float16)
        for size in (8, 4)
    ]
    handoff = object.__new__(HiddenStateHandoff)
    handoff.device = torch.device("cpu")
    handoff.dtype = torch.float16
    handoff.layout = layout
    handoff.layer_views = [cache.view(torch.uint8) for cache in caches]
    handoff.head_bytes = sum(v.shape[-1] for v in handoff.layer_views)
    handoff.record_bytes = layout.num_slots * layout.hidden_size * 2
    handoff.tail_tokens = cdiv(handoff.record_bytes, handoff.head_bytes)
    assert handoff.tail_tokens == 4

    # Request r's prompt ends at token 3 of block 2r: its slots run on into
    # block 2r + 1.
    def tail_slots(idx_mapping_np, prompt_len_np):
        slots = (
            torch.from_numpy(idx_mapping_np)[:, None] * 2 * block_size
            + torch.from_numpy(prompt_len_np)[:, None]
            + torch.arange(handoff.tail_tokens)[None, :]
        )
        return slots // block_size, slots % block_size

    handoff._tail_slots = tail_slots
    records = torch.randn(3, layout.num_slots, layout.hidden_size).half()
    idx_mapping, prompt_len = np.arange(3), np.full(3, 3)
    handoff._write_records(idx_mapping, prompt_len, records)
    # Token slots up to the prompt's end are untouched.
    for cache in caches:
        assert not cache[0::2, :, :3].any()
    if num_heads_read == 1:
        handoff.layer_views = [v[:, 1:] for v in handoff.layer_views]
    assert torch.equal(handoff._read_records(idx_mapping, prompt_len), records)
