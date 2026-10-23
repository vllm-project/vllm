# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests for request identity, rejection/reordering and graph padding."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.looped_self_draft.activation_cache import (
    LoopedSelfDraftActivationCache,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_cache():
    return LoopedSelfDraftActivationCache(
        SimpleNamespace(
            num_speculative_steps=3,
            hidden_size=7,
            max_num_reqs=4,
            device=torch.device("cuda"),
            dtype=torch.float32,
            idx_mapping=torch.tensor([2, 0, -1, -1], device="cuda", dtype=torch.int32),
            current_draft_step=torch.tensor(0, device="cuda", dtype=torch.int64),
        )
    )


def scheduler(**kwargs):
    fields = dict(
        num_scheduled_tokens={"a": 4},
        scheduled_spec_decode_tokens={"a": [1, 2, 3]},
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "change",
    [
        {"num_scheduled_tokens": {"a": 3}},
        {"num_scheduled_tokens": {"unknown": 4}},
        {"scheduled_spec_decode_tokens": {}},
        {"finished_req_ids": {"a"}},
        {"preempted_req_ids": {"a"}},
        {"scheduled_new_reqs": [SimpleNamespace(req_id="a")]},
        {"scheduled_cached_reqs": SimpleNamespace(resumed_req_ids={"a"})},
    ],
)
def test_fallback_gate(change):
    cache = make_cache()
    cache.requests = {"a"}
    assert cache.can_reuse(scheduler())
    assert not cache.can_reuse(scheduler(**change))


def test_record_merge_and_graph_replay():
    cache = make_cache()
    values = torch.arange(28, device="cuda", dtype=torch.float32).reshape(4, 7)
    state = values + 100
    for step in range(3):
        cache.draft_step.fill_(step)
        cache.record(values, state)
    # Padded rows must not overwrite a valid request slot.
    torch.testing.assert_close(
        cache.input_hidden_states[0], values[1].expand(3, 7), rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.input_hidden_states[2], values[0].expand(3, 7), rtol=0, atol=0
    )
    cache.verify_indices[:2] = torch.tensor([0, 2], device="cuda")
    tail = torch.full((2, 7), 42.0, device="cuda")
    merged, states = cache.merge(tail, tail + 100)
    expected = torch.stack(
        [
            values[1],
            values[1],
            values[1],
            tail[0],
            values[0],
            values[0],
            values[0],
            tail[1],
        ]
    )
    torch.testing.assert_close(merged, expected, rtol=0, atol=0)
    torch.testing.assert_close(states, expected + 100, rtol=0, atol=0)

    # Replay reads new request mapping, step and padding from GPU buffers.
    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        cache.record(values, state)
    cache.idx_mapping[0] = 1
    cache.draft_step.fill_(1)
    cache.idx_mapping[1:] = -1
    values.fill_(77)
    state.fill_(88)
    before = cache.input_hidden_states[0].clone()
    graph.replay()
    torch.testing.assert_close(
        cache.input_hidden_states[1, 1], values[0], rtol=0, atol=0
    )
    torch.testing.assert_close(cache.hidden_states[1, 1], state[0], rtol=0, atol=0)
    torch.testing.assert_close(cache.input_hidden_states[0], before, rtol=0, atol=0)


def test_prepare_inputs_uses_correction_and_prefill_lookahead():
    """Drafts start after accepted KV, including zero-sample prefill and padding."""
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.looped_self_draft.speculator import (
        _prepare_draft_inputs_kernel,
    )

    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=8, device=torch.device("cuda")
    )
    indices = torch.tensor([2, 0, -1], device="cuda")
    target_seq_lens = torch.tensor([12, 8, 0], device="cuda", dtype=torch.int32)
    num_sampled = torch.tensor([1, 0, 0], device="cuda", dtype=torch.int32)
    num_rejected = torch.tensor([3, 0, 0], device="cuda", dtype=torch.int32)
    last_sampled = torch.tensor([100, 101, 102, 103], device="cuda")
    next_prefill = torch.tensor([[200, 201, 202, 203]], device="cuda")
    sample_src_positions = torch.empty(4, device="cuda", dtype=torch.int64)
    _prepare_draft_inputs_kernel[(4,)](
        buffers.input_ids,
        buffers.positions,
        sample_src_positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        indices,
        target_seq_lens,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill,
        32,
        4,
        BLOCK_SIZE=32,
    )
    assert buffers.input_ids[:3].tolist() == [102, 200, 0]
    assert buffers.positions[:3].tolist() == [9, 8, 0]
    assert sample_src_positions[:3].tolist() == [9, 8, 0]
    assert buffers.seq_lens.tolist() == [10, 9, 0, 0]
    assert buffers.query_start_loc.tolist() == [0, 1, 2, 3, 3]
