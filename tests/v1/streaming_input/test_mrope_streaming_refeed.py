# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for stale state after an in-place streaming re-feed.

vllm-project/vllm#50731: `NewRequestData` hands the runner the scheduler's own
`prompt_token_ids` list object, and a streaming re-feed grows that list in
place. A re-feed routed through WAITING is repaired by
`_update_streaming_request`, but one that grows the prompt while the request
stays RUNNING comes back through `scheduled_cached_reqs`, where the runner's
derived state still described the pre-growth prompt: `mrope_positions` kept its
old width (raising "Target sizes: [3, N]. Tensor sizes: [3, 0]"), and the
persistent batch kept stale token ids and counters.

A `SimpleNamespace` stands in for `self` so the cached-request path can be
driven without a CUDA runner; `InputBatch` itself is real numpy/torch CPU state.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _unpinned_input_batch(monkeypatch):
    """Pinned allocations need a device; InputBatch also forwards this to its
    block table, so one patch keeps the whole batch host-only."""
    monkeypatch.setattr("vllm.v1.worker.gpu_input_batch.PIN_MEMORY", False)


MAX_REQS = 4
MAX_TOKENS = 128
BLOCK_SIZE = 16
REQ_ID = "streaming-refeed-0"


class _FakeMRoPEModel(SupportsMRoPE):
    """Positions are just the token index, replicated over the 3 M-RoPE axes."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        seq_len = len(input_tokens)
        return torch.arange(seq_len).unsqueeze(0).expand(3, -1).clone(), 0


def _make_req_state(
    *, prompt_token_ids=None, prompt_embeds=None, output_token_ids=None
):
    return CachedRequestState(
        req_id=REQ_ID,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=prompt_embeds,
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=0,
        output_token_ids=output_token_ids if output_token_ids is not None else [],
    )


def _make_runner(req_state: CachedRequestState):
    """Seats `req_state` in a real InputBatch and returns (runner, init_calls)."""
    input_batch = InputBatch(
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_TOKENS,
        max_num_batched_tokens=MAX_TOKENS,
        device=torch.device("cpu"),
        vocab_size=32000,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        max_num_blocks_per_req=[MAX_TOKENS // BLOCK_SIZE],
        logitsprocs=None,
        is_pooling_model=False,
    )
    input_batch.add_request(req_state)

    init_calls: list[str] = []
    runner = SimpleNamespace(
        requests={req_state.req_id: req_state},
        input_batch=input_batch,
        uses_mrope=True,
        uses_xdrope_dim=0,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        num_prompt_logprobs={},
        late_interaction_runner=SimpleNamespace(
            on_requests_finished=lambda req_ids: None
        ),
        get_model=lambda: _FakeMRoPEModel(),
        mrope_positions=SimpleNamespace(
            cpu=torch.zeros(3, MAX_TOKENS, dtype=torch.int64)
        ),
        _process_encoder_cache_scheduler_output=lambda scheduler_output: None,
        _may_reorder_batch=lambda scheduler_output: None,
    )

    def _init_mrope_positions(state):
        init_calls.append(state.req_id)
        GPUModelRunner._init_mrope_positions(runner, state)

    runner._init_mrope_positions = _init_mrope_positions
    runner._refresh_grown_prompt_state = (
        lambda *args: GPUModelRunner._refresh_grown_prompt_state(runner, *args)
    )
    # Seed the pre-growth M-RoPE state the way the new-request path would.
    runner._init_mrope_positions(req_state)
    init_calls.clear()
    return runner, init_calls


def _scheduler_output(*, num_computed: int, num_scheduled: int, num_output: int = 0):
    return SimpleNamespace(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        num_scheduled_tokens={REQ_ID: num_scheduled},
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[REQ_ID],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed],
            num_output_tokens=[num_output],
        ),
    )


def _update_states(runner, scheduler_output):
    with patch(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        GPUModelRunner._update_states(runner, scheduler_output)


def _grow_prompt_and_update(prompt_len: int, grow_by: int):
    """Runs one cached-request step across an in-place prompt growth."""
    req_state = _make_req_state(prompt_token_ids=list(range(1, prompt_len + 1)))
    req_state.num_computed_tokens = prompt_len
    runner, init_calls = _make_runner(req_state)
    assert req_state.mrope_positions.shape == (3, prompt_len)

    # What the scheduler does to the list object it shares with the runner.
    req_state.prompt_token_ids.extend(range(100, 100 + grow_by))
    scheduler_output = _scheduler_output(num_computed=prompt_len, num_scheduled=grow_by)
    _update_states(runner, scheduler_output)
    return runner, req_state, scheduler_output, init_calls


def test_growth_does_not_break_mrope_calc():
    """The crash from the issue: slicing a pre-growth M-RoPE tensor with
    post-growth bookkeeping yields an empty source ([3, 0])."""
    prompt_len, grow_by = 12, 7
    runner, _, scheduler_output, _ = _grow_prompt_and_update(prompt_len, grow_by)

    GPUModelRunner._calc_mrope_positions(runner, scheduler_output)

    expected = torch.arange(prompt_len, prompt_len + grow_by).unsqueeze(0).expand(3, -1)
    torch.testing.assert_close(runner.mrope_positions.cpu[:, :grow_by], expected)


def test_growth_refreshes_mrope_state():
    """A prompt grown in place while RUNNING must re-derive M-RoPE state."""
    prompt_len, grow_by = 12, 7
    _, req_state, _, init_calls = _grow_prompt_and_update(prompt_len, grow_by)

    assert init_calls == [REQ_ID]
    assert req_state.mrope_positions.shape == (3, prompt_len + grow_by)
    assert req_state.num_prompt_tokens == prompt_len + grow_by


def test_growth_mid_prefill_repairs_batch_token_state():
    """Growth with no output tokens to discard: the realignment further down
    in `_update_states` does not fire, so the batch row must be repaired here.

    `num_tokens_no_spec` is the write index for sampled and draft tokens, so a
    stale value lands them inside the grown prompt region.
    """
    prompt_len, grow_by = 12, 7
    new_len = prompt_len + grow_by
    req_state = _make_req_state(prompt_token_ids=list(range(1, prompt_len + 1)))
    req_state.num_computed_tokens = 8  # mid chunked prefill
    runner, _ = _make_runner(req_state)
    batch = runner.input_batch
    assert batch.num_tokens_no_spec[0] == prompt_len

    grown_span = list(range(100, 100 + grow_by))
    req_state.prompt_token_ids.extend(grown_span)
    _update_states(runner, _scheduler_output(num_computed=8, num_scheduled=new_len - 8))

    assert batch.num_prompt_tokens[0] == new_len
    assert batch.num_tokens_no_spec[0] == new_len
    np.testing.assert_array_equal(
        batch.token_ids_cpu[0, prompt_len:new_len], np.array(grown_span)
    )
    assert batch.is_token_ids[0, prompt_len:new_len].all()


def test_growth_after_output_fold_keeps_batch_consistent():
    """The scheduler folds computed outputs into the prompt and clears them.

    The folded span must keep its values (the rewrite is idempotent) and the
    realignment below must land on the refreshed prompt length.
    """
    prompt_len, chunk = 12, 4
    outputs = [901, 902, 903]
    new_len = prompt_len + len(outputs) + chunk
    req_state = _make_req_state(
        prompt_token_ids=list(range(1, prompt_len + 1)),
        output_token_ids=list(outputs),
    )
    req_state.num_computed_tokens = prompt_len + len(outputs)
    runner, _ = _make_runner(req_state)
    batch = runner.input_batch

    # Scheduler-side fold: outputs become prompt, then the new chunk lands.
    req_state.prompt_token_ids.extend(outputs)
    req_state.prompt_token_ids.extend(range(200, 200 + chunk))
    _update_states(
        runner,
        _scheduler_output(
            num_computed=prompt_len + len(outputs), num_scheduled=chunk, num_output=0
        ),
    )

    assert req_state.output_token_ids == []
    assert batch.num_prompt_tokens[0] == new_len
    assert batch.num_tokens_no_spec[0] == new_len
    np.testing.assert_array_equal(
        batch.token_ids_cpu[0, prompt_len : prompt_len + len(outputs)],
        np.array(outputs),
    )


def test_growth_detected_when_mrope_tensor_is_wider_than_prompt():
    """Implementations that precompute decode positions return a tensor wider
    than the prompt; the growth check must not key off that width."""
    prompt_len, grow_by = 12, 7
    req_state = _make_req_state(prompt_token_ids=list(range(1, prompt_len + 1)))
    req_state.num_computed_tokens = prompt_len
    runner, init_calls = _make_runner(req_state)
    req_state.mrope_positions = torch.zeros(3, prompt_len + 50, dtype=torch.int64)

    req_state.prompt_token_ids.extend(range(100, 100 + grow_by))
    _update_states(
        runner, _scheduler_output(num_computed=prompt_len, num_scheduled=grow_by)
    )

    assert init_calls == [REQ_ID]
    assert req_state.num_prompt_tokens == prompt_len + grow_by
    assert runner.input_batch.num_prompt_tokens[0] == prompt_len + grow_by


def test_unchanged_prompt_takes_the_fast_path():
    """The common case must not pay for a re-init or touch the batch row."""
    prompt_len = 12
    req_state = _make_req_state(prompt_token_ids=list(range(1, prompt_len + 1)))
    req_state.num_computed_tokens = prompt_len
    runner, init_calls = _make_runner(req_state)
    positions_before = req_state.mrope_positions
    batch = runner.input_batch
    tokens_before = batch.token_ids_cpu[0].copy()

    _update_states(runner, _scheduler_output(num_computed=prompt_len, num_scheduled=1))

    assert init_calls == []
    assert req_state.mrope_positions is positions_before
    assert req_state.num_prompt_tokens == prompt_len
    assert batch.num_prompt_tokens[0] == prompt_len
    np.testing.assert_array_equal(batch.token_ids_cpu[0], tokens_before)


def test_prompt_embeds_request_is_not_treated_as_grown():
    """Embeds-only requests have no token ids; the length check must use them."""
    prompt_len = 9
    req_state = _make_req_state(prompt_embeds=torch.randn(prompt_len, 16))
    req_state.num_computed_tokens = prompt_len
    runner, init_calls = _make_runner(req_state)
    positions_before = req_state.mrope_positions
    assert positions_before.shape == (3, prompt_len)

    _update_states(runner, _scheduler_output(num_computed=prompt_len, num_scheduled=1))

    assert init_calls == []
    assert req_state.mrope_positions is positions_before
