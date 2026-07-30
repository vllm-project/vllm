# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch ordering and classification in Model Runner V2 (vllm.v1.worker.gpu).

split_decodes_and_prefills assumes decode -> short_extend -> prefill request
ordering. With spec decode (decode_query_len > 1), a shorter chunked-prefill
tail sorted in front of the uniform decodes misclassifies every decode as a
prefill.

Conversely, a prompt chunk of exactly decode_query_len tokens has a decode
batch's shape, and must not be classified as a uniform decode batch.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids
from vllm.v1.worker.utils import get_uniform_decode_token_count


def _make_common_attn_metadata(query_lens: list[int]) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int32), 0, out=query_start_loc[1:]
    )
    seq_lens = torch.tensor([1000 + q for q in query_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        max_seq_len=int(seq_lens.max()),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        block_table_tensor=torch.zeros(num_reqs, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


def _make_runner(req_states: dict[str, tuple[int, int]]) -> Any:
    """Build a runner stub from {req_id: (num_computed_tokens, prefill_len)}."""
    prefill_lens = np.array([s[1] for s in req_states.values()], dtype=np.int32)
    num_computed = np.array([s[0] for s in req_states.values()], dtype=np.int32)
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_states)},
        # The runner keeps this as min(num_computed_tokens, prefill_len).
        num_computed_prefill_tokens=np.minimum(num_computed, prefill_lens),
        prefill_len=SimpleNamespace(np=prefill_lens),
    )
    return runner


def _uniform_token_count(
    req_states: dict[str, tuple[int, int]],
    query_len: int,
    dummy_run: bool = False,
) -> int | None:
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={req_id: query_len for req_id in req_states},
        total_num_scheduled_tokens=query_len * len(req_states),
    )
    return GPUModelRunner._get_uniform_token_count(
        _make_runner(req_states),
        scheduler_output,
        len(req_states),
        query_len * len(req_states),
        query_len,
        dummy_run,
    )


def test_spec_decode_batch_is_uniform_decode():
    # decode_query_len == 8 (K=7): every request past its prefill.
    decodes = {f"d{i}": (16, 16) for i in range(6)}
    assert _uniform_token_count(decodes, 8) == 8


def test_prompt_chunk_of_decode_query_len_is_not_uniform_decode():
    # Two requests are 8 tokens into a 40-token prompt, so their query length
    # only coincides with the K+1 spec-decode query length. Replaying the
    # decode graph here corrupts the six decodes as well, so the whole batch
    # must be rejected. See https://github.com/vllm-project/vllm/issues/49918.
    batch = {f"d{i}": (16, 16) for i in range(6)}
    batch.update({f"p{i}": (8, 40) for i in range(2)})
    assert _uniform_token_count(batch, 8) is None

    # Same collision without spec decoding: a 1-token prompt looks like a
    # 1-token decode step.
    assert _uniform_token_count({"p0": (0, 1)}, 1) is None

    # A fresh prompt that happens to be exactly decode_query_len tokens long.
    assert _uniform_token_count({"p0": (0, 8)}, 8) is None


def test_dummy_batches_stay_uniform_decode():
    # Dummy batches (DP padding, profiling, warmup) are uniform by
    # construction and their requests have no state to consult, so they must be
    # classified without one: looking one up would raise KeyError here, and
    # rejecting them would make graph capture and replay disagree.
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={f"_dummy_req_{i}": 8 for i in range(4)},
        total_num_scheduled_tokens=32,
    )
    assert (
        GPUModelRunner._get_uniform_token_count(
            _make_runner({}), scheduler_output, 4, 32, 8, True
        )
        == 8
    )


def test_sort_batch_req_ids_no_spec():
    # decode_query_len == 1: plain ascending order (decodes first).
    num_tokens_per_req = {"p1": 100, "d1": 1, "p2": 7, "d2": 1}
    assert sort_batch_req_ids(num_tokens_per_req, 1) == ["d1", "d2", "p2", "p1"]


def test_sort_batch_req_ids_spec_decode():
    # decode_query_len == 2 (MTP k=1): uniform decodes lead, then the 1-token
    # chunked-prefill tail, then longer prefills.
    num_tokens_per_req = {"tail": 1, "d1": 2, "p1": 100, "d2": 2}
    assert sort_batch_req_ids(num_tokens_per_req, 2) == ["d1", "d2", "tail", "p1"]


def test_spec_decodes_lead_short_prefill_tail():
    # With the fixed ordering, split_decodes_and_prefills classifies the
    # uniform 2-token decodes as decodes even when a 1-token prefill tail is
    # in the batch (indexer-style: require_uniform, threshold=1+k).
    num_tokens_per_req = {"tail": 1, **{f"d{i}": 2 for i in range(8)}}
    req_ids = sort_batch_req_ids(num_tokens_per_req, 2)
    query_lens = [num_tokens_per_req[r] for r in req_ids]
    assert query_lens == [2] * 8 + [1]

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            _make_common_attn_metadata(query_lens),
            decode_threshold=2,
            require_uniform=True,
        )
    )
    assert (num_decodes, num_prefills) == (8, 1)
    assert (num_decode_tokens, num_prefill_tokens) == (16, 1)


def test_uniform_decode_uses_state_index_not_batch_position():
    """The predicate must read each request's own state, not its batch slot.

    `req_id_to_index` is a persistent map into the runner's state arrays, so a
    request's batch position and its state index diverge as requests come and
    go. Here the only prefilling request sits at state index 0 while the two
    scheduled decodes sit at 1 and 2, so a gather that used batch position
    would read the prefilling row and wrongly reject the batch.
    """
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    # State arrays in state-index order: a prefilling request, then two decodes.
    runner.req_states = SimpleNamespace(
        req_id_to_index={"prefilling": 0, "decode_a": 1, "decode_b": 2},
        num_computed_prefill_tokens=np.array([8, 16, 16], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([40, 16, 16], dtype=np.int32)),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"decode_a": 8, "decode_b": 8},
        total_num_scheduled_tokens=16,
    )

    assert (
        GPUModelRunner._get_uniform_token_count(
            runner, scheduler_output, 2, 16, 8, False
        )
        == 8
    )

    # Scheduling the prefilling request alongside them must reject the batch.
    scheduler_output.num_scheduled_tokens = {
        "decode_a": 8,
        "prefilling": 8,
        "decode_b": 8,
    }
    scheduler_output.total_num_scheduled_tokens = 24
    assert (
        GPUModelRunner._get_uniform_token_count(
            runner, scheduler_output, 3, 24, 8, False
        )
        is None
    )


def test_predicate_agrees_with_is_prefilling():
    """The drafter's call site passes `InputBatch`'s two arrays directly.

    `InputBatch.is_prefilling_np` is documented as
    `num_computed_prefill_tokens_np < prefill_len_np`, so the shared predicate
    must reject a batch exactly when that flag is set for any request. This
    pins the contract the drafter relies on without standing up a speculator.
    """
    cases = [
        (np.array([16, 16], dtype=np.int32), np.array([16, 16], dtype=np.int32)),
        (np.array([8, 16], dtype=np.int32), np.array([40, 16], dtype=np.int32)),
        (np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)),
        (np.array([5, 5, 5], dtype=np.int32), np.array([5, 5, 5], dtype=np.int32)),
    ]
    for num_computed_prefill_tokens, prefill_lens in cases:
        num_reqs = len(prefill_lens)
        is_prefilling = num_computed_prefill_tokens < prefill_lens
        result = get_uniform_decode_token_count(
            num_reqs,
            8 * num_reqs,
            8,
            num_computed_prefill_tokens,
            prefill_lens,
        )
        assert (result is None) == bool(is_prefilling.any()), (
            f"{num_computed_prefill_tokens} vs {prefill_lens} -> {result}"
        )


def test_non_uniform_shape_is_rejected_without_request_state():
    """The shape test must decide on its own, ahead of any state lookup.

    Most batches are mixed prefill/decode and fail on shape alone, so the O(1)
    shape test runs before the per-request prefill state is gathered. A token
    count that is not `num_reqs * max_query_len` has to be rejected even when
    every request has finished prefilling, as they have here.
    """
    num_computed_prefill_tokens = np.array([16, 16], dtype=np.int32)
    prefill_lens = np.array([16, 16], dtype=np.int32)
    assert (
        get_uniform_decode_token_count(
            2,
            12,  # No shared query length over 2 requests produces 12 tokens.
            8,
            num_computed_prefill_tokens,
            prefill_lens,
        )
        is None
    )
