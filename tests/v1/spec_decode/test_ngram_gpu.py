# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 GPU-accelerated n-gram speculator.

These tests target the Triton proposer in
``vllm.v1.worker.gpu.spec_decode.ngram.speculator`` and complement the CPU
``NgramProposer`` tests in ``test_ngram.py``. The GPU speculator follows a
slightly different policy than the CPU one: when multiple n-gram matches of
the same length exist, the GPU kernel picks the right-most (most recent)
match inside the active context, whereas the CPU implementation returns the
left-most. The expectations below reflect the GPU behavior.

Also covers the GPU draft-trimming layout helpers in
``adaptive_verification`` that ngram_gpu shares with DSpark.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    VariableDraftTrimmer,
    build_verification_layout,
)
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.states import RequestState

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for NgramGPUSpeculator tests",
        allow_module_level=True,
    )

DEVICE = torch.device("cuda")


def _make_vllm_config(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 8,
    max_model_len: int = 64,
    method: str = "ngram_gpu",
) -> VllmConfig:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    speculative_config = SpeculativeConfig(
        method=method,
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
    )


def _make_request_state(cfg: VllmConfig) -> RequestState:
    return RequestState(
        max_num_reqs=cfg.scheduler_config.max_num_seqs,
        max_model_len=cfg.model_config.max_model_len,
        max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,
        num_speculative_steps=cfg.speculative_config.num_speculative_tokens,
        vocab_size=cfg.model_config.get_vocab_size(),
        device=DEVICE,
        use_dense_all_token_ids=True,
    )


def _make_speculator(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 8,
    max_model_len: int = 32,
) -> NgramGPUSpeculator:
    cfg = _make_vllm_config(
        min_n=min_n,
        max_n=max_n,
        k=k,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    return NgramGPUSpeculator(cfg, DEVICE, _make_request_state(cfg))


def _propose(
    spec: NgramGPUSpeculator,
    rows: list[list[int]],
    seq_lens: list[int] | None = None,
    num_sampled: list[int] | None = None,
    last_sampled: list[int] | None = None,
    slots: list[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Place each batch row at a request slot and run propose().

    Returns (drafts, num_valid) as python lists in batch order.
    """
    B = len(rows)
    if seq_lens is None:
        seq_lens = [len(r) for r in rows]
    if num_sampled is None:
        num_sampled = [1] * B
    if last_sampled is None:
        last_sampled = [0] * B
    if slots is None:
        slots = list(range(B))

    max_num_reqs = spec.max_num_reqs
    all_token_ids = spec.req_states.all_token_ids.gpu
    total_len = spec.req_states.total_len.gpu
    all_token_ids.zero_()
    total_len.zero_()
    last_sampled_t = torch.zeros((max_num_reqs, 1), dtype=torch.int64, device=DEVICE)
    for row, slot, seq_len, last in zip(rows, slots, seq_lens, last_sampled):
        if row:
            all_token_ids[slot, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=DEVICE
            )
        total_len[slot] = seq_len
        last_sampled_t[slot, 0] = last

    idx_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    input_batch = SimpleNamespace(num_reqs=B, idx_mapping=idx_mapping)

    drafts = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device=DEVICE),
        aux_hidden_states=None,
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE),
        num_rejected=torch.zeros(B, dtype=torch.int32, device=DEVICE),
        last_sampled=last_sampled_t,
        next_prefill_tokens=torch.zeros(B, dtype=torch.int32, device=DEVICE),
        temperature=torch.zeros(B, dtype=torch.float32, device=DEVICE),
        seeds=torch.zeros(B, dtype=torch.int64, device=DEVICE),
    )
    num_valid = spec.num_valid_drafts_for_trim[idx_mapping]
    return drafts.cpu().tolist(), num_valid.cpu().tolist()


# ---------------------------------------------------------------------------
# Proposal behavior
# ---------------------------------------------------------------------------


def test_no_match_returns_zero_valid():
    """No 2-gram match in [1,2,3,4,5] → 0 valid drafts, last_sampled fill."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts, num_valid = _propose(spec, [[1, 2, 3, 4, 5]], last_sampled=[42])
    assert num_valid == [0]
    assert drafts == [[42, 42]]


def test_no_4gram_match_only():
    """No 4-gram match in [1,2,3,4,1,2,3] → 0 valid drafts."""
    spec = _make_speculator(min_n=4, max_n=4, k=2)
    drafts, num_valid = _propose(spec, [[1, 2, 3, 4, 1, 2, 3]], last_sampled=[7])
    assert num_valid == [0]
    assert drafts == [[7, 7]]


def test_falls_back_to_3gram_when_4gram_missing():
    """No 4-gram match but a 3-gram match exists → propose [4, 1]."""
    spec = _make_speculator(min_n=3, max_n=4, k=2)
    drafts, num_valid = _propose(spec, [[1, 2, 3, 4, 1, 2, 3]])
    assert num_valid == [2]
    assert drafts == [[4, 1]]


def test_prefers_longer_ngram():
    """Both a 4-gram and a 3-gram match exist → prefer the 4-gram match."""
    spec = _make_speculator(min_n=3, max_n=4, k=2)
    drafts, num_valid = _propose(spec, [[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]])
    assert num_valid == [2]
    assert drafts == [[1, 2]]


def test_picks_longest_match_among_2_3_4_grams():
    """2-gram and 3-gram match, 4-gram does not → propose 3-gram match [1, 2]."""
    spec = _make_speculator(min_n=2, max_n=4, k=2)
    drafts, num_valid = _propose(spec, [[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]])
    assert num_valid == [2]
    assert drafts == [[1, 2]]


def test_picks_rightmost_when_multiple_matches():
    """Multiple 3-gram matches for suffix (1,2,3) → pick the right-most."""
    spec = _make_speculator(min_n=3, max_n=3, k=2)
    drafts, num_valid = _propose(
        spec, [[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]]
    )
    assert num_valid == [2]
    assert drafts == [[300, 1]]


def test_short_context_yields_zero_valid():
    """The only length-2 window overlaps the suffix itself → no match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts, num_valid = _propose(spec, [[5, 6]], last_sampled=[99])
    assert num_valid == [0]
    assert drafts == [[99, 99]]


def test_zero_sampled_disables_proposal():
    """num_sampled==0 disables proposals for that request regardless of match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts, num_valid = _propose(
        spec, [[1, 2, 3, 1, 2]], num_sampled=[0], last_sampled=[77]
    )
    assert num_valid == [0]
    assert drafts == [[77, 77]]


def test_truncates_num_valid_when_few_tokens_after_match():
    """Fewer than k tokens after the match → num_valid < k, tail falls back.

    Tokens: [1, 2, 1, 2] (seq_len=4). Suffix (1, 2) matches at position 0
    (the match at position 2 is the suffix itself and is excluded). With
    k=3, only 2 slots map to tokens inside the context.
    """
    spec = _make_speculator(min_n=2, max_n=2, k=3)
    drafts, num_valid = _propose(spec, [[1, 2, 1, 2]], last_sampled=[55])
    assert num_valid == [2]
    assert drafts[0][:2] == [1, 2]
    assert drafts[0][2] == 55


def test_multibatch_mixed():
    """Mixed batch: row 0 matches, row 1 has no match."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts, num_valid = _propose(
        spec,
        [[1, 2, 3, 1, 2], [4, 5, 6]],
        last_sampled=[10, 20],
    )
    assert num_valid == [2, 0]
    assert drafts[0] == [3, 1]
    assert drafts[1] == [20, 20]


def test_multibatch_independent_choice_of_n():
    """Each row independently picks its longest matched n."""
    spec = _make_speculator(min_n=2, max_n=3, k=2)
    drafts, num_valid = _propose(
        spec,
        [
            [9, 1, 2, 3, 8, 1, 2, 3],  # 3-gram (1,2,3) at idx 1 → [8, 1]
            [7, 1, 2, 9, 1, 2],  # 2-gram (1,2) at idx 1 → [9, 1]
        ],
    )
    assert num_valid == [2, 2]
    assert drafts[0] == [8, 1]
    assert drafts[1] == [9, 1]


def test_min_n_eq_1():
    """min_n=max_n=1 — single-token n-grams always match if context > 1."""
    spec = _make_speculator(min_n=1, max_n=1, k=2)
    drafts, num_valid = _propose(spec, [[1, 2, 3, 4, 1]])
    assert num_valid == [2]
    assert drafts == [[2, 3]]


def test_noncontiguous_idx_mapping():
    """propose() reads token rows in place via idx_mapping (non-contiguous)."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    drafts, num_valid = _propose(
        spec,
        [[7, 8, 9, 7, 8], [1, 2, 3, 1, 2]],
        slots=[3, 0],
    )
    assert drafts == [[9, 7], [3, 1]]
    assert num_valid == [2, 2]


def test_num_valid_written_to_request_slots():
    """num_valid_drafts_for_trim is req-slot indexed for the draft trimmer."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    _propose(
        spec,
        [[7, 8, 9, 7, 8], [1, 2, 3, 4, 5]],
        slots=[5, 2],
    )
    nv = spec.num_valid_drafts_for_trim.cpu()
    assert nv[5].item() == 2  # match
    assert nv[2].item() == 0  # no match


def test_dummy_run_does_not_touch_state():
    """Dummy runs must not mutate persistent request or drafter state."""
    spec = _make_speculator(min_n=2, max_n=2, k=2)
    _propose(spec, [[1, 2, 3, 1, 2]], slots=[1])
    before = spec.num_valid_drafts_for_trim.clone()

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([1], dtype=torch.int64, device=DEVICE),
    )
    drafts = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device=DEVICE),
        aux_hidden_states=None,
        num_sampled=torch.ones(1, dtype=torch.int32, device=DEVICE),
        num_rejected=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        last_sampled=torch.zeros((8, 1), dtype=torch.int64, device=DEVICE),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        temperature=torch.zeros(1, dtype=torch.float32, device=DEVICE),
        seeds=torch.zeros(1, dtype=torch.int64, device=DEVICE),
        dummy_run=True,
    )
    assert drafts.shape == (1, 2)
    assert torch.equal(spec.num_valid_drafts_for_trim.cpu(), before.cpu())


def test_construction_validates_speculative_config():
    spec = _make_speculator(min_n=2, max_n=3, k=2)
    assert spec.min_n == 2
    assert spec.max_n == 3
    assert spec.num_speculative_steps == 2
    # Inherited no-op hooks must not raise.
    spec.init_cudagraph_manager(None)
    spec.capture()


# ---------------------------------------------------------------------------
# GPU draft trimming (shared verification-layout machinery)
# ---------------------------------------------------------------------------


def test_build_verification_layout_exact_and_gpu_tail():
    """Layout cumsums match a numpy reference; padding tail equals the total."""
    capacities = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEVICE)
    non_draft = torch.tensor([1, 5, 1], dtype=torch.int32, device=DEVICE)
    num_bonus = 1
    max_num_reqs = 6
    cu_num_logits = torch.empty(max_num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl = torch.empty(max_num_reqs + 1, dtype=torch.int32, device=DEVICE)

    for num_tokens in (10, None):  # exact CPU total vs GPU cumsum tail
        cnl, out_qsl = build_verification_layout(
            capacities, non_draft, num_bonus, cu_num_logits, qsl, num_tokens
        )
        assert cnl.cpu().tolist() == [0, 3, 4, 6]
        assert out_qsl.cpu().tolist()[:4] == [0, 3, 8, 10]
        # Trailing (padding) entries hold the batch total.
        assert out_qsl.cpu().tolist()[4:] == [10, 10, 10]


def test_variable_draft_trimmer_clamps_to_num_valid():
    """Scheduled draft slots are clamped per request to the drafter's counts."""
    max_num_reqs = 8
    num_valid_drafts = torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
    num_valid_drafts[4] = 1  # drafter produced 1 valid draft for slot 4
    num_valid_drafts[2] = 3  # more than scheduled for slot 2
    qsl_buf = torch.empty(max_num_reqs + 1, dtype=torch.int32, device=DEVICE)

    trimmer = VariableDraftTrimmer(
        num_valid_drafts,
        qsl_buf,
        num_bonus_tokens=1,
        max_num_reqs=max_num_reqs,
        max_total_logits=1024,
        device=DEVICE,
    )
    # Batch: [slot 4 (2 drafts scheduled), slot 2 (2 drafts), slot 0 (prefill)].
    idx_mapping = torch.tensor([4, 2, 0], dtype=torch.int64, device=DEVICE)
    num_draft_tokens_per_req = np.array([2, 2, 0], dtype=np.int32)
    num_scheduled_tokens = np.array([3, 3, 7], dtype=np.int32)

    cu_num_logits, qsl = trimmer.trim(
        idx_mapping, num_draft_tokens_per_req, num_scheduled_tokens
    )
    # capacities = min(scheduled, num_valid) = [1, 2, 0]
    assert cu_num_logits.cpu().tolist() == [0, 2, 5, 6]
    # query lens = non-draft + capacities = [1+1, 1+2, 7+0]
    assert qsl.cpu().tolist()[:4] == [0, 2, 5, 12]
    # Padding tail equals the (GPU) batch total.
    assert qsl.cpu().tolist()[4:] == [12] * (max_num_reqs - 3)
