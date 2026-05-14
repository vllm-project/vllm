# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 GPU-accelerated n-gram speculator.

These tests target the kernel and proposer logic in
``vllm.v1.worker.gpu.spec_decode.ngram.speculator`` and complement the CPU
``NgramProposer`` tests in ``test_ngram.py``. The GPU speculator follows a
slightly different policy than the CPU one: when multiple n-gram matches of the
same length exist, the GPU kernel picks the right-most (most recent) match
inside the active context, whereas the CPU implementation returns the
left-most. The expectations below reflect the GPU behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
    NgramGPUSpeculator,
    _NgramKernel,
)

# The kernel uses ``@support_torch_compile`` and ``set_forward_context``, both
# of which expect a real CUDA device. Skip these tests gracefully on
# CPU-only or non-CUDA platforms.
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for NgramGPUSpeculator tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 8,
    max_model_len: int = 64,
) -> VllmConfig:
    """Build a minimal VllmConfig configured for ngram_gpu speculative decoding.

    ``enforce_eager=True`` is used so the kernel runs in eager mode (no
    torch.compile) — sufficient for unit-testing the math.
    """
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
        method="ngram_gpu",
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
    )


def _make_kernel(min_n: int, max_n: int, k: int) -> _NgramKernel:
    """Construct a ``_NgramKernel`` bound to a minimal ``VllmConfig``.

    The kernel is decorated by ``@support_torch_compile`` whose generated
    ``__init__`` accepts ``vllm_config`` as a kwarg.
    """
    vllm_config = _make_vllm_config(min_n=min_n, max_n=max_n, k=k)
    with set_current_vllm_config(vllm_config):
        kernel = _NgramKernel(
            min_n=min_n,
            max_n=max_n,
            k=k,
            vllm_config=vllm_config,
        )
    return kernel.to("cuda").eval()


def _pad_tokens(rows: list[list[int]], pad_to: int) -> torch.Tensor:
    """Right-pad ragged ``rows`` to ``pad_to`` length with zeros (int32)."""
    out = torch.zeros((len(rows), pad_to), dtype=torch.int32)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return out


def _run_kernel(
    kernel: _NgramKernel,
    rows: list[list[int]],
    seq_lens: list[int],
    valid_mask: list[bool] | None = None,
    last_sampled: list[int] | None = None,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build inputs from python lists, run the kernel, return (drafts, num_valid)."""
    B = len(rows)
    if pad_to is None:
        pad_to = max((len(r) for r in rows), default=1)
        # The kernel needs at least 1 column to satisfy unfold(1, n, 1).
        pad_to = max(pad_to, kernel.max_n)

    if valid_mask is None:
        valid_mask = [True] * B
    if last_sampled is None:
        last_sampled = [0] * B

    token_ids = _pad_tokens(rows, pad_to).cuda()
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool, device="cuda")
    last_sampled_t = torch.tensor(last_sampled, dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        drafts, num_valid = kernel(
            token_ids,
            seq_lens_t,
            valid_mask_t,
            last_sampled_t,
        )
    return drafts.cpu(), num_valid.cpu()


# ---------------------------------------------------------------------------
# _NgramKernel tests (mirror & extend tests/v1/spec_decode/test_ngram.py)
# ---------------------------------------------------------------------------


def test_kernel_no_match_returns_zero_valid():
    """No 2-gram match in [1,2,3,4,5] → kernel reports 0 valid drafts."""
    kernel = _make_kernel(min_n=2, max_n=2, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 4, 5]],
        seq_lens=[5],
        last_sampled=[42],
    )
    assert num_valid.tolist() == [0]
    # Padding positions must fall back to ``last_sampled`` for invalid rows.
    assert drafts.tolist() == [[42, 42]]


def test_kernel_no_4gram_match_only():
    """No 4-gram match in [1,2,3,4,1,2,3] → 0 valid drafts."""
    kernel = _make_kernel(min_n=4, max_n=4, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 4, 1, 2, 3]],
        seq_lens=[7],
        last_sampled=[7],
    )
    assert num_valid.tolist() == [0]
    assert drafts.tolist() == [[7, 7]]


def test_kernel_falls_back_to_3gram_when_4gram_missing():
    """No 4-gram match but a 3-gram match exists → propose [4, 1]."""
    kernel = _make_kernel(min_n=3, max_n=4, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 4, 1, 2, 3]],
        seq_lens=[7],
    )
    assert num_valid.tolist() == [2]
    assert drafts.tolist() == [[4, 1]]


def test_kernel_prefers_longer_ngram():
    """Both a 4-gram (1,2,3,4) and a 3-gram match are present.

    The kernel must prefer the longer match, returning [1, 2] instead of [5, 1].
    """
    kernel = _make_kernel(min_n=3, max_n=4, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]],
        seq_lens=[12],
    )
    assert num_valid.tolist() == [2]
    assert drafts.tolist() == [[1, 2]]


def test_kernel_picks_longest_match_among_2_3_4_grams():
    """2-gram and 3-gram match, 4-gram does not → propose 3-gram match [1, 2]."""
    kernel = _make_kernel(min_n=2, max_n=4, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]],
        seq_lens=[10],
    )
    assert num_valid.tolist() == [2]
    assert drafts.tolist() == [[1, 2]]


def test_kernel_picks_rightmost_when_multiple_matches():
    """Multiple 3-gram matches exist for suffix (1,2,3).

    The GPU kernel picks the RIGHT-most (most recent) match, unlike the CPU
    proposer which picks the left-most. Tokens after the right-most match
    starting at index 8 are [300, 1].
    """
    kernel = _make_kernel(min_n=3, max_n=3, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]],
        seq_lens=[15],
    )
    assert num_valid.tolist() == [2]
    assert drafts.tolist() == [[300, 1]]


def test_kernel_short_context_yields_zero_valid():
    """Context length < min_n → kernel returns 0 valid drafts.

    Note: callers (NgramGPUSpeculator.propose) gate this via ``valid_mask``.
    Here we exercise the kernel directly with a row whose seq_len < min_n.
    The only window of length 2 cannot match a length-2 suffix because it
    overlaps the suffix itself, so no valid match is produced.
    """
    kernel = _make_kernel(min_n=2, max_n=2, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[5, 6]],
        seq_lens=[2],
        last_sampled=[99],
        pad_to=4,
    )
    assert num_valid.tolist() == [0]
    assert drafts.tolist() == [[99, 99]]


def test_kernel_valid_mask_disables_row():
    """``valid_mask=False`` disables proposals for that row regardless of match."""
    kernel = _make_kernel(min_n=2, max_n=2, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 1, 2]],  # has a 2-gram match for suffix (1,2) → [3, 1]
        seq_lens=[5],
        valid_mask=[False],
        last_sampled=[77],
    )
    assert num_valid.tolist() == [0]
    # With the row disabled, fallback ``last_sampled`` fills the draft slots.
    assert drafts.tolist() == [[77, 77]]


def test_kernel_truncates_num_valid_when_few_tokens_after_match():
    """When fewer than k tokens are available after the match, num_valid < k.

    Tokens: [1, 2, 1, 2] (seq_len=4). Suffix (1, 2) matches at position 0
    (the right-most match at position 2 is the suffix itself and is excluded
    by ``max_valid_pos = seq_len - n - 1``). With k=3, only the first 2 slots
    fit within the active context; the third slot must fall back to
    ``last_sampled``.
    """
    kernel = _make_kernel(min_n=2, max_n=2, k=3)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 1, 2]],
        seq_lens=[4],
        last_sampled=[55],
        pad_to=4,
    )
    # Only the first 2 speculative slots map to tokens inside the context.
    assert num_valid.tolist() == [2]
    drafts_row = drafts.tolist()[0]
    # The first 2 drafts come from the tokens following the match.
    assert drafts_row[:2] == [1, 2]
    # The third slot must fall back to ``last_sampled`` since it is invalid.
    assert drafts_row[2] == 55


def test_kernel_multibatch_mixed():
    """Mixed batch: row 0 matches, row 1 has no match."""
    kernel = _make_kernel(min_n=2, max_n=2, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 1, 2], [4, 5, 6, 0, 0]],
        seq_lens=[5, 3],
        last_sampled=[10, 20],
    )
    assert num_valid.tolist() == [2, 0]
    assert drafts.tolist()[0] == [3, 1]
    # Row 1: no match → all slots filled with last_sampled[1] = 20.
    assert drafts.tolist()[1] == [20, 20]


def test_kernel_multibatch_independent_choice_of_n():
    """Row 0 matches as 3-gram, row 1 only matches as 2-gram.

    Verifies that each row independently picks its longest matched n.
    """
    kernel = _make_kernel(min_n=2, max_n=3, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[
            [9, 1, 2, 3, 8, 1, 2, 3],  # 3-gram (1,2,3) matches at idx 1 → [8, 1]
            [7, 1, 2, 9, 1, 2, 0, 0],  # 2-gram (1,2) matches at idx 1 → [9, 1]
        ],
        seq_lens=[8, 6],
    )
    assert num_valid.tolist() == [2, 2]
    assert drafts.tolist()[0] == [8, 1]
    assert drafts.tolist()[1] == [9, 1]


def test_kernel_min_n_eq_1():
    """min_n=max_n=1 — single-token n-grams always match if context > 1."""
    kernel = _make_kernel(min_n=1, max_n=1, k=2)
    drafts, num_valid = _run_kernel(
        kernel,
        rows=[[1, 2, 3, 4, 1]],  # suffix (1,) matches at idx 0 → tokens after = [2, 3]
        seq_lens=[5],
    )
    assert num_valid.tolist() == [2]
    assert drafts.tolist() == [[2, 3]]


# ---------------------------------------------------------------------------
# NgramGPUSpeculator.propose tests
# ---------------------------------------------------------------------------


class _FakeStaged:
    """Lightweight stand-in for ``StagedWriteTensor`` used by ``RequestState``.

    Only ``.gpu`` is consulted by ``NgramGPUSpeculator.propose``.
    """

    def __init__(self, tensor: torch.Tensor):
        self.gpu = tensor


class _FakeRequestState:
    """Minimal duck-typed ``RequestState`` exposing only the fields propose() reads."""

    def __init__(self, all_token_ids: torch.Tensor, total_len: torch.Tensor):
        self.all_token_ids = _FakeStaged(all_token_ids)
        self.total_len = _FakeStaged(total_len)


def _make_speculator(
    min_n: int,
    max_n: int,
    k: int,
    max_num_seqs: int = 4,
    max_model_len: int = 32,
) -> NgramGPUSpeculator:
    cfg = _make_vllm_config(
        min_n=min_n,
        max_n=max_n,
        k=k,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    with set_current_vllm_config(cfg):
        spec = NgramGPUSpeculator(vllm_config=cfg, device=torch.device("cuda"))
    return spec


def test_speculator_propose_basic_single_request():
    """propose() should return drafts plus per-request valid counts."""
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_num_seqs=4, max_model_len=16)

    # State for a single active request with tokens [1, 2, 3, 1, 2]; suffix (1,2)
    # matches at index 0 → expected draft tokens [3, 1].
    all_token_ids = torch.zeros((4, 16), dtype=torch.int32, device="cuda")
    all_token_ids[2, :5] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
    total_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    total_len[2] = 5

    spec.req_states = _FakeRequestState(all_token_ids, total_len)

    # The active batch contains a single request mapping to slot index 2.
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([2], dtype=torch.int32, device="cuda"),
    )
    last_sampled = torch.full((4, 1), 99, dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")

    drafts, num_valid = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device="cuda"),
        aux_hidden_states=None,
        num_sampled=num_sampled,
        num_rejected=torch.zeros(1, dtype=torch.int32, device="cuda"),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        temperature=torch.zeros(1, dtype=torch.float32, device="cuda"),
        seeds=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )

    assert drafts.shape == (1, 2)
    assert drafts.cpu().tolist() == [[3, 1]]
    assert num_valid is not None
    assert num_valid.cpu().tolist() == [2]


def test_speculator_propose_zero_sampled_disables_proposal():
    """When num_sampled==0 for a request, valid_mask is False → 0 valid drafts."""
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_num_seqs=4, max_model_len=16)

    all_token_ids = torch.zeros((4, 16), dtype=torch.int32, device="cuda")
    all_token_ids[0, :5] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
    total_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    total_len[0] = 5

    spec.req_states = _FakeRequestState(all_token_ids, total_len)

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    last_sampled = torch.full((4, 1), 7, dtype=torch.int64, device="cuda")
    # num_sampled=0 must disable speculation for this request.
    num_sampled = torch.tensor([0], dtype=torch.int32, device="cuda")

    _drafts, num_valid = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device="cuda"),
        aux_hidden_states=None,
        num_sampled=num_sampled,
        num_rejected=torch.zeros(1, dtype=torch.int32, device="cuda"),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        temperature=torch.zeros(1, dtype=torch.float32, device="cuda"),
        seeds=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )

    assert num_valid is not None
    assert num_valid.cpu().tolist() == [0]


def test_speculator_propose_multibatch_noncontiguous_idx_mapping():
    """Verify that propose() correctly reads via idx_mapping (non-contiguous)."""
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_num_seqs=4, max_model_len=16)

    all_token_ids = torch.zeros((4, 16), dtype=torch.int32, device="cuda")
    # Request A at slot 0: tokens [1,2,3,1,2] → drafts [3,1]
    all_token_ids[0, :5] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
    # Request B at slot 3: tokens [7,8,9,7,8] → drafts [9,7]
    all_token_ids[3, :5] = torch.tensor([7, 8, 9, 7, 8], dtype=torch.int32)

    total_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    total_len[0] = 5
    total_len[3] = 5

    spec.req_states = _FakeRequestState(all_token_ids, total_len)

    # Batch order is [slot 3, slot 0] — verify the idx_mapping is honored.
    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([3, 0], dtype=torch.int32, device="cuda"),
    )
    last_sampled = torch.zeros((4, 1), dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

    drafts, num_valid = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device="cuda"),
        aux_hidden_states=None,
        num_sampled=num_sampled,
        num_rejected=torch.zeros(2, dtype=torch.int32, device="cuda"),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.zeros(2, dtype=torch.int32, device="cuda"),
        temperature=torch.zeros(2, dtype=torch.float32, device="cuda"),
        seeds=torch.zeros(2, dtype=torch.int64, device="cuda"),
    )

    assert drafts.cpu().tolist() == [[9, 7], [3, 1]]
    assert num_valid is not None
    assert num_valid.cpu().tolist() == [2, 2]


def test_speculator_propose_returns_num_valid_matching_batch_size():
    """num_valid should be shaped [num_reqs], independent of max_num_seqs."""
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_num_seqs=4, max_model_len=16)

    all_token_ids = torch.zeros((4, 16), dtype=torch.int32, device="cuda")
    all_token_ids[0, :5] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
    total_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    total_len[0] = 5

    spec.req_states = _FakeRequestState(all_token_ids, total_len)

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    last_sampled = torch.zeros((4, 1), dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")

    drafts, num_valid = spec.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device="cuda"),
        aux_hidden_states=None,
        num_sampled=num_sampled,
        num_rejected=torch.zeros(1, dtype=torch.int32, device="cuda"),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        temperature=torch.zeros(1, dtype=torch.float32, device="cuda"),
        seeds=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )

    # Returned tensors match the active batch size, not max_num_seqs.
    assert drafts.shape == (1, 2)
    assert num_valid is not None
    assert num_valid.shape == (1,)
    assert num_valid.cpu().tolist() == [2]


def test_speculator_propose_requires_req_states():
    """propose() must assert that req_states has been injected by the model runner."""
    spec = _make_speculator(min_n=2, max_n=2, k=2, max_num_seqs=2, max_model_len=8)
    # req_states is None by default and should trigger an AssertionError.
    assert spec.req_states is None

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    last_sampled = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")

    with pytest.raises(AssertionError, match="req_states"):
        spec.propose(
            input_batch=input_batch,
            attn_metadata=None,
            slot_mappings=None,
            last_hidden_states=torch.empty(0, device="cuda"),
            aux_hidden_states=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros(1, dtype=torch.int32, device="cuda"),
            last_sampled=last_sampled,
            next_prefill_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
            temperature=torch.zeros(1, dtype=torch.float32, device="cuda"),
            seeds=torch.zeros(1, dtype=torch.int64, device="cuda"),
        )


def test_speculator_construction_validates_speculative_config():
    """NgramGPUSpeculator requires prompt_lookup_min and prompt_lookup_max to be set."""
    cfg = _make_vllm_config(min_n=2, max_n=3, k=2)
    with set_current_vllm_config(cfg):
        spec = NgramGPUSpeculator(vllm_config=cfg, device=torch.device("cuda"))
    assert spec.min_n == 2
    assert spec.max_n == 3
    assert spec.num_speculative_steps == 2
    # No-op hooks must not raise.
    spec.load_model(target_model=None)
    spec.set_attn()
    spec.capture_model()
