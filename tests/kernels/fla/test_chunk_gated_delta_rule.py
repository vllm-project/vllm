# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged-state I/O and mid-sequence checkpoints of the FLA chunk kernel.

The paged cache is a strided view whose rows sit inside wider pages, as in
vLLM's hybrid KV cache, so a contiguous copy anywhere on the way would silently
drop the in-place writes.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type
NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 2, 4, 128
ROW_SIZE = NUM_V_HEADS * HEAD_DIM * HEAD_DIM
NUM_SLOTS = 12


def _make_inputs(seq_lens: list[int]) -> tuple[torch.Tensor, ...]:
    num_tokens = sum(seq_lens)
    q, k = (
        F.normalize(
            torch.randn(1, num_tokens, NUM_K_HEADS, HEAD_DIM), dim=-1
        ).bfloat16()
        for _ in range(2)
    )
    v = torch.randn(1, num_tokens, NUM_V_HEADS, HEAD_DIM).bfloat16()
    g = F.logsigmoid(torch.randn(1, num_tokens, NUM_V_HEADS))
    beta = torch.sigmoid(torch.randn(1, num_tokens, NUM_V_HEADS))
    return q, k, v, g, beta


def _cu_seqlens(seq_lens: list[int]) -> torch.Tensor:
    return torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)


def _make_pages(dtype: torch.dtype) -> torch.Tensor:
    # Each page holds one state row followed by padding, like a Mamba page
    # padded to the attention page size.
    return (0.1 * torch.randn(NUM_SLOTS, ROW_SIZE + 96)).to(dtype)


def _state_rows(pages: torch.Tensor) -> torch.Tensor:
    return pages[:, :ROW_SIZE].view(NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM)


def _run(inputs: tuple[torch.Tensor, ...], seq_lens: list[int], **kwargs):
    q, k, v, g, beta = inputs
    return chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=_cu_seqlens(seq_lens), **kwargs
    )


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Exact: the variants compute every element the same way, whatever
    # autotune config each picks.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_state_cache_matches_gathered_states(
    state_dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-place paged state I/O equals gathering and scattering the rows.

    The sequences end mid-chunk, on a chunk boundary and after one token. One
    starts fresh from a row holding stale values; row 0 (NULL_BLOCK_ID) is
    neither read nor written, so that sequence also starts from zero. The rows
    come as a strided block-table column. The
    in-place launch is autotuned afresh, which replays it once per config and
    benchmark repetition: each replay must start from the same rows. vLLM
    caches autotuning on disk, so without a fresh in-process cache only a cold
    disk cache would replay.
    """
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    seq_lens = [200, 64, 1, 130]
    # A block-table column, as the GDN metadata builder passes it.
    block_table = torch.tensor([[5, 3], [2, 3], [0, 3], [9, 3]], dtype=torch.int32)
    state_indices = block_table[:, 0]
    has_initial_state = torch.tensor([True, False, True, True])
    inputs = _make_inputs(seq_lens)
    pages = _make_pages(state_dtype)

    initial_state = _state_rows(pages)[state_indices.long()].float()
    initial_state[~has_initial_state | (state_indices <= 0)] = 0
    o_ref, final_ref = _run(
        inputs, seq_lens, initial_state=initial_state, output_final_state=True
    )
    written = state_indices > 0
    expected_pages = pages.clone()
    _state_rows(expected_pages)[state_indices[written].long()] = final_ref[written].to(
        state_dtype
    )

    autotuner = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    monkeypatch.setattr(autotuner, "cache", {})
    monkeypatch.setattr(autotuner, "cache_results", False)
    o, final_state = _run(
        inputs,
        seq_lens,
        state_cache=_state_rows(pages),
        state_indices=state_indices,
        has_initial_state=has_initial_state,
    )

    assert final_state is None
    _assert_equal(o, o_ref)
    # Whole pages: the written rows, and everything else (padding included) left
    # alone.
    _assert_equal(pages, expected_pages)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("paged", [True, False])
def test_checkpoints_are_states_at_their_offsets(
    state_dtype: torch.dtype, paged: bool
) -> None:
    """Each snapshot is the final state of its sequence cut at the offset.

    Paged: the snapshots land in rows of the paged state cache the sequences
    also read and write, as an align-mode prefix-cache checkpoint does.
    Otherwise they fill a staging buffer indexed by sequence. Only offsets on
    the 64-token chunk grid strictly inside a sequence, with a destination row
    > 0, are stored: 0 means none, 128 is the whole 128-token sequence, 100 is
    off the grid, and row 0 is NULL_BLOCK_ID. Snapshots leave the outputs as
    they are.
    """
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    seq_lens = [300, 50, 257, 128, 200, 192]
    offsets = [128, 0, 256, 128, 100, 64]
    checkpoint_rows = [7, 8, 3, 10, 11, 0]
    stored = [0, 2] if paged else [0, 2, 5]
    state_indices = torch.tensor([1, 4, 6, 2, 5, 9], dtype=torch.int32)
    has_initial_state = torch.tensor([True, True, False, True, True, True])
    inputs = _make_inputs(seq_lens)
    pages = _make_pages(state_dtype)
    cache = _state_rows(pages)
    initial_state = cache[state_indices.long()].float()
    initial_state[~has_initial_state] = 0
    checkpoint_offsets = torch.tensor(offsets, dtype=torch.int32)

    o_ref, _ = _run(
        inputs, seq_lens, initial_state=initial_state, output_final_state=True
    )
    if paged:
        before = cache.clone()
        o, _ = _run(
            inputs,
            seq_lens,
            state_cache=cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            checkpoint_state=cache,
            checkpoint_offsets=checkpoint_offsets,
            checkpoint_state_indices=torch.tensor(checkpoint_rows, dtype=torch.int32),
        )
        snapshots = cache[checkpoint_rows]
        skipped = [row for i, row in enumerate(checkpoint_rows) if i not in stored]
        assert torch.equal(cache[skipped], before[skipped])
    else:
        snapshots = torch.full(
            (len(seq_lens), NUM_V_HEADS, HEAD_DIM, HEAD_DIM), 7.0, dtype=state_dtype
        )
        o, _ = _run(
            inputs,
            seq_lens,
            initial_state=initial_state,
            output_final_state=True,
            checkpoint_state=snapshots,
            checkpoint_offsets=checkpoint_offsets,
        )
        skipped = [i for i in range(len(seq_lens)) if i not in stored]
        assert (snapshots[skipped] == 7.0).all()

    _assert_equal(o, o_ref)
    starts = _cu_seqlens(seq_lens).tolist()
    for i in stored:
        start, offset = starts[i], offsets[i]
        _, expected = _run(
            tuple(x[:, start : start + offset] for x in inputs),
            [offset],
            initial_state=initial_state[i : i + 1],
            output_final_state=True,
        )
        _assert_equal(snapshots[i], expected[0].to(state_dtype))


def test_offset_checkpoint_metadata_reuses_warmed_kernel(monkeypatch):
    """Slicing prefill metadata must reuse the kernel warmed on dense buffers."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    inputs = _make_inputs([257])
    pages = _make_pages(torch.float32)
    original = pages.clone()
    cache = _state_rows(pages)
    metadata = dict(
        cu_seqlens=_cu_seqlens([257]),
        state_indices=torch.tensor([1], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        checkpoint_offsets=torch.tensor([128], dtype=torch.int32),
        checkpoint_state_indices=torch.tensor([2], dtype=torch.int32),
    )

    def run(metadata):
        return chunk_gated_delta_rule(
            *inputs, state_cache=cache, checkpoint_state=cache, **metadata
        )[0]

    expected = run(metadata)
    expected_pages = pages.clone()
    pages.copy_(original)
    for name, tensor in metadata.items():
        view = tensor.new_empty(tensor.numel() + 1)[1:]
        view.copy_(tensor)
        metadata[name] = view

    compiled = []

    def record_compile(**kwargs):
        compiled.append(getattr(kwargs.get("fn"), "name", None))

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", record_compile)
    actual = run(metadata)
    _assert_equal(actual, expected)
    _assert_equal(pages, expected_pages)
    assert "chunk_gated_delta_rule_fwd_kernel_h_blockdim64" not in compiled
