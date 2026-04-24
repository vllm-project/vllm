# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AuxPoolState (worker-side prompt mean-pool state)."""

import pytest
import torch

from vllm.v1.worker.aux_pool_state import AuxPoolState


@pytest.fixture
def device():
    return torch.device(
        "cuda"
        if hasattr(torch, "accelerator") and torch.accelerator.is_available()
        else "cpu"
    )


@pytest.fixture
def state(device):
    s = AuxPoolState(block_size=4, hidden_size=8, device=device)
    s.enable(num_blocks=16)
    return s


def _hidden(num_tokens: int, hidden_size: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(num_tokens, hidden_size, device=device, generator=g)


def test_single_step_full_prompt(state, device):
    """A request whose entire prompt is computed in one step gets the
    correct mean."""
    prompt_len = 8  # 2 full blocks at block_size=4
    h = _hidden(prompt_len, 8, device)
    slot_mapping = torch.arange(prompt_len, dtype=torch.int64, device=device)

    state.init_request(
        req_id="r0",
        prompt_len=prompt_len,
        cached_block_ids=[],
        num_computed_tokens=0,
    )
    state.update_block_sums(slot_mapping, h)
    pooled = state.update_request("r0", h)

    assert pooled is not None
    expected = h.float().mean(dim=0)
    torch.testing.assert_close(pooled, expected, atol=1e-5, rtol=1e-5)


def test_chunked_prefill_across_steps(state, device):
    """Same prompt arriving over multiple steps gives the same final
    mean as a single-step run."""
    prompt_len = 12  # 3 full blocks
    h = _hidden(prompt_len, 8, device, seed=7)
    slot_mapping = torch.arange(prompt_len, dtype=torch.int64, device=device)

    state.init_request(
        req_id="r0",
        prompt_len=prompt_len,
        cached_block_ids=[],
        num_computed_tokens=0,
    )
    # Step 1: tokens [0..5)
    state.update_block_sums(slot_mapping[:5], h[:5])
    out1 = state.update_request("r0", h[:5])
    assert out1 is None
    # Step 2: tokens [5..9)
    state.update_block_sums(slot_mapping[5:9], h[5:9])
    out2 = state.update_request("r0", h[5:9])
    assert out2 is None
    # Step 3: tokens [9..12)
    state.update_block_sums(slot_mapping[9:], h[9:])
    out3 = state.update_request("r0", h[9:])

    assert out3 is not None
    expected = h.float().mean(dim=0)
    torch.testing.assert_close(out3, expected, atol=1e-5, rtol=1e-5)


def test_prefix_cache_reuse_matches_cold(state, device):
    """Two requests share the first 8 prompt tokens. Request A computes
    them; request B cache-hits those blocks. B's pooled vector must
    equal a fresh mean over its full prompt."""
    block_size = 4
    a_prompt_len = 8  # 2 full blocks
    b_prompt_len = 12  # shares 2 blocks with A, plus 4 new tokens

    h_shared = _hidden(8, 8, device, seed=11)
    h_b_extra = _hidden(4, 8, device, seed=12)
    h_b_full = torch.cat([h_shared, h_b_extra], dim=0)

    # Request A: blocks 5 and 6 (arbitrary).
    a_slot = torch.tensor(
        [5 * block_size + i for i in range(block_size)]
        + [6 * block_size + i for i in range(block_size)],
        dtype=torch.int64,
        device=device,
    )
    state.init_request("rA", a_prompt_len, cached_block_ids=[], num_computed_tokens=0)
    state.update_block_sums(a_slot, h_shared)
    pooled_a = state.update_request("rA", h_shared)
    assert pooled_a is not None
    state.cleanup("rA")

    # Request B: prefix-cache hit on blocks [5, 6]; new block 9 for the
    # last 4 tokens.
    b_slot_new = torch.tensor(
        [9 * block_size + i for i in range(block_size)],
        dtype=torch.int64,
        device=device,
    )
    state.init_request(
        "rB",
        b_prompt_len,
        cached_block_ids=[5, 6, 9],  # last is the new (uncached) block
        num_computed_tokens=8,  # 2 blocks * block_size
    )
    state.update_block_sums(b_slot_new, h_b_extra)
    pooled_b = state.update_request("rB", h_b_extra)
    assert pooled_b is not None
    expected = h_b_full.float().mean(dim=0)
    torch.testing.assert_close(pooled_b, expected, atol=1e-5, rtol=1e-5)


def test_block_reallocation_zeros_aux(state, device):
    """A block_id reused for a new tenancy must not carry stale aux from
    its previous tenant — slot 0 write zeros it."""
    block_size = 4
    block_id = 3
    slot = torch.tensor(
        [block_id * block_size + i for i in range(block_size)],
        dtype=torch.int64,
        device=device,
    )

    # First tenant.
    h_old = _hidden(block_size, 8, device, seed=21)
    state.update_block_sums(slot, h_old)

    # Second tenant: same block_id, different content. The slot==0 write
    # must zero aux first so we don't double-count h_old.
    h_new = _hidden(block_size, 8, device, seed=22)
    state.update_block_sums(slot, h_new)

    # Now a new opted-in request with a cache-hit on this block must see
    # only h_new's contribution.
    state.init_request(
        "rC",
        prompt_len=block_size,
        cached_block_ids=[block_id],
        num_computed_tokens=block_size,
    )
    pooled = state.update_request("rC", torch.zeros(0, 8, device=device))
    assert pooled is not None
    torch.testing.assert_close(pooled, h_new.float().mean(dim=0), atol=1e-5, rtol=1e-5)


def test_padding_slots_masked(state, device):
    """slot_mapping == -1 (padding) must not corrupt aux."""
    h = _hidden(6, 8, device, seed=33)
    slot = torch.tensor([0, 1, 2, 3, -1, -1], dtype=torch.int64, device=device)
    state.update_block_sums(slot, h)

    state.init_request(
        "rE",
        prompt_len=4,
        cached_block_ids=[0],
        num_computed_tokens=4,
    )
    pooled = state.update_request("rE", torch.zeros(0, 8, device=device))
    assert pooled is not None
    expected = h[:4].float().mean(dim=0)
    torch.testing.assert_close(pooled, expected, atol=1e-5, rtol=1e-5)


def test_cleanup_removes_state(state, device):
    state.init_request("rF", 4, [], 0)
    assert state.has_request("rF")
    state.cleanup("rF")
    assert not state.has_request("rF")
