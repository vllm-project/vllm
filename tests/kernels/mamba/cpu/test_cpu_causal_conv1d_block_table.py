# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-table (mamba prefix caching) support in the CPU causal-conv1d ops.

With ``mamba_cache_mode=all`` the scheduler hands the conv ops a 2-D
per-sequence block table plus pointer tensors instead of the legacy 1-D
slot-per-sequence ``cache_indices``. These tests pin the wrapper-level
contract (mirroring the GPU kernel semantics documented in
``mamba_mixer2.conv_ssm_forward``):

  prefill: initial state read from ``table[seq, initial_state_idx[seq]]``,
           final state written to ``table[seq, block_idx_last_scheduled]``,
           and a snapshot written at every ``block_size_to_align`` boundary;
  decode:  state read via ``initial_state_idx``, written via
           ``block_idx_last_scheduled_token`` (migrating on transitions);
           a (batch, 1) table with no pointers degrades to the legacy layout.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn_cpu,
    causal_conv1d_update_cpu,
)

DIM = 16
KERNEL = 4
STATE_LEN = KERNEL - 1
BLOCK = 8


def _ref_conv(seq: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              initial_state: torch.Tensor) -> torch.Tensor:
    """Monolithic reference: silu(depthwise-causal-conv(seq)) given state."""
    full = torch.cat([initial_state, seq], dim=-1)
    out = F.conv1d(full.unsqueeze(0), weight.unsqueeze(1), bias, padding=0,
                   groups=DIM)[0, :, -seq.shape[-1]:]
    return F.silu(out)


def _make_inputs(seqlen: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(DIM, seqlen, generator=g)
    weight = torch.randn(DIM, KERNEL, generator=g) * 0.3
    bias = torch.randn(DIM, generator=g) * 0.1
    return x, weight, bias


def test_prefill_block_table_snapshots_and_final_state():
    seqlen = 3 * BLOCK + 5  # crosses three aligned boundaries, partial tail
    x, weight, bias = _make_inputs(seqlen)
    num_slots = 8
    conv_states = torch.zeros(num_slots, DIM, STATE_LEN)
    table = torch.tensor([[2, 4, 5, 7]], dtype=torch.int32)

    out = causal_conv1d_fn_cpu(
        x.clone(),
        weight,
        bias,
        conv_states,
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32),
        cache_indices=table,
        has_initial_state=torch.tensor([False]),
        activation="silu",
        block_idx_first_scheduled_token=torch.tensor([0], dtype=torch.int32),
        block_idx_last_scheduled_token=torch.tensor([3], dtype=torch.int32),
        initial_state_idx=torch.tensor([0], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        block_size_to_align=BLOCK,
    )

    zero_state = torch.zeros(DIM, STATE_LEN)
    ref_out = _ref_conv(x, weight, bias, zero_state)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)

    # Snapshot at boundary P holds the conv window ending at token P;
    # final (partial) state lives at the last scheduled block's slot.
    full = torch.cat([zero_state, x], dim=-1)
    for boundary_block, pos in ((0, BLOCK), (1, 2 * BLOCK), (2, 3 * BLOCK)):
        slot = int(table[0, boundary_block])
        torch.testing.assert_close(
            conv_states[slot], full[:, pos:pos + STATE_LEN],
            rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        conv_states[int(table[0, 3])], full[:, -STATE_LEN:],
        rtol=1e-6, atol=1e-6)


def test_prefill_resume_from_cached_block_matches_monolithic():
    """Warm resume: prefill the tail from a boundary snapshot; outputs must
    match the monolithic cold prefill exactly."""
    seqlen = 2 * BLOCK + 3
    x, weight, bias = _make_inputs(seqlen, seed=1)
    num_slots = 8
    zero_state = torch.zeros(DIM, STATE_LEN)

    # Cold: full prefill, capturing the 2*BLOCK boundary snapshot.
    cold_states = torch.zeros(num_slots, DIM, STATE_LEN)
    table = torch.tensor([[1, 3, 6]], dtype=torch.int32)
    cold_out = causal_conv1d_fn_cpu(
        x.clone(), weight, bias, cold_states,
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32),
        cache_indices=table,
        has_initial_state=torch.tensor([False]),
        activation="silu",
        block_idx_first_scheduled_token=torch.tensor([0], dtype=torch.int32),
        block_idx_last_scheduled_token=torch.tensor([2], dtype=torch.int32),
        initial_state_idx=torch.tensor([0], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        block_size_to_align=BLOCK,
    )

    # Warm: first 2*BLOCK tokens are a cache hit; resume the 3-token tail
    # reading the initial state from block 1's snapshot.
    warm_states = cold_states.clone()
    tail = x[:, 2 * BLOCK:]
    warm_out = causal_conv1d_fn_cpu(
        tail.clone(), weight, bias, warm_states,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        cache_indices=table,
        has_initial_state=torch.tensor([True]),
        activation="silu",
        block_idx_first_scheduled_token=torch.tensor([2], dtype=torch.int32),
        block_idx_last_scheduled_token=torch.tensor([2], dtype=torch.int32),
        initial_state_idx=torch.tensor([1], dtype=torch.int32),
        num_computed_tokens=torch.tensor([2 * BLOCK], dtype=torch.int32),
        block_size_to_align=BLOCK,
    )

    torch.testing.assert_close(warm_out, cold_out[:, 2 * BLOCK:],
                               rtol=1e-6, atol=1e-6)
    ref_out = _ref_conv(x, weight, bias, zero_state)
    torch.testing.assert_close(cold_out, ref_out, rtol=1e-5, atol=1e-5)


def test_decode_block_table_gather_and_migration():
    """Decode with a block table must read the current block's slot and
    migrate state when the write block differs from the read block."""
    _, weight, bias = _make_inputs(1, seed=2)
    num_slots = 8
    conv_state = torch.zeros(num_slots, DIM, STATE_LEN)
    running = torch.randn(DIM, STATE_LEN)
    conv_state[5] = running  # state lives in block idx 1 -> slot 5
    table = torch.tensor([[2, 5, 7]], dtype=torch.int32)
    xt = torch.randn(1, DIM, 1)

    ref = _ref_conv(xt[0], weight, bias, running.clone())

    out = causal_conv1d_update_cpu(
        xt.clone(), conv_state, weight, bias, "silu",
        conv_state_indices=table,
        block_idx_last_scheduled_token=torch.tensor([2], dtype=torch.int32),
        initial_state_idx=torch.tensor([1], dtype=torch.int32),
    )
    torch.testing.assert_close(out[0], ref, rtol=1e-5, atol=1e-5)
    # State migrated to the write block's slot and advanced by one token.
    expected_state = torch.cat([running, xt[0]], dim=-1)[:, -STATE_LEN:]
    torch.testing.assert_close(conv_state[7], expected_state,
                               rtol=1e-6, atol=1e-6)


def test_decode_width_one_table_without_pointers_is_legacy():
    """Without prefix caching each sequence owns one block: a (batch, 1)
    table and no pointer tensors must behave like 1-D cache_indices."""
    _, weight, bias = _make_inputs(1, seed=3)
    num_slots = 4
    running = torch.randn(DIM, STATE_LEN)
    xt = torch.randn(1, DIM, 1)

    state_2d = torch.zeros(num_slots, DIM, STATE_LEN)
    state_2d[3] = running
    out_2d = causal_conv1d_update_cpu(
        xt.clone(), state_2d, weight, bias, "silu",
        conv_state_indices=torch.tensor([[3]], dtype=torch.int32),
    )

    state_1d = torch.zeros(num_slots, DIM, STATE_LEN)
    state_1d[3] = running
    out_1d = causal_conv1d_update_cpu(
        xt.clone(), state_1d, weight, bias, "silu",
        conv_state_indices=torch.tensor([3], dtype=torch.int32),
    )

    torch.testing.assert_close(out_2d, out_1d, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_2d, state_1d, rtol=0.0, atol=0.0)
