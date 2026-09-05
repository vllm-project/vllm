# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level tests for GDN all-mode speculative decode (dual anchor).

All-mode spec decode gathers per-spec-token state slots from the full block
table: READ slots at the previous step's last-scheduled anchor + [0..num_spec]
(where that step wrote its per-token states; the kernel picks the resume slot
via num_accepted_tokens), WRITE slots at the current last-scheduled anchor +
[0..num_spec]. The spec conv reads its initial state from the last computed
block and writes to the last scheduled block in-kernel.

Align mode's sliced spec table addresses the same physical blocks whenever the
anchors are in-block, so all-vs-align must be bit-identical there; at a
boundary crossing the read window shifts to the previous anchor and the two
modes are compared with equivalently seeded state sequences (the slot the
kernel selects is the same in both).

Reuses the harness of test_gdn_all_mode_prefill (real ``_forward_core`` +
real ``GDNAttentionMetadataBuilder`` metadata on GPU).
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN all-mode spec tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.kernels.mamba import test_gdn_all_mode_prefill as harness  # noqa: E402
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

BLOCK = 64
NUM_SPEC = 2


def _make_spec_config(mode):
    cfg = harness._make_vllm_config(BLOCK, mode)
    cfg.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return cfg


def _build_spec_metadata(cfg, batch, device, drafts, accepted, prev=None):
    """Build spec metadata on a table widened by the speculative blocks
    (real tables carry num_speculative_blocks extra columns; the synthetic
    arange table must too, or the anchor+offset gathers go out of range)."""
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, device, arange_block_indices=True
    )
    table = common.block_table_tensor
    n, w = table.shape
    extra = torch.arange(n * NUM_SPEC, dtype=table.dtype, device=table.device).reshape(
        n, NUM_SPEC
    ) + int(table.max().item() + 1)
    common.block_table_tensor = torch.cat([table, extra], dim=1)
    # Shift past the reserved NULL block (id 0), as in the base harness.
    common.block_table_tensor.add_(1)
    kwargs = {
        "num_decode_draft_tokens_cpu": torch.tensor(drafts, dtype=torch.int32),
        "num_accepted_tokens": torch.tensor(accepted, dtype=torch.int32, device=device),
    }
    if prev is not None:
        kwargs["prev_last_scheduled_idx"] = torch.tensor(
            prev, dtype=torch.int32, device=device
        )
    with set_current_vllm_config(cfg):
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)
    return meta, common


def _build_spec_layer(cfg, conv_state, ssm_state, weights):
    layer = harness._build_layer(cfg, BLOCK, conv_state, ssm_state, weights)
    layer.num_spec = NUM_SPEC
    layer._decode_state_offsets = torch.arange(
        1 + NUM_SPEC, dtype=torch.int64, device=ssm_state.device
    ).unsqueeze(0)
    return layer


def _run_spec(mode, batch, inputs, weights, drafts, accepted, seed_blocks, prev=None):
    """seed_blocks: (seq_row, table_col) -> (conv, ssm) states to pre-place."""
    device = torch.device("cuda")
    mixed_qkv, b, a = inputs
    num_tokens = mixed_qkv.shape[0]
    cfg = _make_spec_config(mode)
    meta, common = _build_spec_metadata(cfg, batch, device, drafts, accepted, prev)
    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv_state, ssm_state = harness._make_pools(pool_size, torch.float32, device)
    for (seq, col), (conv_src, ssm_src) in seed_blocks.items():
        blk = int(common.block_table_tensor[seq, col].item())
        if conv_src is not None:
            conv_state[blk] = conv_src
        if ssm_src is not None:
            ssm_state[blk] = ssm_src
    layer = _build_spec_layer(cfg, conv_state, ssm_state, weights)
    out = harness._run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens)
    return out, conv_state, ssm_state, common.block_table_tensor


def _rand_states(device, n):
    conv0 = harness._make_pools(1, torch.float32, device)
    return [
        (torch.randn_like(conv0[0][0]), torch.randn_like(conv0[1][0])) for _ in range(n)
    ]


def test_spec_in_block_matches_align():
    """First spec step inside a block: read anchor == write anchor == align's
    sliced table; all-vs-align must be bit-identical (outputs and the written
    per-token states)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    # seq 103, 3 spec tokens: ncomp=100 -> anchors all in block 1; slots
    # [1, 2, 3] in both modes.
    batch = BatchSpec(seq_lens=[103], query_lens=[3])
    inputs = harness._make_inputs(3, device)
    (conv_c, _), *states = _rand_states(device, 4)
    seeds = {
        (0, 1): (conv_c, states[0][1]),
        (0, 2): (None, states[1][1]),
        (0, 3): (None, states[2][1]),
    }

    out_all, _, ssm_all, table = _run_spec(
        "all", batch, inputs, weights, [NUM_SPEC], [2], seeds
    )
    out_align, _, ssm_align, _ = _run_spec(
        "align", batch, inputs, weights, [NUM_SPEC], [2], seeds
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
    for col in (1, 2, 3):
        blk = int(table[0, col].item())
        torch.testing.assert_close(ssm_all[blk], ssm_align[blk], atol=0, rtol=0)


def test_spec_crossing_reads_prev_anchor():
    """Spec step just after a block-boundary crossing: the read window sits at
    the PREVIOUS step's anchor (plumbed via prev_last_scheduled_idx), the
    write window at the current anchor. Equivalently-seeded align run (same
    state sequence at its own read window) must match bit-exactly."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    # ncomp=129 (crossed 128), 3 spec tokens -> seq 132. Current anchor =
    # block 2 (write slots [2, 3, 4]); previous step's anchor = 1 (read slots
    # [1, 2, 3]). Align reads and writes at its sliced window [2, 3, 4].
    batch = BatchSpec(seq_lens=[132], query_lens=[3])
    inputs = harness._make_inputs(3, device)
    (conv_c, _), *states = _rand_states(device, 4)
    s0, s1, s2 = (s[1] for s in states)

    # all-mode: prev-step states at cols [1, 2, 3]; conv state at the last
    # computed block (col 2).
    out_all, _, ssm_all, table = _run_spec(
        "all",
        batch,
        inputs,
        weights,
        [NUM_SPEC],
        [2],
        {(0, 1): (None, s0), (0, 2): (conv_c, s1), (0, 3): (None, s2)},
        prev=[1],
    )
    # align: the same state sequence at its read window [2, 3, 4]; conv state
    # in its current block (col 2).
    out_align, _, ssm_align, _ = _run_spec(
        "align",
        batch,
        inputs,
        weights,
        [NUM_SPEC],
        [2],
        {(0, 2): (conv_c, s0), (0, 3): (None, s1), (0, 4): (None, s2)},
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
    # Both modes write the new per-token states at cols [2, 3, 4].
    for col in (2, 3, 4):
        blk = int(table[0, col].item())
        torch.testing.assert_close(ssm_all[blk], ssm_align[blk], atol=0, rtol=0)
    # The all-mode read-window head (col 1) keeps its checkpoint: it is the
    # APC entry for its token range.
    blk1 = int(table[0, 1].item())
    torch.testing.assert_close(ssm_all[blk1], s0.to(ssm_all.dtype))


def test_spec_pure_batch_uses_request_level_tensors():
    """Pure-spec batch (the cudagraph-captured shape) takes the full
    request-level tensor path — two rows, both in-block, must match align
    bit-exactly."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    # Rows: seq 103 (anchors block 1) and seq 231 (anchors block 3).
    batch = BatchSpec(seq_lens=[103, 231], query_lens=[3, 3])
    inputs = harness._make_inputs(6, device)
    st = _rand_states(device, 8)
    seeds = {}
    for i, col in enumerate((1, 2, 3)):
        seeds[(0, col)] = (st[i][0] if col == 1 else None, st[i][1])
    for i, col in enumerate((3, 4, 5)):
        seeds[(1, col)] = (st[4 + i][0] if col == 3 else None, st[4 + i][1])

    out_all, _, ssm_all, table = _run_spec(
        "all", batch, inputs, weights, [NUM_SPEC, NUM_SPEC], [2, 1], seeds
    )
    out_align, _, ssm_align, _ = _run_spec(
        "align", batch, inputs, weights, [NUM_SPEC, NUM_SPEC], [2, 1], seeds
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
    for seq, cols in ((0, (1, 2, 3)), (1, (3, 4, 5))):
        for col in cols:
            blk = int(table[seq, col].item())
            torch.testing.assert_close(ssm_all[blk], ssm_align[blk], atol=0, rtol=0)


def test_spec_mixed_with_prefill_matches_align():
    """Mixed spec + prefill batch (eager): the spec rows are boolean-selected
    from the request-level tensors; whole-batch outputs must match align
    bit-exactly (the prefill half is covered by the prefill tests)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    # Row 0: fresh 192-token prefill; row 1: in-block spec row (seq 103).
    batch = BatchSpec(seq_lens=[192, 103], query_lens=[192, 3])
    inputs = harness._make_inputs(195, device)
    (conv_c, _), *states = _rand_states(device, 4)
    seeds = {
        (1, 1): (conv_c, states[0][1]),
        (1, 2): (None, states[1][1]),
        (1, 3): (None, states[2][1]),
    }

    out_all, _, _, _ = _run_spec(
        "all", batch, inputs, weights, [-1, NUM_SPEC], [1, 2], seeds
    )
    out_align, _, _, _ = _run_spec(
        "align", batch, inputs, weights, [-1, NUM_SPEC], [1, 2], seeds
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
