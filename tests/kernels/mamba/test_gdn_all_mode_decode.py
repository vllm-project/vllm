# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level tests for GDN all-mode decode (dual read/write anchor).

All-mode decode must READ the running conv/SSM state from the last *computed*
block and WRITE to the last *scheduled* block. Within a mamba block the two
anchors are the same physical block, so in-block decode must be bit-identical
to align mode; exactly at a block-boundary crossing they differ and the state
must migrate into the fresh block (the runner does not copy mamba state across
blocks in all-mode). Covered paths:

* the packed non-spec decode fast path (``_forward_core_decode_non_spec``,
  in-place kernels -> carry-copy),
* the non-packed decode branches in ``_forward_core`` (kernel-side dual
  index: conv block args + K2 ``ssm_state_indices_output``),
* the peeled decode rows of a mixed decode+prefill batch.

Reuses the harness of test_gdn_all_mode_prefill (real ``_forward_core`` +
real ``GDNAttentionMetadataBuilder`` metadata on GPU).
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN all-mode decode tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.kernels.mamba import test_gdn_all_mode_prefill as harness  # noqa: E402
from tests.v1.attention.utils import BatchSpec  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    QwenGatedDeltaNetAttention,
)

BLOCK = 64


def _build_decode_layer(cfg, conv_state, ssm_state, weights, packed):
    layer = harness._build_layer(cfg, BLOCK, conv_state, ssm_state, weights)
    layer.enable_packed_recurrent_decode = packed
    layer._forward_core_decode_non_spec = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_decode_non_spec, layer
    )
    return layer


def _run_mode(mode, batch, inputs, packed, seed_blocks, weights):
    """Run one decode batch in `mode`; seed_blocks maps (seq, block_idx) in
    the block table -> (conv, ssm) source states to pre-place."""
    device = torch.device("cuda")
    mixed_qkv, b, a = inputs
    num_tokens = mixed_qkv.shape[0]
    cfg = harness._make_vllm_config(BLOCK, mode)
    meta, common = harness._build_metadata(cfg, batch, BLOCK, device)
    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv_state, ssm_state = harness._make_pools(pool_size, torch.float32, device)
    for (seq, block_idx), (conv_src, ssm_src) in seed_blocks.items():
        blk = int(common.block_table_tensor[seq, block_idx].item())
        conv_state[blk] = conv_src
        ssm_state[blk] = ssm_src
    layer = _build_decode_layer(cfg, conv_state, ssm_state, weights, packed)
    out = harness._run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens)
    return out, conv_state, ssm_state, common.block_table_tensor


@pytest.mark.parametrize("packed", [True, False])
def test_decode_in_block_matches_align(packed):
    """Within a mamba block, read == write anchor: all-mode decode must be
    bit-identical to align mode (same seeded state, same physical block)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    inputs = harness._make_inputs(1, device)
    # seq 101 with 1 new token: ncomp=100 -> mid-block of block 1 for both
    # modes (align's sliced table also lands on block 1).
    batch = BatchSpec(seq_lens=[101], query_lens=[1])
    conv_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[0][0])
    ssm_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[1][0])

    out_all, conv_all, ssm_all, table = _run_mode(
        "all", batch, inputs, packed, {(0, 1): (conv_s, ssm_s)}, weights
    )
    out_align, conv_align, ssm_align, _ = _run_mode(
        "align", batch, inputs, packed, {(0, 1): (conv_s, ssm_s)}, weights
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
    blk = int(table[0, 1].item())
    torch.testing.assert_close(ssm_all[blk], ssm_align[blk], atol=0, rtol=0)
    torch.testing.assert_close(conv_all[blk], conv_align[blk], atol=0, rtol=0)


@pytest.mark.parametrize("packed", [True, False])
def test_decode_boundary_crossing_carries_state(packed):
    """At a block-boundary crossing (ncomp == 2*BLOCK), all-mode must read
    the running state from the previous block and write the updated state
    into the fresh block — matching an align run whose state already sits in
    the current block."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    inputs = harness._make_inputs(1, device)
    # seq 129 with 1 new token: ncomp=128=2*BLOCK. Read anchor = block 1
    # (state after 128 tokens), write anchor = block 2 (fresh).
    batch = BatchSpec(seq_lens=[129], query_lens=[1])
    conv_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[0][0])
    ssm_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[1][0])

    # all-mode: running state lives in block 1; block 2 is empty.
    out_all, conv_all, ssm_all, table = _run_mode(
        "all", batch, inputs, packed, {(0, 1): (conv_s, ssm_s)}, weights
    )
    # align reference: the runner has already placed the state in block 2.
    out_align, conv_align, ssm_align, _ = _run_mode(
        "align", batch, inputs, packed, {(0, 2): (conv_s, ssm_s)}, weights
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
    # The updated running state must land in block 2 in both runs.
    blk2 = int(table[0, 2].item())
    torch.testing.assert_close(ssm_all[blk2], ssm_align[blk2], atol=0, rtol=0)
    torch.testing.assert_close(conv_all[blk2], conv_align[blk2], atol=0, rtol=0)
    # And block 1 (the read anchor) must keep its checkpoint intact: it is
    # the content-addressed cache entry for tokens [64, 128).
    blk1 = int(table[0, 1].item())
    torch.testing.assert_close(ssm_all[blk1], ssm_s.to(ssm_all.dtype))


@pytest.mark.parametrize("packed", [True, False])
def test_prefill_then_decode_matches_single_prefill(packed):
    """Continuation parity across a boundary: all-mode prefill of 120 tokens
    followed by 16 single-token decode steps (crossing the 128 boundary) must
    reproduce the single-shot 136-token prefill: outputs and the final block
    state."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    total, prefill_len = 136, 120
    mixed_qkv, b, a = harness._make_inputs(total, device)
    cfg = harness._make_vllm_config(BLOCK, "all")

    # Reference: single-shot prefill.
    ref_batch = BatchSpec(seq_lens=[total], query_lens=[total])
    meta_ref, common_ref = harness._build_metadata(cfg, ref_batch, BLOCK, device)
    pool_size = int(common_ref.block_table_tensor.max().item()) + 1
    conv_ref, ssm_ref = harness._make_pools(pool_size, torch.float32, device)
    layer_ref = _build_decode_layer(cfg, conv_ref, ssm_ref, weights, packed)
    out_ref = harness._run_forward_core(layer_ref, meta_ref, mixed_qkv, b, a, total)

    # Prefill 120, then decode token-by-token on the same pools.
    conv_state, ssm_state = harness._make_pools(pool_size, torch.float32, device)
    layer = _build_decode_layer(cfg, conv_state, ssm_state, weights, packed)
    p_batch = BatchSpec(seq_lens=[prefill_len], query_lens=[prefill_len])
    meta_p, common_p = harness._build_metadata(cfg, p_batch, BLOCK, device)
    torch.testing.assert_close(
        common_p.block_table_tensor,
        common_ref.block_table_tensor[:, : common_p.block_table_tensor.shape[1]],
    )
    harness._run_forward_core(
        layer,
        meta_p,
        mixed_qkv[:prefill_len],
        b[:prefill_len],
        a[:prefill_len],
        prefill_len,
    )

    decode_outs = []
    for i in range(prefill_len, total):
        d_batch = BatchSpec(seq_lens=[i + 1], query_lens=[1])
        meta_d, _ = harness._build_metadata(cfg, d_batch, BLOCK, device)
        assert meta_d.num_decodes == 1
        out_d = harness._run_forward_core(
            layer, meta_d, mixed_qkv[i : i + 1], b[i : i + 1], a[i : i + 1], 1
        )
        decode_outs.append(out_d)

    out_decode = torch.cat(decode_outs, dim=0)
    # Recurrent decode vs chunk prefill accumulate differently; use the
    # kernel-parity tolerances of the forward-core split test.
    atol = rtol = 2e-2
    torch.testing.assert_close(out_decode, out_ref[prefill_len:], atol=atol, rtol=rtol)
    # Final running state (block 2, after 136 tokens) must match.
    blk2 = int(common_ref.block_table_tensor[0, 2].item())
    torch.testing.assert_close(ssm_state[blk2], ssm_ref[blk2], atol=atol, rtol=rtol)
    # Interior checkpoint for block 1 (boundary 128) was written by the
    # decode path at the crossing in the continuation run, by the prefill
    # scatter in the reference run.
    blk1 = int(common_ref.block_table_tensor[0, 1].item())
    torch.testing.assert_close(ssm_state[blk1], ssm_ref[blk1], atol=atol, rtol=rtol)


def test_mixed_decode_prefill_all_mode_matches_align():
    """The peeled decode rows of a mixed decode+prefill batch use the K2
    dual-index; in-block (read == write) the whole batch must match align
    bit-exactly (prefill part already proven by the prefill tests)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = harness._make_weights(device)
    # Row 0: in-block decode (seq 101, block 1); row 1: fresh 192-token
    # prefill (3 blocks).
    batch = BatchSpec(seq_lens=[101, 192], query_lens=[1, 192])
    inputs = harness._make_inputs(193, device)
    conv_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[0][0])
    ssm_s = torch.randn_like(harness._make_pools(1, torch.float32, device)[1][0])

    out_all, _, _, _ = _run_mode(
        "all", batch, inputs, False, {(0, 1): (conv_s, ssm_s)}, weights
    )
    out_align, _, _, _ = _run_mode(
        "align", batch, inputs, False, {(0, 1): (conv_s, ssm_s)}, weights
    )
    torch.testing.assert_close(out_all, out_align, atol=0, rtol=0)
