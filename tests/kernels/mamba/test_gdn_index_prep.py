# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CS1 tests for the GDN all-mode block-index prep kernel + A3 table-walk
trims (design doc allmode_decode_opt_design.md, tests T1-T12).

- T1-T5: the fused prep kernel reproduces the eager anchor chain
  (``compute_mamba_prefix_caching_block_indices`` + the spec read-anchor
  fallback of gdn_attn.py) elementwise (torch.equal) across block sizes,
  boundary patterns, prev/fallback resolution and padded rows.
- T6: ``GDNAttentionMetadataBuilder.build()`` with the shared prep buffers
  supplied is field-bit-identical to the eager build, and issues zero
  aten.floor_divide / aten.neg kernels (TorchDispatchMode counter); the
  layer-level wiring (packed anchors through the real ``_forward_core``)
  is exercised on decode and spec batches.
- T7-T11: packed (read, write) anchor single-load parity for the three decode
  kernels (sigmoid gating incl. the hoisted write-anchor loop, conv update
  incl. the skip-second-lookup branch, packed recurrent decode), bit-exact
  against the separate-anchor path (which is code-identical to the pre-change
  kernels, constexpr-pruned) and against the host-gathered index oracle;
  NULL rows skipped.
- T12: wrapper assertion tests (mutual exclusivity, shape/dtype/contiguity).
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN index-prep tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.kernels.mamba import test_gdn_all_mode_prefill as harness  # noqa: E402
from tests.kernels.mamba import (  # noqa: E402
    test_gdn_all_mode_spec_decode as spec_harness,
)
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gdn_index_prep import (  # noqa: E402
    GDNBlockIdxPrepBuffers,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba_attn import (  # noqa: E402
    compute_mamba_prefix_caching_block_indices,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# T1-T5: prep kernel == eager anchor chain
# --------------------------------------------------------------------------


def _eager_chain(num_computed, seq_lens, prev, block_size):
    """The exact eager formulas the prep kernel replaces
    (mamba_attn.py:91-101 + gdn_attn.py:497-506)."""
    lc, fs, ls = compute_mamba_prefix_caching_block_indices(
        num_computed, seq_lens, block_size
    )
    fallback = torch.clamp((num_computed - 1) // block_size, min=0)
    ra = torch.where(prev >= 0, prev, fallback) if prev is not None else fallback
    return lc, fs, ls, ra


def _run_prep(seq_lens, num_computed, prev, block_size):
    bufs = GDNBlockIdxPrepBuffers(seq_lens.shape[0], seq_lens.device)
    bufs.prepare(seq_lens, num_computed, prev, block_size, seq_lens.shape[0])
    return bufs


def _assert_prep_matches(num_computed, seq_lens, prev, block_size):
    lc, fs, ls, ra = _eager_chain(num_computed, seq_lens, prev, block_size)
    bufs = _run_prep(
        seq_lens,
        num_computed,
        prev if prev is not None else torch.full_like(num_computed, -1),
        block_size,
    )
    n = seq_lens.shape[0]
    assert torch.equal(bufs.block_idx_last_computed_token[:n], lc)
    assert torch.equal(bufs.block_idx_first_scheduled_token[:n], fs)
    assert torch.equal(bufs.block_idx_last_scheduled_token[:n], ls)
    assert torch.equal(bufs.block_idx_last_scheduled_token_prev_step[:n], ra)
    # Packed pairs mirror the scalar outputs exactly.
    assert torch.equal(bufs.packed_anchors[:n, 0], lc)
    assert torch.equal(bufs.packed_anchors[:n, 1], ls)
    assert torch.equal(bufs.packed_anchors_spec[:n, 0], ra)
    assert torch.equal(bufs.packed_anchors_spec[:n, 1], ls)
    return bufs


@pytest.mark.parametrize("block_size", [2, 4, 64, 576, 1152])
@pytest.mark.parametrize("num_reqs", [1, 2, 17, 64, 257])
def test_prep_matches_eager_dense_grid(block_size, num_reqs):
    """T1: elementwise formula fidelity (cdiv + clamp placement, unclamped
    first_scheduled) over boundary-straddling num_computed patterns, T=1/4."""
    torch.manual_seed(0)
    b = block_size
    pattern = [0, 1, b - 1, b, b + 1, 3 * b - 1, 3 * b, 3 * b + 1, 100 * b + 7]
    nct = torch.tensor(
        [pattern[i % len(pattern)] for i in range(num_reqs)],
        dtype=torch.int32,
        device=DEVICE,
    )
    prev = torch.tensor(
        [-1 if i % 3 == 0 else i % 5 for i in range(num_reqs)],
        dtype=torch.int32,
        device=DEVICE,
    )
    for t in (1, 4):
        seq_lens = nct + t
        _assert_prep_matches(nct, seq_lens, prev, block_size)


@pytest.mark.parametrize(
    "block_size,nct_vals,expected",
    [
        # b=2, T=4: windows advancing the write anchor by 1 and 2 blocks.
        (2, [0, 1, 2, 5], [1, 2, 2, 2]),
        # b=4, T=4: 0-, and 1-block windows (incl. exactly-aligned starts).
        (4, [0, 1, 4, 6], [0, 1, 1, 1]),
    ],
)
def test_prep_multi_crossing_window(block_size, nct_vals, expected):
    """T2: T=4 windows spanning 0, 1 and 2 block boundaries; the
    last_scheduled - last_computed delta equals the crossing count the
    decode kernels consume."""
    t = 4
    nct = torch.tensor(nct_vals, dtype=torch.int32, device=DEVICE)
    seq_lens = nct + t
    prev = torch.full_like(nct, -1)
    bufs = _assert_prep_matches(nct, seq_lens, prev, block_size)
    n = nct.shape[0]
    crossings = bufs.block_idx_last_scheduled_token[
        :n
    ] - bufs.block_idx_last_computed_token[:n]
    assert crossings.tolist() == expected


def test_prep_first_block_edge():
    """T3: request birth (num_computed=0, prev=-1) clamps last_computed and
    the fallback to 0; seq_len=0 rows clamp last_scheduled to 0."""
    nct = torch.tensor([0, 0, 0], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([1, 4, 0], dtype=torch.int32, device=DEVICE)
    prev = torch.tensor([-1, -1, -1], dtype=torch.int32, device=DEVICE)
    bufs = _assert_prep_matches(nct, seq_lens, prev, 4)
    assert bufs.block_idx_last_computed_token[:3].tolist() == [0, 0, 0]
    assert bufs.block_idx_last_scheduled_token_prev_step[:3].tolist() == [0, 0, 0]
    assert bufs.block_idx_last_scheduled_token[:3].tolist() == [0, 0, 0]


def test_prep_spec_prev_fallback_resolution():
    """T4: where(prev >= 0, prev, clamp((nct-1)//B, 0)) reproduced for
    all-fallback, all-tracked and mixed prev vectors."""
    nct = torch.tensor([0, 5, 128, 129, 300], dtype=torch.int32, device=DEVICE)
    seq_lens = nct + 4
    for prev_vals in (
        [-1, -1, -1, -1, -1],
        [3, 0, 1, 2, 4],
        [-1, 2, -1, 0, -1],
    ):
        prev = torch.tensor(prev_vals, dtype=torch.int32, device=DEVICE)
        _assert_prep_matches(nct, seq_lens, prev, 64)


def test_prep_padded_tail_contract():
    """T5: padded rows (seq_len=0, query_len=0, prev=-1) produce exactly the
    fill values (0) of the pre-change staging contract."""
    nct = torch.tensor([100, 0, 0], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([104, 0, 0], dtype=torch.int32, device=DEVICE)
    prev = torch.tensor([1, -1, -1], dtype=torch.int32, device=DEVICE)
    bufs = _run_prep(seq_lens, nct, prev, 64)
    for buf in (
        bufs.block_idx_last_computed_token,
        bufs.block_idx_first_scheduled_token,
        bufs.block_idx_last_scheduled_token,
        bufs.block_idx_last_scheduled_token_prev_step,
    ):
        assert buf[1:3].tolist() == [0, 0]
    assert bufs.packed_anchors[1:3].flatten().tolist() == [0, 0, 0, 0]
    assert bufs.packed_anchors_spec[1:3].flatten().tolist() == [0, 0, 0, 0]


# --------------------------------------------------------------------------
# T6: builder integration parity + eager-kernel-count assertion
# --------------------------------------------------------------------------


class _OpCounter(TorchDispatchMode):
    """Counts CUDA-tensor aten ops (the eager GDN block-index math; CPU-side
    chunk-index prep of prefill builds is out of scope)."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in args):
            self.counts[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))


BLOCK = spec_harness.BLOCK
NUM_SPEC = spec_harness.NUM_SPEC


def _prep_from_common(common, prev, block_size):
    n = common.num_reqs
    bufs = GDNBlockIdxPrepBuffers(n, DEVICE)
    if prev is None:
        prev = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    bufs.prepare(
        common.seq_lens,
        common.compute_num_computed_tokens(),
        prev,
        block_size,
        n,
    )
    return bufs


def _meta_fields(meta):
    return {
        f: getattr(meta, f)
        for f in (
            "block_idx_last_scheduled_token",
            "block_idx_first_scheduled_token",
            "block_idx_last_computed_token",
            "block_idx_last_scheduled_token_prev_step",
            "num_computed_tokens",
            "all_state_indices_tensor",
            "spec_state_indices_tensor",
            "non_spec_state_indices_tensor",
            "num_accepted_tokens",
        )
    }


def _assert_meta_equal(meta_eager, meta_prep):
    for name, ref in _meta_fields(meta_eager).items():
        got = getattr(meta_prep, name)
        if ref is None:
            assert got is None, name
        else:
            assert got is not None, name
            assert torch.equal(got, ref), name


@pytest.mark.parametrize(
    "batch",
    [
        BatchSpec(seq_lens=[101, 231], query_lens=[1, 1]),  # pure decode
        BatchSpec(seq_lens=[320, 192], query_lens=[320, 192]),  # prefill-only
        BatchSpec(seq_lens=[101, 192], query_lens=[1, 192]),  # mixed
    ],
)
def test_builder_parity_non_spec(batch):
    """T6 (non-spec): flag-on build (prep buffers supplied) is bit-identical
    per field to the eager build, with zero floor_divide/neg eager kernels."""
    torch.manual_seed(0)
    cfg = harness._make_vllm_config(BLOCK, "all")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(cfg):
        meta_eager = builder.build(0, common)
        bufs = _prep_from_common(common, None, BLOCK)
        # Prime the num_computed cache the way the runner does (its subs are
        # issued once per step by the prep helper, not per build).
        common.compute_num_computed_tokens()
        with _OpCounter() as counter:
            meta_prep = builder.build(0, common, block_idx_prep=bufs)
    _assert_meta_equal(meta_eager, meta_prep)
    assert counter.counts["floor_divide"] == 0
    assert counter.counts["neg"] == 0
    assert meta_prep.block_idx_packed_anchors is not None
    assert torch.equal(
        meta_prep.block_idx_packed_anchors[:, 0],
        meta_eager.block_idx_last_computed_token,
    )
    assert torch.equal(
        meta_prep.block_idx_packed_anchors[:, 1],
        meta_eager.block_idx_last_scheduled_token,
    )


@pytest.mark.parametrize(
    "batch,drafts,accepted,prev",
    [
        (
            BatchSpec(seq_lens=[103, 231], query_lens=[3, 3]),
            [NUM_SPEC, NUM_SPEC],
            [2, 1],
            [1, -1],
        ),
        (
            BatchSpec(seq_lens=[192, 103], query_lens=[192, 3]),
            [-1, NUM_SPEC],
            [1, 2],
            [-1, 1],
        ),
    ],
)
def test_builder_parity_spec(batch, drafts, accepted, prev):
    """T6 (spec): pure-spec and mixed spec+prefill builds bit-identical with
    the prep buffers supplied, including the prev-step read anchor."""
    torch.manual_seed(0)
    cfg = spec_harness._make_spec_config("all")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    table = common.block_table_tensor
    n, _ = table.shape
    extra = torch.arange(
        n * NUM_SPEC, dtype=table.dtype, device=table.device
    ).reshape(n, NUM_SPEC) + int(table.max().item() + 1)
    common.block_table_tensor = torch.cat([table, extra], dim=1)
    common.block_table_tensor.add_(1)
    prev_t = torch.tensor(prev, dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        num_accepted_tokens=torch.tensor(
            accepted, dtype=torch.int32, device=DEVICE
        ),
        prev_last_scheduled_idx=prev_t,
    )
    with set_current_vllm_config(cfg):
        meta_eager = builder.build(0, common, **kwargs)
        bufs = _prep_from_common(common, prev_t, BLOCK)
        common.compute_num_computed_tokens()
        with _OpCounter() as counter:
            meta_prep = builder.build(0, common, block_idx_prep=bufs, **kwargs)
    _assert_meta_equal(meta_eager, meta_prep)
    assert counter.counts["floor_divide"] == 0
    assert counter.counts["neg"] == 0
    assert meta_prep.block_idx_packed_anchors_spec is not None
    assert torch.equal(
        meta_prep.block_idx_packed_anchors_spec[:, 0],
        meta_eager.block_idx_last_scheduled_token_prev_step,
    )


def _run_layer_with_meta(cfg, meta, common, inputs, weights, seed_blocks,
                         num_spec=0):
    mixed_qkv, b, a = inputs
    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv_state, ssm_state = harness._make_pools(pool_size, torch.float32, DEVICE)
    for (seq, col), (conv_src, ssm_src) in seed_blocks.items():
        blk = int(common.block_table_tensor[seq, col].item())
        if conv_src is not None:
            conv_state[blk] = conv_src
        if ssm_src is not None:
            ssm_state[blk] = ssm_src
    layer = harness._build_layer(cfg, BLOCK, conv_state, ssm_state, weights)
    if num_spec:
        layer.num_spec = num_spec
    out = harness._run_forward_core(
        layer, meta, mixed_qkv, b, a, mixed_qkv.shape[0]
    )
    return out, conv_state, ssm_state


def test_layer_decode_prep_metadata_parity():
    """T6 (layer wiring, decode): the real _forward_core on flag-on metadata
    (packed anchors populated) is bit-identical to the eager metadata run,
    across an in-block row and a boundary-crossing row."""
    torch.manual_seed(0)
    weights = harness._make_weights(DEVICE)
    inputs = harness._make_inputs(2, DEVICE)
    # Row 0 in-block (ncomp=100); row 1 crossing (ncomp=128=2*BLOCK).
    batch = BatchSpec(seq_lens=[101, 129], query_lens=[1, 1])
    cfg = harness._make_vllm_config(BLOCK, "all")
    builder_spec = MambaSpec(
        block_size=BLOCK, shapes=((16, 64),), dtypes=(torch.float16,)
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=builder_spec,
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(cfg):
        meta_eager = builder.build(0, common)
        bufs = _prep_from_common(common, None, BLOCK)
        meta_prep = builder.build(0, common, block_idx_prep=bufs)
    assert meta_prep.block_idx_packed_anchors is not None

    conv_s = torch.randn_like(harness._make_pools(1, torch.float32, DEVICE)[0][0])
    ssm_s = torch.randn_like(harness._make_pools(1, torch.float32, DEVICE)[1][0])
    seeds = {(0, 1): (conv_s, ssm_s), (1, 1): (conv_s, ssm_s)}
    out_e, conv_e, ssm_e = _run_layer_with_meta(
        cfg, meta_eager, common, inputs, weights, seeds
    )
    out_p, conv_p, ssm_p = _run_layer_with_meta(
        cfg, meta_prep, common, inputs, weights, seeds
    )
    assert torch.equal(out_p, out_e)
    assert torch.equal(conv_p, conv_e)
    assert torch.equal(ssm_p, ssm_e)


def test_layer_spec_prep_metadata_parity():
    """T6 (layer wiring, spec): pure-spec batch through the real
    _forward_core — flag-on metadata (packed spec anchors) bit-identical to
    the eager metadata run, including a prev-step read anchor crossing."""
    torch.manual_seed(0)
    weights = harness._make_weights(DEVICE)
    batch = BatchSpec(seq_lens=[132], query_lens=[3])
    inputs = harness._make_inputs(3, DEVICE)
    cfg = spec_harness._make_spec_config("all")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    table = common.block_table_tensor
    n, _ = table.shape
    extra = torch.arange(
        n * NUM_SPEC, dtype=table.dtype, device=table.device
    ).reshape(n, NUM_SPEC) + int(table.max().item() + 1)
    common.block_table_tensor = torch.cat([table, extra], dim=1)
    common.block_table_tensor.add_(1)
    prev_t = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        prev_last_scheduled_idx=prev_t,
    )
    with set_current_vllm_config(cfg):
        meta_eager = builder.build(0, common, **kwargs)
        bufs = _prep_from_common(common, prev_t, BLOCK)
        meta_prep = builder.build(0, common, block_idx_prep=bufs, **kwargs)
    assert meta_prep.block_idx_packed_anchors_spec is not None

    (conv_c, _), *states = spec_harness._rand_states(DEVICE, 4)
    seeds = {
        (0, 1): (None, states[0][1]),
        (0, 2): (conv_c, states[1][1]),
        (0, 3): (None, states[2][1]),
    }
    out_e, conv_e, ssm_e = _run_layer_with_meta(
        cfg, meta_eager, common, inputs, weights, seeds, num_spec=NUM_SPEC
    )
    out_p, conv_p, ssm_p = _run_layer_with_meta(
        cfg, meta_prep, common, inputs, weights, seeds, num_spec=NUM_SPEC
    )
    assert torch.equal(out_p, out_e)
    assert torch.equal(conv_p, conv_e)
    assert torch.equal(ssm_p, ssm_e)


# --------------------------------------------------------------------------
# T7-T11: packed-anchor kernel parity (A3 trims)
# --------------------------------------------------------------------------


def _gating_inputs(num_reqs, seq, h, hv, k, v, dtype, seed=0):
    torch.manual_seed(seed)
    num_tokens = num_reqs * seq
    q = torch.rand(1, num_tokens, h, k, dtype=dtype, device=DEVICE)
    key = torch.rand(1, num_tokens, h, k, dtype=dtype, device=DEVICE)
    val = torch.rand(1, num_tokens, hv, v, dtype=dtype, device=DEVICE)
    return dict(
        A_log=torch.rand(hv, dtype=dtype, device=DEVICE),
        dt_bias=torch.rand(hv, dtype=dtype, device=DEVICE),
        a=torch.rand(num_tokens, hv, dtype=dtype, device=DEVICE),
        b=torch.rand(num_tokens, hv, dtype=dtype, device=DEVICE),
        q=q,
        k=key,
        v=val,
        cu_seqlens=torch.arange(
            0, num_tokens + 1, seq, dtype=torch.int32, device=DEVICE
        ),
    )


def _anchored_table(num_reqs, width, max_anchor, null_row=None, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(4 * num_reqs * width - 1, generator=g).to(torch.int32) + 1
    table = perm[: num_reqs * width].view(num_reqs, width).contiguous().to(DEVICE)
    if null_row is not None:
        table[null_row] = 0
    read = torch.randint(
        0, max_anchor + 1, (num_reqs,), generator=g, dtype=torch.int32
    ).to(DEVICE)
    write = torch.randint(
        0, max_anchor + 1, (num_reqs,), generator=g, dtype=torch.int32
    ).to(DEVICE)
    packed = torch.stack([read, write], dim=1).contiguous()
    assert packed.shape == (num_reqs, 2) and packed.stride() == (2, 1)
    return table, read, write, packed


@pytest.mark.parametrize("geom", [(4, 8, 128, 128), (2, 16, 64, 64)])
@pytest.mark.parametrize("seq", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 2, 7])
def test_sigmoid_gating_packed_anchor_parity(geom, seq, num_reqs):
    """T7: packed-anchor + hoisted-write-loop path vs the separate-anchor
    path (code-identical to the pre-change kernel): outputs and the full
    state pool bit-equal, incl. NULL rows and every accepted count."""
    h, hv, k, v = geom
    ins = _gating_inputs(num_reqs, seq, h, hv, k, v, torch.bfloat16)
    width = 2 * seq + 4
    null_row = num_reqs - 1 if num_reqs > 2 else None
    table, read, write, packed = _anchored_table(
        num_reqs, width, width - seq, null_row=null_row
    )
    accepted = (
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE) % seq + 1
        if seq > 1
        else None
    )
    base_state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )

    def call(state, use_packed):
        anchor_kw = (
            dict(packed_anchors=packed)
            if use_packed
            else dict(read_anchor=read, write_anchor=write)
        )
        return fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=ins["cu_seqlens"],
            block_table=table,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
            **anchor_kw,
        )

    state_ref = base_state.clone()
    out_ref, _ = call(state_ref, use_packed=False)
    state_packed = base_state.clone()
    out_packed, _ = call(state_packed, use_packed=True)
    if null_row is not None:
        # NULL rows return without stores: compare only valid rows' outputs.
        valid = [i for i in range(num_reqs) if i != null_row]
        tok = torch.tensor(
            [i * seq + t for i in valid for t in range(seq)], device=DEVICE
        )
        assert torch.equal(out_packed[:, tok], out_ref[:, tok])
    else:
        assert torch.equal(out_packed, out_ref)
    assert torch.equal(state_packed, state_ref)


def test_sigmoid_gating_packed_anchor_vs_gather_oracle():
    """T10: packed-anchor result also equals the ORIGINAL host-gathered
    index-tensor path (ssm_state_indices/_output), tying the whole in-kernel
    chain back to the pre-fixC semantics."""
    h, hv, k, v = 4, 8, 128, 128
    num_reqs, seq = 2, 3
    ins = _gating_inputs(num_reqs, seq, h, hv, k, v, torch.bfloat16, seed=1234)
    table, read, write, packed = _anchored_table(num_reqs, 2 * seq, seq)
    accepted = torch.tensor([2, 3], dtype=torch.int32, device=DEVICE)
    base_state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )
    offs = torch.arange(seq, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    gather_in = table.gather(1, read.long().unsqueeze(1) + offs)
    gather_out = table.gather(1, write.long().unsqueeze(1) + offs)

    def base_kwargs():
        return dict(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            inplace_final_state=True,
            cu_seqlens=ins["cu_seqlens"],
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )

    state_gather = base_state.clone()
    out_gather, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=state_gather,
        ssm_state_indices=gather_in.to(torch.int32).contiguous(),
        ssm_state_indices_output=gather_out.to(torch.int32).contiguous(),
        **base_kwargs(),
    )
    state_packed = base_state.clone()
    out_packed, _ = fused_sigmoid_gating_delta_rule_update(
        initial_state=state_packed,
        block_table=table,
        packed_anchors=packed,
        **base_kwargs(),
    )
    assert torch.equal(out_packed, out_gather)
    assert torch.equal(state_packed, state_gather)


CONV_DIM = 1300  # deliberately not a multiple of BLOCK_N=256
CONV_WIDTH = 4


def _conv_pool(num_lines, state_len):
    return torch.rand(
        num_lines, CONV_DIM, state_len, dtype=torch.float32, device=DEVICE
    )


@pytest.mark.parametrize("crossing", [False, True])
@pytest.mark.parametrize("spec", [False, True])
def test_conv_update_packed_anchor_parity(crossing, spec):
    """T8: conv update with packed anchors vs separate anchors, on both sides
    of the skip-second-lookup predicate (write == / != read), spec + nospec,
    dim not a multiple of BLOCK_N."""
    torch.manual_seed(0)
    num_reqs, seq = 3, (3 if spec else 1)
    width = 6
    table, read, write, packed = _anchored_table(num_reqs, width, width - seq - 1)
    if not crossing:
        write = read.clone()
        packed = torch.stack([read, write], dim=1).contiguous()
    weight = torch.rand(CONV_DIM, CONV_WIDTH, dtype=torch.float32, device=DEVICE)
    bias = torch.rand(CONV_DIM, dtype=torch.float32, device=DEVICE)
    state_len = CONV_WIDTH - 1 + (seq - 1 if spec else 0)
    base_pool = _conv_pool(int(table.max().item()) + 1, state_len)
    if spec:
        x = torch.rand(num_reqs * seq, CONV_DIM, dtype=torch.float32, device=DEVICE)
        qsl = torch.arange(
            0, num_reqs * seq + 1, seq, dtype=torch.int32, device=DEVICE
        )
        accepted = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEVICE)
        extra = dict(
            num_accepted_tokens=accepted, query_start_loc=qsl, max_query_len=seq
        )
    else:
        x = torch.rand(num_reqs, CONV_DIM, dtype=torch.float32, device=DEVICE)
        extra = {}

    def call(pool, use_packed):
        anchor_kw = (
            dict(packed_anchors=packed)
            if use_packed
            else dict(
                block_idx_last_scheduled_token=write, initial_state_idx=read
            )
        )
        return causal_conv1d_update(
            x.clone(),
            pool,
            weight,
            bias,
            activation="silu",
            conv_state_indices=table,
            validate_data=False,
            **anchor_kw,
            **extra,
        )

    pool_ref = base_pool.clone()
    out_ref = call(pool_ref, use_packed=False)
    pool_packed = base_pool.clone()
    out_packed = call(pool_packed, use_packed=True)
    assert torch.equal(out_packed, out_ref)
    assert torch.equal(pool_packed, pool_ref)


def test_packed_recurrent_packed_anchor_parity():
    """T9: packed recurrent decode (T=1) with packed anchors vs separate
    anchors; boundary-crossing rows must leave the read block untouched and
    populate the write block identically."""
    torch.manual_seed(0)
    h, hv, k, v = 4, 8, 128, 128
    num_reqs = 4
    packed_dim = 2 * h * k + hv * v
    mixed_qkv = torch.rand(
        num_reqs, packed_dim, dtype=torch.bfloat16, device=DEVICE
    )
    a = torch.rand(num_reqs, hv, dtype=torch.bfloat16, device=DEVICE)
    b = torch.rand(num_reqs, hv, dtype=torch.bfloat16, device=DEVICE)
    a_log = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    dt_bias = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    table, read, write, packed = _anchored_table(num_reqs, 5, 4)
    # Force crossing rows (read != write) and one in-block row.
    read[0], write[0] = 1, 2
    read[1], write[1] = 1, 1
    packed = torch.stack([read, write], dim=1).contiguous()
    base_state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )

    def call(state, use_packed):
        out = torch.zeros(
            num_reqs, 1, hv, v, dtype=mixed_qkv.dtype, device=DEVICE
        )
        anchor_kw = (
            dict(packed_anchors=packed)
            if use_packed
            else dict(read_anchor=read, write_anchor=write)
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=k**-0.5,
            initial_state=state,
            out=out,
            block_table=table,
            use_qk_l2norm_in_kernel=True,
            **anchor_kw,
        )
        return out

    state_ref = base_state.clone()
    out_ref = call(state_ref, use_packed=False)
    state_packed = base_state.clone()
    out_packed = call(state_packed, use_packed=True)
    assert torch.equal(out_packed, out_ref)
    assert torch.equal(state_packed, state_ref)
    # Crossing row 0: read block untouched, write block updated.
    read_blk = int(table[0, 1].item())
    write_blk = int(table[0, 2].item())
    assert torch.equal(state_packed[read_blk], base_state[read_blk])
    assert not torch.equal(state_packed[write_blk], base_state[write_blk])


def test_packed_recurrent_null_row_skipped():
    """T11: a NULL (block id 0) row under packed anchors stores a zero output
    and leaves the state pool byte-identical."""
    torch.manual_seed(0)
    h, hv, k, v = 4, 8, 128, 128
    num_reqs = 2
    packed_dim = 2 * h * k + hv * v
    mixed_qkv = torch.rand(num_reqs, packed_dim, dtype=torch.bfloat16, device=DEVICE)
    a = torch.rand(num_reqs, hv, dtype=torch.bfloat16, device=DEVICE)
    b = torch.rand(num_reqs, hv, dtype=torch.bfloat16, device=DEVICE)
    a_log = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    dt_bias = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    table, read, write, packed = _anchored_table(num_reqs, 4, 3, null_row=1)
    base_state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )
    state = base_state.clone()
    out = torch.full(
        (num_reqs, 1, hv, v), 7.0, dtype=mixed_qkv.dtype, device=DEVICE
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k**-0.5,
        initial_state=state,
        out=out,
        block_table=table,
        packed_anchors=packed,
        use_qk_l2norm_in_kernel=True,
    )
    assert (out[1] == 0).all()  # NULL row: zeros stored
    # No state slot belonging to the NULL row was written (its table is all
    # zeros, and slot 0 is the reserved null block, not in row 0's table).
    row0_write = int(table[0, int(write[0].item())].item())
    untouched = [
        i for i in range(base_state.shape[0]) if i != row0_write
    ]
    assert torch.equal(state[untouched], base_state[untouched])


# --------------------------------------------------------------------------
# T12: wrapper assertions
# --------------------------------------------------------------------------


def _tiny_gating_case():
    h, hv, k, v = 4, 8, 128, 128
    ins = _gating_inputs(2, 1, h, hv, k, v, torch.bfloat16)
    table, read, write, packed = _anchored_table(2, 3, 2)
    state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )
    return ins, table, read, write, packed, state


def test_wrapper_rejects_packed_plus_separate_anchors():
    ins, table, read, write, packed, state = _tiny_gating_case()
    with pytest.raises(AssertionError, match="mutually exclusive"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            initial_state=state,
            cu_seqlens=ins["cu_seqlens"],
            block_table=table,
            read_anchor=read,
            write_anchor=write,
            packed_anchors=packed,
        )


def test_wrapper_rejects_packed_without_table():
    ins, table, read, write, packed, state = _tiny_gating_case()
    with pytest.raises(AssertionError, match="requires block_table"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            initial_state=state,
            cu_seqlens=ins["cu_seqlens"],
            ssm_state_indices=table[:, :1].contiguous(),
            packed_anchors=packed,
        )


@pytest.mark.parametrize(
    "mutate",
    ["dtype", "shape", "stride"],
)
def test_wrapper_rejects_malformed_packed_anchors(mutate):
    ins, table, read, write, packed, state = _tiny_gating_case()
    if mutate == "dtype":
        bad = packed.to(torch.int64)
    elif mutate == "shape":
        bad = torch.cat([packed, packed], dim=1)  # (N, 4)
    else:
        bad = torch.stack([read, write], dim=0).t()  # stride (1, N)
    with pytest.raises(AssertionError, match="packed_anchors"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            initial_state=state,
            cu_seqlens=ins["cu_seqlens"],
            block_table=table,
            packed_anchors=bad,
        )


def test_conv_wrapper_rejects_packed_plus_separate_anchors():
    torch.manual_seed(0)
    table, read, write, packed = _anchored_table(2, 4, 2)
    pool = _conv_pool(int(table.max().item()) + 1, CONV_WIDTH - 1)
    weight = torch.rand(CONV_DIM, CONV_WIDTH, dtype=torch.float32, device=DEVICE)
    x = torch.rand(2, CONV_DIM, dtype=torch.float32, device=DEVICE)
    with pytest.raises(AssertionError, match="mutually exclusive"):
        causal_conv1d_update(
            x,
            pool,
            weight,
            None,
            activation="silu",
            conv_state_indices=table,
            initial_state_idx=read,
            block_idx_last_scheduled_token=write,
            packed_anchors=packed,
        )


def test_recurrent_wrapper_rejects_packed_misuse():
    torch.manual_seed(0)
    h, hv, k, v = 4, 8, 128, 128
    packed_dim = 2 * h * k + hv * v
    mixed_qkv = torch.rand(2, packed_dim, dtype=torch.bfloat16, device=DEVICE)
    a = torch.rand(2, hv, dtype=torch.bfloat16, device=DEVICE)
    b = torch.rand(2, hv, dtype=torch.bfloat16, device=DEVICE)
    a_log = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    dt_bias = torch.rand(hv, dtype=torch.bfloat16, device=DEVICE)
    table, read, write, packed = _anchored_table(2, 4, 3)
    state = torch.rand(
        int(table.max().item()) + 1, hv, v, k, dtype=torch.float32, device=DEVICE
    )
    out = torch.zeros(2, 1, hv, v, dtype=mixed_qkv.dtype, device=DEVICE)
    kwargs = dict(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k**-0.5,
        initial_state=state,
        out=out,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        fused_recurrent_gated_delta_rule_packed_decode(
            block_table=table,
            read_anchor=read,
            write_anchor=write,
            packed_anchors=packed,
            **kwargs,
        )
    with pytest.raises(ValueError, match="requires `block_table`"):
        fused_recurrent_gated_delta_rule_packed_decode(
            ssm_state_indices=table[:, 0].contiguous(),
            packed_anchors=packed,
            **kwargs,
        )
