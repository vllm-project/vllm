# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CS2 tests for GDN all-mode in-kernel checkpoint writes
(``VLLM_GDN_INKERNEL_CKPT_WRITE``; design doc allmode_decode_opt_design.md,
tests T13-T17).

- T13-T15: the single-launch Triton scatter
  (``gdn_scatter_block_checkpoints_triton``) reproduces the Python-looped
  ``gdn_scatter_block_checkpoints`` (run as golden) bit-exactly on the full
  state pool: every interior-block-count/crossing pattern, multi-sequence
  adjacent chunk ranges (the seq_hi clamp), resumed prefills, the
  unaligned-skip (never-poison-APC) rule, and the fp32 -> pool-dtype cast.
- T16: builder staging elimination — flag-on metadata aliases the runner
  block table (data_ptr equality), flag-off keeps the builder-local copy;
  values identical either way.
- T17: DtoD copy_-count assertion — the anchor + table staging copies
  actually disappear under the flags (delta == 4 spec / 3 nospec per build).
- Layer level: the real ``_forward_core`` prefill under WRITE=1 equals the
  WRITE=0 run bit-exactly (outputs + full pools).
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN in-kernel ckpt-write tests require CUDA (Triton kernels).",
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
from vllm.config import CUDAGraphMode, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.all_mode_utils import (  # noqa: E402
    gdn_scatter_block_checkpoints,
)
from vllm.model_executor.layers.mamba.ops.gdn_scatter import (  # noqa: E402
    gdn_scatter_block_checkpoints_triton,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

DEVICE = torch.device("cuda")
CHUNK = 4
BLK = 8  # mamba block = 2 chunks (same grid as test_gdn_scatter)
STATE_SHAPE = (3, 5, 7)  # deliberately odd per-block state shape


# --------------------------------------------------------------------------
# T13-T15: scatter kernel == Python-looped golden
# --------------------------------------------------------------------------


def _run_both(seqs, state_dtype=torch.float32, pool_blocks=64, seed=0):
    """seqs: list of (first, last, ncomp, num_chunks). Returns (golden, run)
    pools after the eager and Triton scatters on identical inputs."""
    torch.manual_seed(seed)
    num_seqs = len(seqs)
    nt = sum(s[3] for s in seqs)
    inter = torch.randn(nt, *STATE_SHAPE, dtype=torch.float32, device=DEVICE)
    final = torch.randn(
        num_seqs, *STATE_SHAPE, dtype=torch.float32, device=DEVICE
    )
    width = max(s[1] for s in seqs) + 2
    perm = torch.randperm(pool_blocks - 1, dtype=torch.int32, device=DEVICE) + 1
    table = perm[: num_seqs * width].view(num_seqs, width).contiguous()
    first = torch.tensor([s[0] for s in seqs], dtype=torch.int32, device=DEVICE)
    last = torch.tensor([s[1] for s in seqs], dtype=torch.int32, device=DEVICE)
    ncomp = torch.tensor([s[2] for s in seqs], dtype=torch.int32, device=DEVICE)
    chunk_offsets = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor([s[3] for s in seqs]), 0)),
        dtype=torch.int32,
        device=DEVICE,
    )
    base_pool = torch.randn(
        pool_blocks, *STATE_SHAPE, dtype=torch.float32, device=DEVICE
    ).to(state_dtype)
    num_prefill_tokens = int(sum(s[1] - s[0] + 1 for s in seqs)) * BLK

    pool_golden = base_pool.clone()
    gdn_scatter_block_checkpoints(
        pool_golden, inter, final, table, first, last, ncomp,
        chunk_offsets, BLK, CHUNK,
    )
    pool_run = base_pool.clone()
    gdn_scatter_block_checkpoints_triton(
        pool_run, inter, final, table, first, last, ncomp,
        chunk_offsets, BLK, CHUNK, num_prefill_tokens,
    )
    return pool_golden, pool_run, base_pool, table


@pytest.mark.parametrize(
    "seqs",
    [
        # n interior blocks = last - first in {0, 1, 2, 5}; fresh + resumed
        # (ncomp = j*BLK) starts; final partial and exactly-full.
        [(0, 0, 0, 1)],  # first == last (final block only)
        [(0, 1, 0, 3)],  # 1 interior
        [(0, 2, 0, 5)],  # 2 interior
        [(0, 5, 0, 11)],  # 5 interior
        [(1, 2, 8, 3)],  # resumed at block boundary (ncomp = BLK)
        [(2, 4, 16, 6)],  # resumed 2 blocks in, exactly-full final chunk
        # multi-sequence, adjacent chunk ranges (seq_hi clamp defense)
        [(0, 2, 0, 5), (0, 1, 0, 4), (1, 3, 8, 5)],
        [(0, 0, 0, 1), (0, 4, 0, 9)],
    ],
)
def test_scatter_kernel_matches_golden(seqs):
    """T13: full-pool bit-equality with the Python-looped scatter across
    crossing patterns, multi-seq ranges and resumed prefills."""
    pool_golden, pool_run, _, _ = _run_both(seqs)
    assert torch.equal(pool_run, pool_golden)


def test_scatter_kernel_unaligned_skip():
    """T14: rows with ncomp % chunk != 0 skip interior checkpoints (pool
    untouched there) but still write the final block; aligned rows in the
    same batch scatter normally."""
    seqs = [(0, 2, 2, 5), (0, 2, 0, 5)]  # row 0 unaligned (2 % 4 != 0)
    pool_golden, pool_run, base_pool, table = _run_both(seqs)
    assert torch.equal(pool_run, pool_golden)
    # Row 0 interior blocks untouched, final written.
    for j in (0, 1):
        blk = int(table[0, j].item())
        assert torch.equal(pool_run[blk], base_pool[blk])
    assert not torch.equal(
        pool_run[int(table[0, 2].item())], base_pool[int(table[0, 2].item())]
    )
    # Row 1 interior blocks written.
    for j in (0, 1):
        blk = int(table[1, j].item())
        assert not torch.equal(pool_run[blk], base_pool[blk])


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
def test_scatter_kernel_dtype_cast(state_dtype):
    """T15: fp32 chunk states stored into a narrower pool dtype cast exactly
    like the eager ``.to(ssm_state.dtype)``."""
    pool_golden, pool_run, _, _ = _run_both(
        [(0, 2, 0, 5), (1, 3, 8, 5)], state_dtype=state_dtype
    )
    assert torch.equal(pool_run, pool_golden)


# --------------------------------------------------------------------------
# T16/T17: builder staging elimination + copy-count assertions
# --------------------------------------------------------------------------

BLOCK = spec_harness.BLOCK
NUM_SPEC = spec_harness.NUM_SPEC


class _CudaCopyCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if name == "copy_" and all(
            (not isinstance(t, torch.Tensor)) or t.is_cuda for t in args
        ):
            self.counts["cuda_copy_"] += 1
        return func(*args, **(kwargs or {}))


def _spec_builder_and_common(batch, drafts, accepted, prev):
    cfg = spec_harness._make_spec_config("all")
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
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
    assert builder.use_full_cuda_graph
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
    kwargs = dict(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        num_accepted_tokens=torch.tensor(
            accepted, dtype=torch.int32, device=DEVICE
        ),
        prev_last_scheduled_idx=torch.tensor(
            prev, dtype=torch.int32, device=DEVICE
        ),
    )
    return cfg, builder, common, kwargs


def _decode_builder_and_common(batch):
    cfg = harness._make_vllm_config(BLOCK, "all")
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    assert builder.use_full_cuda_graph
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    return cfg, builder, common


def test_staging_elimination_metadata_aliasing(monkeypatch):
    """T16: WRITE=1 metadata aliases the (runner-owned) block table tensor
    (data_ptr equality); WRITE=0 stages a builder-local copy; contents
    identical either way."""
    torch.manual_seed(0)
    batch = BatchSpec(seq_lens=[101, 231], query_lens=[1, 1])

    monkeypatch.setenv("VLLM_GDN_INKERNEL_CKPT_WRITE", "0")
    cfg, builder, common = _decode_builder_and_common(batch)
    with set_current_vllm_config(cfg):
        meta_off = builder.build(0, common)
    assert (
        meta_off.all_state_indices_tensor.data_ptr()
        != common.block_table_tensor.data_ptr()
    )
    assert (
        meta_off.all_state_indices_tensor.data_ptr()
        == builder.all_state_indices_tensor.data_ptr()
    )

    monkeypatch.setenv("VLLM_GDN_INKERNEL_CKPT_WRITE", "1")
    with set_current_vllm_config(cfg):
        meta_on = builder.build(0, common)
    assert (
        meta_on.all_state_indices_tensor.data_ptr()
        == common.block_table_tensor.data_ptr()
    )
    assert torch.equal(
        meta_on.all_state_indices_tensor, meta_off.all_state_indices_tensor
    )
    for f in (
        "block_idx_last_scheduled_token",
        "block_idx_last_computed_token",
        "non_spec_state_indices_tensor",
    ):
        assert torch.equal(getattr(meta_on, f), getattr(meta_off, f)), f


@pytest.mark.parametrize("spec", [True, False])
def test_staging_copy_count_drops(monkeypatch, spec):
    """T17: the anchor + table staging copies disappear under the flags —
    CUDA copy_ count per build drops by exactly 4 (spec: table + 3 anchors)
    or 3 (nospec decode: table + 2 anchors), with bit-identical metadata."""
    torch.manual_seed(0)
    from tests.kernels.mamba.test_gdn_index_prep import _prep_from_common

    if spec:
        batch = BatchSpec(seq_lens=[103, 231], query_lens=[3, 3])
        cfg, builder, common, kwargs = _spec_builder_and_common(
            batch, [NUM_SPEC, NUM_SPEC], [2, 1], [1, -1]
        )
        prev = kwargs["prev_last_scheduled_idx"]
        expected_delta = 4
    else:
        batch = BatchSpec(seq_lens=[101, 231], query_lens=[1, 1])
        cfg, builder, common = _decode_builder_and_common(batch)
        kwargs = {}
        prev = None
        expected_delta = 3

    def build(flags_on):
        monkeypatch.setenv(
            "VLLM_GDN_INKERNEL_CKPT_WRITE", "1" if flags_on else "0"
        )
        extra = dict(kwargs)
        if flags_on:
            extra["block_idx_prep"] = _prep_from_common(common, prev, BLOCK)
            common.compute_num_computed_tokens()
        with set_current_vllm_config(cfg), _CudaCopyCounter() as counter:
            meta = builder.build(0, common, **extra)
        return meta, counter.counts["cuda_copy_"]

    meta_off, copies_off = build(False)
    meta_on, copies_on = build(True)
    assert copies_off - copies_on == expected_delta, (copies_off, copies_on)
    for f in (
        "all_state_indices_tensor",
        "block_idx_last_scheduled_token",
        "block_idx_last_computed_token",
        "block_idx_last_scheduled_token_prev_step",
    ):
        ref, got = getattr(meta_off, f), getattr(meta_on, f)
        if ref is None:
            assert got is None, f
        else:
            assert torch.equal(got, ref), f


# --------------------------------------------------------------------------
# Layer-level: prefill scatter A/B through the real _forward_core
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resumed", [False, True])
def test_layer_prefill_scatter_flag_ab(monkeypatch, resumed):
    """WRITE=1 vs WRITE=0 through the real prefill path: outputs and the
    full conv/SSM pools bit-identical (fresh and resumed prefills)."""
    torch.manual_seed(0)
    mamba_block_size = 64
    total = 5 * mamba_block_size
    cached = 3 * mamba_block_size if resumed else 0
    weights = harness._make_weights(DEVICE)
    mixed_qkv, b, a = harness._make_inputs(total, DEVICE)
    cfg = harness._make_vllm_config(mamba_block_size, "all")

    def run():
        batch = BatchSpec(seq_lens=[total], query_lens=[total - cached])
        meta, common = harness._build_metadata(
            cfg, batch, mamba_block_size, DEVICE
        )
        pool_size = int(common.block_table_tensor.max().item()) + 1
        conv_state, ssm_state = harness._make_pools(
            pool_size, torch.float32, DEVICE
        )
        layer = harness._build_layer(
            cfg, mamba_block_size, conv_state, ssm_state, weights
        )
        out = harness._run_forward_core(
            layer, meta, mixed_qkv[cached:], b[cached:], a[cached:],
            total - cached,
        )
        return out, conv_state, ssm_state

    monkeypatch.setenv("VLLM_GDN_INKERNEL_CKPT_WRITE", "0")
    out_off, conv_off, ssm_off = run()
    monkeypatch.setenv("VLLM_GDN_INKERNEL_CKPT_WRITE", "1")
    out_on, conv_on, ssm_on = run()
    assert torch.equal(out_on, out_off)
    assert torch.equal(conv_on, conv_off)
    assert torch.equal(ssm_on, ssm_off)
