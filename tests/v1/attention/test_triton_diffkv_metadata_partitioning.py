# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for Triton DiffKV metadata partitioning.

Pins the DiffKV builder's V-shaped 3D softmax scratch, build()'s partition
threshold and buffer wiring, the forward handoff of partition buffers with the
kernel stubbed, and the launcher's 3D/2D partition dispatch with both kernels
stubbed.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.triton_attn_diffkv as diffkv_backend
import vllm.v1.attention.ops.triton_unified_attention_diffkv as diffkv_ops
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import CUDAGraphMode
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.triton_attn import (
    MIN_LAUNCH_GRID_SIZE_2D,
    NUM_PAR_SOFTMAX_SEGMENTS,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
    TritonAttentionDiffKVMetadataBuilder,
)

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = torch.device("cpu")
BLOCK_SIZE = 16
HEAD_SIZE_QK = 128
HEAD_SIZE_V = 64
NUM_BLOCKS = 8
SEQ_THRESHOLD_3D = MIN_LAUNCH_GRID_SIZE_2D // 8


class _KernelRecorder:
    """Captures grid and kwargs of every stubbed Triton kernel launch."""

    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], dict]] = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


class _TritonShim:
    next_power_of_2 = staticmethod(next_power_of_2)


@pytest.fixture(scope="module")
def env():
    original_head_size_v = TritonAttentionDiffKVBackend.head_size_v
    TritonAttentionDiffKVBackend.set_head_size_v(HEAD_SIZE_V)
    config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    spec = create_standard_kv_cache_spec(config)
    builder = TritonAttentionDiffKVMetadataBuilder(spec, ["layer0"], config, DEVICE)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 200], query_lens=[1, 1]), BLOCK_SIZE, DEVICE
    )
    yield SimpleNamespace(config=config, spec=spec, builder=builder, common=common)
    TritonAttentionDiffKVBackend.set_head_size_v(original_head_size_v)


@pytest.fixture
def launcher(monkeypatch):
    main, reduce = _KernelRecorder(), _KernelRecorder()
    monkeypatch.setattr(diffkv_ops, "triton", _TritonShim)
    monkeypatch.setattr(diffkv_ops, "kernel_unified_attention_diffkv", main)
    monkeypatch.setattr(diffkv_ops, "kernel_reduce_segments_diffkv", reduce)
    return main, reduce


def _make_impl() -> TritonAttentionDiffKVImpl:
    return TritonAttentionDiffKVImpl(
        num_heads=16,
        head_size=HEAD_SIZE_QK,
        scale=0.25,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _run_launcher(num_seqs, max_seqlen_q, with_partition_buffers):
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32) * max_seqlen_q
    kwargs = dict(
        q=torch.zeros(num_seqs * max_seqlen_q, 16, HEAD_SIZE_QK),
        k=torch.zeros(NUM_BLOCKS, BLOCK_SIZE, 8, HEAD_SIZE_QK),
        v=torch.zeros(NUM_BLOCKS, BLOCK_SIZE, 8, HEAD_SIZE_V),
        out=torch.zeros(num_seqs * max_seqlen_q, 16, HEAD_SIZE_V),
        cu_seqlens_q=query_start_loc,
        seqused_k=torch.full((num_seqs,), 64, dtype=torch.int32),
        softmax_scale=0.25,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.zeros(num_seqs, 2, dtype=torch.int32),
        softcap=0,
        max_seqlen_q=max_seqlen_q,
    )
    if with_partition_buffers:
        kwargs.update(
            seq_threshold_3D=SEQ_THRESHOLD_3D,
            num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
            softmax_segm_output=torch.zeros(
                SEQ_THRESHOLD_3D, 16, NUM_PAR_SOFTMAX_SEGMENTS, HEAD_SIZE_V
            ),
            softmax_segm_max=torch.zeros(
                SEQ_THRESHOLD_3D, 16, NUM_PAR_SOFTMAX_SEGMENTS
            ),
            softmax_segm_expsum=torch.zeros(
                SEQ_THRESHOLD_3D, 16, NUM_PAR_SOFTMAX_SEGMENTS
            ),
        )
    diffkv_ops.unified_attention_diffkv(**kwargs)


def test_builder_softmax_scratch_padded_to_v_head_size(env):
    """DiffKV scratch last dim follows head_size_v, not the QK head size."""
    builder = env.builder
    assert builder.headdim == HEAD_SIZE_QK
    assert builder.headdim != HEAD_SIZE_V
    assert builder.softmax_segm_output.shape == (
        builder.seq_threshold_3D,
        builder.num_heads_q,
        builder.num_par_softmax_segments,
        next_power_of_2(HEAD_SIZE_V),
    )
    parent = TritonAttentionMetadataBuilder(env.spec, ["layer0"], env.config, DEVICE)
    assert parent.softmax_segm_output.shape[-1] == next_power_of_2(HEAD_SIZE_QK)


def test_build_attaches_partition_buffers_and_threshold(env):
    """build() threads the partition threshold and scratch buffers through."""
    builder = env.builder
    meta = builder.build(0, env.common)
    assert meta.seq_threshold_3D == MIN_LAUNCH_GRID_SIZE_2D // builder.num_heads_kv
    assert meta.num_par_softmax_segments == NUM_PAR_SOFTMAX_SEGMENTS
    assert meta.softmax_segm_output is builder.softmax_segm_output
    assert meta.softmax_segm_max is builder.softmax_segm_max
    assert meta.softmax_segm_expsum is builder.softmax_segm_expsum
    assert meta.softmax_segm_output.shape == (
        meta.seq_threshold_3D,
        builder.num_heads_q,
        NUM_PAR_SOFTMAX_SEGMENTS,
        next_power_of_2(HEAD_SIZE_V),
    )
    assert not meta.use_cascade
    assert meta.num_actual_tokens == 2
    assert meta.max_query_len == 1
    assert meta.query_start_loc is env.common.query_start_loc
    assert meta.seq_lens is env.common.seq_lens
    assert meta.block_table is env.common.block_table_tensor


def test_build_threshold_follows_cudagraph_capture_sizes(env):
    """With CUDA graph capture, the threshold snaps to the nearest size."""
    config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config.compilation_config.cudagraph_capture_sizes = [12, 64]
    spec = create_standard_kv_cache_spec(config)
    builder = TritonAttentionDiffKVMetadataBuilder(spec, ["layer0"], config, DEVICE)
    base = MIN_LAUNCH_GRID_SIZE_2D // builder.num_heads_kv
    assert base == 16
    assert builder.seq_threshold_3D == 12
    assert builder.softmax_segm_output.shape[0] == 12
    assert builder.build(0, env.common).seq_threshold_3D == 12


def test_forward_slices_kv_and_hands_partition_buffers_to_launcher(env, monkeypatch):
    """forward() splits the packed cache at QK/V sizes and forwards scratch."""
    captured = {}
    monkeypatch.setattr(
        diffkv_backend,
        "unified_attention_diffkv",
        lambda **kwargs: captured.update(kwargs),
    )
    builder, meta = env.builder, env.builder.build(0, env.common)
    impl = _make_impl()
    num_tokens = 4
    query = torch.randn(num_tokens, 16, HEAD_SIZE_QK)
    key = torch.randn(num_tokens, 8, HEAD_SIZE_QK)
    value = torch.randn(num_tokens, 8, HEAD_SIZE_V)
    kv_cache = torch.zeros(NUM_BLOCKS, 8, BLOCK_SIZE, HEAD_SIZE_QK + HEAD_SIZE_V)
    output = torch.ones(num_tokens, 16, HEAD_SIZE_V)

    result = impl.forward(None, query, key, value, kv_cache, meta, output)

    assert result is output
    assert captured["q"].shape == (meta.num_actual_tokens, 16, HEAD_SIZE_QK)
    assert captured["out"].shape == (meta.num_actual_tokens, 16, HEAD_SIZE_V)
    assert captured["k"].shape == (NUM_BLOCKS, BLOCK_SIZE, 8, HEAD_SIZE_QK)
    assert captured["v"].shape == (NUM_BLOCKS, BLOCK_SIZE, 8, HEAD_SIZE_V)
    assert output[meta.num_actual_tokens :].eq(1).all()
    assert captured["cu_seqlens_q"] is meta.query_start_loc
    assert captured["seqused_k"] is meta.seq_lens
    assert captured["block_table"] is meta.block_table
    assert captured["max_seqlen_q"] == meta.max_query_len
    assert captured["softmax_scale"] == impl.scale
    assert captured["window_size"] == impl.sliding_window
    assert captured["softcap"] == impl.logits_soft_cap
    assert captured["causal"] is True
    assert captured["seq_threshold_3D"] == builder.seq_threshold_3D
    assert captured["num_par_softmax_segments"] == builder.num_par_softmax_segments
    assert captured["softmax_segm_output"] is builder.softmax_segm_output
    assert captured["softmax_segm_max"] is builder.softmax_segm_max
    assert captured["softmax_segm_expsum"] is builder.softmax_segm_expsum


def test_forward_without_metadata_returns_zeros(env):
    impl = _make_impl()
    output = torch.ones(4, 16, HEAD_SIZE_V)
    result = impl.forward(None, torch.zeros(0), None, None, None, None, output)
    assert result is output
    assert result.eq(0).all()


def test_forward_rejects_output_scale(env, monkeypatch):
    monkeypatch.setattr(diffkv_backend, "unified_attention_diffkv", lambda **_: None)
    impl = _make_impl()
    meta = env.builder.build(0, env.common)
    output = torch.ones(4, 16, HEAD_SIZE_V)
    kv_cache = torch.zeros(NUM_BLOCKS, 8, BLOCK_SIZE, HEAD_SIZE_QK + HEAD_SIZE_V)
    with pytest.raises(NotImplementedError):
        impl.forward(
            None,
            torch.zeros(4, 16, HEAD_SIZE_QK),
            None,
            None,
            kv_cache,
            meta,
            output,
            output_scale=torch.ones(1),
        )


def test_forward_rejects_cascade_metadata(env, monkeypatch):
    monkeypatch.setattr(diffkv_backend, "unified_attention_diffkv", lambda **_: None)
    impl = _make_impl()
    meta = dataclasses.replace(env.builder.build(0, env.common), use_cascade=True)
    output = torch.ones(4, 16, HEAD_SIZE_V)
    kv_cache = torch.zeros(NUM_BLOCKS, 8, BLOCK_SIZE, HEAD_SIZE_QK + HEAD_SIZE_V)
    with pytest.raises(AssertionError, match="Cascade attention"):
        impl.forward(
            None,
            torch.zeros(4, 16, HEAD_SIZE_QK),
            None,
            None,
            kv_cache,
            meta,
            output,
        )


def test_launcher_takes_3d_partition_path_on_small_decode_batch(launcher):
    """Fewer sequences than the threshold: 3D grid plus per-segment reduce."""
    main, reduce = launcher
    _run_launcher(num_seqs=4, max_seqlen_q=1, with_partition_buffers=True)
    assert len(main.calls) == 1
    grid, kwargs = main.calls[0]
    assert grid == (4, 8, NUM_PAR_SOFTMAX_SEGMENTS)
    assert kwargs["IS_3D"] is True
    assert kwargs["NUM_SEGMENTS_PER_SEQ"] == NUM_PAR_SOFTMAX_SEGMENTS
    assert len(reduce.calls) == 1
    reduce_grid, reduce_kwargs = reduce.calls[0]
    assert reduce_grid == (4, 16)
    assert reduce_kwargs["NUM_SEGMENTS_PER_SEQ"] == NUM_PAR_SOFTMAX_SEGMENTS


def test_launcher_falls_back_to_2d_above_threshold(launcher):
    """More sequences than the threshold: 2D grid, no reduce launch."""
    main, reduce = launcher
    _run_launcher(
        num_seqs=SEQ_THRESHOLD_3D + 1, max_seqlen_q=1, with_partition_buffers=True
    )
    assert len(main.calls) == 1
    grid, kwargs = main.calls[0]
    assert grid == (19, 8)
    assert kwargs["IS_3D"] is False
    assert kwargs["NUM_SEGMENTS_PER_SEQ"] == 1
    assert reduce.calls == []


def test_launcher_falls_back_to_2d_on_prefill(launcher):
    """Multi-token queries: 2D grid, no reduce launch."""
    main, reduce = launcher
    _run_launcher(num_seqs=4, max_seqlen_q=8, with_partition_buffers=True)
    assert len(main.calls) == 1
    grid, kwargs = main.calls[0]
    assert grid == (8, 8)
    assert kwargs["IS_3D"] is False
    assert kwargs["NUM_SEGMENTS_PER_SEQ"] == 1
    assert reduce.calls == []


def test_launcher_uses_2d_without_partition_buffers(launcher):
    """Absent partition metadata disables the 3D path entirely."""
    main, reduce = launcher
    _run_launcher(num_seqs=4, max_seqlen_q=1, with_partition_buffers=False)
    assert len(main.calls) == 1
    grid, kwargs = main.calls[0]
    assert grid == (4, 8)
    assert kwargs["IS_3D"] is False
    assert kwargs["NUM_SEGMENTS_PER_SEQ"] == 1
    assert reduce.calls == []
