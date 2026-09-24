# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer SM90 sparse MLA backend wiring and index packing.

The FlashInfer wrapper and top-k conversion are replaced by CPU recorders;
the tests pin the contract between the impl and the kernel API: page_size=1
varlen rows, reserved-buffer refresh, plan parameters (dims, NoPE/rope scale,
causality), ckv/kpe cache splitting, and the backend's model-shape gates.
"""

from types import SimpleNamespace

import pytest
import torch

# isort: off
import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as sm90_mod
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Backend,
    FlashInferMLASparseSM90Builder,
    FlashInferMLASparseSM90Impl,
)
# isort: on

BLOCK_SIZE = 64
HEAD = 512
TOPK = 128  # triton convert requires width % 128 == 0


def ref_convert(req_id, block_table, token_indices, BLOCK_SIZE=64, **_):
    out = torch.full_like(token_indices, -1)
    counts = torch.zeros(token_indices.shape[0], dtype=torch.int32)
    for t in range(token_indices.shape[0]):
        vals = []
        for j in range(token_indices.shape[1]):
            pos = int(token_indices[t, j])
            if pos == -1:
                continue
            blk = int(block_table[int(req_id[t]), pos // BLOCK_SIZE])
            if blk < 0:
                continue
            vals.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
        out[t, : len(vals)] = torch.tensor(vals, dtype=out.dtype)
        counts[t] = len(vals)
    return out, counts


class FakeWrapper:
    def __init__(self):
        self.plan_args = None
        self.run_args = None

    def plan(self, *args, **kwargs):
        self.plan_args = (args, kwargs)

    def run(self, *args, **kwargs):
        q_nope, q_pe, ckv, kpe = args
        self.run_args = (q_nope, q_pe, ckv, kpe, kwargs)
        return torch.zeros(
            q_nope.shape[0], q_nope.shape[1], ckv.shape[-1], dtype=torch.bfloat16
        )


class FakeState:
    def __init__(self, width, max_tokens=64):
        self.kv_indices = torch.zeros(max_tokens * width, dtype=torch.int32)
        self.kv_len_arr = torch.zeros(max_tokens, dtype=torch.int32)
        self.kv_indptr = torch.zeros(max_tokens + 1, dtype=torch.int32)
        self.wrapper = FakeWrapper()
        self.plan_calls = []

    def plan(self, num_tokens, kv_lens):
        self.plan_calls.append((num_tokens, kv_lens))

    def pack_indices(self, slots):
        for row in range(slots.shape[0]):
            start, end = self.kv_indptr[row : row + 2].tolist()
            self.kv_indices[start:end] = slots[row, : end - start].clamp(min=0)


def make_impl(qk_rope, kv_dtype="fp8_e4m3", num_heads=2, topk_width=TOPK):
    impl = object.__new__(FlashInferMLASparseSM90Impl)
    impl.num_heads = num_heads
    impl.head_size = HEAD + qk_rope
    impl.scale = (HEAD + qk_rope) ** -0.5
    impl.kv_lora_rank = HEAD
    impl.qk_rope_head_dim = qk_rope
    impl.kv_cache_dtype = kv_dtype
    impl.use_fp8_kv_cache = kv_dtype in ("fp8", "fp8_e4m3")
    rows = 4
    impl.topk_indices_buffer = torch.full((rows, topk_width), -1, dtype=torch.int32)
    return impl, rows


def make_batch(rows, topk_rows, own_blocks):
    req_id = torch.tensor([0] * rows, dtype=torch.int32)
    block_table = torch.zeros(1, 16, dtype=torch.int32)
    block_table[:, 0] = own_blocks[0]
    topk = torch.full((rows, TOPK), -1, dtype=torch.int32)
    for t, row in enumerate(topk_rows):
        topk[t, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return SimpleNamespace(
        req_id_per_token=req_id, block_table=block_table, block_size=BLOCK_SIZE
    )


@pytest.mark.parametrize("qk_rope,kv_dtype", [(0, "fp8_e4m3"), (64, "auto")])
def test_forward_wiring(monkeypatch, qk_rope, kv_dtype):
    impl, rows = make_impl(qk_rope, kv_dtype)
    state = FakeState(TOPK)
    monkeypatch.setattr(
        sm90_mod, "triton_convert_req_index_to_global_index", ref_convert
    )

    # req with context 10 < topk: 8 valid + -1 padding.
    topk_rows = [
        [7, 3, 1, 9, 0, 2, 5, 8] + [-1] * (TOPK - 8),
        [4, 0, 2, 3, 1] + [-1] * (TOPK - 5),
        [6, 5, 4] + [-1] * (TOPK - 3),
        [2] + [-1] * (TOPK - 1),
    ]
    impl.topk_indices_buffer.copy_(torch.tensor(topk_rows, dtype=torch.int32))
    state.kv_indptr[1:5] = torch.tensor([8, 13, 16, 17], dtype=torch.int32)
    state.kv_indptr[5:] = 17
    meta = make_batch(rows, topk_rows, [3])
    meta.state = state
    q_nope = torch.randn(rows, impl.num_heads, HEAD)
    q_rope = torch.randn(rows, impl.num_heads, qk_rope)
    cache = torch.zeros(
        8 * BLOCK_SIZE,
        impl.head_size,
        dtype=torch.uint8 if impl.use_fp8_kv_cache else torch.bfloat16,
    )

    out, lse = impl.forward_mqa(
        (q_nope, q_rope), cache, meta, SimpleNamespace(_k_scale_float=0.5)
    )
    assert lse is None and out.shape == (rows, impl.num_heads, HEAD)

    # Reserved buffers carry this step's slots; lengths are NOT refreshed
    # here (the builder plans them host-side before capture/replay).
    ref_slots, ref_counts = ref_convert(
        meta.req_id_per_token, meta.block_table, impl.topk_indices_buffer
    )
    offset = 0
    for t in range(rows):
        k = int(ref_counts[t])
        assert (
            state.kv_indices[offset : offset + k].tolist() == ref_slots[t, :k].tolist()
        )
        offset += k
    assert state.plan_calls == []

    assert state.wrapper.run_args is not None
    q_pe, ckv, kpe, kwargs = state.wrapper.run_args[1:]
    assert q_pe.shape == (rows, impl.num_heads, qk_rope)
    assert ckv.shape == (8 * BLOCK_SIZE, 1, HEAD)
    assert kpe.shape[-1] == qk_rope
    if impl.use_fp8_kv_cache:
        assert kwargs["ckv_scale"] == 0.5 and kwargs["kpe_scale"] == 1.0
    else:
        assert kwargs == {}


@pytest.mark.parametrize("use_mha", [False, True])
@pytest.mark.parametrize("num_decodes", [0, 1])
def test_builder_plans_only_rows_dispatched_to_mqa(monkeypatch, use_mha, num_decodes):
    """MHA prefill rows must not make the MQA kernel read beyond its query."""
    builder = object.__new__(FlashInferMLASparseSM90Builder)
    builder._index_topk = 2048
    builder._index_kpool = 4
    builder._async_scheduling = False
    builder.state = FakeState(TOPK)
    builder._attention_layer = SimpleNamespace(_use_sparse_mha=lambda _: use_mha)
    metadata = object.__new__(sm90_mod.FlashInferMLASparseSM90Metadata)
    metadata.state = None
    metadata.num_prefills = 1
    metadata.num_decode_tokens = num_decodes
    monkeypatch.setattr(
        sm90_mod.FlashInferMLASparseMetadataBuilder,
        "build",
        lambda *_args, **_kwargs: metadata,
    )
    cam = SimpleNamespace(
        num_reqs=num_decodes + 1,
        query_start_loc_cpu=torch.tensor(
            [0, 1, 6] if num_decodes else [0, 5], dtype=torch.int32
        ),
        seq_lens_cpu_upper_bound=torch.tensor(
            [1402, 5] if num_decodes else [5], dtype=torch.int32
        ),
        positions=None,
    )

    result = builder.build(0, cam)

    assert result.state is builder.state
    expected_lens = [1402] if num_decodes else []
    if not use_mha:
        expected_lens += [1, 2, 3, 4, 5]
    assert builder.state.plan_calls[0][0] == len(expected_lens)
    assert builder.state.plan_calls[0][1].tolist() == expected_lens


def test_plan_uses_state_params(monkeypatch):
    """The NoPE/rope dims and scale live on the builder state, not the layer.

    plan() takes exact per-row KV lengths; the schedule is rebuilt on every
    call (contexts grow between steps) and the indptrs are always full-size
    with zero-query padding rows past num_tokens.
    """
    impl, rows = make_impl(64, "auto")
    wrapper = FakeWrapper()
    state = sm90_mod._SM90State.__new__(sm90_mod._SM90State)
    state.device = torch.device("cpu")
    state.wrapper = wrapper
    state.num_heads = 4
    state.kv_dtype = torch.bfloat16
    state.kv_lora_rank = HEAD
    state.qk_rope_head_dim = 64
    state.sm_scale = 576**-0.5
    state.max_tokens = 4
    state.topk_width = TOPK
    state.kv_indices = torch.zeros(4 * TOPK)
    state._arange_cpu = torch.arange(5, dtype=torch.int32)
    state._qo_cpu = torch.empty(5, dtype=torch.int32)
    state._kv_cpu = torch.empty(5, dtype=torch.int32)
    state._lens_cpu = torch.full((4,), TOPK, dtype=torch.int32)

    state.plan(3, torch.tensor([2, 5, 7], dtype=torch.int32))
    assert wrapper.plan_args is not None
    args, kwargs = wrapper.plan_args
    (qo, kv, indices, kv_len, heads, ckv, kpe, page, causal, scale) = args
    assert qo.tolist() == [0, 1, 2, 3, 3]  # clamp: rows past 3 have no queries
    assert kv.tolist() == [0, 2, 7, 14, 14]
    assert kv_len.tolist() == [2, 5, 7, 0]
    assert (heads, ckv, kpe, page, causal) == (4, HEAD, 64, 1, False)
    assert scale == 576**-0.5
    assert kwargs["q_data_type"] == torch.bfloat16
    assert kwargs["kv_data_type"] == torch.bfloat16

    # Replanning a smaller batch must clear the previous rows' lengths.
    state.plan(1, torch.tensor([TOPK], dtype=torch.int32))
    assert state._kv_cpu.tolist() == [0, TOPK, TOPK, TOPK, TOPK]
    assert state._lens_cpu.tolist() == [TOPK, 0, 0, 0]
    state.plan(0, torch.empty(0, dtype=torch.int32))
    assert state._kv_cpu.tolist() == [0, 0, 0, 0, 0]
    assert state._lens_cpu.tolist() == [0, 0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("width", [128, 2048, 2176])
def test_pack_indices_replays_with_updated_offsets(width):
    """Graph replay packs exact prefixes after row lengths and slots change."""
    state = sm90_mod._SM90State.__new__(sm90_mod._SM90State)
    state.kv_indptr = torch.zeros(5, dtype=torch.int32, device="cuda")
    state.kv_indices = torch.full((4 * width,), -99, dtype=torch.int32, device="cuda")
    # Exercise a contiguous view with an unaligned starting address.
    slots = torch.arange(4 * width + 1, dtype=torch.int32, device="cuda")[1:]
    slots = slots.view(4, width)
    state.kv_indptr.copy_(torch.tensor([0, 1, 4, 4, 4]))
    state.pack_indices(slots)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.pack_indices(slots)

    for lengths in ([1, 3, 0, 0], [width, 0, 17, width - 1], [0, 0, 0, 0]):
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
        state.kv_indptr.copy_(offsets)
        state.kv_indices.fill_(-99)
        slots.add_(1)
        graph.replay()
        expected = torch.cat(
            [slots[row, :length] for row, length in enumerate(lengths)]
        )
        torch.testing.assert_close(state.kv_indices[: expected.numel()], expected)
        assert (state.kv_indices[expected.numel() :] == -99).all()


def test_kv_lens_host_formula():
    """Per-row host lengths: context == position + 1; capped at
    index_topk + trailing-pool remainder past the sparse threshold."""
    builder = object.__new__(FlashInferMLASparseSM90Builder)
    builder._index_topk = 2048
    builder._index_kpool = 4
    builder._async_scheduling = False
    cam = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 5, 7, 10], dtype=torch.int32),
        seq_lens=torch.tensor([100, 9, 3000], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100, 9, 3000], dtype=torch.int32),
        positions=None,
    )
    num_rows, lens = builder._kv_lens_host(cam)
    assert num_rows == 10
    # req0: positions 95..99 -> ctx 96..100 (all <= 2048: full context)
    # req1: positions 7,8 -> ctx 8,9
    # req2: positions 2997..2999 -> ctx 2998..3000 (> 2048: topk + ctx%4)
    assert lens.tolist() == [96, 97, 98, 99, 100, 8, 9, 2050, 2051, 2048]


def test_kv_lens_host_empty():
    builder = object.__new__(FlashInferMLASparseSM90Builder)
    builder._index_topk = 2048
    builder._index_kpool = 4
    cam = SimpleNamespace(
        num_reqs=0,
        query_start_loc_cpu=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.zeros(0, dtype=torch.int32),
    )
    num_rows, lens = builder._kv_lens_host(cam)
    assert num_rows == 0 and lens.numel() == 0


def test_supports_combination_gates(monkeypatch, default_vllm_config):
    monkeypatch.setattr(sm90_mod, "has_flashinfer_sm90_nope_mla", lambda: True)
    call = lambda **kw: FlashInferMLASparseSM90Backend.supports_combination(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_e4m3",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=SimpleNamespace(major=9),
        **kw,
    )
    assert call() is None  # no model config: only the feature gate applies

    import vllm.config as cfg

    monkeypatch.setattr(
        cfg,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )
    assert call() is None
    monkeypatch.setattr(sm90_mod, "has_flashinfer_sm90_nope_mla", lambda: False)
    assert "requires FlashInfer" in (call() or "")
