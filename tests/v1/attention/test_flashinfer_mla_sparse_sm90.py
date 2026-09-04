# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FlashInfer SM90 sparse MLA backend wiring (no GPU).

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
        self.wrapper = FakeWrapper()
        self.plan_calls = []

    def plan(self, num_tokens, kv_lens):
        self.plan_calls.append((num_tokens, kv_lens))


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
    width = TOPK
    got_slots = state.kv_indices[: rows * width].view(rows, width)
    for t in range(rows):
        k = int(ref_counts[t])
        assert got_slots[t, :k].tolist() == ref_slots[t, :k].tolist()
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


def test_builder_attaches_its_state(monkeypatch):
    builder = object.__new__(FlashInferMLASparseSM90Builder)
    builder._index_topk = 2048
    builder._index_kpool = 4
    builder._async_scheduling = False
    builder.state = FakeState(TOPK)
    metadata = object.__new__(sm90_mod.FlashInferMLASparseSM90Metadata)
    metadata.state = None
    monkeypatch.setattr(
        sm90_mod.FlashInferMLASparseMetadataBuilder,
        "build",
        lambda *_args, **_kwargs: metadata,
    )
    cam = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
        positions=None,
    )

    result = builder.build(0, cam)

    assert result.state is builder.state
    assert builder.state.plan_calls[0][0] == 1
    assert builder.state.plan_calls[0][1].tolist() == [1]


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
    assert kv.tolist() == [i * TOPK for i in (0, 1, 2, 3, 3)]
    assert kv_len.tolist() == [2, 5, 7, TOPK]  # padded row keeps full width
    assert (heads, ckv, kpe, page, causal) == (4, HEAD, 64, 1, False)
    assert scale == 576**-0.5
    assert kwargs["q_data_type"] == torch.bfloat16
    assert kwargs["kv_data_type"] == torch.bfloat16


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
