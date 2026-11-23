# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1 DSA packed input projection (VLLM_K3_DSA_PACKED_PROJ) on CPU.

The packed [q_a | kv_a | gate | wk | wv] GEMM against the separate GEMMs it
replaces (bitwise: CPU matmul of the same rows), the weight-loader path into
the packed rows, the quantized/odd-layout fallbacks, the wrappers' packed
forwards against the plain ``MultiHeadLatentAttentionWrapper.forward`` (with
stub attention), the multi-stream serial path and the post-load hook.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.amd import dsa

# Small stand-ins for K3.1-Pro TP8 (hidden 7168, q_lora 1536, kv_lora 512,
# rope 64, gate 12 x 128, index head dim 128, 64 index heads).
HIDDEN = 96
Q_LORA = 48
KV_LORA = 32
ROPE = 8
NUM_HEADS = 2
NOPE = 8
V_HEAD = 16
GATE = NUM_HEADS * V_HEAD
INDEX_DIM = 16
INDEX_HEADS = 4


def _cpu_gemm(layer: nn.Module) -> nn.Module:
    # The CPU UnquantizedLinearMethod GEMM reads layer.cpu_linear (set by its
    # post-load dispatch, which also drops the weight; not wanted here).
    layer.cpu_linear = F.linear
    return layer


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return {
        "q_a_proj": rand(Q_LORA, HIDDEN),
        "kv_a_proj_with_mqa": rand(KV_LORA + ROPE, HIDDEN),
        "g_proj": rand(GATE, HIDDEN),
        "wk_index": rand(INDEX_DIM, HIDDEN),
        "wv_index": rand(INDEX_HEADS, HIDDEN),
    }


class _Parts:
    """The four DSA input projections, built and loaded as vLLM does."""

    def __init__(self) -> None:
        self.fused_qkv_a_proj = _cpu_gemm(
            MergedColumnParallelLinear(
                HIDDEN,
                [Q_LORA, KV_LORA + ROPE],
                bias=False,
                params_dtype=torch.bfloat16,
                prefix="layers.0.self_attn.fused_qkv_a_proj",
                disable_tp=True,
            )
        )
        self.g_proj = _cpu_gemm(
            ColumnParallelLinear(
                HIDDEN,
                GATE,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix="layers.0.self_attn.g_proj",
            )
        )
        self.wk_index = _cpu_gemm(
            ReplicatedLinear(
                HIDDEN,
                INDEX_DIM,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix="layers.0.self_attn.dsa_indexer.wk_index",
            )
        )
        self.wv_index = _cpu_gemm(
            ReplicatedLinear(
                HIDDEN,
                INDEX_HEADS,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix="layers.0.self_attn.dsa_indexer.wv_index",
            )
        )

    def load(self, ckpt: dict[str, torch.Tensor]) -> None:
        """The checkpoint loader's calls: q_a/kv_a as fused shards 0/1."""
        w = self.fused_qkv_a_proj.weight
        w.weight_loader(w, ckpt["q_a_proj"], 0)
        w.weight_loader(w, ckpt["kv_a_proj_with_mqa"], 1)
        for name in ("g_proj", "wk_index", "wv_index"):
            p = getattr(self, name).weight
            p.weight_loader(p, ckpt[name])

    def layers(self, gate: bool = True) -> list[nn.Module]:
        layers = [self.fused_qkv_a_proj]
        if gate:
            layers.append(self.g_proj)
        return layers + [self.wk_index, self.wv_index]


@pytest.fixture
def parts(dist_init) -> _Parts:
    p = _Parts()
    p.load(_checkpoint(0))
    return p


def _x(num_tokens: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, HIDDEN, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 3, 17, 64])
def test_packed_gemm_matches_separate(parts: _Parts, gate: bool, num_tokens: int):
    layers = parts.layers(gate)
    x = _x(num_tokens)
    ref = [layer(x)[0] for layer in layers]
    assert dsa.packed_proj_unsupported_reason(layers) is None

    packed = dsa.pack_linear_weights(layers)
    assert packed.shape == (sum(r.shape[1] for r in ref), HIDDEN)
    assert packed.is_contiguous()
    # Each parameter is a row view of the packed weight (no extra memory).
    offset = 0
    for layer in layers:
        w = layer.weight
        assert isinstance(w, nn.Parameter)
        assert w.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert w.data_ptr() == packed[offset].data_ptr()
        assert w.is_contiguous()
        offset += w.shape[0]

    outs = F.linear(x, packed).split([r.shape[1] for r in ref], dim=-1)
    for out, r in zip(outs, ref):
        assert torch.equal(out, r)
    # The parts' own GEMMs still read the same values.
    for layer, r in zip(layers, ref):
        assert torch.equal(layer(x)[0], r)


def test_weight_reload_updates_packed_rows(parts: _Parts):
    layers = parts.layers()
    packed = dsa.pack_linear_weights(layers)
    # Repeated post-load calls keep the packed tensor.
    assert dsa.pack_linear_weights(layers, packed) is packed

    ckpt = _checkpoint(7)
    parts.load(ckpt)
    assert dsa.pack_linear_weights(layers, packed) is packed
    expected = torch.cat(
        [
            ckpt["q_a_proj"],
            ckpt["kv_a_proj_with_mqa"],
            ckpt["g_proj"],
            ckpt["wk_index"],
            ckpt["wv_index"],
        ]
    )
    assert torch.equal(packed, expected)

    # A reload that swaps a parameter's storage gets a fresh pack.
    parts.wk_index.weight.data = parts.wk_index.weight.data.clone()
    repacked = dsa.pack_linear_weights(layers, packed)
    assert repacked is not packed
    assert torch.equal(repacked, expected)


def test_unsupported_parts_fall_back(parts: _Parts, monkeypatch):
    layers = parts.layers()
    assert dsa.packed_proj_unsupported_reason(layers) is None

    w = parts.g_proj.weight
    saved = w.data
    w.data = saved.to(torch.float8_e4m3fn)
    assert "float8" in dsa.packed_proj_unsupported_reason(layers)
    w.data = saved.t().contiguous().t()
    assert "contiguous" in dsa.packed_proj_unsupported_reason(layers)
    w.data = saved

    class _OtherMethod(UnquantizedLinearMethod):
        pass

    method = parts.wk_index.quant_method
    parts.wk_index.quant_method = _OtherMethod()
    assert "_OtherMethod" in dsa.packed_proj_unsupported_reason(layers)
    parts.wk_index.quant_method = method

    parts.wv_index.bias = nn.Parameter(torch.zeros(INDEX_HEADS))
    assert "bias" in dsa.packed_proj_unsupported_reason(layers)
    parts.wv_index.bias = None

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert "BATCH_INVARIANT" in dsa.packed_proj_unsupported_reason(layers)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    # Any active offloader (UVA or prefetch) re-points the parts' storage.
    monkeypatch.setattr(dsa, "get_offloader", lambda: object())
    assert "offloaded" in dsa.packed_proj_unsupported_reason(layers)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), 1e-6).to(
            x.dtype
        )


class _Indexer(nn.Module):
    """Records the indexer's key/weight inputs (running wk/wv when not given),
    as ``KimiK3DSAIndexer.forward`` does."""

    topk_tokens = 4

    def __init__(self, wk: nn.Module, wv: nn.Module) -> None:
        super().__init__()
        self.wk_index, self.wv_index = wk, wv
        self.index_head_dim, self.index_num_heads = INDEX_DIM, INDEX_HEADS
        self.k_index_norm = _RMSNorm(INDEX_DIM)
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def forward(
        self, hidden, q_c, positions=None, rotary_emb=None, index_k=None, index_v=None
    ):
        if index_k is None:
            index_k = self.wk_index(hidden)[0]
        if index_v is None:
            index_v = self.wv_index(hidden)[0]
        self.calls.append((q_c.clone(), index_k.clone(), index_v.clone()))
        return None


class _MLAAttn:
    """Stub attention: deterministic in (q, kv_c_normed, k_pe)."""

    def __init__(self) -> None:
        self.impl = SimpleNamespace(record_logical_topk_ready=lambda: None)
        self.layer_name = "layers.0.self_attn.attn"

    def __call__(self, q, kv_c_normed, k_pe, output_shape, q_dcp_replicated=None):
        t = q.shape[0]
        mix = q.float().sum(-1) + kv_c_normed.float().sum(-1, keepdim=True)
        mix = mix + k_pe.float().sum(-1)
        out = mix.repeat_interleave(V_HEAD, dim=-1).reshape(output_shape)
        return out[:t].to(q.dtype)


class _FakeWrapper(dsa._PackedProjMixin):
    """The attributes ``MultiHeadLatentAttentionWrapper.forward`` reads."""

    def __init__(self, parts: _Parts, gate: bool) -> None:
        torch.manual_seed(3)
        self.q_lora_rank = Q_LORA
        self.kv_lora_rank = KV_LORA
        self.qk_rope_head_dim = ROPE
        self.qk_nope_head_dim = NOPE
        self.qk_head_dim = NOPE + ROPE
        self.v_head_dim = V_HEAD
        self.num_heads = NUM_HEADS
        self.fused_qkv_a_proj = parts.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = None
        self.q_a_layernorm = _RMSNorm(Q_LORA)
        self.kv_a_layernorm = _RMSNorm(KV_LORA)
        self.q_b_proj = _cpu_gemm(
            ReplicatedLinear(
                Q_LORA,
                NUM_HEADS * (NOPE + ROPE),
                bias=False,
                params_dtype=torch.bfloat16,
            )
        )
        nn.init.normal_(self.q_b_proj.weight)
        self.q_proj = None
        self.o_proj = _cpu_gemm(
            ReplicatedLinear(
                NUM_HEADS * V_HEAD, HIDDEN, bias=False, params_dtype=torch.bfloat16
            )
        )
        nn.init.normal_(self.o_proj.weight)
        self.g_proj = parts.g_proj if gate else None
        self.indexer = _Indexer(parts.wk_index, parts.wv_index)
        self.indexer_rope_emb = None
        self.rotary_emb = None
        self.is_sparse = True
        self.skip_topk = False
        self.fuse_qkv_rmsnorm = False
        self.fuse_output_gate = False
        self.dcp_q_replicate = False
        self.mla_attn = _MLAAttn()
        self._init_packed_proj(include_gate=True)


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 5, 33])
def test_packed_wrapper_forward_matches_plain(
    parts: _Parts, gate: bool, num_tokens: int
):
    fake = _FakeWrapper(parts, gate)
    x = _x(num_tokens, seed=num_tokens)
    positions = torch.arange(num_tokens)
    ref = MultiHeadLatentAttentionWrapper.forward(fake, positions, x)
    ref_attn = MultiHeadLatentAttentionWrapper.forward(
        fake, positions, x, skip_o_proj=True
    )
    ref_calls = list(fake.indexer.calls)
    fake.indexer.calls.clear()

    fake.pack_projections()
    assert fake._packed_w is not None
    split = [Q_LORA, KV_LORA + ROPE] + ([GATE] if gate else [])
    assert fake._packed_split == split + [INDEX_DIM, INDEX_HEADS]
    out = dsa.KimiK3DSAPackedMLAWrapper.forward(fake, positions, x)
    out_attn = dsa.KimiK3DSAPackedMLAWrapper.forward(
        fake, positions, x, skip_o_proj=True
    )
    assert torch.equal(out, ref)
    assert torch.equal(out_attn, ref_attn)
    assert len(fake.indexer.calls) == len(ref_calls) == 2
    for got, want in zip(fake.indexer.calls, ref_calls):
        for a, b in zip(got, want):
            assert torch.equal(a, b)


def _ref_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps, max_tokens=None):
    """CPU reference of fused_q_kv_rmsnorm / mla_bounded_q_kv_rmsnorm."""

    def norm(x, w):
        return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(x.dtype)

    return norm(q, q_weight), norm(kv, kv_weight)


def _ref_output_gate(attn_out, gate, fused=False):
    if fused:
        # The fused kernel reads the gate rows with unit column stride.
        assert gate.stride(-1) == 1
        return (attn_out.float() * gate.float().sigmoid()).to(attn_out.dtype)
    return attn_out * gate.sigmoid()


@pytest.mark.parametrize("max_tokens", [None, 4])
@pytest.mark.parametrize("fuse_gate", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_packed_wrapper_fused_norm_gate_matches_plain(
    parts: _Parts, monkeypatch, max_tokens: int | None, fuse_gate: bool, num_tokens
):
    """VLLM_K3_DSA_FUSED_NORM_GATE=1 (the recipe default): the fused and the
    token-bounded q_a/kv_a RMSNorm and the fused output gate on the packed
    views, with the GPU kernels replaced by CPU references."""
    from vllm.model_executor.layers import mla as mla_mod

    for mod in (mla_mod, dsa):
        monkeypatch.setattr(mod, "fused_q_kv_rmsnorm", _ref_q_kv_rmsnorm)
        monkeypatch.setattr(mod, "mla_output_gate", _ref_output_gate)
    monkeypatch.setattr(
        torch.ops.vllm, "mla_bounded_q_kv_rmsnorm", _ref_q_kv_rmsnorm, raising=False
    )
    fake = _FakeWrapper(parts, gate=True)
    fake.q_a_layernorm.variance_epsilon = 1e-6
    fake.kv_a_layernorm.variance_epsilon = 1e-6
    fake.fuse_qkv_rmsnorm = True
    fake.fuse_qkv_rmsnorm_max_tokens = max_tokens
    fake.fuse_output_gate = fuse_gate
    x = _x(num_tokens, seed=num_tokens)
    positions = torch.arange(num_tokens)
    ref = MultiHeadLatentAttentionWrapper.forward(fake, positions, x)
    ref_calls = list(fake.indexer.calls)
    fake.indexer.calls.clear()

    fake.pack_projections()
    assert fake._packed_w is not None
    out = dsa.KimiK3DSAPackedMLAWrapper.forward(fake, positions, x)
    assert torch.equal(out, ref)
    for got, want in zip(fake.indexer.calls, ref_calls, strict=True):
        for a, b in zip(got, want):
            assert torch.equal(a, b)


class _StridedAttn(_MLAAttn):
    """Non-contiguous output: ``_ms_join`` takes its torch path on CPU."""

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).t().contiguous().t()


class _FakeMSWrapper(_FakeWrapper):
    """``KimiK3DSAMultiStreamMLAWrapper``'s state on the fake wrapper."""

    def __init__(self, parts: _Parts, name: str) -> None:
        super().__init__(parts, gate=True)
        self.mla_attn = _StridedAttn()
        self.mla_attn.layer_name = name
        self.ms_name = name
        self.ms_state = dsa._MSState()
        self.ms_comm = None
        self._init_packed_proj(include_gate=False)


@pytest.mark.parametrize("num_tokens", [1, 6])
def test_ms_wrapper_forward_packed_matches_unpacked(
    parts: _Parts, monkeypatch, num_tokens: int
):
    """``KimiK3DSAMultiStreamMLAWrapper.forward`` through the registered
    fork/indexer/join ops (serial path: no aux comm), packed vs unpacked."""

    class _GateMul:
        def __getitem__(self, grid):
            def run(attn_out, gate, out, n, BLOCK):
                gated = attn_out.float() * gate.float().sigmoid()
                out.view(-1).copy_(gated.to(out.dtype).view(-1))

            return run

    monkeypatch.setattr(dsa, "_gate_mul_kernel", _GateMul())
    name = f"layers.0.self_attn.attn.test_ms_forward_{num_tokens}"
    fake = _FakeMSWrapper(parts, name)
    dsa._MS_LAYERS[name] = fake
    try:
        x = _x(num_tokens, seed=num_tokens)
        positions = torch.arange(num_tokens)
        ref = dsa.KimiK3DSAMultiStreamMLAWrapper.forward(fake, positions, x)
        ref_calls = list(fake.indexer.calls)
        fake.indexer.calls.clear()

        fake.pack_projections()
        assert fake._packed_split == [Q_LORA, KV_LORA + ROPE, INDEX_DIM, INDEX_HEADS]
        out = dsa.KimiK3DSAMultiStreamMLAWrapper.forward(fake, positions, x)
        assert torch.equal(out, ref)
        for got, want in zip(fake.indexer.calls, ref_calls, strict=True):
            for a, b in zip(got, want):
                assert torch.equal(a, b)
        assert fake.ms_state.raw_kv is None
    finally:
        del dsa._MS_LAYERS[name]


def _sparse_impl(monkeypatch):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as be

    parent = be.ROCMAiterMLASparseImpl.__mro__[1]
    monkeypatch.setattr(parent, "process_weights_after_loading", lambda self, d: None)
    impl = object.__new__(be.ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    return impl


def test_install_packed_proj_hook(parts: _Parts, monkeypatch):
    """KimiK3DSAAttention's wiring: the hook goes on the sparse MLA impl and
    packs from its process_weights_after_loading; other backends and (without
    VLLM_K3_DSA_PACKED_PROJ_MS) the multi-stream wrapper stay unpacked."""
    monkeypatch.setattr(dsa, "_PACKED_PROJ_LAYERS", {})
    monkeypatch.delenv("VLLM_K3_DSA_PACKED_PROJ_MS", raising=False)

    # Any other backend: no hook attribute, nothing installed.
    other = _FakeWrapper(parts, gate=True)
    assert not dsa.install_packed_proj_hook(other)
    assert not hasattr(other.mla_attn.impl, "_dsa_post_load")

    fake = _FakeWrapper(parts, gate=True)
    impl = _sparse_impl(monkeypatch)
    fake.mla_attn.impl = impl
    assert dsa.install_packed_proj_hook(fake)
    assert {fake.mla_attn.layer_name: None} == dsa._PACKED_PROJ_LAYERS
    impl.process_weights_after_loading(torch.bfloat16)
    assert fake._packed_w is not None
    assert {fake.mla_attn.layer_name: ""} == dsa._PACKED_PROJ_LAYERS

    # The multi-stream wrapper packs only with VLLM_K3_DSA_PACKED_PROJ_MS=1.
    monkeypatch.setattr(dsa, "KimiK3DSAMultiStreamMLAWrapper", _FakeMSWrapper)
    ms = _FakeMSWrapper(parts, "layers.1.self_attn.attn")
    ms.mla_attn.impl = _sparse_impl(monkeypatch)
    assert not dsa.install_packed_proj_hook(ms)
    assert ms.mla_attn.impl._dsa_post_load is None
    monkeypatch.setenv("VLLM_K3_DSA_PACKED_PROJ_MS", "1")
    assert dsa.install_packed_proj_hook(ms)
    assert ms.mla_attn.impl._dsa_post_load == ms.pack_projections


def test_pack_projections_quantized_part_keeps_plain_path(parts: _Parts):
    fake = _FakeWrapper(parts, gate=True)
    w = parts.wv_index.weight
    w.data = w.data.to(torch.float16)
    before = [layer.weight.data_ptr() for layer in fake._packed_parts]
    fake.pack_projections()
    assert fake._packed_w is None
    assert [layer.weight.data_ptr() for layer in fake._packed_parts] == before


def test_indexer_forward_takes_given_key_and_weights(parts: _Parts):
    """``KimiK3DSAIndexer.forward`` (FP8, non-DCP) with index_k/index_v skips
    wk/wv and feeds the given (strided) views to the same ops."""
    x = _x(6)
    proj = F.linear(
        x, torch.cat([parts.wk_index.weight.data, parts.wv_index.weight.data])
    )
    index_k, index_v = proj.split([INDEX_DIM, INDEX_HEADS], dim=-1)
    seen: dict[str, torch.Tensor] = {}

    def no_gemm(_):
        raise AssertionError("wk/wv must not run")

    def quantize(q, weights):
        seen["weights"] = weights
        return q, weights

    k_norm = _RMSNorm(INDEX_DIM)
    fake = SimpleNamespace(
        index_num_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM,
        wq_index=lambda t: (t[:, : INDEX_HEADS * INDEX_DIM],),
        wk_index=no_gemm,
        wv_index=no_gemm,
        use_fp4=False,
        k_index_norm=k_norm,
        _quantize_index_q_and_weights=quantize,
        dcp_scores=None,
        indexer_op=SimpleNamespace(
            forward_hip=lambda h, q, k, w: seen.setdefault("k", k)
        ),
    )
    dsa.KimiK3DSAIndexer.forward(fake, x, x, index_k=index_k, index_v=index_v)
    assert torch.equal(seen["k"], k_norm(parts.wk_index(x)[0]))
    assert torch.equal(seen["weights"], parts.wv_index(x)[0])


def test_ms_serial_path_uses_packed_key_and_weights(parts: _Parts):
    """Multi-stream wrapper outside its aux-stream decode (no aux comm): the
    packed fork hands wk/wv's views to the serial indexer call."""
    name = "layers.0.self_attn.attn.test_dsa_packing"
    indexer = _Indexer(parts.wk_index, parts.wv_index)
    layer = SimpleNamespace(
        ms_state=dsa._MSState(),
        ms_comm=None,
        indexer=indexer,
        g_proj=parts.g_proj,
    )
    dsa._MS_LAYERS[name] = layer
    try:
        layers = parts.layers(gate=False)
        x = _x(4)
        q_c = _x(4)[:, :Q_LORA].contiguous()
        packed = dsa.pack_linear_weights(layers)
        proj = F.linear(x, packed)
        token = dsa._ms_fork_packed(x, proj, name)
        assert token.numel() == 0 and layer.ms_state.stage == 0
        raw_k, raw_v = layer.ms_state.raw_kv
        assert torch.equal(raw_k, parts.wk_index(x)[0])
        assert torch.equal(raw_v, parts.wv_index(x)[0])

        gate = dsa._ms_indexer(x, q_c, token, name)
        assert layer.ms_state.raw_kv is None
        assert torch.equal(gate, parts.g_proj(x)[0])
        (got,) = indexer.calls
        assert torch.equal(got[0], q_c)
        assert torch.equal(got[1], raw_k) and torch.equal(got[2], raw_v)
    finally:
        del dsa._MS_LAYERS[name]


def test_sparse_impl_post_load_hook_runs(monkeypatch):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as be

    parent = be.ROCMAiterMLASparseImpl.__mro__[1]
    monkeypatch.setattr(parent, "process_weights_after_loading", lambda self, d: None)
    impl = object.__new__(be.ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    # No hook set: unchanged.
    be.ROCMAiterMLASparseImpl.process_weights_after_loading(impl, torch.bfloat16)
    calls = []
    impl._dsa_post_load = lambda: calls.append(1)
    be.ROCMAiterMLASparseImpl.process_weights_after_loading(impl, torch.bfloat16)
    assert calls == [1]


def test_packed_proj_flag_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_K3_DSA_PACKED_PROJ", raising=False)
    assert not dsa.packed_proj_requested()
    monkeypatch.setenv("VLLM_K3_DSA_PACKED_PROJ", "0")
    assert not dsa.packed_proj_requested()
    monkeypatch.setenv("VLLM_K3_DSA_PACKED_PROJ", "1")
    assert dsa.packed_proj_requested()
    monkeypatch.delenv("VLLM_K3_DSA_PACKED_PROJ_MS", raising=False)
    assert not dsa.packed_proj_ms_requested()
