# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiered context-KV projection for quantized DFlash drafters (#51581).

Pins down:
- the strategy decision (fused / scaled_mm / dequant / per_layer),
- the per-layer fallback's output layout,
- the invariant that packed weights never reach a bare ``F.linear``,
- the real-FP8 drafter precompute on GPU (per-layer / dequant / scaled_mm).
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.models.qwen3_dflash import (
    ContextKVStrategy,
    DFlashQwen3Model,
    _decide_context_kv_strategy,
)

HIDDEN = 32
HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_LAYERS = 3
NUM_CTX = 5
Q_SIZE = NUM_HEADS * HEAD_DIM      # 32
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 16


def _cpu_rms_norm(out, input, weight, eps):
    """Reference RMSNorm writing into ``out`` (vLLM's fused kernel is CUDA-only).
    Handles 1-D ([H]) and grouped [L, H] weights."""
    var = input.float().pow(2).mean(-1, keepdim=True)
    normed = (input * (var + eps).rsqrt()).to(input.dtype)
    if weight.ndim == 2 and input.ndim >= 3:
        weight = weight.reshape(weight.shape[0], *([1] * (input.ndim - 2)),
                                weight.shape[1])
    out.copy_(normed * weight)


@pytest.fixture
def cpu_rms_norm(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash as mod

    monkeypatch.setattr(mod.ops, "rms_norm", _cpu_rms_norm)


def _ints(*shape):
    """Small-integer float32 data: additions are exact for bit-level compares."""
    return torch.randint(-3, 4, shape).float()


class _PackedLinearMethod(LinearMethodBase):
    """Placeholder for an arbitrary weight-quantized method (layout-agnostic)."""

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, x, bias=None):
        out = torch.matmul(x, layer.unpacked_weight.t())
        return out if bias is None else out + bias


class _Projection(nn.Module):
    """Stand-in for QKVParallelLinear. A quantized ``.weight`` access raises,
    mirroring GPTQ/AWQ packed schemes where that attribute does not exist."""

    def __init__(self, weight, bias, quant_method):
        super().__init__()
        self.unpacked_weight = weight
        self.bias = bias
        self.quant_method = quant_method
        self.forward_calls = 0

    @property
    def weight(self):
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise AssertionError("fused precompute must not slice a quantized weight")
        return self.unpacked_weight

    def forward(self, x):
        self.forward_calls += 1
        return self.quant_method.apply(self, x, self.bias), None


def _attn(proj):
    return SimpleNamespace(
        qkv_proj=proj,
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        k_norm=SimpleNamespace(weight=torch.ones(HEAD_DIM)),
    )


def _build_model(quantized: bool, *, seed: int = 0):
    torch.manual_seed(seed)
    weights = [_ints(Q_SIZE + 2 * KV_SIZE, HIDDEN) for _ in range(NUM_LAYERS)]
    method = _PackedLinearMethod() if quantized else UnquantizedLinearMethod()
    projections = [_Projection(w, None, method) for w in weights]
    layers_attn = [_attn(p) for p in projections]

    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(weight=torch.ones(HIDDEN))
    model._rms_norm_eps = 1e-6
    model.compute_dtype = torch.float32
    model.layers = [
        SimpleNamespace(input_layernorm=SimpleNamespace(weight=torch.ones(HIDDEN)))
        for _ in layers_attn
    ]
    model._build_context_kv_buffers(layers_attn, has_bias=False)
    return model, projections


def _project(model):
    return model._project_context_kv(
        _ints(NUM_CTX, HIDDEN), NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )


def test_decide_strategy_unquantized():
    p = _Projection(_ints(4, 4), None, UnquantizedLinearMethod())
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.FUSED


def test_decide_strategy_quantized_and_mixed():
    p = _Projection(_ints(4, 4), None, _PackedLinearMethod())
    u = _Projection(_ints(4, 4), None, UnquantizedLinearMethod())
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.PER_LAYER
    # Mixed (one quantized, one not) -> conservative per-layer.
    assert _decide_context_kv_strategy([u, p]) is ContextKVStrategy.PER_LAYER


def _simple_fp8(*, use_marlin=False, block_quant=False, use_deep_gemm=False):
    """Real ``Fp8LinearMethod`` without running ``__init__`` (needs a config /
    layer); only the attributes the strategy check reads are set."""
    m = object.__new__(Fp8LinearMethod)
    m.use_marlin = use_marlin
    m.block_quant = block_quant
    m.use_deep_gemm = use_deep_gemm
    return m


def test_decide_strategy_simple_fp8_gates_on_cutlass(monkeypatch):
    """Simple FP8 (non-Marlin) -> SCALED_MM when cutlass-fp8 is available,
    FUSED_DEQUANT otherwise."""
    from vllm.model_executor.models import qwen3_dflash as mod

    p = _Projection(_ints(4, 4), None, _simple_fp8())
    monkeypatch.setattr(mod, "cutlass_fp8_supported", lambda: True)
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.SCALED_MM
    monkeypatch.setattr(mod, "cutlass_fp8_supported", lambda: False)
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.FUSED_DEQUANT


def test_decide_strategy_deep_gemm_goes_per_layer():
    """FP8 on a DeepGEMM path transforms its scales -> PER_LAYER."""
    p = _Projection(_ints(4, 4), None, _simple_fp8(use_deep_gemm=True))
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.PER_LAYER


def test_decide_strategy_marlin_goes_per_layer():
    """Marlin (weight-only FP8) cannot be fused -> PER_LAYER."""
    p = _Projection(_ints(4, 4), None, _simple_fp8(use_marlin=True))
    assert _decide_context_kv_strategy([p, p]) is ContextKVStrategy.PER_LAYER


def test_fused_matches_per_layer_unquantized(cpu_rms_norm):
    """Unquantized: the FUSED fused path and the PER_LAYER path agree
    bit-for-bit and produce the same [L, num_ctx, nkv, hd] layout."""
    fused_model, _ = _build_model(False)
    assert fused_model._kv_strategy is ContextKVStrategy.FUSED
    per_model, projections = _build_model(True)
    assert per_model._kv_strategy is ContextKVStrategy.PER_LAYER

    torch.manual_seed(1234)
    fk, fv = _project(fused_model)
    torch.manual_seed(1234)
    pk, pv = _project(per_model)

    assert all(p.forward_calls == 1 for p in projections)
    for fused, per in ((fk, pk), (fv, pv)):
        assert fused.shape == per.shape == \
            (NUM_LAYERS, NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
        assert torch.equal(fused, per)


def test_quantized_projection_is_never_sliced(monkeypatch, cpu_rms_norm):
    """Regression #51581: packed weights must never reach F.linear."""
    model, _ = _build_model(True)
    assert model._kv_strategy is ContextKVStrategy.PER_LAYER
    assert not hasattr(model, "_fused_kv_weight")

    monkeypatch.setattr(
        F,
        "linear",
        lambda *a, **kw: pytest.fail("F.linear reached a packed weight"),
    )
    try:
        all_k, all_v = _project(model)
    finally:
        monkeypatch.undo()
    assert all_k.shape == (NUM_LAYERS, NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
    assert all_v.shape == all_k.shape


# ---------------------------------------------------------------------------
# GPU: real FP8 drafter (the issue's reproduction scenario)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_drafter_precompute(dist_init, default_vllm_config):
    """Real FP8 QKV projections -> the strategy follows the platform:
    - Marlin (A100/sm80): PER_LAYER
    - cutlass-fp8 (sm89+): SCALED_MM
    - non-Marlin, no cutlass-fp8: FUSED_DEQUANT

    In every case the projected K/V must match the per-layer
    ``quant_method.apply`` ground truth within quantization tolerance.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        cutlass_fp8_supported,
    )

    dtype, device = torch.bfloat16, "cuda"
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
    )

    def _make_proj():
        proj = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS,
            total_num_kv_heads=NUM_KV_HEADS,
            bias=False,
            params_dtype=dtype,
            quant_config=quant_config,
        ).to(device)
        # Fill real fp8 weights + per-shard scales (offline-serialized format).
        for name, p in proj.named_parameters():
            if p.dtype == torch.float8_e4m3fn:
                w = torch.randn(p.shape, dtype=torch.float32, device=device)
                p.data.copy_(
                    (w / w.abs().max() * 200).to(torch.float8_e4m3fn))
        proj.weight_scale.data.copy_(
            torch.tensor([1e-3, 1e-3, 1e-3], device=device))
        proj.quant_method.process_weights_after_loading(proj)
        return proj

    projections = [_make_proj() for _ in range(NUM_LAYERS)]
    layers_attn = [_attn(p) for p in projections]

    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(
        weight=torch.ones(HIDDEN, device=device, dtype=dtype))
    model._rms_norm_eps = 1e-6
    model.compute_dtype = dtype
    model.layers = [SimpleNamespace() for _ in layers_attn]
    model._build_context_kv_buffers(layers_attn, has_bias=False)

    use_marlin = any(
        getattr(p.quant_method, "use_marlin", False) for p in projections
    )
    if use_marlin:
        expected = ContextKVStrategy.PER_LAYER
    elif cutlass_fp8_supported():
        expected = ContextKVStrategy.SCALED_MM
    else:
        expected = ContextKVStrategy.FUSED_DEQUANT
    assert model._kv_strategy is expected, model._kv_strategy

    context_states = torch.randn(NUM_CTX, HIDDEN, device=device, dtype=dtype)
    all_k, all_v = model._project_context_kv(
        context_states, NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )

    # Per-layer ground truth (goes through quant_method.apply).
    normed = torch.empty_like(context_states)
    import vllm._custom_ops as ops

    ops.rms_norm(normed, context_states, model.hidden_norm.weight, 1e-6)
    per_k = []
    per_v = []
    for p in projections:
        out = p.quant_method.apply(p, normed, bias=None)
        per_k.append(out[..., Q_SIZE:Q_SIZE + KV_SIZE].view(
            NUM_CTX, NUM_KV_HEADS, HEAD_DIM))
        per_v.append(out[..., Q_SIZE + KV_SIZE:].view(
            NUM_CTX, NUM_KV_HEADS, HEAD_DIM))
    torch.testing.assert_close(all_k, torch.stack(per_k), rtol=0.02, atol=0.02)
    torch.testing.assert_close(all_v, torch.stack(per_v), rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_compute_dtype_no_crash(dist_init, default_vllm_config):
    """P1 regression guard: the fused buffer must use the model dtype
    (an fp16 engine must not crash)."""
    from vllm.model_executor.layers.linear import QKVParallelLinear

    def _proj():
        p = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS,
            total_num_kv_heads=NUM_KV_HEADS,
            bias=False,
            params_dtype=torch.float16,
        ).cuda()
        p.weight.data.normal_(std=0.1)
        return p

    projections = [_proj() for _ in range(NUM_LAYERS)]
    layers_attn = [_attn(p) for p in projections]
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(
        weight=torch.ones(HIDDEN, device="cuda", dtype=torch.float16))
    model._rms_norm_eps = 1e-6
    model.compute_dtype = torch.float16
    model.layers = [SimpleNamespace() for _ in layers_attn]
    model._build_context_kv_buffers(layers_attn, has_bias=False)

    assert model._kv_strategy is ContextKVStrategy.FUSED
    assert model._fused_kv_weight.dtype == torch.float16
    context_states = torch.randn(
        NUM_CTX, HIDDEN, device="cuda", dtype=torch.float16
    )
    all_k, all_v = model._project_context_kv(
        context_states, NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )
    assert all_k.dtype == all_v.dtype == torch.float16
