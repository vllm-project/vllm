# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-KV precompute for weight-quantized DFlash/DSpark drafters.

The precompute fuses every layer's KV projection into one GEMM built by slicing
raw ``.weight`` tensors, which is only meaningful while those projections are
unquantized. These tests pin the gate that detects that, the per-layer fallback's
output layout, and the fact that a quantized projection is never sliced.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.gemma4_dspark import Gemma4DSparkModel
from vllm.model_executor.models.laguna_dflash import DFlashLagunaModel
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

HIDDEN = 32
HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_LAYERS = 3
NUM_CTX = 5
EPS = 1e-6

Q_SIZE = NUM_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM


def _copy_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Stand-in for the fused RMSNorm kernel, which needs an accelerator.

    Every assertion below compares two projection paths over the same normed
    input, so the norm cancels out; copying keeps the comparison exact.
    """
    out.copy_(x)


def _ints(*shape: int) -> torch.Tensor:
    """Small integer-valued float32 data.

    Sums of these are exact in float32 regardless of accumulation order, so the
    fused GEMM and the per-layer GEMMs are comparable bit for bit.
    """
    return torch.randint(-3, 4, shape).float()


class _PackedLinearMethod(LinearMethodBase):
    """Stands in for any weight-quantized method.

    The gate only asks whether a projection is unquantized, so the specific
    scheme does not matter. ``apply`` avoids ``F.linear`` so tests can assert
    the fused path's ``F.linear`` is never reached.
    """

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, x, bias=None):
        out = torch.matmul(x, layer.unpacked_weight.t())
        return out if bias is None else out + bias


class _Projection(nn.Module):
    """Stand-in for QKVParallelLinear / ColumnParallelLinear.

    ``.weight`` raises while quantized, mirroring packed schemes (GPTQ/AWQ)
    where the attribute does not exist at all.
    """

    def __init__(self, weight, bias, quant_method):
        super().__init__()
        self.unpacked_weight = weight
        self.bias = bias
        self.quant_method = quant_method
        self.skip_bias_add = False
        self.forward_calls = 0

    @property
    def weight(self):
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise AssertionError("fused precompute must not slice a quantized weight")
        return self.unpacked_weight

    def forward(self, x):
        self.forward_calls += 1
        return self.quant_method.apply(self, x, self.bias), None


def _attn(proj, *, proj_attr: str) -> SimpleNamespace:
    return SimpleNamespace(
        **{proj_attr: proj},
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        k_norm=SimpleNamespace(weight=torch.ones(HEAD_DIM)),
    )


def _model(cls, layers_attn, *, has_bias: bool):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(weight=torch.ones(HIDDEN))
    model._rms_norm_eps = EPS
    model.layers = [
        SimpleNamespace(input_layernorm=SimpleNamespace(weight=torch.ones(HIDDEN)))
        for _ in layers_attn
    ]
    model._build_context_kv_buffers(layers_attn, has_bias)
    return model


def _project(model) -> tuple[torch.Tensor, torch.Tensor]:
    return model._project_context_kv(
        _ints(NUM_CTX, HIDDEN), NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )


def _build_qkv_case(quantized: bool, *, cls, has_bias: bool, seed: int = 0):
    """Build a qwen3/laguna-shaped model over deterministic shared weights."""
    torch.manual_seed(seed)
    weights = [_ints(Q_SIZE + 2 * KV_SIZE, HIDDEN) for _ in range(NUM_LAYERS)]
    biases = [
        _ints(Q_SIZE + 2 * KV_SIZE) if has_bias else None for _ in range(NUM_LAYERS)
    ]
    method = _PackedLinearMethod() if quantized else UnquantizedLinearMethod()
    projections = [_Projection(w, b, method) for w, b in zip(weights, biases)]
    layers_attn = [_attn(p, proj_attr="qkv_proj") for p in projections]
    return _model(cls, layers_attn, has_bias=has_bias), projections


def _build_k_proj_case(quantized: bool, *, has_bias: bool, seed: int = 0):
    """Build a gemma4_dspark-shaped model (k_proj only, V derived via k_eq_v)."""
    torch.manual_seed(seed)
    weights = [_ints(KV_SIZE, HIDDEN) for _ in range(NUM_LAYERS)]
    biases = [_ints(KV_SIZE) if has_bias else None for _ in range(NUM_LAYERS)]
    method = _PackedLinearMethod() if quantized else UnquantizedLinearMethod()
    projections = [_Projection(w, b, method) for w, b in zip(weights, biases)]
    layers_attn = [_attn(p, proj_attr="k_proj") for p in projections]
    return _model(Gemma4DSparkModel, layers_attn, has_bias=has_bias), projections


_CASES = {
    "qwen3_dflash": lambda q, b: _build_qkv_case(q, cls=DFlashQwen3Model, has_bias=b),
    "laguna_dflash": lambda q, b: _build_qkv_case(q, cls=DFlashLagunaModel, has_bias=b),
    "gemma4_dspark": lambda q, b: _build_k_proj_case(q, has_bias=b),
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("has_bias", [False, True])
def test_per_layer_fallback_matches_fused_path(monkeypatch, case, has_bias):
    """The fallback must reproduce the fused path exactly for unquantized weights."""
    monkeypatch.setattr(ops, "rms_norm", _copy_norm)
    build = _CASES[case]

    fused_model, _ = build(False, has_bias)
    fallback_model, projections = build(True, has_bias)

    assert fused_model._fuse_context_kv is True
    assert fallback_model._fuse_context_kv is False

    torch.manual_seed(1234)
    fused_k, fused_v = _project(fused_model)
    torch.manual_seed(1234)
    fallback_k, fallback_v = _project(fallback_model)

    assert all(p.forward_calls == 1 for p in projections)
    for fused, fallback in ((fused_k, fallback_k), (fused_v, fallback_v)):
        assert fallback.shape == fused.shape
        assert fallback.dtype == fused.dtype
        assert fallback.is_contiguous()
        assert torch.equal(fallback, fused)


@pytest.mark.parametrize("case", list(_CASES))
def test_quantized_projection_is_never_sliced(monkeypatch, case):
    """Regression for #51581: a packed weight must not reach the fused GEMM.

    Building the fused buffer used to read ``.weight`` and hand the packed
    tensor to ``F.linear``, which raised at engine init.
    """
    monkeypatch.setattr(ops, "rms_norm", _copy_norm)
    model, projections = _CASES[case](True, False)

    assert model._fuse_context_kv is False
    assert not hasattr(model, "_fused_kv_weight")
    assert not hasattr(model, "_fused_k_weight")
    assert not hasattr(model, "_kv_weights")

    monkeypatch.setattr(
        F, "linear", lambda *a, **kw: pytest.fail("F.linear reached a packed weight")
    )
    all_k, all_v = _project(model)

    assert all(p.forward_calls == 1 for p in projections)
    assert all_k.shape == (NUM_LAYERS, NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
    assert all_v.shape == all_k.shape


def test_gate_ignores_layers_the_quant_config_left_unquantized():
    """A drafter that quantizes only its MLP keeps the fused fast path."""
    from vllm.model_executor.models.qwen3_dflash import _can_fuse_context_kv

    unquantized = _Projection(_ints(4, 4), None, UnquantizedLinearMethod())
    quantized = _Projection(_ints(4, 4), None, _PackedLinearMethod())

    assert _can_fuse_context_kv([unquantized, unquantized]) is True
    assert _can_fuse_context_kv([unquantized, quantized]) is False


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fp8 weight quantization requires CUDA"
)
@pytest.mark.parametrize("case", ["qwen3_dflash", "gemma4_dspark"])
def test_fp8_drafter_matches_dequantized_reference(
    default_vllm_config, dist_init, case
):
    """Online-fp8 projections run cleanly and stay numerically sane.

    Mirrors the issue's repro: ``{"quant_method": "fp8",
    "activation_scheme": "dynamic"}``, which vLLM applies at load time.
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        QKVParallelLinear,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    dtype, device = torch.bfloat16, "cuda"
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
    )
    is_qkv = case == "qwen3_dflash"

    def _make_projection() -> nn.Module:
        if is_qkv:
            proj = QKVParallelLinear(
                hidden_size=HIDDEN,
                head_size=HEAD_DIM,
                total_num_heads=NUM_HEADS,
                total_num_kv_heads=NUM_KV_HEADS,
                bias=False,
                params_dtype=dtype,
                quant_config=quant_config,
            )
        else:
            proj = ColumnParallelLinear(
                input_size=HIDDEN,
                output_size=KV_SIZE,
                bias=False,
                params_dtype=dtype,
                quant_config=quant_config,
            )
        proj = proj.to(device)
        if getattr(proj.quant_method, "use_marlin", False):
            pytest.skip("fp8 marlin repacks weights; no dequant reference available")
        proj.weight.data.normal_(std=0.1)
        proj.quant_method.process_weights_after_loading(proj)
        return proj

    projections = [_make_projection() for _ in range(NUM_LAYERS)]
    # The packed weight is exactly what the fused path used to slice.
    assert all(p.weight.dtype == torch.float8_e4m3fn for p in projections)

    layers_attn = [
        _attn(p, proj_attr="qkv_proj" if is_qkv else "k_proj") for p in projections
    ]
    for attn in layers_attn:
        attn.k_norm.weight = attn.k_norm.weight.to(device=device, dtype=dtype)

    model_cls = DFlashQwen3Model if is_qkv else Gemma4DSparkModel
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(
        weight=torch.ones(HIDDEN, device=device, dtype=dtype)
    )
    model._rms_norm_eps = EPS
    model.layers = list(layers_attn)
    model._build_context_kv_buffers(layers_attn, False)
    assert model._fuse_context_kv is False

    context_states = torch.randn(NUM_CTX, HIDDEN, device=device, dtype=dtype)
    all_k, all_v = model._project_context_kv(
        context_states, NUM_CTX, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
    )

    normed = torch.empty_like(context_states)
    ops.rms_norm(normed, context_states, model.hidden_norm.weight, EPS)
    for i, proj in enumerate(projections):
        weight = get_and_maybe_dequant_weights(proj, out_dtype=torch.float32)
        out = F.linear(normed.float(), weight[Q_SIZE:] if is_qkv else weight)
        expected_k = out[:, :KV_SIZE].view(NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
        torch.testing.assert_close(all_k[i].float(), expected_k, rtol=0.1, atol=0.1)
        if is_qkv:
            expected_v = out[:, KV_SIZE:].view(NUM_CTX, NUM_KV_HEADS, HEAD_DIM)
            torch.testing.assert_close(all_v[i].float(), expected_v, rtol=0.1, atol=0.1)
