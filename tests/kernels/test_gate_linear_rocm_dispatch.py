# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear's ROCm router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. ``allow_cublas_router_gemm`` selects the
bf16xbf16->fp32 ``torch.mm`` epilogue, while ``allow_fp32_router_gemm`` selects
the gfx950 low-M kernel with fp32 weights and output.

The ROCm branch is guarded on ``not bias`` because ``torch.mm`` has no bias
term; a biased gate must fall back so the bias is not silently dropped.
"""

import inspect

import pytest
import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    input_size: int = 2048,
    output_size: int = 64,
    on_gfx950: bool = False,
    parallel_world_size: int = 1,
    force_fp32_compute: bool = False,
    quant_config=None,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: parallel_world_size,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(platform, "is_device_capability", lambda *a, **k: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)
    if is_rocm:
        import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
        force_fp32_compute=force_fp32_compute,
        quant_config=quant_config,
    )


def test_rocm_no_bias_bf16_fp32_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False)
    assert not gate.allow_specialized_router_gemm
    assert gate.allow_cublas_router_gemm


def test_rocm_bias_disables_fused_gemm(monkeypatch):
    # torch.mm cannot add a bias, so a biased gate must not take the fused path.
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True)
    assert not gate.allow_cublas_router_gemm


def test_rocm_fp32_weight_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, params_dtype=torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_rocm_non_fp32_out_dtype_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, out_dtype=torch.bfloat16)
    assert not gate.allow_cublas_router_gemm


def test_non_rocm_non_cuda_disables_fused_gemm(monkeypatch):
    # Neither the CUDA specialized path nor the ROCm branch applies.
    gate = _make_gate(monkeypatch, is_rocm=False, is_cuda=False)
    assert not gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False, out_dtype=None)
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_respects_bias_guard(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(3072, 256), (4096, 8), (4096, 192), (6144, 128), (6144, 256)],
)
def test_rocm_gfx950_enables_fp32_router_gemm(
    monkeypatch, input_size: int, output_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=True,
    )
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_rocm_fp32_router_gemm_is_replicated_across_ep_sizes(
    monkeypatch, ep_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        on_gfx950=True,
        parallel_world_size=ep_size,
    )
    assert gate.weight.shape == (128, 6144)
    assert gate.allow_fp32_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size", "params_dtype", "bias", "on_gfx950"),
    [
        pytest.param(6144, 128, torch.float32, False, False, id="gfx942"),
        pytest.param(2048, 64, torch.float32, False, True, id="shape"),
        pytest.param(6144, 128, torch.bfloat16, False, True, id="bf16-weight"),
        pytest.param(6144, 128, torch.float32, True, True, id="bias"),
    ],
)
def test_rocm_fp32_router_gemm_rejects_unsupported_configs(
    monkeypatch,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    bias: bool,
    on_gfx950: bool,
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=params_dtype,
        bias=bias,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=on_gfx950,
    )
    assert not gate.allow_fp32_router_gemm


# ---------------------------------------------------------------------------
# ROCm BF16x3 tier (fp32 router weights)
# ---------------------------------------------------------------------------
def _make_bf16x3_gate(monkeypatch, *, supported=True, **kwargs):
    """Build a gate with the ROCm bf16x3 platform check mocked."""
    from vllm.model_executor.layers.fused_moe.router import (
        bf16x3_router_gemm_rocm as rocm_bf16x3,
    )

    monkeypatch.setattr(rocm_bf16x3, "platform_supported", lambda: supported)
    kwargs.setdefault("params_dtype", torch.float32)
    return _make_gate(monkeypatch, is_rocm=True, **kwargs)


def test_rocm_bf16x3_enabled_for_fp32_weight(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch)
    assert gate.allow_rocm_bf16x3_router_gemm
    # Built by process_weights_after_loading, not lazily in forward().
    # Registered up front but still None, so named_buffers() skips it.
    assert gate._bf16x3_weight is None
    assert "_bf16x3_weight" in gate._buffers


def test_rocm_bf16x3_disabled_by_bias(monkeypatch):
    # torch.mm has no bias term.
    gate = _make_bf16x3_gate(monkeypatch, bias=True)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_bf16_weight(monkeypatch):
    # A bf16 weight belongs to the cuBLAS tier; there is nothing to split.
    gate = _make_bf16x3_gate(monkeypatch, params_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_for_non_fp32_out_dtype(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    assert not gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_disabled_on_unsupported_arch(monkeypatch):
    # Only gfx950 is benchmarked; gfx942 has a different fp32:bf16 MFMA ratio.
    gate = _make_bf16x3_gate(monkeypatch, supported=False)
    assert not gate.allow_rocm_bf16x3_router_gemm
    assert not gate._rocm_bf16x3_weight_eligible


def test_rocm_bf16x3_no_split_built_when_tier_disabled(monkeypatch):
    """An ineligible out_dtype must not allocate a split that is never read;
    it is 1.5x the router weight on every layer."""
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=torch.bfloat16)
    gate.quant_method.process_weights_after_loading(gate)
    assert gate._bf16x3_weight is None


def test_rocm_bf16x3_set_out_dtype_enables_tier(monkeypatch):
    gate = _make_bf16x3_gate(monkeypatch, out_dtype=None)
    assert not gate.allow_rocm_bf16x3_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_rocm_bf16x3_router_gemm


def test_rocm_bf16x3_dispatch_is_a_custom_op():
    """The num_tokens branch must stay opaque to Dynamo; see the op's impl."""
    assert hasattr(torch.ops.vllm, "rocm_bf16x3_router_gemm_dispatch")
    x = torch.zeros(8192, 6144, dtype=torch.bfloat16, device="meta")
    w = torch.zeros(128, 6144, dtype=torch.float32, device="meta")
    split = torch.zeros(3, 128, 6144, dtype=torch.bfloat16, device="meta")
    out = torch.ops.vllm.rocm_bf16x3_router_gemm_dispatch(x, w, split)
    assert out.shape == (8192, 128)
    assert out.dtype == torch.float32


def test_rocm_bf16x3_is_reachable_for_low_m_shapes(monkeypatch):
    """The gfx950 low-M fp32 tier claims these shapes too and returns
    unconditionally, so bf16x3 is only reachable if checked first."""
    gate = _make_bf16x3_gate(
        monkeypatch, input_size=6144, output_size=128, on_gfx950=True
    )
    assert gate.allow_fp32_router_gemm
    assert gate.allow_rocm_bf16x3_router_gemm

    source = inspect.getsource(type(gate).forward)
    assert source.index("allow_rocm_bf16x3_router_gemm") < source.index(
        "allow_fp32_router_gemm"
    )


# ---------------------------------------------------------------------------
# ROCm AITER tuned GEMM tier (bf16 router weights)
# ---------------------------------------------------------------------------
def _make_aiter_gate(monkeypatch, *, tgemm=True, tuned=True, **kwargs):
    """Build a gate with the AITER tgemm and tuned-shape checks mocked."""
    from vllm._aiter_ops import rocm_aiter_ops

    tuned_calls = []

    def is_bf16_gemm_tuned(n, k, out_dtype):
        tuned_calls.append((n, k, out_dtype))
        return tuned

    monkeypatch.setattr(rocm_aiter_ops, "is_tgemm_enabled", staticmethod(lambda: tgemm))
    monkeypatch.setattr(
        rocm_aiter_ops, "is_bf16_gemm_tuned", staticmethod(is_bf16_gemm_tuned)
    )
    return _make_gate(monkeypatch, is_rocm=True, **kwargs), tuned_calls


class _OnlineQuantMethod:
    """Quantizes after loading, so the weight is still bf16 at construction."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        weight = torch.empty(
            sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype
        )
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )


class _OnlineQuantConfig:
    online_quantization_config = None

    def get_quant_method(self, layer, prefix):
        return _OnlineQuantMethod()


def test_rocm_aiter_router_gemm_enabled_for_tuned_bf16_gate(monkeypatch):
    gate, tuned_calls = _make_aiter_gate(monkeypatch, input_size=6144, output_size=256)
    assert gate.allow_aiter_router_gemm
    assert tuned_calls == [(256, 6144, torch.float32)]


def test_rocm_aiter_router_gemm_rechecks_tuning_on_set_out_dtype(monkeypatch):
    """Tuned rows are keyed by output dtype, so a gate built without an
    out_dtype probes bf16 first and must re-probe once fp32 is set."""
    gate, tuned_calls = _make_aiter_gate(
        monkeypatch, out_dtype=None, input_size=6144, output_size=256
    )
    gate.set_out_dtype(torch.float32)
    assert gate.allow_aiter_router_gemm
    assert tuned_calls == [
        (256, 6144, torch.bfloat16),
        (256, 6144, torch.float32),
    ]


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_rocm_aiter_bf16_gemm_passes_out_dtype_as_otype(monkeypatch, out_dtype):
    """The op must ask tgemm for out_dtype directly, not run a bf16 GEMM and
    upcast: tgemm keys its tuned rows by output dtype."""
    import sys
    import types

    from vllm._aiter_ops import _rocm_aiter_bf16_gemm_impl

    calls = []

    class _FakeTgemm:
        def mm(self, inp, weights, bias=None, otype=None):
            calls.append((bias, otype))
            return inp.new_zeros(inp.shape[0], weights.shape[0], dtype=otype)

    fake_module = types.ModuleType("aiter.tuned_gemm")
    fake_module.tgemm = _FakeTgemm()
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", fake_module)

    x = torch.zeros(4, 16, dtype=torch.bfloat16)
    weight = torch.zeros(8, 16, dtype=torch.bfloat16)
    out = _rocm_aiter_bf16_gemm_impl(x, weight, out_dtype)

    assert calls == [(None, out_dtype)]
    assert out.dtype == out_dtype


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"tgemm": False}, id="tgemm-disabled"),
        pytest.param({"tuned": False}, id="untuned-shape"),
        pytest.param({"bias": True}, id="bias"),
        pytest.param({"params_dtype": torch.float32}, id="fp32-weight"),
        pytest.param({"force_fp32_compute": True}, id="force-fp32-compute"),
        pytest.param({"quant_config": _OnlineQuantConfig()}, id="quantized"),
    ],
)
def test_rocm_aiter_router_gemm_rejects_ineligible_gates(monkeypatch, kwargs):
    gate, _ = _make_aiter_gate(monkeypatch, **kwargs)
    assert not gate.allow_aiter_router_gemm


def test_rocm_aiter_router_gemm_disabled_by_batch_invariant(monkeypatch):
    # The AITER tuned GEMM picks its kernel from the token count.
    monkeypatch.setattr(gate_linear_mod.envs, "VLLM_BATCH_INVARIANT", True)
    gate, _ = _make_aiter_gate(monkeypatch, input_size=6144, output_size=256)
    assert not gate.allow_aiter_router_gemm


@pytest.mark.parametrize(("tuned", "expected"), [(True, "aiter"), (False, "torch.mm")])
def test_rocm_aiter_router_gemm_forward_dispatch(monkeypatch, tuned, expected):
    """The torch.mm epilogue is eligible on ROCm too, so a tuned gate must take
    the AITER op instead of it."""
    gate, _ = _make_aiter_gate(monkeypatch, tuned=tuned)
    assert gate.allow_cublas_router_gemm

    calls = []

    def fake_aiter(x, weight, out_dtype):
        calls.append("aiter")
        return x.new_zeros(x.shape[0], weight.shape[0], dtype=out_dtype)

    def fake_mm(x, weight_t, out_dtype=None):
        calls.append("torch.mm")
        return x.new_zeros(x.shape[0], weight_t.shape[1], dtype=out_dtype)

    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "bf16_gemm", staticmethod(fake_aiter))
    monkeypatch.setattr(torch, "mm", fake_mm)

    x = torch.zeros(4, gate.input_size, dtype=torch.bfloat16)
    output, _ = gate(x)

    assert calls == [expected]
    assert output.shape == (4, gate.output_size)
    assert output.dtype == torch.float32


# ---------------------------------------------------------------------------
# ROCm AITER tuned GEMM tier on hardware, shape tuned for only one M
# ---------------------------------------------------------------------------
# A router shape AITER tunes with fp32 output (GateLinear's out_dtype).
ROUTER_N, ROUTER_K = 128, 6144


@pytest.fixture
def aiter_router_shape_tuned_only_at_m1(monkeypatch):
    """Keep only the M=1 fp32-output row of AITER's bf16 tuned table for the
    router shape."""
    if not current_platform.is_rocm():
        pytest.skip("requires ROCm")
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if not rocm_aiter_ops.is_tgemm_enabled():
        pytest.skip("requires the AITER tuned GEMM on gfx950")

    import aiter.tuned_gemm as tuned_gemm

    # Keys are (gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle).
    table = {
        key: row
        for key, row in tuned_gemm.get_GEMM_A16W16_config_().items()
        if key[3:5] != (ROUTER_N, ROUTER_K) or key[2] == 1
    }
    fp32 = str(torch.float32)
    if not any(key[2:5] == (1, ROUTER_N, ROUTER_K) and key[7] == fp32 for key in table):
        pytest.skip("AITER has no M=1 fp32-output tuned row for the router shape")

    monkeypatch.setattr(tuned_gemm, "get_GEMM_A16W16_config_", lambda: table)
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    rocm_aiter_ops.is_bf16_gemm_tuned.cache_clear()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    rocm_aiter_ops.is_bf16_gemm_tuned.cache_clear()


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 33, 300])
@torch.inference_mode()
def test_rocm_aiter_router_gemm_runs_for_untuned_m(
    aiter_router_shape_tuned_only_at_m1, monkeypatch, num_tokens: int
) -> None:
    """One tuned M enables the tier for every M, so an untuned M must fall back
    inside AITER instead of failing."""
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        on_gfx950=True,
        input_size=ROUTER_K,
        output_size=ROUTER_N,
    ).cuda()
    assert gate.allow_aiter_router_gemm

    from aiter.tuned_gemm import TunedGemm

    calls = []
    tgemm_mm = TunedGemm.mm

    def spy(self, inp, *args, **kwargs):
        calls.append(inp.shape[0])
        return tgemm_mm(self, inp, *args, **kwargs)

    monkeypatch.setattr(TunedGemm, "mm", spy)

    set_random_seed(num_tokens)
    gate.weight.normal_(0, 0.02)
    x = torch.randn(num_tokens, ROUTER_K, dtype=torch.bfloat16, device="cuda")
    output, _ = gate(x)

    reference = torch.nn.functional.linear(x.float(), gate.weight.float())
    assert calls == [num_tokens]
    assert output.shape == (num_tokens, ROUTER_N)
    assert output.dtype == torch.float32
    rel_err = (output - reference).abs().max() / reference.abs().max()
    assert rel_err < 2e-2
