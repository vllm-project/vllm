# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode GEMM plans of Qwen4Exp's specialized decode
(`QWEN4_EXP_DECODE_GEMM_PLANS`, `PlannedGemms`): the table's
configurations fit their shapes; only the specialized decode reads it (the
stock installer and other GPUs do not); `PlannedGemms` plans only the
unquantized BF16 layers of listed shapes and swaps their quant methods only
within `active`; and on GB300, every plan matches cuBLAS within BF16
rounding, falls back to the layer's own method at other row counts, compiles
nothing in a capture after its warm-up and replays bitwise."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import low_latency_gemm
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
    QWEN4_EXP_SM103_GEMM_PLANS,
    enable_qwen4_exp_low_latency_gemm,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import gemm
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gemm import (
    QWEN4_EXP_DECODE_GEMM_PLANS,
    PlannedGemms,
    PlannedLinearMethod,
    decode_gemm_plans,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the decode GEMM plans are CUDA kernels"
)

GB300_PLANS = QWEN4_EXP_DECODE_GEMM_PLANS[(10, 3)]
POINTS = [
    (shape, rows) for shape, plans in sorted(GB300_PLANS.items()) for rows in plans
]

on_gb300 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability(103)
    or not shape_dynamic_skinny_gemm.is_available(),
    reason="the decode GEMM plans are GB300 (SM103) CuTe DSL kernels",
)


class _Linear(nn.Module):
    """The parts of a linear layer that `PlannedGemms` reads: a quant
    method and a weight."""

    def __init__(
        self,
        n: int,
        k: int,
        quant_method: object | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__()
        # Any: the stock or planned linear method, or a bare object standing
        # in for a quantized one, as a layer's quant method can be any of them.
        self.quant_method: Any = quant_method or UnquantizedLinearMethod()
        generator = torch.Generator(device=device).manual_seed(seed)
        self.weight = nn.Parameter(
            (0.02 * torch.randn(n, k, generator=generator, device=device)).to(dtype),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output through its quant method, as `LinearBase`."""
        return self.quant_method.apply(self, x)


def test_the_plans_fit_their_shapes():
    assert set(QWEN4_EXP_DECODE_GEMM_PLANS) == {(10, 3)}
    assert POINTS
    for (n, k), plans in GB300_PLANS.items():
        for rows, config in plans.items():
            assert config.num_rows == rows and 1 <= rows <= 16
            assert n % config.outputs_per_block == 0
            tile = config.block_size * config.vector_width
            assert k % tile == 0
            assert config.static_k in (None, k)
            if config.static_k is not None:
                assert k // tile >= 2


def test_the_plans_differ_from_the_stock_table_where_both_plan():
    """A decode plan at a point the stock table plans replaces it with another
    configuration; repeating the stock plan would only add a dispatch."""
    for shape, plans in GB300_PLANS.items():
        for rows, config in plans.items():
            assert QWEN4_EXP_SM103_GEMM_PLANS.get(shape, {}).get(rows) != config


def test_only_gb300_has_decode_plans(monkeypatch):
    monkeypatch.setattr(shape_dynamic_skinny_gemm, "is_available", lambda: True)
    for capability in ((10, 0), (9, 0), (12, 1)):
        monkeypatch.setattr(
            gemm.current_platform,
            "get_device_capability",
            lambda capability=capability: capability,
        )
        assert decode_gemm_plans() == {}
    monkeypatch.setattr(gemm.current_platform, "get_device_capability", lambda: (10, 3))
    plans = decode_gemm_plans()
    assert plans == GB300_PLANS
    # A copy: changing it does not change the table.
    plans.clear()
    assert QWEN4_EXP_DECODE_GEMM_PLANS[(10, 3)]


def test_the_stock_installer_does_not_read_the_decode_plans(monkeypatch):
    """The default decode's layers take only the stock table: a shape that
    only the decode table plans keeps the plain unquantized method."""
    decode_only = [
        shape for shape in GB300_PLANS if shape not in QWEN4_EXP_SM103_GEMM_PLANS
    ]
    if not decode_only:
        pytest.skip("every decode-planned shape is also in the stock table")
    root = nn.Module()
    for index, (n, k) in enumerate(decode_only):
        root.add_module(f"decode_only_{index}", _Linear(n, k))
    stock_shape = next(iter(QWEN4_EXP_SM103_GEMM_PLANS))
    root.stock = _Linear(*stock_shape)
    monkeypatch.setattr(low_latency_gemm, "LinearBase", _Linear)
    monkeypatch.setattr(
        low_latency_gemm.current_platform, "get_device_capability", lambda: (10, 3)
    )
    monkeypatch.setattr(shape_dynamic_skinny_gemm, "is_available", lambda: True)
    monkeypatch.setattr(
        shape_dynamic_skinny_gemm, "request_warmup_configs", lambda *a, **k: None
    )
    enable_qwen4_exp_low_latency_gemm(root, torch.bfloat16)
    assert isinstance(
        root.stock.quant_method, low_latency_gemm.Qwen4ExpLowLatencyLinearMethod
    )
    for index in range(len(decode_only)):
        layer = root.get_submodule(f"decode_only_{index}")
        assert type(layer.quant_method) is UnquantizedLinearMethod


def test_build_plans_only_unquantized_bf16_layers_of_listed_shapes(monkeypatch):
    plan = {4: SkinnyGemmConfig(4, 64, 2, vector_width=4, static_k=512)}
    monkeypatch.setattr(gemm, "decode_gemm_plans", lambda: {(64, 512): plan})
    monkeypatch.setattr(gemm, "LinearBase", _Linear)
    quantized = object()
    layer = nn.Module()
    layer.planned = _Linear(64, 512)
    layer.quantized = _Linear(64, 512, quant_method=quantized)
    layer.fp32 = _Linear(64, 512, dtype=torch.float32)
    layer.unlisted = _Linear(32, 512)
    # The same layer reached twice is planned once.
    planned = PlannedGemms.build([layer, layer.planned])
    assert planned.sites == [(layer.planned, plan)]
    assert planned.configs() == [(plan[4], layer.planned.weight)]


def test_active_swaps_the_quant_methods_only_within_the_block():
    stock = UnquantizedLinearMethod()
    layer = _Linear(64, 512, quant_method=stock)
    plan = {4: SkinnyGemmConfig(4, 64, 2, vector_width=4, static_k=512)}
    planned = PlannedGemms([(layer, plan)])
    with planned.active():
        method = layer.quant_method
        assert isinstance(method, PlannedLinearMethod)
        assert method.stock is stock and method.plans == plan
    assert layer.quant_method is stock
    with pytest.raises(RuntimeError), planned.active():
        raise RuntimeError
    assert layer.quant_method is stock
    # A method installed after build (the MTP draft model installs the stock
    # table's after building its layers) is the one other calls take.
    later = low_latency_gemm.Qwen4ExpLowLatencyLinearMethod()
    layer.quant_method = later
    with planned.active():
        assert layer.quant_method.stock is later
    assert layer.quant_method is later


def test_a_call_without_a_plan_takes_the_stock_method():
    """Row counts without a plan and calls with a bias run the stock method."""
    layer = _Linear(64, 512, device="cuda", seed=1)
    plan = {4: SkinnyGemmConfig(4, 64, 2, vector_width=4, static_k=512)}
    method = PlannedLinearMethod(layer.quant_method, plan)
    x = torch.randn(3, 512, device="cuda").to(torch.bfloat16)
    bias = torch.randn(64, device="cuda").to(torch.bfloat16)
    assert torch.equal(method.apply(layer, x), layer.quant_method.apply(layer, x))
    x4 = torch.randn(4, 512, device="cuda").to(torch.bfloat16)
    assert torch.equal(
        method.apply(layer, x4, bias), layer.quant_method.apply(layer, x4, bias)
    )


class _NoStockMethod(UnquantizedLinearMethod):
    """A layer's own method that fails if called: a planned call must not
    reach it."""

    def apply(self, layer, x, bias=None):
        raise AssertionError("the call took the layer's own method, not the plan")


@on_gb300
@pytest.mark.parametrize("shape, rows", POINTS)
def test_each_plan_matches_cublas(shape, rows):
    n, k = shape
    layer = _Linear(n, k, device="cuda", seed=n + rows)
    method = PlannedLinearMethod(_NoStockMethod(), GB300_PLANS[shape])
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, k, generator=generator, device="cuda").to(torch.bfloat16)
    planned = method.apply(layer, x)
    # The plan ran: the output is the planned kernel's, bit for bit.
    config = GB300_PLANS[shape][rows]
    assert torch.equal(planned, shape_dynamic_skinny_gemm(x, layer.weight, config))
    got = planned.float()
    ref = F.linear(x, layer.weight).float()
    exact = x.float() @ layer.weight.float().T
    assert got.shape == ref.shape
    # FP32 accumulation in another K order, rounded once to BF16: half a BF16
    # unit in the last place from the FP32 result, so within 2**-7 of the
    # largest output (one unit there) of it, and within 1 % of the largest
    # output of the standard implementation's BF16 output.
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    assert cos >= 0.99999
    assert (got - exact).abs().max() <= 2**-7 * exact.abs().max()
    assert (got - ref).abs().max() <= 0.01 * ref.abs().max()
    # Rows of a row-strided activation view take the plan too.
    wide = torch.randn(rows, k + 64, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    view = wide[:, :k]
    assert torch.equal(method.apply(layer, view), method.apply(layer, view.clone()))


@on_gb300
def test_planned_layers_compile_nothing_in_a_capture_and_replay_bitwise(monkeypatch):
    monkeypatch.setattr(gemm, "LinearBase", _Linear)
    root = nn.Module()
    for index, (n, k) in enumerate(sorted(GB300_PLANS)):
        root.add_module(f"layer_{index}", _Linear(n, k, device="cuda", seed=index))
    planned = PlannedGemms.build([root])
    assert len(planned.sites) == len(GB300_PLANS)
    warmed = planned.warm_up()
    assert set(warmed) == {
        config for plans in GB300_PLANS.values() for config in plans.values()
    }
    compiled = len(shape_dynamic_skinny_gemm._compiled)
    generator = torch.Generator(device="cuda").manual_seed(7)
    inputs = [
        torch.randn(rows, k, generator=generator, device="cuda").to(torch.bfloat16)
        for (n, k), rows in POINTS
    ]
    layers = {tuple(layer.weight.shape): layer for layer in root.children()}
    with planned.active():
        eager = [layers[shape](x) for (shape, _), x in zip(POINTS, inputs)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = [layers[shape](x) for (shape, _), x in zip(POINTS, inputs)]
    assert len(shape_dynamic_skinny_gemm._compiled) == compiled
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        for got, ref in zip(captured, eager):
            assert torch.equal(got, ref)
    # Outside the block the layers run their own method again.
    for (shape, _), x in zip(POINTS, inputs):
        assert torch.equal(layers[shape](x), F.linear(x, layers[shape].weight))
