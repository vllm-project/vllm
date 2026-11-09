# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 quantization choices per NVIDIA GPU: the activation and the
padding of the MXFP4 routed experts and the kernel of the MXFP8 linear
layers."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.nvidia import quantization


def _platform(capability):
    return SimpleNamespace(
        is_device_capability=lambda c: c == capability,
        is_device_capability_family=lambda f: f == capability // 10 * 10,
    )


@pytest.mark.parametrize(
    ("capability", "intermediate", "expected"),
    [
        # Hopper, TP2: 1152 columns per rank, whole 128-element groups.
        (90, 1152, kMxfp8Dynamic),
        # Hopper, TP8: 288 columns per rank, no block FP8 schedule.
        (90, 288, None),
        # Blackwell keeps MXFP8 at any split.
        (100, 288, kMxfp8Dynamic),
    ],
)
def test_mxfp4_moe_activation_key(monkeypatch, capability, intermediate, expected):
    monkeypatch.setattr(quantization, "current_platform", _platform(capability))
    moe = SimpleNamespace(hidden_dim=4096, intermediate_size_per_partition=intermediate)
    assert quantization.mxfp4_moe_activation_key(moe) == expected


@pytest.mark.parametrize(
    ("capability", "backend", "per_quant", "expected"),
    [
        (80, "auto", None, True),
        (90, "auto", None, True),
        (89, "auto", None, False),
        (100, "auto", None, False),
        # An explicit MXFP8 backend keeps vLLM's choice on SM90 ...
        (90, "marlin", None, False),
        (90, "auto", {"mxfp8": "humming"}, False),
        (90, "auto", {"nvfp4_w4a16": "humming"}, True),
        # ... but not on SM80, whose attention reads the BF16 wo_a weight.
        (80, "marlin", None, True),
    ],
)
def test_mxfp8_linear_as_bf16(monkeypatch, capability, backend, per_quant, expected):
    monkeypatch.setattr(quantization, "current_platform", _platform(capability))
    kernel_config = SimpleNamespace(
        linear_backend=backend, linear_backend_per_quant=per_quant
    )
    monkeypatch.setattr(
        quantization,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=kernel_config),
    )
    assert quantization.mxfp8_linear_as_bf16() == expected


def test_bf16_mxfp8_linear_method_takes_emulation(monkeypatch):
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization import modelopt

    def create_weights(self, layer, *args, **kwargs):
        layer.weight = SimpleNamespace(shape=(1024, 4096))
        self.kernel = "platform kernel"

    monkeypatch.setattr(modelopt.ModelOptLinearMethod, "create_weights", create_weights)
    method_cls = quantization.bf16_mxfp8_linear_method_cls()
    assert issubclass(method_cls, modelopt.ModelOptLinearMethod)
    method = method_cls.__new__(method_cls)
    method.fmt = modelopt.FormatScheme()
    method.create_weights(SimpleNamespace())  # type: ignore[call-arg]
    assert isinstance(method.kernel, EmulationMxfp8LinearKernel)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (80, quantization.MarlinMxfp4MoEMethod),
        (90, quantization.MarlinMxfp4MoEMethod),
        (100, Mxfp4MoEMethod),
    ],
)
def test_mxfp4_moe_method(monkeypatch, capability, expected):
    monkeypatch.setattr(quantization, "current_platform", _platform(capability))
    created = {}

    def init(self, moe, activation_key=None):
        created["activation_key"] = activation_key
        self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
        self.experts_cls = None

    monkeypatch.setattr(Mxfp4MoEMethod, "__init__", init)
    moe = SimpleNamespace(
        hidden_dim=4096, intermediate_size_per_partition=288, moe_backend="auto"
    )
    method = quantization.mxfp4_moe_method(moe)
    assert type(method) is expected
    assert created["activation_key"] == quantization.mxfp4_moe_activation_key(moe)


@pytest.mark.parametrize(
    ("capability", "moe_backend", "activation_key", "stock", "expected"),
    [
        # H100, TP8: BF16 activations, so Marlin replaces Humming.
        (90, "auto", None, Mxfp4MoeBackend.HUMMING, Mxfp4MoeBackend.MARLIN),
        # H100, TP2: Humming's block FP8 activations stay.
        (90, "auto", kMxfp8Dynamic, Mxfp4MoeBackend.HUMMING, Mxfp4MoeBackend.HUMMING),
        # An explicit backend keeps vLLM's choice.
        (90, "humming", None, Mxfp4MoeBackend.HUMMING, Mxfp4MoeBackend.HUMMING),
        (80, "auto", None, Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.MARLIN),
    ],
)
def test_marlin_in_place_of_humming(
    monkeypatch, capability, moe_backend, activation_key, stock, expected
):
    monkeypatch.setattr(quantization, "current_platform", _platform(capability))

    def init(self, moe, activation_key=None):
        self.moe = moe
        self.mxfp4_backend = stock
        self.experts_cls = "stock experts"

    def select(config):
        assert config is not moe and config.moe_backend == "marlin"
        return Mxfp4MoeBackend.MARLIN, "marlin experts"

    monkeypatch.setattr(Mxfp4MoEMethod, "__init__", init)
    monkeypatch.setattr(quantization, "select_deepseek_v4_mxfp4_moe_backend", select)
    moe = SimpleNamespace(moe_backend=moe_backend)
    method = quantization.MarlinMxfp4MoEMethod(moe, activation_key=activation_key)
    assert method.mxfp4_backend == expected
    assert method.experts_cls == (
        "stock experts" if expected == stock else "marlin experts"
    )
    # The layer pads its config in place later; the method keeps that object.
    assert method.moe is moe and moe.moe_backend == moe_backend


@pytest.mark.parametrize(
    ("backend", "intermediate", "stock", "sm80"),
    [
        # Marlin, TP8: the 288 columns per rank pad to 320, not 384.
        (Mxfp4MoeBackend.MARLIN, 288, 384, 320),
        # Marlin, TP2: 1152 is a multiple of both.
        (Mxfp4MoeBackend.MARLIN, 1152, 1152, 1152),
        # Other backends keep their padding.
        (Mxfp4MoeBackend.BATCHED_MARLIN, 288, 384, 384),
        (Mxfp4MoeBackend.TRITON, 288, None, None),
    ],
)
def test_marlin_intermediate_padding(backend, intermediate, stock, sm80):
    def sizes(cls: type[Mxfp4MoEMethod]) -> tuple[int, int]:
        method = object.__new__(cls)
        method.mxfp4_backend = backend
        method.moe = SimpleNamespace(activation=None)
        parallel = FusedMoEParallelConfig.make_no_parallel()
        return method.maybe_roundup_sizes(5120, intermediate, torch.bfloat16, parallel)

    want = sizes(Mxfp4MoEMethod)
    got = sizes(quantization.MarlinMxfp4MoEMethod)
    assert got[0] == want[0]  # the hidden size keeps the stock padding
    if stock is not None:
        assert want[1] == stock
    assert got[1] == (want[1] if sm80 is None else sm80)


needs_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an SM80-or-newer CUDA GPU",
)


def _mxfp8_layer(n: int, k: int, seed: int) -> torch.nn.Module:
    """A linear layer as the MXFP8 loader leaves it: E4M3 weights and UE8M0
    scales of 32 x 32 blocks, expanded to one row of scales per weight row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] -= 1  # no NaN codes
    blocks = torch.randint(
        112, 131, (n // 32, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        codes.view(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        blocks.repeat_interleave(32, 0), requires_grad=False
    )
    return layer


def _linear_kernel(monkeypatch, layer: torch.nn.Module, bmm_batch_size=None):
    """The kernel the BF16 MXFP8 linear method gives a layer."""
    from vllm.model_executor.layers.quantization import modelopt

    def create_weights(self, layer, *args, **kwargs):
        self.kernel = "platform kernel"

    monkeypatch.setattr(modelopt.ModelOptLinearMethod, "create_weights", create_weights)
    method_cls = quantization.bf16_mxfp8_linear_method_cls()
    method = method_cls.__new__(method_cls)
    method.fmt = modelopt.FormatScheme()
    layer.bmm_batch_size = bmm_batch_size
    method.create_weights(layer)  # type: ignore[call-arg]
    return method.kernel


@needs_sm80
# cuBLAS reduces split-K partial sums in FP32, as the bound below assumes.
@pytest.mark.usefixtures("fp32_cublas_reduction")
@pytest.mark.parametrize("rows", [6, 24, 25, 64])
def test_gemv_linear_dispatch(monkeypatch, rows):
    """Decode steps of up to GEMV_MAX_ROWS rows take the GEMV on the packed
    FP8 copy, larger ones cuBLAS on the BF16 copy; both compute the product
    of the load-time dequantized weights."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    n, k = 576, 1024
    layer = _mxfp8_layer(n, k, seed=rows)
    kernel = _linear_kernel(monkeypatch, layer)
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert kernel.gemv is not None
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    got = kernel.apply_weights(layer, x)
    if rows <= quantization.GEMV_MAX_ROWS:
        blocks, factors, _ = kernel.gemv
        assert torch.equal(got, gemv.mxfp8_gemv(x, blocks, factors))
    else:
        assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))
    want = x.double() @ layer.weight.double().t()
    magnitude = x.double().abs() @ layer.weight.double().abs().t()
    bound = want.abs() * 2.0**-7 + magnitude * 2.0**-12
    assert ((got.double() - want).abs() <= bound).all()


@needs_sm80
def test_gemv_linear_unaligned_rows(monkeypatch):
    """Rows that are not 16-byte aligned keep cuBLAS."""
    n, k = 576, 1024
    layer = _mxfp8_layer(n, k, seed=0)
    kernel = _linear_kernel(monkeypatch, layer)
    kernel.process_weights_after_loading(layer)
    x = torch.randn(6, k + 8, dtype=torch.bfloat16, device="cuda")[:, 1 : k + 1]
    got = kernel.apply_weights(layer, x)
    assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))


@needs_sm80
@pytest.mark.parametrize("case", ["row scales differ", "scale above 2^7", "bmm"])
def test_gemv_linear_declines(monkeypatch, case):
    """Weights the GEMV cannot take stay on cuBLAS alone, with no packed copy."""
    n, k = 576, 1024
    layer = _mxfp8_layer(n, k, seed=1)
    if case == "row scales differ":
        layer.weight_scale.data[1, 0] += 1
    elif case == "scale above 2^7":
        layer.weight_scale.data[:32, 0] = 135
    if case != "bmm":
        packed = quantization.pack_for_gemv(layer.weight.data, layer.weight_scale.data)
        assert packed is None
    # A batched layer (wo_a) never gets a packed copy; its 2-D weight here
    # only stands in for the real [B, N, K] one.
    kernel = _linear_kernel(
        monkeypatch, layer, bmm_batch_size=2 if case == "bmm" else None
    )
    kernel.process_weights_after_loading(layer)
    assert kernel.gemv is None
    x = torch.randn(6, k, dtype=torch.bfloat16, device="cuda")
    got = kernel.apply_weights(layer, x)
    assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))


@needs_sm80
def test_gemv_linear_reload(monkeypatch):
    """A reload of the weights (reload_weights: the FP8 parameters again,
    `process_weights_after_loading` again, the processed parameters copied
    into the first ones) writes the new packed copy into the first one, which
    a captured CUDA graph reads."""
    n, k, rows = 576, 1024, 6
    layer = _mxfp8_layer(n, k, seed=0)
    kernel = _linear_kernel(monkeypatch, layer)
    kernel.process_weights_after_loading(layer)
    first = kernel.gemv
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    out.copy_(kernel.apply_weights(layer, x))
    before = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(kernel.apply_weights(layer, x))

    weight = layer.weight
    fresh = _mxfp8_layer(n, k, seed=1)
    layer.weight, layer.weight_scale = fresh.weight, fresh.weight_scale
    kernel.process_weights_after_loading(layer)
    weight.data.copy_(layer.weight.data)
    layer.weight = weight
    assert kernel.gemv is first

    graph.replay()
    assert torch.equal(out, kernel.apply_weights(layer, x))
    assert not torch.equal(out, before)
    want = x.double() @ layer.weight.double().t()
    magnitude = x.double().abs() @ layer.weight.double().abs().t()
    bound = want.abs() * 2.0**-7 + magnitude * 2.0**-12
    assert ((out.double() - want).abs() <= bound).all()

    # Weights a reload can no longer pack fail loudly.
    stale = _mxfp8_layer(n, k, seed=2)
    stale.weight_scale.data[:32, 0] = 135
    layer.weight, layer.weight_scale = stale.weight, stale.weight_scale
    with pytest.raises(RuntimeError, match="no longer fit"):
        kernel.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (90, quantization.DeferringMarlinExperts),
        (80, quantization.DeferringMarlinExperts),
        (100, None),
    ],
)
def test_marlin_experts_can_defer(monkeypatch, capability, expected):
    """On SM80 and SM90 the Marlin experts are the ones that can leave the
    top-k reduction to the boundaries' all-reduce; elsewhere vLLM's stay."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts,
    )

    monkeypatch.setattr(quantization, "current_platform", _platform(capability))

    def init(self, moe, activation_key=None):
        self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
        self.experts_cls = MarlinExperts

    monkeypatch.setattr(Mxfp4MoEMethod, "__init__", init)
    method = quantization.MarlinMxfp4MoEMethod(SimpleNamespace(moe_backend="auto"))
    assert method.experts_cls is (expected or MarlinExperts)


@needs_sm80
@pytest.mark.parametrize("rows", [6, 49])
def test_deferring_marlin_experts(rows):
    """On Marlin itself: a call the layer defers hands over GEMM2's rows
    unweighted, one per route, bitwise the rows Marlin's post-GEMM2 hook
    gets, with the identity row index; their weighted sum is Marlin's
    finalized output within BF16 rounding. A larger call finalizes as Marlin
    does."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts,
    )
    from vllm.model_executor.layers.fused_moe.moe_output import (
        UnfinalizedMoEOutput,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        rand_marlin_weight_mxfp4_like,
    )

    torch.manual_seed(0)
    num_experts, top_k, hidden, intermediate = 16, 6, 512, 256

    def mxfp4_experts(n, k):
        packed = [
            rand_marlin_weight_mxfp4_like(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16), 32
            )
            for _ in range(num_experts)
        ]
        return (
            torch.stack([qweight for _, qweight, _ in packed]).contiguous(),
            torch.stack([scales for _, _, scales in packed]),
        )

    w1, w1_scale = mxfp4_experts(2 * intermediate, hidden)
    w2, w2_scale = mxfp4_experts(hidden, intermediate)
    moe_config = SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        use_deferred_moe_finalize=True,
        defer_moe_finalize_max_num_tokens=48,
        experts_per_token=top_k,
        should_defer_moe_finalize=lambda n: 0 < n <= 48,
    )
    quant_config = mxfp4_w4a16_moe_quant_config(w1_scale, w2_scale)
    experts = quantization.DeferringMarlinExperts(moe_config, quant_config)
    experts.process_weights_after_loading(torch.nn.Module())
    # A weight reload builds new experts; graphs keep reading the same index.
    rebuilt = quantization.DeferringMarlinExperts(moe_config, quant_config)
    rebuilt.process_weights_after_loading(torch.nn.Module())
    assert rebuilt.route_index is experts.route_index
    marlin = MarlinExperts(moe_config, quant_config)

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) / 4
    topk_weights = torch.rand(rows, top_k, device="cuda")
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(rows)]
    ).to(torch.int32)

    def run(kernel):
        output = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
        args = (topk_weights, topk_ids, MoEActivation.SILU, num_experts, None)
        got = kernel.apply(output, x, w1, w2, *args, *[None] * 5, False)
        return got, output

    finalized = run(marlin)[1]
    got, output = run(experts)
    if rows > 48:
        assert got is None
        torch.testing.assert_close(output, finalized, atol=0, rtol=0)
        return
    hooked: list[UnfinalizedMoEOutput] = []
    marlin.post_gemm2_hook = lambda unfinalized, _: hooked.append(unfinalized)
    run(marlin)
    assert isinstance(got, UnfinalizedMoEOutput) and got.route_major
    assert torch.equal(got.gemm2_permuted, hooked[0].gemm2_permuted)
    assert torch.equal(
        got.expanded_idx_to_permuted_idx, hooked[0].expanded_idx_to_permuted_idx
    )
    assert got.expert_weights is topk_weights
    summed = (
        got.gemm2_permuted.view(rows, top_k, hidden).float()
        * topk_weights.unsqueeze(-1)
    ).sum(1)
    torch.testing.assert_close(summed.bfloat16(), finalized, atol=2e-2, rtol=2e-2)


def test_deferring_marlin_experts_refuse_unsupported():
    """A call the layer defers fails loudly where the deferred path does not
    apply, instead of finalizing behind the layer's back."""
    experts = object.__new__(quantization.DeferringMarlinExperts)
    experts.moe_config = SimpleNamespace(should_defer_moe_finalize=lambda n: True)
    experts._lora_context = object()
    with pytest.raises(NotImplementedError):
        experts.apply(None, torch.empty(1, 8), *[None] * 12, False)  # type: ignore[call-arg]
