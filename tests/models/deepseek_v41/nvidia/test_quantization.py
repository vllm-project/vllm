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
    ("capability", "backend", "per_quant", "expected", "tp_size"),
    [
        (80, "auto", None, True, 8),
        (90, "auto", None, True, 8),
        (89, "auto", None, False, 8),
        (100, "auto", None, False, 8),
        # An explicit MXFP8 backend keeps vLLM's choice on SM90 ...
        (90, "marlin", None, False, 8),
        (90, "auto", {"mxfp8": "humming"}, False, 8),
        (90, "auto", {"nvfp4_w4a16": "humming"}, True, 8),
        # ... but not on SM80, whose attention reads the BF16 wo_a weight.
        (80, "marlin", None, True, 8),
        # At TP1 on SM90 every rank holds whole dense layers, and the two extra
        # copies cost 12.4 GiB per GPU; TP2 and up keep BF16 and the GEMV.
        (90, "auto", None, False, 1),
        (90, "auto", None, True, 2),
        (90, "auto", None, True, 16),
        (80, "auto", None, True, 1),
        (100, "auto", None, False, 1),
    ],
)
def test_mxfp8_linear_as_bf16(
    monkeypatch, capability, backend, per_quant, expected, tp_size
):
    monkeypatch.setattr(quantization, "current_platform", _platform(capability))
    monkeypatch.setattr(
        quantization, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    kernel_config = SimpleNamespace(
        linear_backend=backend, linear_backend_per_quant=per_quant
    )
    monkeypatch.setattr(
        quantization,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=kernel_config),
    )
    assert quantization.mxfp8_linear_as_bf16() == expected


@pytest.mark.parametrize(
    ("capability", "backend", "per_quant", "tp_size", "batch_invariant", "expected"),
    [
        # TP1 on SM90 keeps vLLM's kernel on the FP8 weights, with the GEMV ...
        (90, "auto", None, 1, False, "gemv"),
        # ... but an explicit MXFP8 backend means that kernel only, and so does
        # batch invariance, which the GEMV's row-count-dependent results break.
        (90, "marlin", None, 1, False, "stock"),
        (90, "auto", {"mxfp8": "humming"}, 1, False, "stock"),
        (90, "auto", {"nvfp4_w4a16": "humming"}, 1, False, "gemv"),
        (90, "auto", None, 1, True, "stock"),
        # TP2 and up, and SM80, keep BF16 GEMMs and the GEMV.
        (90, "auto", None, 2, False, "bf16"),
        (90, "auto", None, 2, True, "bf16"),
        (90, "auto", None, 8, False, "bf16"),
        (80, "auto", None, 1, False, "bf16"),
        (80, "auto", None, 8, False, "bf16"),
        (90, "marlin", None, 8, False, "stock"),
        (100, "auto", None, 1, False, "stock"),
        (100, "auto", None, 8, False, "stock"),
    ],
)
def test_mxfp8_linear_method_cls(
    monkeypatch, capability, backend, per_quant, tp_size, batch_invariant, expected
):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod

    monkeypatch.setattr(quantization, "current_platform", _platform(capability))
    monkeypatch.setattr(
        quantization, "envs", SimpleNamespace(VLLM_BATCH_INVARIANT=batch_invariant)
    )
    monkeypatch.setattr(
        quantization, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    kernel_config = SimpleNamespace(
        linear_backend=backend, linear_backend_per_quant=per_quant
    )
    monkeypatch.setattr(
        quantization,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kernel_config=kernel_config),
    )
    bf16 = quantization.gemv_mxfp8_linear_method_cls(True)
    gemv = quantization.gemv_mxfp8_linear_method_cls(False)
    assert issubclass(bf16, ModelOptLinearMethod)
    assert issubclass(gemv, ModelOptLinearMethod)
    assert bf16 is not gemv
    want = {"bf16": bf16, "gemv": gemv, "stock": ModelOptLinearMethod}[expected]
    assert quantization.mxfp8_linear_method_cls() is want


def _gemv_method(monkeypatch, bf16, kernel=None):
    """A GEMV method as vLLM builds it, `create_weights` stubbed down to the
    selection of `kernel` as the platform's MXFP8 kernel."""
    from vllm.model_executor.layers.quantization import modelopt

    def create_weights(self, layer, *args, **kwargs):
        self.kernel = kernel or "platform kernel"

    monkeypatch.setattr(modelopt.ModelOptLinearMethod, "create_weights", create_weights)
    method_cls = quantization.gemv_mxfp8_linear_method_cls(bf16)
    method = method_cls.__new__(method_cls)
    method.fmt = modelopt.FormatScheme()
    return method


def test_bf16_gemv_linear_method_takes_emulation(monkeypatch):
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )

    method = _gemv_method(monkeypatch, bf16=True)
    layer = SimpleNamespace(weight=SimpleNamespace(shape=(1024, 4096)))
    method.create_weights(layer)  # type: ignore[call-arg]
    assert isinstance(method.kernel, EmulationMxfp8LinearKernel)


def test_stock_gemv_linear_method_keeps_the_platform_kernel(monkeypatch):
    method = _gemv_method(monkeypatch, bf16=False)
    method.create_weights(SimpleNamespace())  # type: ignore[call-arg]
    assert method.kernel == "platform kernel"


def test_gemv_linear_methods_weight_loader():
    """The stock variant keeps `ModelOptLinearMethod`'s v2 weight loader; the
    BF16 one keeps v1."""
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    stock = quantization.gemv_mxfp8_linear_method_cls(False)
    bf16 = quantization.gemv_mxfp8_linear_method_cls(True)
    assert stock.__name__ in WEIGHT_LOADER_V2_SUPPORTED
    assert bf16.__name__ not in WEIGHT_LOADER_V2_SUPPORTED


def test_gemv_linear_method_forwards_quantized_activation(monkeypatch):
    """A pre-quantized activation is no input of the GEMV: the kernel gets it."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

    method = _gemv_method(
        monkeypatch,
        bf16=False,
        kernel=SimpleNamespace(apply_weights=lambda layer, x, bias: "kernel"),
    )
    method.create_weights(SimpleNamespace())  # type: ignore[call-arg]
    method.gemv = (None, None, None)  # type: ignore[assignment]
    x = QuantizedActivation(
        torch.zeros(2, 32),
        torch.zeros(2),
        torch.bfloat16,
        torch.Size((2, 32)),
        kMxfp8Dynamic,
    )
    assert method.apply(SimpleNamespace(), x) == "kernel"  # type: ignore[arg-type]


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
    assert created["activation_key"] == kMxfp8Dynamic


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


def _linear_method(monkeypatch, layer: torch.nn.Module, bf16, bmm_batch_size=None):
    """The GEMV method of a layer. The stock one runs the emulation kernel as
    the platform's: BF16 GEMMs on the weights dequantized at load, as the BF16
    one does, but with the GEMV packed before it, as for any platform kernel."""
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization import modelopt

    method = _gemv_method(
        monkeypatch,
        bf16,
        EmulationMxfp8LinearKernel(
            Mxfp8LinearLayerConfig(weight_shape=tuple(layer.weight.shape))
        ),
    )
    monkeypatch.setattr(
        modelopt.ModelOptLinearMethod,
        "process_weights_after_loading",
        lambda self, layer: self.kernel.process_weights_after_loading(layer),
    )
    layer.bmm_batch_size = bmm_batch_size
    method.create_weights(layer)  # type: ignore[call-arg]
    return method


@needs_sm80
# cuBLAS reduces split-K partial sums in FP32, as the bound below assumes.
@pytest.mark.usefixtures("fp32_cublas_reduction")
@pytest.mark.parametrize("bf16", [True, False])
@pytest.mark.parametrize("rows", [6, 24, 25, 64])
def test_gemv_linear_dispatch(monkeypatch, bf16, rows):
    """Decode steps of up to GEMV_MAX_ROWS rows take the GEMV on the packed
    FP8 copy, larger ones the kernel on the BF16 copy; both compute the product
    of the load-time dequantized weights."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    n, k = 576, 1024
    layer = _mxfp8_layer(n, k, seed=rows)
    method = _linear_method(monkeypatch, layer, bf16)
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert method.gemv is not None
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    got = method.apply(layer, x)
    if rows <= quantization.GEMV_MAX_ROWS:
        blocks, factors, _ = method.gemv
        assert torch.equal(got, gemv.mxfp8_gemv(x, blocks, factors))
    else:
        assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))
    want = x.double() @ layer.weight.double().t()
    magnitude = x.double().abs() @ layer.weight.double().abs().t()
    bound = want.abs() * 2.0**-7 + magnitude * 2.0**-12
    assert ((got.double() - want).abs() <= bound).all()


@needs_sm80
def test_gemv_linear_packs_the_same_bytes_with_any_kernel(monkeypatch):
    """The stock method packs the FP8 weights before its kernel rewrites them
    (as Humming does, into its own layout), and a decode step takes the GEMV,
    not that kernel: the same packed bytes and output as the BF16 method."""
    n, k = 576, 1024
    layers = [_mxfp8_layer(n, k, seed=3) for _ in range(2)]
    bf16 = _linear_method(monkeypatch, layers[0], True)
    stock = _linear_method(monkeypatch, layers[1], False)

    def rewrite(layer):
        layer.weight.data.view(torch.uint8).zero_()

    stock.kernel = SimpleNamespace(
        process_weights_after_loading=rewrite,
        apply_weights=lambda layer, x, bias: torch.zeros(
            x.shape[0], n, dtype=x.dtype, device=x.device
        ),
    )
    for method, layer in zip((bf16, stock), layers):
        method.process_weights_after_loading(layer)
    assert not layers[1].weight.data.view(torch.uint8).any()
    assert bf16.gemv is not None and stock.gemv is not None
    assert torch.equal(bf16.gemv[0], stock.gemv[0])
    assert torch.equal(bf16.gemv[1], stock.gemv[1])
    x = torch.randn(6, k, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(bf16.apply(layers[0], x), stock.apply(layers[1], x))


@needs_sm80
@pytest.mark.parametrize("bf16", [True, False])
def test_gemv_linear_unaligned_rows(monkeypatch, bf16):
    """Rows that are not 16-byte aligned keep the kernel."""
    n, k = 576, 1024
    layer = _mxfp8_layer(n, k, seed=0)
    method = _linear_method(monkeypatch, layer, bf16)
    method.process_weights_after_loading(layer)
    x = torch.randn(6, k + 8, dtype=torch.bfloat16, device="cuda")[:, 1 : k + 1]
    got = method.apply(layer, x)
    assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))


@pytest.mark.parametrize(("bmm_batch_size", "is_bmm"), [(2, False), (None, True)])
def test_gemv_linear_packs_no_batched_layer(monkeypatch, bmm_batch_size, is_bmm):
    """A batched layer (wo_a) is told by its batch size or by `is_bmm` alone; the
    layer here has no weights to pack."""
    layer = SimpleNamespace(weight=SimpleNamespace(shape=(576, 1024)), is_bmm=is_bmm)
    method = _linear_method(monkeypatch, layer, False, bmm_batch_size)
    method.kernel = SimpleNamespace(process_weights_after_loading=lambda layer: None)
    method.process_weights_after_loading(layer)
    assert method.gemv is None


@needs_sm80
@pytest.mark.parametrize("bf16", [True, False])
@pytest.mark.parametrize("case", ["row scales differ", "scale above 2^7", "bmm"])
def test_gemv_linear_declines(monkeypatch, bf16, case):
    """Weights the GEMV cannot take stay on the kernel alone, with no packed
    copy."""
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
    method = _linear_method(
        monkeypatch, layer, bf16, bmm_batch_size=2 if case == "bmm" else None
    )
    method.process_weights_after_loading(layer)
    assert method.gemv is None
    x = torch.randn(6, k, dtype=torch.bfloat16, device="cuda")
    got = method.apply(layer, x)
    assert torch.equal(got, torch.nn.functional.linear(x, layer.weight))


@needs_sm80
@pytest.mark.parametrize("bf16", [True, False])
def test_gemv_linear_reload(monkeypatch, bf16):
    """A reload of the weights (reload_weights: the FP8 parameters again,
    `process_weights_after_loading` again, the processed parameters copied
    into the first ones) writes the new packed copy into the first one, which
    a captured CUDA graph reads."""
    n, k, rows = 576, 1024, 6
    layer = _mxfp8_layer(n, k, seed=0)
    method = _linear_method(monkeypatch, layer, bf16)
    method.process_weights_after_loading(layer)
    first = method.gemv
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    out.copy_(method.apply(layer, x))
    before = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(method.apply(layer, x))

    weight = layer.weight
    fresh = _mxfp8_layer(n, k, seed=1)
    layer.weight, layer.weight_scale = fresh.weight, fresh.weight_scale
    method.process_weights_after_loading(layer)
    weight.data.copy_(layer.weight.data)
    layer.weight = weight
    assert method.gemv is first

    graph.replay()
    assert torch.equal(out, method.apply(layer, x))
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
        method.process_weights_after_loading(layer)


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
