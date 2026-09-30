# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import ReLUSquaredActivation, SiluAndMul
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
QUANT_DTYPES = [current_platform.fp8_dtype()]
NUM_TOKENS = [1, 17, 86, 1234, 3045]  # Arbitrary values for testing
HIDDEN_SIZES = [16, 48, 128, 1562, 4096]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def ref_impl(
    silu_and_mul: SiluAndMul, x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    silu_and_mul_out = silu_and_mul.forward_native(x)
    out, scales = ops.scaled_fp8_quant(silu_and_mul_out, scale)
    return out


def ops_impl(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 2)
    out = torch.empty(out_shape, dtype=current_platform.fp8_dtype(), device=x.device)
    torch.ops._C.silu_and_mul_quant(out, x, scale)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    layer = SiluAndMul()

    # Make inputs
    scale = torch.randn((1), device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    ref_out = ref_impl(layer, x, scale)
    ops_out = ops_impl(x, scale)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ref_out.shape == ops_out.shape
    assert torch.allclose(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )
    opcheck(torch.ops._C.silu_and_mul_quant, (ops_out, x, scale))


# ---------------------------------------------------------------------------
# Tests for maybe_fused_act_quant interface
# ---------------------------------------------------------------------------


class MockLinearFp8Static(torch.nn.Module):
    """Mock linear layer advertising kFp8StaticTensorSym."""

    def __init__(self, input_scale: torch.Tensor):
        super().__init__()
        self._input_quant_key = kFp8StaticTensorSym
        self.input_scale = input_scale


class MockLinearFp8Dynamic128(torch.nn.Module):
    """Mock linear layer advertising kFp8Dynamic128Sym."""

    def __init__(self):
        super().__init__()
        self._input_quant_key = kFp8Dynamic128Sym


class MockLinearNoQuant(torch.nn.Module):
    """Mock linear layer with no input quantization key."""

    pass


class MockLinearNvfp4(torch.nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self._input_quant_key = kNvfp4Dynamic
        self.input_global_scale_inv = scale


@pytest.fixture
def native_vllm_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["none"]
        )
    )
    with set_current_vllm_config(config):
        yield config


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("shape", [(1, 16), (17, 80), (129, 1344), (256, 5376)])
@pytest.mark.parametrize("scale_value", [0.0009765625, 1.0, 17.125, 224.0])
@torch.inference_mode()
def test_relu2_nvfp4_exact_and_changed_graph(native_vllm_config, shape, scale_value):
    """Preserve BF16 rounding, packing, padded scales, and replay data dependence."""
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales

    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(scale_value, dtype=torch.float32, device=x.device)
    linear = MockLinearNvfp4(scale)
    act = ReLUSquaredActivation(compile_native=False)
    compiled_relu = torch.compile(act.forward_native, fullgraph=True)
    expected = ops.scaled_fp4_quant(compiled_relu(x), scale)
    before = x.clone()
    actual = maybe_fused_act_quant(act, x, linear)
    assert isinstance(actual, QuantizedActivation)
    assert torch.equal(x, before)
    assert torch.equal(actual.data, expected[0])
    expected_scale = recover_swizzled_scales(expected[1], *shape)
    actual_scale = recover_swizzled_scales(actual.scale, *shape, include_padding=True)
    torch.testing.assert_close(
        actual_scale[: shape[0], : shape[1] // 16], expected_scale, rtol=0, atol=0
    )
    actual_scale[: shape[0], : shape[1] // 16] = 0
    assert not torch.count_nonzero(actual_scale)
    old_data = actual.data.clone()
    old_scale = actual.scale.view(torch.uint8).clone()
    for _ in range(3):
        maybe_fused_act_quant(act, x, linear)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = maybe_fused_act_quant(act, x, linear)
    x.normal_()
    scale.mul_(1.25)
    graph.replay()
    expected = ops.scaled_fp4_quant(compiled_relu(x), scale)
    assert torch.equal(replayed.data, expected[0])
    torch.testing.assert_close(
        recover_swizzled_scales(replayed.scale, *shape),
        recover_swizzled_scales(expected[1], *shape),
        rtol=0,
        atol=0,
    )
    assert torch.equal(actual.data, old_data)
    assert torch.equal(actual.scale.view(torch.uint8), old_scale)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("width", [0, 15, 17, 33])
@torch.inference_mode()
def test_relu2_nvfp4_unsupported_width_preserves_fallback(native_vllm_config, width):
    x = torch.randn((3, width), dtype=torch.bfloat16, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device=x.device)
    act = ReLUSquaredActivation(compile_native=False)
    result = maybe_fused_act_quant(act, x, MockLinearNvfp4(scale))
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, act.forward_native(x), rtol=0, atol=0)


class MockLinearRequiresUnquantized(MockLinearFp8Static):
    """Mock consumer that needs the original activation for another branch."""

    requires_unquantized_input = True


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 3.0e20, 100.0])
@pytest.mark.parametrize("scale_value", [2.0**-30, 1.0, 64.0, 1.0e20])
@torch.inference_mode()
def test_relu2_nvfp4_nonfinite_and_saturation(native_vllm_config, value, scale_value):
    x = torch.tensor([[value] + [1.0] * 15], dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(scale_value, dtype=torch.float32, device=x.device)
    act = ReLUSquaredActivation(compile_native=False)
    compiled_relu = torch.compile(act.forward_native, fullgraph=True)
    expected = ops.scaled_fp4_quant(
        compiled_relu(x), scale, backend="flashinfer-cutedsl"
    )
    actual = maybe_fused_act_quant(act, x, MockLinearNvfp4(scale))
    assert isinstance(actual, QuantizedActivation)
    assert torch.equal(actual.data, expected[0])
    assert torch.equal(actual.scale.view(torch.uint8), expected[1].view(torch.uint8))


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("mode", ["native", "compiled_native", "cuda"])
@torch.inference_mode()
def test_relu2_nvfp4_preserves_selected_activation(mode):
    """CUDA ReLU2 suppresses NaN and must not use the native-semantics producer."""
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["all" if mode == "cuda" else "none"],
        )
    )
    with set_current_vllm_config(config):
        act = ReLUSquaredActivation(compile_native=mode == "compiled_native")
        x = torch.tensor(
            [[float("nan")] + [1.0] * 15], device="cuda", dtype=torch.bfloat16
        )
        scale = torch.ones((), device=x.device, dtype=torch.float32)
        expected = act(x)
        dispatch = maybe_fused_act_quant
        if mode == "compiled_native":
            dispatch = torch.compile(dispatch, fullgraph=True)
        actual = dispatch(act, x, MockLinearNvfp4(scale))
        if mode == "cuda":
            assert isinstance(actual, torch.Tensor)
            assert expected[0, 0].item() == 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert isinstance(actual, QuantizedActivation)
            expected_data, expected_scale = ops.scaled_fp4_quant(expected, scale)
            assert torch.equal(actual.data, expected_data)
            assert torch.equal(
                actual.scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )


def test_relu2_native_guard_tracks_callable_changes(native_vllm_config):
    """Fullgraph tracing must guard the selected callable, not a cached flag."""
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        _uses_native_relu_squared,
    )

    act = ReLUSquaredActivation(compile_native=False)

    def select(x):
        return x + 1 if _uses_native_relu_squared(act) else x - 1

    class Lookalike:
        __func__ = ReLUSquaredActivation.forward_native

        def __call__(self, x):
            return x

        def __eq__(self, other):
            return True

    def forward_native(x):
        return x

    forward_native.__dict__["__func__"] = ReLUSquaredActivation.forward_native
    compiled = torch.compile(select, backend="eager", fullgraph=True)
    x = torch.ones(2)
    for forward, supported in (
        (act.forward_native, True),
        (torch.compile(act.forward_native, backend="eager"), True),
        (act.forward_cuda, False),
        (torch.compile(act.forward_cuda, backend="eager"), False),
        (forward_native, False),
        (Lookalike(), False),
        (act.forward_native, True),
    ):
        act._forward_method = forward
        assert _uses_native_relu_squared(act) is supported
        torch.testing.assert_close(
            compiled(x), x + (1 if supported else -1), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@torch.inference_mode()
@pytest.mark.parametrize(
    "use_aot,use_bytecode_hook", [(False, False), (False, True), (True, False)]
)
def test_relu2_nvfp4_fullgraph_changed_rows(
    native_vllm_config, monkeypatch, use_aot, use_bytecode_hook
):
    """One guard-free vLLM graph must handle unseen M, including K padding."""
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales
    from vllm.compilation.counter import compilation_counter
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.config import CUDAGraphMode

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", str(int(use_aot)))
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", str(int(use_bytecode_hook)))
    native_vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    native_vllm_config.compilation_config.compile_sizes = []
    native_vllm_config.scheduler_config.max_num_batched_tokens = 8192
    native_vllm_config.compilation_config.compile_ranges_endpoints = [8192]

    class Wrapper(TorchCompileWithNoGuardsWrapper):
        def __init__(self):
            self.act = ReLUSquaredActivation(compile_native=True)
            self.linear = MockLinearNvfp4(torch.tensor(17.125, device="cuda"))
            super().__init__()

        def forward(self, x):
            qa = maybe_fused_act_quant(self.act, x, self.linear)
            return qa.data, qa.scale

    torch._dynamo.reset()
    dispatch = Wrapper()
    with compilation_counter.expect(num_graphs_seen=1, num_inductor_compiles=1):
        for m in (128, 17, 129, 1, 257, 33, 4097):
            x = torch.randn((m, 80), device="cuda", dtype=torch.bfloat16)
            if m == 128:
                torch._dynamo.mark_dynamic(x, 0)
            actual_data, actual_scale = dispatch(x)
            expected_data, expected_scale = ops.scaled_fp4_quant(
                dispatch.act.forward_native(x),
                dispatch.linear.input_global_scale_inv,
                backend="flashinfer-cutedsl",
            )
            assert torch.equal(actual_data, expected_data)
            scales = recover_swizzled_scales(
                actual_scale, *x.shape, include_padding=True
            )
            torch.testing.assert_close(
                scales[:m, :5],
                recover_swizzled_scales(expected_scale, *x.shape),
                rtol=0,
                atol=0,
            )
            scales[:m, :5] = 0
            assert not torch.count_nonzero(scales)
    opcheck(
        torch.ops.vllm.relu_squared_nvfp4_quant,
        (x, dispatch.linear.input_global_scale_inv),
    )
    torch._dynamo.reset()


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fp8_static(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant with FP8 static per-tensor quantization."""
    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    linear = MockLinearFp8Static(scale)

    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)
    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, QuantizedActivation)
    assert result.quant_key == kFp8StaticTensorSym
    assert result.data.dtype == current_platform.fp8_dtype()
    assert result.orig_dtype == dtype
    assert result.orig_shape == (num_tokens, hidden_size)

    ref_out = ref_impl(act_fn, x, scale)
    torch.testing.assert_close(result.data.to(torch.float32), ref_out.to(torch.float32))


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fp8_dynamic_block(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant with FP8 dynamic per-block quantization."""
    group_size = 128  # We only support 128 for now

    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    linear = MockLinearFp8Dynamic128()

    scale = 1 / hidden_size
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device) * scale
    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, QuantizedActivation)
    assert result.quant_key == kFp8Dynamic128Sym
    assert result.data.dtype == current_platform.fp8_dtype()
    assert result.orig_dtype == dtype
    assert result.orig_shape == (num_tokens, hidden_size)

    num_groups = hidden_size // group_size
    assert result.scale.shape == (num_tokens, num_groups)

    gate, up = x.split(hidden_size, dim=-1)
    silu_out = F.silu(gate) * up
    ref_out, ref_scales = per_token_group_quant_fp8(
        silu_out, group_size=group_size, use_ue8m0=False
    )

    torch.testing.assert_close(result.scale, ref_scales, rtol=1e-5, atol=1e-5)

    ref_deq = ref_out.to(torch.float32) * ref_scales.repeat_interleave(
        group_size, dim=1
    )
    result_deq = result.data.to(torch.float32) * result.scale.repeat_interleave(
        group_size, dim=1
    )
    torch.testing.assert_close(ref_deq, result_deq, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("hidden_size", [128, 512])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fallback(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant falls back without an input quantization key."""
    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    linear = MockLinearNoQuant()
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)

    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, QuantizedActivation)
    assert result.dtype == dtype
    assert result.shape == (num_tokens, hidden_size)

    ref_out = act_fn(x)
    torch.testing.assert_close(result, ref_out)


@torch.inference_mode()
def test_maybe_fused_act_quant_preserves_required_unquantized_input(
    default_vllm_config,
) -> None:
    device = "cuda:0"
    act_fn = SiluAndMul()
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    linear = MockLinearRequiresUnquantized(scale)
    x = torch.randn(17, 256, dtype=torch.bfloat16, device=device)

    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, act_fn(x))


def _relu2_nvfp4_warmup_mlp(width):
    module = torch.nn.Module()
    module.act_fn = ReLUSquaredActivation(compile_native=False)
    module.down_proj = MockLinearNvfp4(
        torch.ones(1, dtype=torch.float32, device="cuda")
    )
    module.down_proj.params_dtype = torch.bfloat16
    module.down_proj.input_size_per_partition = width
    return module


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize(
    "case",
    ["ready", "lora", "activation", "dtype", "scale_dtype", "scale_device", "width"],
)
def test_relu2_nvfp4_warmup_resolves_post_load_consumer(
    native_vllm_config, monkeypatch, case
):
    """Registration must not allocate/compile or select a stale pre-LoRA linear."""
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        register_relu_squared_nvfp4_quant_warmup,
    )
    from vllm.model_executor.layers.fusion.relu2_nvfp4_quant import (
        _relu_squared_nvfp4_quant_kernel,
        _relu_squared_nvfp4_warmup_inputs,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor

    native_vllm_config.kernel_config.enable_jit_warmup = True
    module = _relu2_nvfp4_warmup_mlp(80)
    scale = module.down_proj.input_global_scale_inv
    del module.down_proj.input_global_scale_inv
    registry = JitWarmupRegistry(native_vllm_config)

    def fail_early(*args, **kwargs):
        pytest.fail("registration allocated tensors or eagerly compiled")

    with registry.activate(), monkeypatch.context() as patch:
        patch.setattr(torch, "empty", fail_early)
        patch.setattr(_relu_squared_nvfp4_quant_kernel, "get_warmup_keys", fail_early)
        register_relu_squared_nvfp4_quant_warmup(module)
        register_relu_squared_nvfp4_quant_warmup(module)
    assert len(registry) == 1
    assert not list(_relu_squared_nvfp4_warmup_inputs(module, 128))

    module.down_proj.input_global_scale_inv = scale
    if case == "lora":
        old_linear = module.down_proj
        module.down_proj = _relu2_nvfp4_warmup_mlp(80).down_proj
        module.down_proj.requires_unquantized_input = True
        assert not getattr(old_linear, "requires_unquantized_input", False)
    elif case == "activation":
        module.act_fn._forward_method = module.act_fn.forward_cuda
    elif case == "dtype":
        module.down_proj.params_dtype = torch.float16
    elif case == "scale_dtype":
        module.down_proj.input_global_scale_inv = scale.half()
    elif case == "scale_device":
        module.down_proj.input_global_scale_inv = scale.cpu()
    elif case == "width":
        module.down_proj.input_size_per_partition = 17

    cases = list(_relu_squared_nvfp4_warmup_inputs(module, 128))
    assert bool(cases) == (case == "ready")
    for inputs in cases:
        assert not any(isinstance(value, torch.Tensor) for value in inputs.values())
        assert isinstance(inputs["x_ptr"], TritonWarmupTensor)

    native_vllm_config.kernel_config.enable_jit_warmup = False
    disabled_registry = JitWarmupRegistry(native_vllm_config)
    with disabled_registry.activate():
        register_relu_squared_nvfp4_quant_warmup(module)
    assert len(disabled_registry) == 0


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("width", [16, 48, 80, 1344, 5376])
def test_relu2_nvfp4_oversized_dispatch_falls_back(
    native_vllm_config, monkeypatch, width
):
    """Reject unsafe index products with fake inputs, without huge allocations."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.model_executor.layers.fusion.fused_act_quant import (
        _FUSED_ACT_QUANT,
        _relu_squared_nvfp4_quant_supported,
    )
    from vllm.model_executor.layers.fusion.relu2_nvfp4_quant import (
        _relu_squared_nvfp4_max_rows,
    )

    def fail_fused(*args, **kwargs):
        pytest.fail("Oversized input reached the int32-indexed producer")

    monkeypatch.setitem(
        _FUSED_ACT_QUANT, (ReLUSquaredActivation, kNvfp4Dynamic), fail_fused
    )
    act = ReLUSquaredActivation(compile_native=False)
    with FakeTensorMode():
        x = torch.empty(
            (_relu_squared_nvfp4_max_rows(width) + 1, width),
            dtype=torch.bfloat16,
            device="cuda",
        )
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        linear = MockLinearNvfp4(scale)
        assert _relu_squared_nvfp4_quant_supported(act, x[:-1], linear)
        actual = maybe_fused_act_quant(act, x, linear)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == x.shape
        assert actual.dtype == x.dtype


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("width", [16, 48, 80, 1344, 5376])
@torch.inference_mode()
def test_relu2_nvfp4_startup_warmup_covers_unseen_rows(
    native_vllm_config, monkeypatch, record_property, width
):
    """Use real Triton keys; startup must cover new M and input/scale alignments."""
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        register_relu_squared_nvfp4_quant_warmup,
    )
    from vllm.model_executor.layers.fusion.relu2_nvfp4_quant import (
        _relu_squared_nvfp4_quant_kernel as owner,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.triton_utils import triton

    native_vllm_config.kernel_config.enable_jit_warmup = True
    native_vllm_config.scheduler_config.max_num_batched_tokens = 32768
    module = _relu2_nvfp4_warmup_mlp(width)
    keys = owner.get_warmup_keys(module=module, max_tokens=32768)
    warmed_keys = set().union(*(key.jit_keys for key in keys))
    assert 0 < len(keys) <= 20  # At most five padding buckets x four alignments.
    record_property("native_compile_keys", len(warmed_keys))

    compiled_keys = []
    original_compile = owner.compile

    def compile_metadata(key):
        assert not any(isinstance(value, torch.Tensor) for _, value in key.inputs)
        compiled_keys.append(key)
        original_compile(key)

    registry = JitWarmupRegistry(native_vllm_config)
    with registry.activate():
        register_relu_squared_nvfp4_quant_warmup(module)
        register_relu_squared_nvfp4_quant_warmup(_relu2_nvfp4_warmup_mlp(width))
    with monkeypatch.context() as patch:
        patch.setattr(owner, "compile", compile_metadata)
        registry.warmup()
    assert set(compiled_keys) == set(keys)
    assert len(compiled_keys) == len(keys)  # Identical layers compile only once.

    def assert_bits(actual, x, scale):
        assert isinstance(actual, QuantizedActivation)
        if x.shape[0] == 0:
            assert actual.data.numel() == actual.scale.numel() == 0
            return
        expected = ops.scaled_fp4_quant(module.act_fn.forward_native(x), scale)
        assert torch.equal(actual.data, expected[0])
        actual_scale = recover_swizzled_scales(
            actual.scale, *x.shape, include_padding=True
        )
        torch.testing.assert_close(
            actual_scale[: x.shape[0], : width // 16],
            recover_swizzled_scales(expected[1], *x.shape),
            rtol=0,
            atol=0,
        )
        actual_scale[: x.shape[0], : width // 16] = 0
        assert not torch.count_nonzero(actual_scale)

    def fail_compile(**kwargs):
        pytest.fail(f"Producer compiled after startup warmup: {kwargs.get('fn')}")

    rows = (0, 1, 2, 7, 17, 33, 65, 127, 128, 129, 255, 256, 257, 1023, 4097, 32768)
    for input_offset, scale_offset in product((0, 1), repeat=2):
        scale = torch.full((2,), 17.125, dtype=torch.float32, device="cuda")
        scale = scale[scale_offset : scale_offset + 1]
        module.down_proj.input_global_scale_inv = scale
        for m in rows:
            x = torch.randn(
                m * width + input_offset, dtype=torch.bfloat16, device="cuda"
            )
            x = x[input_offset:].view(m, width)
            with monkeypatch.context() as patch:
                patch.setattr(
                    triton.knobs.runtime, "jit_post_compile_hook", fail_compile
                )
                actual = maybe_fused_act_quant(module.act_fn, x, module.down_proj)
            assert_bits(actual, x, scale)
            if m == 129 and input_offset == scale_offset == 1:
                graph = torch.cuda.CUDAGraph()
                with monkeypatch.context() as patch:
                    patch.setattr(
                        triton.knobs.runtime, "jit_post_compile_hook", fail_compile
                    )
                    with torch.cuda.graph(graph):
                        replayed = maybe_fused_act_quant(
                            module.act_fn, x, module.down_proj
                        )
                x.neg_()
                scale.mul_(1.25)
                graph.replay()
                assert_bits(replayed, x, scale)
    record_property("post_warmup_dispatch_cases", len(rows) * 4)
    record_property("unseen_row_alignment_cases", sum(m > 128 for m in rows) * 4)
