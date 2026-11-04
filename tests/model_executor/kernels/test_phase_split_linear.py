# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for per-phase linear activations (``QuantSpec.phase``)."""

from __future__ import annotations

import types

import pytest
import torch

import vllm.model_executor.kernels.linear.phase_split as phase_split_mod
from tests.quantization.utils import quant_spec
from vllm.model_executor.kernels.linear import check_linear_phase_served
from vllm.model_executor.kernels.linear.phase_split import (
    DualActivationKernel,
    PhaseSplitLinearKernel,
    maybe_phase_split,
    padding_rows,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
    kMxfp8Dynamic,
    kNvfp4DynamicTensor,
)
from vllm.platforms import current_platform

MAX_DECODE_TOKENS = 8


def _phase(**overrides):
    phase = {
        "decode": "unquantized",
        "prefill": "nvfp4",
        "max_decode_tokens": MAX_DECODE_TOKENS,
        **overrides,
    }
    return quant_spec(phase=phase).phase


_OFFSET = {None: 0, kNvfp4DynamicTensor: 1, kMxfp8Dynamic: 2}


class _DualKernel(DualActivationKernel):
    """Adds 1 to the output of NVFP4 calls and 2 to MXFP8 ones so the phases
    differ."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    @classmethod
    def activation_modes(cls):
        return frozenset(_OFFSET)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(self, layer, x, bias=None, *, mode=None):
        self.calls.append((x.shape[0], mode))
        output = x @ layer.weight.T + _OFFSET[mode]
        return output if bias is None else output + bias


def _layer(prefix: str = "model.layers.0.proj", n: int = 6, k: int = 4, cls=None):
    layer = (cls or torch.nn.Module)()
    layer.prefix = prefix
    layer.weight = torch.nn.Parameter(torch.randn(n, k), requires_grad=False)
    layer.output_size_per_partition = n
    return layer


@pytest.fixture(autouse=True)
def _platform_device():
    # The op is registered for the platform's dispatch key only.
    with torch.device(current_platform.device_type):
        yield


@pytest.fixture
def vllm_config(default_vllm_config):
    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    return default_vllm_config


def _loaded(vllm_config):
    inner = _DualKernel()
    kernel = maybe_phase_split(inner, _phase(), weight="nvfp4")
    assert isinstance(kernel, PhaseSplitLinearKernel)
    layer = _layer()
    kernel.process_weights_after_loading(layer)
    return kernel, inner, layer


@torch.inference_mode()
def test_phase_split_dispatches_on_rows(vllm_config) -> None:
    """Calls with at most max_decode_tokens rows use the decode activations,
    with no forward context, as in multimodal encoders run before it is set."""
    kernel, inner, layer = _loaded(vllm_config)
    assert kernel.input_quant_key() is None
    bias = torch.randn(6)
    for m in (MAX_DECODE_TOKENS, MAX_DECODE_TOKENS + 1):
        x = torch.randn(1, m, 4)
        out = kernel.apply_weights(layer, x, bias)
        expected = x @ layer.weight.T + bias + (m > MAX_DECODE_TOKENS)
        torch.testing.assert_close(out, expected)
    assert inner.calls == [
        (MAX_DECODE_TOKENS, None),
        (MAX_DECODE_TOKENS + 1, kNvfp4DynamicTensor),
    ]


_CELLS = [
    ("unquantized", "nvfp4"),
    ("nvfp4", "nvfp4"),
    ("nvfp4", "unquantized"),
    ("unquantized", "unquantized"),
    ("unquantized", "mxfp8"),
    ("mxfp8", "mxfp8"),
    ("mxfp8", "unquantized"),
]


@pytest.mark.parametrize(
    "decode,prefill,cutoff",
    [
        (decode, prefill, cutoff)
        for decode, prefill in _CELLS
        for cutoff in (MAX_DECODE_TOKENS, None)
        if decode == prefill or cutoff is not None
    ],
)
@torch.inference_mode()
def test_phase_split_dispatches_every_cell(vllm_config, decode, prefill, cutoff):
    """Each (decode, prefill) pair reaches its own activations on each side of
    max_decode_tokens, eagerly and through vllm::phase_split_linear. With one
    format for both phases the cutoff is optional."""
    phase = _phase(decode=decode, prefill=prefill, max_decode_tokens=cutoff)
    if phase.quantized_format == "mxfp8" or phase.quantized_format is None:
        assert phase.scale is None
    inner = _DualKernel()
    kernel = maybe_phase_split(inner, phase, weight=phase.quantized_format or "any")
    assert isinstance(kernel, PhaseSplitLinearKernel)
    layer = _layer(prefix=f"cell.{decode}.{prefill}.{cutoff}")
    kernel.process_weights_after_loading(layer)
    decode_mode, prefill_mode = phase.activation_modes()
    key = layer.phase_split_key
    for m in (1, MAX_DECODE_TOKENS, MAX_DECODE_TOKENS + 1, 40):
        x = torch.randn(m, 4)
        mode = decode_mode if cutoff is None or m <= cutoff else prefill_mode
        expected = x @ layer.weight.T + _OFFSET[mode]
        torch.testing.assert_close(kernel.apply_weights(layer, x), expected)
        out = torch.ops.vllm.phase_split_linear(x, None, key, 6)
        torch.testing.assert_close(out, expected)
        assert inner.calls[-2:] == [(m, mode), (m, mode)]


@torch.inference_mode()
def test_phase_split_dispatches_at_runtime_under_torch_compile(vllm_config) -> None:
    """One graph with a dynamic row count reaches both phases: a branch on the
    row count in traced code would be fixed for every size. Compiled encoders
    run without a forward context, so the op must not need one."""
    from torch._dynamo.testing import CompileCounterWithBackend

    kernel, inner, layer = _loaded(vllm_config)
    counter = CompileCounterWithBackend("aot_eager")
    run = torch.compile(
        lambda x: kernel.apply_weights(layer, x),
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    for m in (3, 40, MAX_DECODE_TOKENS, MAX_DECODE_TOKENS + 1):
        x = torch.randn(m, 4)
        expected = x @ layer.weight.T + (m > MAX_DECODE_TOKENS)
        torch.testing.assert_close(run(x), expected)
    assert counter.frame_count == 1
    assert [mode for _, mode in inner.calls] == [
        None,
        kNvfp4DynamicTensor,
        None,
        kNvfp4DynamicTensor,
    ]
    torch.library.opcheck(
        torch.ops.vllm.phase_split_linear.default,
        (torch.randn(20, 4), None, layer.phase_split_key, 6),
    )


@torch.inference_mode()
def test_phase_split_serves_mla_style_eye_dequant(vllm_config) -> None:
    """MLA dequantizes kv_b_proj at load time by applying the layer to an
    identity matrix, outside a forward pass; rows above max_decode_tokens use
    the prefill activations."""
    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    for k in (4, 2 * MAX_DECODE_TOKENS):
        layer = _layer(prefix=f"model.layers.0.self_attn.kv_b_proj_{k}", k=k)
        layer.input_size_per_partition = k
        layer.quant_method = types.SimpleNamespace(apply=kernel.apply_weights)
        kernel.process_weights_after_loading(layer)
        expected = layer.weight + (k > MAX_DECODE_TOKENS)
        dequant = get_and_maybe_dequant_weights(layer)
        torch.testing.assert_close(dequant, expected)


@torch.inference_mode()
def test_phase_split_resolves_each_layer_by_its_key(vllm_config) -> None:
    """Each layer runs its own weight through the op, whatever its prefix, and
    keeps its key across weight reloads."""

    class _PrefixlessLinear(torch.nn.Module):
        pass

    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    layers = [_layer(prefix="dup.proj"), _layer(prefix="dup.proj")]
    layers += [_layer(prefix="", cls=_PrefixlessLinear) for _ in range(2)]
    for layer in layers:
        kernel.process_weights_after_loading(layer)
    key = layers[1].phase_split_key
    kernel.process_weights_after_loading(layers[1])
    assert layers[1].phase_split_key is key
    assert key.device.type == "cpu"
    assert "phase_split_key" not in layers[1].state_dict()
    x = torch.randn(MAX_DECODE_TOKENS + 1, 4)
    for layer in layers:
        out = torch.ops.vllm.phase_split_linear(x, None, layer.phase_split_key, 6)
        torch.testing.assert_close(out, x @ layer.weight.T + 1)


@torch.inference_mode()
def test_phase_split_compiled_region_serves_several_layers(vllm_config) -> None:
    """vLLM reuses one compiled region for repeated blocks, such as encoder
    layers. The layer reaches the op as a graph input, so the reused graph
    runs each block's own weight without recompiling."""
    from torch._dynamo.testing import CompileCounterWithBackend

    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    layers = [_layer(prefix=f"block.{i}.proj") for i in range(3)]
    for scale, layer in zip((1.0, 10.0, 100.0), layers):
        layer.weight.data = torch.eye(6, 4) * scale
        kernel.process_weights_after_loading(layer)
    counter = CompileCounterWithBackend("aot_eager")
    run = torch.compile(
        lambda x, key: torch.ops.vllm.phase_split_linear(x, None, key, 6),
        backend=counter,
        fullgraph=True,
    )
    x = torch.ones(2, 4)
    outputs = [run(x, layer.phase_split_key)[0, 0].item() for layer in layers]
    assert outputs == [1.0, 10.0, 100.0]
    assert counter.frame_count == 1


def test_phase_split_rejects_kernel_without_both_modes(vllm_config) -> None:
    class _A16OnlyKernel(_DualKernel):
        @classmethod
        def activation_modes(cls):
            return frozenset((None,))

    with pytest.raises(NotImplementedError) as error:
        maybe_phase_split(_A16OnlyKernel(), _phase(), weight="nvfp4", candidates=[])
    message = str(error.value)
    assert "weight=nvfp4" in message
    assert "decode=unquantized, prefill=nvfp4, max_decode_tokens=8" in message
    assert "_A16OnlyKernel cannot serve both phases" in message
    with pytest.raises(NotImplementedError, match=r"backend\): _DualKernel\."):
        maybe_phase_split(object(), _phase(), weight="nvfp4", candidates=[_DualKernel])


def test_phase_split_without_phase_returns_kernel() -> None:
    kernel = object()
    assert maybe_phase_split(kernel, None, weight="nvfp4") is kernel


@pytest.mark.parametrize(
    "setting,match",
    [
        ("batch_invariant", "VLLM_BATCH_INVARIANT"),
        ("capture_sizes", "largest CUDA graph capture size 16"),
        ("data_parallel", "does not support data parallelism"),
        ("elastic_ep", "does not support data parallelism"),
        ("encoder_cudagraphs", "encoder CUDA graphs"),
    ],
)
def test_phase_split_rejects_unsafe_configs(
    monkeypatch, vllm_config, setting, match
) -> None:
    """Padded rows are uninitialized and must not reach a shared scale."""
    if setting == "batch_invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    elif setting == "capture_sizes":
        vllm_config.compilation_config.cudagraph_capture_sizes = [8, 16]
    elif setting == "data_parallel":
        vllm_config.parallel_config.data_parallel_size = 2
    elif setting == "elastic_ep":
        vllm_config.parallel_config.enable_elastic_ep = True
    else:
        vllm_config.compilation_config.cudagraph_mm_encoder = True
    with pytest.raises(ValueError, match=match) as error:
        maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    assert "quantization_config.linear.phase" in str(error.value)


@pytest.mark.parametrize(
    "decode,prefill", [("nvfp4", "nvfp4"), ("nvfp4", "unquantized")]
)
def test_phase_split_nvfp4_decode_needs_padding_mask(
    monkeypatch, vllm_config, decode, prefill
) -> None:
    """NVFP4 decode calls run in CUDA graphs, so they need the runner's padding
    mask, which the V2 runner only refreshes with VLLM_MOE_SKIP_PADDING. They
    do not need max_decode_tokens above the capture sizes."""
    vllm_config.compilation_config.cudagraph_capture_sizes = [8, 16]
    phase = _phase(decode=decode, prefill=prefill)
    assert isinstance(
        maybe_phase_split(_DualKernel(), phase, weight="nvfp4"), PhaseSplitLinearKernel
    )
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")
    with pytest.raises(ValueError, match="requires VLLM_MOE_SKIP_PADDING=1"):
        maybe_phase_split(_DualKernel(), phase, weight="nvfp4")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    vllm_config.parallel_config.data_parallel_size = 2
    with pytest.raises(ValueError, match="does not support data parallelism"):
        maybe_phase_split(_DualKernel(), phase, weight="nvfp4")


@pytest.mark.parametrize(
    "decode,prefill",
    [("unquantized", "mxfp8"), ("mxfp8", "mxfp8"), ("mxfp8", "unquantized")],
)
def test_phase_split_mxfp8_ignores_padding(monkeypatch, vllm_config, decode, prefill):
    """MXFP8 scales are per row block, so padding rows cannot reach a real
    row's scale: CUDA graph sizes and data parallelism are unconstrained. A
    split still makes outputs depend on the rows in a step."""
    vllm_config.compilation_config.cudagraph_capture_sizes = [8, 16]
    vllm_config.parallel_config.data_parallel_size = 2
    phase = _phase(decode=decode, prefill=prefill)
    assert isinstance(
        maybe_phase_split(_DualKernel(), phase, weight="mxfp8"), PhaseSplitLinearKernel
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    if decode == prefill:
        maybe_phase_split(_DualKernel(), phase, weight="mxfp8")
    else:
        with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
            maybe_phase_split(_DualKernel(), phase, weight="mxfp8")


def test_padding_rows_reads_the_runner_mask(monkeypatch, vllm_config) -> None:
    """Only calls over the step's token rows get the mask; it is None without
    a forward context, without a mask, or with VLLM_MOE_SKIP_PADDING=0."""
    from vllm.forward_context import set_forward_context

    assert padding_rows(4) is None
    is_padding = torch.tensor([False, False, True, True])
    with set_forward_context(None, vllm_config, is_padding=is_padding):
        assert padding_rows(4) is is_padding
        assert padding_rows(3) is None
        monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")
        assert padding_rows(4) is None
    with set_forward_context(None, vllm_config):
        monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
        assert padding_rows(4) is None


class _Linear(LinearBase):
    def __init__(self, quant_method) -> None:
        torch.nn.Module.__init__(self)
        self.quant_method = quant_method
        self.input_size_per_partition = 4
        self.output_size_per_partition = 6


@pytest.fixture
def _phase_set(monkeypatch, vllm_config):
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod, "_user_linear_phase", _phase)
    monkeypatch.setattr(phase_split_mod, "_LAYERS", type(phase_split_mod._LAYERS)())


def test_check_phase_served_warns_for_unquantized_model(
    _phase_set, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unquantized model must not silently ignore phase."""
    import vllm.model_executor.kernels.linear as linear_mod
    from vllm.logger import _print_warning_once

    # Keep the real warning_once, whose cache hashes the arguments.
    _print_warning_once.cache_clear()
    warnings: list[str] = []
    monkeypatch.setattr(
        linear_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )
    model = torch.nn.Sequential(_Linear(UnquantizedLinearMethod()))
    check_linear_phase_served(model)
    assert any("no linear layer serves it" in w for w in warnings)


def test_check_phase_served_keeps_other_quantized_linears(
    _phase_set, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linears in other formats (e.g. Q8_0 or FP8) keep their own activations
    once at least one layer serves phase."""
    import vllm.model_executor.kernels.linear as linear_mod
    from vllm.logger import _print_info_once, _print_warning_once

    _print_info_once.cache_clear()
    _print_warning_once.cache_clear()
    logged: dict[str, list[str]] = {"info": [], "warning": []}
    for level in logged:
        monkeypatch.setattr(
            linear_mod.logger,
            level,
            lambda msg, *args, _level=level, **kwargs: logged[_level].append(
                msg % args
            ),
        )
    served = _Linear(object())
    served.weight = torch.nn.Parameter(torch.randn(6, 4), requires_grad=False)
    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    kernel.process_weights_after_loading(served)
    model = torch.nn.Sequential(
        served, _Linear(UnquantizedLinearMethod()), _Linear(object())
    )
    check_linear_phase_served(model)
    assert not logged["warning"]
    assert any("serves 1 of 3 linear layers" in m for m in logged["info"])
