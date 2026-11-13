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
from vllm.utils.torch_utils import _encode_layer_name

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
        if self.phase is not None:
            self.register_phase_layer(layer)

    def apply_weights(self, layer, x, bias=None):
        assert self.phase is not None
        return self.apply_phase(layer, x, bias)

    def apply_mode(self, layer, x, bias=None, *, mode):
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
    kernel = _DualKernel()
    # The oracle returns the selected kernel itself, with its phase set.
    assert maybe_phase_split(kernel, _phase(), weight="nvfp4") is kernel
    assert kernel.phase == _phase()
    layer = _layer()
    kernel.process_weights_after_loading(layer)
    return kernel, layer


@torch.inference_mode()
def test_phase_split_dispatches_on_rows(vllm_config) -> None:
    """Calls with at most max_decode_tokens rows use the decode activations,
    with no forward context, as in multimodal encoders run before it is set."""
    kernel, layer = _loaded(vllm_config)
    bias = torch.randn(6)
    for m in (MAX_DECODE_TOKENS, MAX_DECODE_TOKENS + 1):
        x = torch.randn(1, m, 4)
        out = kernel.apply_weights(layer, x, bias)
        expected = x @ layer.weight.T + bias + (m > MAX_DECODE_TOKENS)
        torch.testing.assert_close(out, expected)
    assert kernel.calls == [
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
    assert phase is not None
    if phase.quantized_format == "mxfp8" or phase.quantized_format is None:
        assert phase.scale is None
    decode_mode, prefill_mode = phase.activation_modes()
    kernel = maybe_phase_split(
        _DualKernel(), phase, weight=phase.quantized_format or "any"
    )
    assert kernel.phase is phase
    layer = _layer(prefix=f"cell.{decode}.{prefill}.{cutoff}")
    kernel.process_weights_after_loading(layer)
    name = _encode_layer_name(layer.phase_split_layer_name)
    for m in (1, MAX_DECODE_TOKENS, MAX_DECODE_TOKENS + 1, 40):
        x = torch.randn(m, 4)
        mode = decode_mode if cutoff is None or m <= cutoff else prefill_mode
        expected = x @ layer.weight.T + _OFFSET[mode]
        torch.testing.assert_close(kernel.apply_weights(layer, x), expected)
        out = torch.ops.vllm.phase_split_linear(x, None, name, 6)
        torch.testing.assert_close(out, expected)
        assert kernel.calls[-2:] == [(m, mode), (m, mode)]


@torch.inference_mode()
def test_phase_split_dispatches_at_runtime_under_torch_compile(vllm_config) -> None:
    """One graph with a dynamic row count reaches both phases: a branch on the
    row count in traced code would be fixed for every size. Compiled encoders
    run without a forward context, so the op must not need one."""
    from torch._dynamo.testing import CompileCounterWithBackend

    kernel, layer = _loaded(vllm_config)
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
    assert [mode for _, mode in kernel.calls] == [
        None,
        kNvfp4DynamicTensor,
        None,
        kNvfp4DynamicTensor,
    ]
    torch.library.opcheck(
        torch.ops.vllm.phase_split_linear.default,
        (
            torch.randn(20, 4),
            None,
            _encode_layer_name(layer.phase_split_layer_name),
            6,
        ),
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
def test_phase_split_names_layers_without_unique_prefix(vllm_config) -> None:
    """Models that build linears without a prefix, or reuse one, still load and
    each layer runs its own weight through the op."""

    class _PrefixlessLinear(torch.nn.Module):
        pass

    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    layers = [_layer(prefix="dup.proj"), _layer(prefix="dup.proj")]
    layers += [_layer(prefix="", cls=_PrefixlessLinear) for _ in range(2)]
    for layer in layers:
        kernel.process_weights_after_loading(layer)
    names = [layer.phase_split_layer_name for layer in layers]
    assert names == [
        "dup.proj",
        "dup.proj#1",
        "_PrefixlessLinear",
        "_PrefixlessLinear#1",
    ]
    # Reprocessing a layer, e.g. on weight reload, keeps its name.
    registered = len(phase_split_mod._LAYERS)
    kernel.process_weights_after_loading(layers[1])
    assert layers[1].phase_split_layer_name == "dup.proj#1"
    assert len(phase_split_mod._LAYERS) == registered
    x = torch.randn(MAX_DECODE_TOKENS + 1, 4)
    for layer in layers:
        out = torch.ops.vllm.phase_split_linear(
            x, None, _encode_layer_name(layer.phase_split_layer_name), 6
        )
        torch.testing.assert_close(out, x @ layer.weight.T + 1)


@torch.inference_mode()
def test_phase_split_compiled_region_serves_several_layers(vllm_config) -> None:
    """A compiled region that vLLM reuses for repeated blocks, such as encoder
    layers, runs each block's own weight: the layer reaches the op as a graph
    input, not a constant baked into the graph."""
    from torch._dynamo.testing import CompileCounterWithBackend

    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    blocks = []
    for i, scale in enumerate((1.0, 10.0, 100.0)):
        block = torch.nn.Module()
        block.proj = _layer(prefix=f"block.{i}.proj")
        block.proj.weight.data = torch.eye(6, 4) * scale
        kernel.process_weights_after_loading(block.proj)
        blocks.append(block)
    counter = CompileCounterWithBackend("aot_eager")
    run = torch.compile(
        lambda block, x: kernel.apply_weights(block.proj, x),
        backend=counter,
        fullgraph=True,
        # vLLM drops every guard, so a reused region never recompiles.
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )
    x = torch.ones(2, 4)
    assert [run(block, x)[0, 0].item() for block in blocks] == [1.0, 10.0, 100.0]
    assert counter.frame_count == 1


@torch.inference_mode()
def test_phase_split_keeps_layer_name_after_reload(
    vllm_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layer keeps its name across reloads even once the layer that held the
    base name is gone, so the registry does not grow."""
    import gc
    import weakref

    # Layers other tests left alive would take the names this test expects.
    monkeypatch.setattr(phase_split_mod, "_LAYERS", weakref.WeakValueDictionary())
    kernel = maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")
    first, second = _layer(prefix="proj"), _layer(prefix="proj")
    kernel.process_weights_after_loading(first)
    kernel.process_weights_after_loading(second)
    assert second.phase_split_layer_name == "proj#1"
    del first
    gc.collect()
    kernel.process_weights_after_loading(second)
    assert second.phase_split_layer_name == "proj#1"
    assert [n for n in phase_split_mod._LAYERS if n.startswith("proj")] == ["proj#1"]


def test_phase_split_requires_layer_name_inputs(
    vllm_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without LayerName the layer name is baked into the graph, so a compiled
    graph that vLLM reuses for other layers would resolve the wrong ones."""
    from vllm.utils import torch_utils

    monkeypatch.setattr(torch_utils, "_USE_LAYERNAME", False)
    with pytest.raises(ValueError, match="requires LayerName"):
        maybe_phase_split(_DualKernel(), _phase(), weight="nvfp4")


def test_phase_split_reports_batch_invariance_first(
    vllm_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VLLM_BATCH_INVARIANT selects its own kernels; the error names it rather
    than the kernel it selected."""

    class _A16OnlyKernel(_DualKernel):
        @classmethod
        def activation_modes(cls):
            return frozenset((None,))

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
        maybe_phase_split(_A16OnlyKernel(), _phase(), weight="nvfp4", candidates=[])


@torch.inference_mode()
def test_phase_split_op_rejects_wrong_output_shape(vllm_config) -> None:
    """The op's output feeds a traced graph, so a kernel returning the wrong
    shape raises even under ``python -O``."""

    class _WrongShapeKernel(_DualKernel):
        def apply_mode(self, layer, x, bias=None, *, mode):
            return super().apply_mode(layer, x, bias, mode=mode)[:, :-1]

    kernel = maybe_phase_split(_WrongShapeKernel(), _phase(), weight="nvfp4")
    layer = _layer(prefix="wrong.shape")
    kernel.process_weights_after_loading(layer)
    name = _encode_layer_name(layer.phase_split_layer_name)
    with pytest.raises(RuntimeError, match="returned shape"):
        torch.ops.vllm.phase_split_linear(torch.randn(3, 4), None, name, 6)


def test_phase_split_rejects_kernel_without_both_modes(vllm_config) -> None:
    class _A16OnlyKernel(_DualKernel):
        @classmethod
        def activation_modes(cls):
            return frozenset((None,))

    # Unsupported kernels are reported before settings that would also fail.
    vllm_config.parallel_config.data_parallel_size = 2
    with pytest.raises(NotImplementedError) as error:
        maybe_phase_split(_A16OnlyKernel(), _phase(), weight="nvfp4", candidates=[])
    vllm_config.parallel_config.data_parallel_size = 1
    message = str(error.value)
    assert "weight=nvfp4" in message
    assert "decode=unquantized, prefill=nvfp4, max_decode_tokens=8" in message
    assert "_A16OnlyKernel cannot serve it" in message
    with pytest.raises(NotImplementedError, match=r"b12x\): _DualKernel\."):
        maybe_phase_split(object(), _phase(), weight="nvfp4", candidates=[_DualKernel])


def test_phase_split_without_phase_returns_kernel() -> None:
    kernel = object()
    assert maybe_phase_split(kernel, None, weight="nvfp4") is kernel


@pytest.mark.parametrize(
    "setting,match",
    [
        ("batch_invariant", "VLLM_BATCH_INVARIANT"),
        ("capture_sizes", "largest CUDA graph capture size 16"),
        ("decoder_replay_sizes", "largest CUDA graph capture size 16"),
        ("data_parallel", "does not support data parallelism"),
        ("elastic_ep", "does not support data parallelism"),
        ("encoder_cudagraphs", "encoder CUDA graphs"),
        ("prompt_embeds", "client-supplied prompt or multimodal embeddings"),
        ("mm_embeds", "client-supplied prompt or multimodal embeddings"),
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
    elif setting == "decoder_replay_sizes":
        vllm_config.compilation_config.decoder_replay_cudagraph_capture_sizes = [16]
    elif setting == "data_parallel":
        vllm_config.parallel_config.data_parallel_size = 2
    elif setting == "elastic_ep":
        vllm_config.parallel_config.enable_elastic_ep = True
    elif setting == "encoder_cudagraphs":
        vllm_config.compilation_config.cudagraph_mm_encoder = True
    else:
        # One request's embeddings would set the scale of every request.
        vllm_config.model_config = types.SimpleNamespace(
            enable_prompt_embeds=setting == "prompt_embeds",
            multimodal_config=types.SimpleNamespace(
                enable_mm_embeds=setting == "mm_embeds"
            ),
        )
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
    assert maybe_phase_split(_DualKernel(), phase, weight="nvfp4").phase is phase
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
    assert maybe_phase_split(_DualKernel(), phase, weight="mxfp8").phase is phase
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
