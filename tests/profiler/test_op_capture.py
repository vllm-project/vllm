# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Meta-device operator capture: what it records, and that hardware agrees."""

from collections import defaultdict
from dataclasses import replace

import pytest
import torch

import vllm.model_executor.layers.attention.attention  # noqa: F401
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    requantize_with_max_scale,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.meta_loader import MetaModelLoader
from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.platforms import current_platform
from vllm.profiler.op_capture import (
    BatchSpec,
    ForwardHarness,
    OpCapture,
    OpRecorder,
    UnsupportedMetaOpError,
    capture_batches,
    capture_model_ops,
    capture_ranks,
    compare_devices,
    format_diff,
    format_report,
    meta_ops,
    register_meta_impls,
    write_capture_files,
)
from vllm.profiler.op_capture import recorder as recorder_module
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import DIRECT_REGISTERED_OPS, direct_register_custom_op
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MAMBA_MODEL = "state-spaces/mamba-130m-hf"
LAYER_PREFIX = "model.layers."
ATTENTION_OP = "vllm::unified_attention_with_output"


def _capture(num_hidden_layers: int, keep_going: bool = False) -> OpCapture:
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=1024,
        hf_overrides={"num_hidden_layers": num_hidden_layers},
    )
    return capture_model_ops(MODEL, engine_args=engine_args, keep_going=keep_going)


def _ops_by_layer(capture: OpCapture) -> dict[str, list[str]]:
    """Operator names issued from each decoder layer, keyed by layer index."""
    by_layer: dict[str, list[str]] = defaultdict(list)
    for op in capture.ops:
        if op.module.startswith(LAYER_PREFIX):
            index = op.module.removeprefix(LAYER_PREFIX).partition(".")[0]
            by_layer[index].append(op.name)
    return by_layer


@pytest.fixture(scope="module")
def capture() -> OpCapture:
    return _capture(num_hidden_layers=2)


@pytest.fixture
def leaf_library(monkeypatch) -> torch.library.Library:
    """A namespace `register_meta_impls` treats as a compiled extension."""
    namespace = "_test_leaf_C"
    namespaces = meta_ops.LEAF_NAMESPACES | {namespace}
    monkeypatch.setattr(meta_ops, "LEAF_NAMESPACES", namespaces)
    monkeypatch.setattr(recorder_module, "LEAF_NAMESPACES", namespaces)
    library = torch.library.Library(namespace, "FRAGMENT")
    yield library
    library._destroy()


@pytest.fixture
def glue_library() -> torch.library.Library:
    """A namespace of Python custom ops, as `direct_register_custom_op` makes."""
    namespace = "_test_glue"
    library = torch.library.Library(namespace, "FRAGMENT")
    yield library
    for name in [name for name in DIRECT_REGISTERED_OPS if name.startswith(namespace)]:
        del DIRECT_REGISTERED_OPS[name]
    library._destroy()


def test_meta_load_format_resolves_to_meta_loader():
    loader = get_model_loader(LoadConfig(load_format="meta"))
    assert isinstance(loader, MetaModelLoader)


def test_capture_leaves_no_process_group(capture):
    """A later in-process worker would otherwise inherit the harness's gloo group."""
    assert not torch.distributed.is_initialized()


def test_batch_over_scheduler_budget_is_refused():
    """A batch no real step could schedule is refused rather than captured."""
    engine_args = EngineArgs(
        model=MODEL, max_model_len=1024, max_num_batched_tokens=1024
    )
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        ForwardHarness(
            MODEL, batch=BatchSpec(num_reqs=2, num_tokens=2048), engine_args=engine_args
        )


def test_batch_beyond_max_model_len_is_refused():
    engine_args = EngineArgs(model=MODEL, max_model_len=1024)
    with pytest.raises(ValueError, match="max_model_len"):
        ForwardHarness(
            MODEL,
            batch=BatchSpec(num_tokens=8, num_computed_tokens=1024),
            engine_args=engine_args,
        )


def test_capture_batches_runs_each_batch_on_one_model():
    """Each capture sees its own batch's shapes, with the model built once."""
    batches = [
        BatchSpec(num_tokens=8),
        BatchSpec(num_reqs=2, num_tokens=2, num_computed_tokens=16),
    ]
    engine_args = EngineArgs(
        model=MODEL, max_model_len=1024, hf_overrides={"num_hidden_layers": 1}
    )
    captures = capture_batches(MODEL, batches, engine_args=engine_args)
    assert [capture.batch for capture in captures] == batches
    embeddings = [
        next(op for op in capture.ops if op.name.startswith("aten::embedding"))
        for capture in captures
    ]
    assert [op.inputs[-1] for op in embeddings] == ["i64[8]", "i64[2]"]


def test_tensor_parallel_needs_one_harness_per_rank():
    engine_args = EngineArgs(model=MODEL, max_model_len=1024, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="capture_ranks"):
        ForwardHarness(MODEL, engine_args=engine_args)


def test_capture_ranks_records_collectives_on_per_rank_shards():
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=1024,
        tensor_parallel_size=2,
        hf_overrides={"num_hidden_layers": 1},
    )
    ranks = capture_ranks(MODEL, [BatchSpec()], engine_args=engine_args)
    assert [captures[0].rank for captures in ranks] == [0, 1]
    for (capture,) in ranks:
        selection = capture.selection
        assert selection.tensor_parallel_size == 2
        heads = (selection.num_query_heads, selection.num_kv_heads, selection.head_size)
        assert heads == (7, 1, 64)
        assert "vllm::all_reduce" in {op.name for op in capture.ops}


class _ShapelessLayer:
    """An attention layer exposing only the attributes it is given."""

    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)


class _CacheOnlyLayer(_ShapelessLayer):
    """A layer owning a KV cache without running attention, as an indexer does."""


def test_head_counts_come_from_the_first_layer_that_has_them():
    """A hybrid model's layers do not all carry attention's head counts.

    Mamba mixers and the indexer of sparse attention are attention layers too,
    and they lead the model, so reading the counts off whichever layer comes
    first reports another layer type's shape -- or a zero -- as attention's.
    """
    first_shaped = ForwardHarness._first_shaped_layer
    indexer = _ShapelessLayer(topk_tokens=2048)
    mamba = _ShapelessLayer(num_heads=128, head_size=64)
    attention = _ShapelessLayer(num_heads=16, num_kv_heads=2, head_size=128)
    assert first_shaped([indexer, mamba, attention]) is attention
    assert first_shaped([attention, mamba]) is attention
    assert first_shaped([indexer, mamba]) is None
    assert first_shaped([]) is None


def test_selection_metadata_leaves_unreported_head_counts_unset(monkeypatch):
    """The whole path, on a model whose layers report no shape at all."""
    engine_args = EngineArgs(
        model=MODEL, max_model_len=1024, hf_overrides={"num_hidden_layers": 1}
    )
    with ForwardHarness(MODEL, engine_args=engine_args) as harness:
        monkeypatch.setattr(
            harness, "_attention_layers", lambda: {"shapeless": _ShapelessLayer()}
        )
        selection = harness.selection_metadata()
    heads = (selection.num_query_heads, selection.num_kv_heads, selection.head_size)
    assert heads == (None, None, None)


def test_a_model_without_head_counts_reports_them_as_unknown(capture):
    """No layer to read them off must read as unknown, not as a shape of zero."""
    selection = replace(
        capture.selection, num_query_heads=None, num_kv_heads=None, head_size=None
    )
    lines = format_report(replace(capture, selection=selection)).splitlines()
    (row,) = [line for line in lines if line.strip().startswith("heads")]
    assert "unknown" in row
    assert "0" not in row


def test_selection_metadata_counts_cache_only_layers_by_kind(monkeypatch):
    """Sparse attention's indexer and compressor are attention layers too.

    They own a KV cache without running attention, so the count exceeds the
    model's depth and only its composition explains why.
    """
    engine_args = EngineArgs(
        model=MODEL, max_model_len=1024, hf_overrides={"num_hidden_layers": 2}
    )
    layers = {
        "layers.0.attn": _ShapelessLayer(num_heads=16, num_kv_heads=2, head_size=128),
        "layers.1.attn": _ShapelessLayer(num_heads=16, num_kv_heads=2, head_size=128),
        "layers.0.attn.indexer.k_cache": _CacheOnlyLayer(),
        "layers.1.attn.indexer.k_cache": _CacheOnlyLayer(),
        "layers.0.attn.compressor.state_cache": _CacheOnlyLayer(),
    }
    with ForwardHarness(MODEL, engine_args=engine_args) as harness:
        monkeypatch.setattr(harness, "_attention_layers", lambda: layers)
        selection = harness.selection_metadata()
    assert selection.num_attention_layers == len(selection.attention_layers) == 5
    assert selection.num_model_layers == 2
    assert selection.layer_kinds == (("_CacheOnlyLayer", 3), ("_ShapelessLayer", 2))


def test_per_layer_counts_do_not_divide_by_cache_only_layers(capture):
    """Operators run once per decoder layer, not once per `AttentionLayerBase`.

    A sparse-attention model reports four times its depth in attention layers,
    which would understate every per-layer operator by that factor.
    """
    inflated = replace(
        capture.selection,
        num_attention_layers=4 * capture.selection.num_attention_layers,
    )
    before = format_report(capture).splitlines()
    after = format_report(replace(capture, selection=inflated)).splitlines()
    assert len(before) == len(after)
    differing = [line for line, other in zip(before, after) if line != other]
    assert len(differing) == 1
    assert differing[0].strip().startswith("attention layers")


def test_layer_kinds_explain_the_attention_layer_count(capture):
    """One number for several layer kinds reads as a wrong number without them."""
    selection = replace(
        capture.selection,
        num_attention_layers=9,
        layer_kinds=(("Attention", 6), ("DeepseekV4IndexerCache", 3)),
    )
    lines = format_report(replace(capture, selection=selection)).splitlines()
    (row,) = [line for line in lines if line.strip().startswith("attention layers")]
    assert row.endswith("9 (6 Attention, 3 DeepseekV4IndexerCache)")


def test_multimodal_items_need_a_multimodal_model():
    engine_args = EngineArgs(model=MODEL, max_model_len=1024)
    with pytest.raises(ValueError, match="no multimodal inputs"):
        ForwardHarness(MODEL, batch=BatchSpec(num_mm_items=1), engine_args=engine_args)


def test_multimodal_items_run_the_encoder():
    """Only a batch with multimodal items reaches the vision tower."""
    engine_args = EngineArgs(
        model=VL_MODEL,
        max_model_len=4096,
        hf_overrides={"text_config": {"num_hidden_layers": 1}},
    )
    batches = [BatchSpec(num_tokens=64), BatchSpec(num_tokens=2048, num_mm_items=1)]
    text, image = capture_batches(VL_MODEL, batches, engine_args=engine_args)
    assert not any(op.module.startswith("visual.") for op in text.ops)
    assert any(op.module.startswith("visual.") for op in image.ops)


def test_capture_allocates_no_accelerator_memory():
    """The point of a meta capture: it runs where the model would not fit."""
    if current_platform.is_cpu():
        pytest.skip("No accelerator to stay off of")
    device_module = torch.get_device_module(current_platform.device_type)
    allocated = device_module.memory_allocated()
    capture = _capture(num_hidden_layers=2)
    assert device_module.memory_allocated() == allocated
    assert capture.selection.device == "meta"


def test_kv_cache_scales_survive_meta_loading():
    """KV-cache scale post-processing branches on values meta tensors lack."""
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=1024,
        quantization="fp8",
        hf_overrides={"num_hidden_layers": 1},
    )
    capture = capture_model_ops(MODEL, engine_args=engine_args)
    assert any(op.name == ATTENTION_OP for op in capture.ops)


def test_attention_is_recorded_once_per_layer(capture):
    attention = [op for op in capture.ops if op.name == ATTENTION_OP]
    layers = capture.selection.num_attention_layers
    assert len(attention) == layers
    assert len({op.module for op in attention}) == layers


def test_op_counts_scale_with_layer_count(capture):
    """Twice the decoder layers, twice the operators, one layer's repeated.

    Only the first layer differs, normalizing its input where the rest add a
    residual to theirs.
    """
    layers = capture.selection.num_model_layers
    by_layer = _ops_by_layer(_capture(num_hidden_layers=2 * layers))
    assert len(by_layer) == 2 * layers
    repeated = by_layer["1"]
    assert all(ops == repeated for index, ops in by_layer.items() if index != "0")


@pytest.mark.parametrize(
    "schema,op_name,returns_input",
    [
        ("fills(Tensor(a!) out) -> ()", "fills", False),
        ("views(Tensor(a) x) -> Tensor(a)", "views", True),
    ],
)
def test_leaf_meta_kernel_derived_from_schema(
    leaf_library, schema, op_name, returns_input
):
    """Out-argument kernels return nothing; aliasing ones return the argument."""
    leaf_library.define(schema)
    register_meta_impls()

    tensor = torch.empty(4, device="meta")
    result = getattr(torch.ops._test_leaf_C, op_name)(tensor)
    assert result is tensor if returns_input else result is None


def test_leaf_op_with_unreadable_shape_fails_loudly(leaf_library):
    """Never guess a shape: an unknown kernel must name itself instead."""
    leaf_library.define("guesses(Tensor x, int n) -> Tensor")
    register_meta_impls()

    with pytest.raises(UnsupportedMetaOpError, match="guesses"):
        torch.ops._test_leaf_C.guesses(torch.empty(4, device="meta"), 2)


def test_keep_going_substitutes_placeholders_for_unknown_shapes(leaf_library):
    """Every gap in one run: an op with no meta shape is marked, not fatal."""
    leaf_library.define("guesses(Tensor x, int n) -> Tensor")
    register_meta_impls()

    x = torch.empty(4, 2, device="meta")
    with OpRecorder(torch.nn.Module(), keep_going=True) as recorder:
        out = torch.ops._test_leaf_C.guesses(x, 2)
        out.add_(1)
    assert out.shape == x.shape
    assert [op.placeholder for op in recorder.ops] == [True, False]


def test_keep_going_finishes_a_failing_custom_op_with_its_fake(glue_library):
    """A kernel reading a tensor's value on meta still yields the ops around it."""

    def reads_a_value(x: torch.Tensor) -> torch.Tensor:
        y = x.abs()
        if y.sum().item():
            y.add_(1)
        return y

    direct_register_custom_op(
        "reads_a_value",
        reads_a_value,
        fake_impl=lambda x: torch.empty_like(x),
        target_lib=glue_library,
    )
    x = torch.empty(4, 2, device="meta")
    with OpRecorder(torch.nn.Module(), keep_going=True) as recorder:
        out = torch.ops._test_glue.reads_a_value(x).mul(2)

    assert out.shape == x.shape
    glue, *inner, after = recorder.ops
    assert glue.body_error.startswith("RuntimeError: ")
    assert [op.name for op in inner[:2]] == ["aten::abs", "aten::sum"]
    assert all(op.depth == 1 for op in inner) and after.depth == 0
    with OpRecorder(torch.nn.Module()), pytest.raises(RuntimeError):
        torch.ops._test_glue.reads_a_value(x)


def test_op_registered_for_another_backend_only_is_flagged(leaf_library):
    """Meta never runs the platform's kernel, so an absent one must be found."""
    leaf_library.define("fills(Tensor(a!) out) -> ()")
    leaf_library.impl("fills", lambda out: None, "CPU")

    assert meta_ops.has_kernel_for("_test_leaf_C::fills", "CPU")
    assert not meta_ops.has_kernel_for("_test_leaf_C::fills", "CUDA")


def test_keep_going_reports_where_the_forward_pass_stopped(monkeypatch):
    """An error no placeholder can bridge ends the capture where it happened."""

    def unsupported(self, x):
        raise RuntimeError("no kernel for this")

    monkeypatch.setattr(Qwen2MLP, "forward", unsupported)
    capture = _capture(num_hidden_layers=1, keep_going=True)

    assert capture.failure is not None
    assert capture.failure.error == "RuntimeError: no kernel for this"
    assert capture.failure.module == "model.layers.0.mlp"
    assert capture.failure.location.startswith("vllm/model_executor/models/qwen2.py:")
    assert capture.ops
    assert all(".mlp" not in op.module for op in capture.ops)
    assert not capture.missing_kernels


def test_aligned_mamba_state_indices_follow_model_runner_v2(monkeypatch):
    """A builder that takes precomputed aligned state indices gets them as the
    V2 runner computes them, or the platform's runner cannot build it."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        Mamba1AttentionMetadataBuilder,
        "mamba_aligned_state_indices",
        None,
        raising=False,
    )
    engine_args = EngineArgs(
        model=MAMBA_MODEL,
        max_model_len=1024,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    harness = ForwardHarness(
        MAMBA_MODEL, batch=BatchSpec(num_reqs=2, num_tokens=8), engine_args=engine_args
    )
    if not current_platform.is_cuda_alike():
        with pytest.raises(NotImplementedError, match="CUDA-only kernel"), harness:
            pass
        return

    with harness:
        (group,) = [group for groups in harness.attn_groups for group in groups]
        indices = group.get_metadata_builder().mamba_aligned_state_indices
    num_state_slots = 1 + group.kv_cache_spec.num_speculative_blocks
    assert indices.shape == (2, num_state_slots)
    assert indices.dtype == torch.int32


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU model runner")
def test_capture_applies_the_model_runner_torch_cuda_aliases(monkeypatch):
    """Model code calls `torch.cuda` on XPU, as the real model runner allows."""
    forward = Qwen2MLP.forward

    def calls_torch_cuda(self, x):
        assert not torch.cuda.is_current_stream_capturing()
        return forward(self, x)

    monkeypatch.setattr(Qwen2MLP, "forward", calls_torch_cuda)
    before = dict(vars(torch.cuda))
    _capture(num_hidden_layers=1)
    assert vars(torch.cuda) == before


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU kernels")
def test_xpu_mhc_overrides_match_the_kernels():
    """Hand-written output shapes must agree with what the kernel returns."""
    tokens, hc_mult, hidden_size = 5, 4, 256
    mix = (2 + hc_mult) * hc_mult
    residual = torch.randn(
        tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device="xpu"
    )
    pre_args = (
        residual,
        torch.randn(mix, hc_mult * hidden_size, device="xpu") * 0.01,
        torch.ones(3, device="xpu"),
        torch.zeros(mix, device="xpu"),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        3,
    )
    post_mix, comb_mix, _ = torch.ops._xpu_C.mhc_pre(*pre_args)
    post_args = (torch.randn_like(residual[:, 0]), residual, post_mix, comb_mix)
    for name, args in (
        ("mhc_pre", pre_args),
        ("mhc_post", post_args),
        ("mhc_fused_post_pre", post_args + pre_args[1:]),
    ):
        op = getattr(torch.ops._xpu_C, name)
        real = op(*args)
        arguments = {
            argument.name: value.to("meta") if torch.is_tensor(value) else value
            for argument, value in zip(op.default._schema.arguments, args)
        }
        fake = meta_ops.OVERRIDES[f"_xpu_C::{name}"](arguments)
        real, fake = (t if isinstance(t, tuple) else (t,) for t in (real, fake))
        assert [(t.shape, t.dtype) for t in fake] == [
            (t.shape, t.dtype) for t in real
        ], name


def test_register_meta_impls_never_replaces_a_kernel():
    """Only gaps are filled, so ops vLLM already faked keep their own kernel."""
    registered = register_meta_impls()
    assert "vllm::unified_kv_cache_update" not in registered
    assert register_meta_impls() == registered


if HAS_TRITON:

    @triton.jit
    def fill_kernel(out_ptr, n: tl.constexpr):
        tl.store(out_ptr + tl.arange(0, n), 1.0)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_meta_triton_launch_is_recorded_not_run():
    """A raw Triton kernel handed meta tensors is dropped before it compiles."""
    out = torch.empty(8, device="meta")
    skipped = []
    with meta_ops.skip_meta_triton_launches(
        lambda name, args, kwargs: skipped.append((name, args, kwargs))
    ):
        fill_kernel[(1,)](out, n=8)
    assert skipped == [(fill_kernel.__qualname__, (out,), {"n": 8})]


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_gather_initial_states_is_recorded_on_meta():
    """The KDA and Kimi GDN prefill gather accepts meta state and is recorded."""
    state = torch.empty(4, 2, 8, device="meta")
    indices = torch.empty(3, dtype=torch.int32, device="meta")
    has_initial_state = torch.empty(3, dtype=torch.bool, device="meta")
    skipped = []
    with meta_ops.skip_meta_triton_launches(lambda name, *_: skipped.append(name)):
        output = gather_initial_states(state, indices, has_initial_state)
    assert output.shape == (3, 2, 8) and output.is_meta
    assert skipped == ["_gather_initial_states_kernel"]


def test_per_tensor_fp8_requantization_passes_meta_weights_through():
    """A static per-tensor FP8 checkpoint loads on meta, as Mistral-Medium-3.5's.

    Whether to requantize fused shards depends on the loaded scale values, which
    meta has none of; either way the shapes are the same.
    """
    weight = torch.empty(6, 4, dtype=torch.float8_e4m3fn, device="meta")
    weight_scale = torch.empty(2, dtype=torch.float32, device="meta")
    max_scale, requantized = requantize_with_max_scale(weight, weight_scale, [4, 2])
    assert max_scale.shape == () and max_scale.is_meta
    assert requantized is weight


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_meta_triton_launch_bypassing_grid_fails_loudly():
    """A launch that cannot be dropped raises rather than run on null pointers."""
    out = torch.empty(8, device="meta")
    with (
        meta_ops.skip_meta_triton_launches(lambda *_: None),
        pytest.raises(UnsupportedMetaOpError, match="fill_kernel"),
    ):
        fill_kernel.run(out, n=8, grid=(1,), warmup=False)


def test_capture_files_list_every_op(capture, tmp_path):
    write_capture_files(capture, tmp_path)

    distinct = (tmp_path / "ops.txt").read_text().splitlines()
    assert distinct == sorted({op.name for op in capture.ops})
    sequence = (tmp_path / "ops.sequence.txt").read_text().splitlines()
    assert len(sequence) == len(capture.ops)
    attention = [line for line in sequence if line.startswith(ATTENTION_OP)]
    assert len(attention) == capture.selection.num_attention_layers
    assert all(line.endswith("[full]") for line in attention)


@pytest.mark.slow_test
def test_meta_capture_matches_hardware():
    """The acceptance check: a meta run dispatches what the hardware run does."""
    if current_platform.is_cpu():
        pytest.skip("No accelerator to compare against")
    diff = compare_devices(
        MODEL, current_platform.device_type, batch=BatchSpec(num_tokens=8)
    )
    assert diff.equal, format_diff(diff)


@pytest.mark.parametrize(
    ("model", "batch", "max_model_len"),
    [
        # Encoder-only attention keeps no KV cache, and a pooler replaces logits.
        ("BAAI/bge-small-en-v1.5", BatchSpec(num_reqs=2, num_tokens=16), 512),
        # Mamba decode reads the prefill flags and the SSU backend.
        (
            MAMBA_MODEL,
            BatchSpec(num_reqs=4, num_tokens=4, num_computed_tokens=16),
            1024,
        ),
        # The vision tower's output replaces the first tokens' embeddings, and
        # M-RoPE takes three rows of positions.
        (VL_MODEL, BatchSpec(num_tokens=2048, num_mm_items=1), 4096),
    ],
)
def test_meta_capture_matches_hardware_beyond_decoders(model, batch, max_model_len):
    """Models the runner prepares differently from a decoder still match."""
    if current_platform.is_cpu():
        pytest.skip("No accelerator to compare against")
    engine_args = EngineArgs(
        model=model, max_model_len=max_model_len, load_format="dummy"
    )
    diff = compare_devices(
        model, current_platform.device_type, batch=batch, engine_args=engine_args
    )
    assert diff.equal, format_diff(diff)
