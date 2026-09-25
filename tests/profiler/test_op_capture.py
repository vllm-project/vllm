# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Meta-device operator capture: what it records, and that hardware agrees."""

from collections import defaultdict

import pytest
import torch

import vllm.model_executor.layers.attention.attention  # noqa: F401
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
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
    capture_model_ops,
    compare_devices,
    format_diff,
    meta_ops,
    register_meta_impls,
)
from vllm.profiler.op_capture import recorder as recorder_module
from vllm.triton_utils import HAS_TRITON, tl, triton

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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


def test_capture_allocates_no_accelerator_memory():
    """The point of a meta capture: it runs where the model would not fit."""
    if current_platform.is_cpu():
        pytest.skip("No accelerator to stay off of")
    device_module = torch.get_device_module(current_platform.device_type)
    allocated = device_module.memory_allocated()
    capture = _capture(num_hidden_layers=2)
    assert device_module.memory_allocated() == allocated
    assert capture.selection.device == "meta"


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
    layers = capture.selection.num_attention_layers
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
def test_meta_triton_launch_bypassing_grid_fails_loudly():
    """A launch that cannot be dropped raises rather than run on null pointers."""
    out = torch.empty(8, device="meta")
    with (
        meta_ops.skip_meta_triton_launches(lambda *_: None),
        pytest.raises(UnsupportedMetaOpError, match="fill_kernel"),
    ):
        fill_kernel.run(out, n=8, grid=(1,), warmup=False)


@pytest.mark.slow_test
def test_meta_capture_matches_hardware():
    """The acceptance check: a meta run dispatches what the hardware run does."""
    if current_platform.is_cpu():
        pytest.skip("No accelerator to compare against")
    diff = compare_devices(
        MODEL, current_platform.device_type, batch=BatchSpec(num_tokens=8)
    )
    assert diff.equal, format_diff(diff)
