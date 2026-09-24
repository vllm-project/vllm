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
from vllm.platforms import current_platform
from vllm.profiler.op_capture import (
    BatchSpec,
    ForwardHarness,
    OpCapture,
    UnsupportedMetaOpError,
    capture_model_ops,
    compare_devices,
    format_diff,
    meta_ops,
    register_meta_impls,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_PREFIX = "model.layers."
ATTENTION_OP = "vllm::unified_attention_with_output"


def _capture(num_hidden_layers: int) -> OpCapture:
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=1024,
        hf_overrides={"num_hidden_layers": num_hidden_layers},
    )
    return capture_model_ops(MODEL, engine_args=engine_args)


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
    monkeypatch.setattr(
        meta_ops, "LEAF_NAMESPACES", meta_ops.LEAF_NAMESPACES | {namespace}
    )
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
