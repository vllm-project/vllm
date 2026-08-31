# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch
from utils import skip_unsupported

import vllm.model_executor.determinism.batch_invariant as batch_invariant_module
import vllm.model_executor.determinism.batch_invariant_configs as config_module
from vllm.model_executor.determinism.batch_invariant import (
    addmm_batch_invariant,
    matmul_batch_invariant,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    _get_tuned_matmul_arch_family,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_batch_invariant_matmul_custom_op_schema(monkeypatch, with_bias):
    monkeypatch.setattr(
        batch_invariant_module, "_warmup_tuned_matmul_configs", lambda *args: None
    )
    a = torch.rand((8, 64), dtype=torch.bfloat16, device=DEVICE_TYPE)
    b = torch.rand((64, 32), dtype=torch.bfloat16, device=DEVICE_TYPE)
    bias = (
        torch.rand((32,), dtype=torch.bfloat16, device=DEVICE_TYPE)
        if with_bias
        else None
    )

    torch.library.opcheck(
        torch.ops.vllm.batch_invariant_matmul.default,
        (a, b, bias),
    )


def test_compiled_addmm_selects_tuned_config_from_runtime_m(monkeypatch):
    has_bias_values = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(a, b, c, *args, **meta):
                has_bias_values.append(meta["HAS_BIAS"])
                c.fill_(meta["BLOCK_SIZE_M"])

            return launch

    monkeypatch.setattr(
        batch_invariant_module, "matmul_kernel_persistent", RecordingKernel()
    )
    monkeypatch.setattr(batch_invariant_module, "num_compute_units", lambda _: 1)
    monkeypatch.setattr(
        config_module,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"],
    )
    monkeypatch.setattr(config_module, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)

    compile_count = 0

    def counting_backend(graph_module, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled_addmm = torch.compile(
        addmm_batch_invariant,
        backend=counting_backend,
        dynamic=True,
        fullgraph=True,
    )
    b = torch.empty((2048, 4096), dtype=torch.bfloat16, device=DEVICE_TYPE)
    bias = torch.empty((4096,), dtype=torch.bfloat16, device=DEVICE_TYPE)

    small = compiled_addmm(
        bias, torch.empty((8, 2048), dtype=torch.bfloat16, device=DEVICE_TYPE), b
    )
    large = compiled_addmm(
        bias, torch.empty((2048, 2048), dtype=torch.bfloat16, device=DEVICE_TYPE), b
    )

    assert compile_count == 1
    assert has_bias_values == [True, True]
    assert torch.all(small == 16)
    assert torch.all(large == 128)


@pytest.mark.parametrize(
    "dtype,b_shape,has_table",
    [
        pytest.param(torch.bfloat16, (64, 32), True, id="untuned-shape"),
        pytest.param(torch.float16, (2048, 4096), True, id="untuned-dtype"),
        pytest.param(torch.bfloat16, (2048, 4096), False, id="untuned-device"),
    ],
)
def test_compiled_matmul_keeps_untuned_paths_in_graph(
    monkeypatch, dtype, b_shape, has_table
):
    monkeypatch.setattr(
        config_module,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"] if has_table else None,
    )
    monkeypatch.setattr(config_module, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    graphs = []

    def inspect_backend(graph_module, example_inputs):
        graphs.append(graph_module.graph)
        return graph_module.forward

    compiled_matmul = torch.compile(
        matmul_batch_invariant,
        backend=inspect_backend,
        dynamic=True,
        fullgraph=True,
    )
    K, N = b_shape
    a = torch.empty((8, K), dtype=dtype, device=DEVICE_TYPE)
    b = torch.empty(b_shape, dtype=dtype, device=DEVICE_TYPE)

    compiled_matmul(a, b)

    assert len(graphs) == 1
    assert "vllm.batch_invariant_matmul" not in str(graphs[0])


@skip_unsupported
def test_compiled_matmul_runtime_m_dispatch_is_correct(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"],
    )
    monkeypatch.setattr(config_module, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    torch._dynamo.reset()

    compiled_matmul = torch.compile(
        matmul_batch_invariant,
        dynamic=True,
        fullgraph=True,
    )
    torch.manual_seed(42)
    b = torch.rand((2048, 4096), dtype=torch.bfloat16, device=DEVICE_TYPE)
    shared_row = torch.rand((2048,), dtype=torch.bfloat16, device=DEVICE_TYPE)
    first_row_outputs = []

    try:
        for m in (8, 256):
            a = torch.rand((m, 2048), dtype=torch.bfloat16, device=DEVICE_TYPE)
            a[0] = shared_row
            actual = compiled_matmul(a, b)
            expected = torch.matmul(a, b)
            torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
            first_row_outputs.append(actual[0])
        assert torch.equal(*first_row_outputs)
    finally:
        torch._dynamo.reset()


def test_tuned_matmul_warmup_covers_all_table_shapes(monkeypatch):
    calls = []

    class RecordingKernel:
        def warmup(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        batch_invariant_module, "matmul_kernel_persistent", RecordingKernel()
    )
    monkeypatch.setattr(batch_invariant_module, "num_compute_units", lambda _: 1)
    monkeypatch.setattr(
        config_module,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"],
    )
    monkeypatch.setattr(config_module, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    monkeypatch.setattr(batch_invariant_module, "_WARMED_TUNED_MATMUL_CONFIGS", set())

    batch_invariant_module._warmup_tuned_matmul_configs(
        current_platform.current_device(), 2048, 1, 1, 2048, False
    )

    def contains(runtime_m, block_m, block_n, num_warps, num_stages):
        return any(
            args[4] == runtime_m
            and kwargs["BLOCK_SIZE_M"] == block_m
            and kwargs["BLOCK_SIZE_N"] == block_n
            and kwargs["num_warps"] == num_warps
            and kwargs["num_stages"] == num_stages
            for args, kwargs in calls
        )

    assert contains(2, 16, 128, 8, 2)
    assert contains(16, 128, 64, 8, 4)
    assert contains(1, 16, 32, 8, 5)
    assert contains(2, 16, 64, 4, 4)
    assert any(kwargs["C_LARGE"] for _, kwargs in calls if kwargs["BLOCK_SIZE_N"] == 64)
    assert {args[9:11] for args, _ in calls} == {(1, 16)}
    assert {kwargs["HAS_BIAS"] for _, kwargs in calls} == {False}


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """
    Compare matmul_batch_invariant against torch.matmul for various shapes.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """
    Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """

    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@skip_unsupported
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["contiguous", "transposed"])
def test_matmul_batch_invariance_across_tuned_m_buckets(m, transpose_b):
    # Tuned M buckets must preserve each row's K-reduction order.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS:
        pytest.skip("No tuned persistent matmul config for this architecture")

    device = torch.device(DEVICE_TYPE)
    n = k = 2048
    torch.manual_seed(42)
    a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    if transpose_b:
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device).t()
    else:
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    single_output = matmul_batch_invariant(a[:1], b)
    batch_output = matmul_batch_invariant(a, b)

    assert torch.equal(single_output[0], batch_output[0])
