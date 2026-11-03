# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and integration contracts for the BF16 gated projection."""

import json

import pytest
import torch
import torch.nn.functional as F

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.gated_linear_silu import (
    GatedLinearSilu,
    create_gated_linear_silu,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="The kernel requires NVIDIA BF16 tensor cores",
)


@pytest.fixture
def single_rank(tmp_path):
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        yield
        return
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{tmp_path / 'distributed-init'}",
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def reference(x, weight):
    projected = F.linear(x, weight)
    out = torch.empty(
        (*projected.shape[:-1], projected.shape[-1] // 2),
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops._C.silu_and_mul(out, projected)
    return out


def operands(rows, hidden=2560, intermediate=9216, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return (
        torch.randn(rows, hidden, dtype=dtype, device="cuda") * 0.05,
        torch.randn(2 * intermediate, hidden, dtype=dtype, device="cuda") * 0.05,
    )


@pytest.mark.parametrize("rows", [768, 769, 1056, 1584, 2048, 4096])
@torch.inference_mode()
def test_gated_projection_preserves_declared_rounding(rows):
    x, weight = operands(rows)
    actual = torch.ops.vllm.gated_linear_silu(x, weight)
    expected = reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    relative_rmse = (actual.float() - expected.float()).square().mean().sqrt()
    relative_rmse /= expected.float().square().mean().sqrt()
    assert relative_rmse <= 0.0035
    projected = (x[:16].double() @ weight.double().T).to(torch.bfloat16)
    gate, up = projected.chunk(2, dim=-1)
    activated = (gate.double() / (1 + torch.exp(-gate.double()))).to(torch.bfloat16)
    precise = (activated.double() * up.double()).to(torch.bfloat16)
    torch.testing.assert_close(actual[:16], precise, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "rows,hidden,intermediate,dtype",
    [
        (16, 2560, 9216, torch.bfloat16),
        (64, 2560, 9216, torch.bfloat16),
        (528, 2560, 9216, torch.bfloat16),
        (767, 2560, 9216, torch.bfloat16),
        (4097, 2560, 9216, torch.bfloat16),
        (768, 128, 65, torch.bfloat16),
        (768, 2560, 9216, torch.float16),
    ],
)
@torch.inference_mode()
def test_ineligible_inputs_preserve_native_output_exactly(
    rows, hidden, intermediate, dtype
):
    x, weight = operands(rows, hidden, intermediate, dtype)
    actual = torch.ops.vllm.gated_linear_silu(x, weight)
    expected = reference(x, weight)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@torch.inference_mode()
def test_strided_input_uses_native_fallback():
    x, weight = operands(768)
    storage = torch.empty((768, 5120), dtype=x.dtype, device=x.device)
    strided = storage[:, ::2]
    strided.copy_(x)
    actual = torch.ops.vllm.gated_linear_silu(strided, weight)
    expected = reference(strided, weight)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@torch.inference_mode()
def test_supported_prefill_executes_the_new_kernel_and_small_rows_do_not(tmp_path):
    for rows, should_fuse in [(768, True), (528, False)]:
        x, weight = operands(rows)
        torch.ops.vllm.gated_linear_silu(x, weight)
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            torch.ops.vllm.gated_linear_silu(x, weight)
            torch.accelerator.synchronize()
        path = tmp_path / f"dispatch-{rows}.json"
        profile.export_chrome_trace(str(path))
        kernels = [
            e["name"]
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ]
        assert kernels
        assert (
            any("_gated_linear_silu_kernel" in name for name in kernels) == should_fuse
        )


@torch.inference_mode()
def test_schema_fake_and_compiled_runtime_shape_dispatch():
    x, weight = operands(768)
    torch.library.opcheck(
        torch.ops.vllm.gated_linear_silu.default,
        (x, weight),
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
    )
    compiled = torch.compile(
        torch.ops.vllm.gated_linear_silu, fullgraph=True, dynamic=True
    )
    for rows in [16, 768, 1584, 528]:
        x, _ = operands(rows)
        actual = compiled(x, weight)
        expected = torch.ops.vllm.gated_linear_silu(x, weight)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@torch.inference_mode()
def test_graph_replay_reads_updated_original_weight_storage():
    x, weight = operands(768)
    pointer = weight.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.gated_linear_silu(x, weight)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = torch.ops.vllm.gated_linear_silu(x, weight)
    weight.zero_()
    output.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    assert weight.data_ptr() == pointer
    assert torch.equal(output, torch.zeros_like(output))


def real_linears():
    gate = MergedColumnParallelLinear(
        2560,
        [9216, 9216],
        bias=False,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    down = RowParallelLinear(
        9216,
        2560,
        bias=False,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    return gate, down, SiluAndMul(compile_native=False)


@pytest.mark.parametrize("mode", ["default", "enabled", "disabled"])
def test_real_linear_factory_respects_custom_op_control(mode, single_rank):
    custom_ops = {
        "default": ["all"],
        "enabled": ["all", "+gated_linear_silu"],
        "disabled": ["all", "-gated_linear_silu"],
    }[mode]
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=custom_ops))
    with set_current_vllm_config(config), torch.device("cuda"):
        fused = create_gated_linear_silu(*real_linears())
        if mode == "enabled" and current_platform.get_device_capability() == (8, 9):
            assert isinstance(fused, GatedLinearSilu)
        else:
            assert fused is None


def test_real_linear_factory_keeps_batch_invariant_path(monkeypatch, single_rank):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+gated_linear_silu"])
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        assert create_gated_linear_silu(*real_linears()) is None


@pytest.mark.parametrize("backend", ["uva", "prefetch"])
def test_weight_offloading_keeps_native_linears(backend, single_rank):
    from vllm.config.offload import (
        OffloadConfig,
        PrefetchOffloadConfig,
        UVAOffloadConfig,
    )

    offload = (
        OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=1))
        if backend == "uva"
        else OffloadConfig(prefetch=PrefetchOffloadConfig(offload_group_size=2))
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+gated_linear_silu"]),
        offload_config=offload,
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        assert create_gated_linear_silu(*real_linears()) is None
