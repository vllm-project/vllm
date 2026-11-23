# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact producer/transport regressions for Humming's opt-in FP8 dispatch."""

from contextlib import nullcontext

import pytest
import torch

from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingIndexedExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.humming.moe import (
    _make_humming_fp8_dispatch_quantizer,
    make_humming_moe_kernel,
    make_humming_moe_quant_config,
    snapshot_humming_fp8_dispatch_enabled,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming
from vllm.v1.worker.workspace import init_workspace_manager

from .modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)


def _quant_config(group_size: int = 0, dtype: torch.dtype = torch.bfloat16):
    from humming import dtypes
    from humming.config import LayerConfig

    native_kwargs = dict(
        num_experts=2,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16 if dtype == torch.bfloat16 else dtypes.float16,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float8e8m0,
        input_quant_mode="dynamic_group" if group_size else "dynamic_token",
        input_scale_group_size=group_size,
        weight_scale_group_size=32,
        weight_scale_type="group",
    )
    return make_humming_moe_quant_config(
        quant_dtype=current_platform.fp8_dtype() if group_size else "float8e4m3",
        weight_dtype="float4e2m1",
        activation_group_shape=GroupShape(1, group_size) if group_size else None,
        humming_configs={
            "w13": LayerConfig(shape_n=256, shape_k=256, **native_kwargs),
            "w2": LayerConfig(shape_n=256, shape_k=128, **native_kwargs),
        },
    )


def _make_kernel(
    vllm_config: VllmConfig, device: torch.device, dtype: torch.dtype = torch.bfloat16
):
    quant_config = _quant_config(dtype=dtype)
    original = vllm_config.parallel_config
    parallel = FusedMoEParallelConfig.make(
        tp_size_=original.tensor_parallel_size,
        pcp_size_=original.prefill_context_parallel_size,
        dp_size_=original.data_parallel_size,
        sp_size_=(
            original.tensor_parallel_size if original.use_sequence_parallel_moe else 1
        ),
        vllm_parallel_config=original,
    )
    num_experts = 2 * parallel.ep_size
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=dtype,
        elastic_ep_max_dp_size=vllm_config.parallel_config.elastic_ep_max_dp_size,
    )
    with set_current_vllm_config(vllm_config):
        enabled = snapshot_humming_fp8_dispatch_enabled()
    # Weight post-processing happens after the model's construction context exits.
    return make_humming_moe_kernel(
        quant_config,
        moe_config,
        HummingIndexedExperts,
        enabled=enabled,
    )


def _inputs(
    rows: int,
    rank: int,
    device: torch.device,
    misaligned: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    num_experts: int = 4,
):
    generator = torch.Generator(device=device).manual_seed(42 + rank)
    x = torch.randn(rows, 256, dtype=dtype, device=device, generator=generator)
    if rows:
        x[0].zero_()
    if misaligned:
        assert rows > 0
        storage = torch.empty(x.numel() + 16, dtype=x.dtype, device=device)
        assert storage.data_ptr() % 16 == 0
        offset = storage[1 : 1 + x.numel()].view_as(x)
        offset.copy_(x)
        x = offset
    assert x.is_contiguous() and x.data_ptr() % 16 == (2 if misaligned else 0)
    first_weight = (
        torch.arange(rows, dtype=torch.float32, device=device) + 1 + rank
    ) / 1024
    weights = torch.stack((first_weight, 1 - first_weight), dim=1)
    ids = (
        torch.arange(rows * 2, device=device, dtype=torch.int32).reshape(rows, 2) + rank
    ) % num_experts
    return x, weights, ids


def _prepare(kernel, inputs):
    return kernel.prepare_finalize.prepare(
        *inputs,
        num_experts=kernel.fused_experts.moe_config.num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=kernel.fused_experts.quant_config,
        defer_input_quant=kernel.fused_experts.expects_unquantized_inputs,
    )


def _dispatch_context(config: VllmConfig, dp_rows: tuple[int, ...], mode):
    parallel = config.parallel_config
    return set_forward_context(
        None,
        config,
        num_tokens=dp_rows[parallel.data_parallel_rank],
        num_tokens_across_dp=torch.tensor(dp_rows, device="cpu"),
        cudagraph_runtime_mode=mode,
    )


def _local_sizes(config: VllmConfig):
    """Use the real runner's DP/PCP/SP row expansion."""
    metadata = get_forward_context().dp_metadata
    parallel = config.parallel_config
    return (
        metadata.sp_local_sizes(
            parallel.tensor_parallel_size if parallel.use_sequence_parallel_moe else 1,
            pcp_size=parallel.prefill_context_parallel_size,
            use_ep=parallel.enable_expert_parallel,
        )
        if metadata is not None
        else nullcontext()
    )


def _local_rows(config: VllmConfig, dp_rows: tuple[int, ...]) -> int:
    parallel = config.parallel_config
    sp = parallel.tensor_parallel_size if parallel.use_sequence_parallel_moe else 1
    return (dp_rows[parallel.data_parallel_rank] + sp - 1) // sp


def _assert_prequantized(kernel, actual, reference):
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[0].shape == reference[0].shape
    assert actual[1].dtype == torch.float32
    assert actual[1].shape == (reference[0].shape[0], 1)
    actual_q, actual_scale = actual[0].clone(), actual[1].clone()
    if reference[0].shape[0]:
        q, scale, _ = kernel.fused_experts.process_input("w13", reference[0], None)
        assert torch.equal(actual_q.view(torch.uint8), q.view(torch.uint8))
        torch.testing.assert_close(actual_scale, scale, atol=0, rtol=0)
    for index in (3, 4):
        torch.testing.assert_close(actual[index], reference[index], atol=0, rtol=0)
    q, scale, _ = kernel.fused_experts.process_input(
        "w13", actual[0], None, input_scale=actual[1]
    )
    assert q is actual[0] and scale is actual[1]


def _dispatch_worker(
    pgi: ProcessGroupInfo, config: VllmConfig, cpu_group, dtype: torch.dtype
):
    init_workspace_manager(pgi.device)
    baseline = _make_kernel(config, pgi.device, dtype)
    enabled_config = VllmConfig(
        parallel_config=config.parallel_config,
        kernel_config=KernelConfig(humming_fp8_dispatch=True),
    )
    enabled = _make_kernel(enabled_config, pgi.device, dtype)
    prepare = enabled.prepare_finalize
    assert isinstance(prepare, MoEPrepareAndFinalizeNaiveDPEPModular)
    assert prepare._input_quantizer is not None
    assert enabled.fused_experts.expects_unquantized_inputs
    parallel = config.parallel_config
    moe = enabled.fused_experts.moe_config
    assert moe.tp_size == 1 and moe.ep_size == pgi.world_size
    assert moe.is_sequence_parallel == parallel.use_sequence_parallel_moe
    assert (
        _make_humming_fp8_dispatch_quantizer(
            _quant_config(128), moe, HummingIndexedExperts, enabled=True
        )
        is None
    )
    with set_current_vllm_config(enabled_config):
        legacy = make_humming_moe_kernel(
            enabled.fused_experts.quant_config, moe, HummingIndexedExperts
        )
    legacy_prepare = legacy.prepare_finalize
    assert isinstance(legacy_prepare, MoEPrepareAndFinalizeNaiveDPEPModular)
    assert legacy_prepare._input_quantizer is None
    dp_cases: list[tuple[int, ...]] = (
        [(127, 128), (128, 128), (128, 129), (0, 257), (257, 0)]
        if parallel.data_parallel_size == 2
        else [(127,), (128,), (129,), (0,)]
    )
    small_cases: list[tuple[int, ...]] = (
        [(1, 1), (0, 1), (1, 0), (0, 0)]
        if parallel.data_parallel_size == 2
        else [(1,), (0,)]
    )
    cases: list[tuple[tuple[int, ...], CUDAGraphMode, int | None]] = [
        (sizes, CUDAGraphMode.NONE, None) for sizes in dp_cases + small_cases
    ]
    full_rows = (128,) * parallel.data_parallel_size
    cases += [
        (full_rows, CUDAGraphMode.FULL, None),
        (full_rows, CUDAGraphMode.PIECEWISE, None),
        (full_rows, CUDAGraphMode.NONE, 0),
        (full_rows, CUDAGraphMode.NONE, pgi.world_size - 1),
    ]
    for sizes, mode, misaligned_rank in cases:
        inputs = _inputs(
            _local_rows(config, sizes),
            pgi.rank,
            pgi.device,
            misaligned=pgi.rank == misaligned_rank,
            dtype=dtype,
            num_experts=moe.num_experts,
        )
        unchanged = inputs[0].clone() if misaligned_rank is not None else None
        with _dispatch_context(config, sizes, mode), _local_sizes(config):
            reference = tuple(
                value.clone() if isinstance(value, torch.Tensor) else value
                for value in _prepare(baseline, inputs)
            )
            assert reference[0].dtype == dtype and reference[1] is None
            metadata = get_forward_context().dp_metadata
            expected_rows = (
                sum(metadata.get_chunk_sizes_across_dp_rank())
                if metadata is not None
                else inputs[0].shape[0] * parallel.prefill_context_parallel_size
            )
            assert reference[0].shape[0] == expected_rows
            actual = _prepare(enabled, inputs)
            _assert_prequantized(baseline, actual, reference)
            deferred = _prepare(legacy, inputs)
            assert deferred[0].dtype == dtype and deferred[1] is None
            torch.testing.assert_close(deferred[0], reference[0], atol=0, rtol=0)
        if unchanged is not None:
            torch.testing.assert_close(inputs[0], unchanged, atol=0, rtol=0)

    inputs = _inputs(
        _local_rows(config, full_rows),
        pgi.rank,
        pgi.device,
        dtype=dtype,
        num_experts=moe.num_experts,
    )
    with (
        _dispatch_context(config, full_rows, CUDAGraphMode.FULL),
        _local_sizes(config),
        graph_capture(device=pgi.device),
    ):
        _prepare(baseline, inputs)
        _prepare(enabled, inputs)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _prepare(enabled, inputs)
            q, scale, _ = enabled.fused_experts.process_input(
                "w13", captured[0], None, input_scale=captured[1]
            )
            assert q is captured[0] and scale is captured[1]
        assert captured[0].dtype == torch.float8_e4m3fn
        assert captured[1].dtype == torch.float32
        for factor in (1.0, -2.0):
            inputs[0].mul_(factor)
            reference = tuple(
                value.clone() if isinstance(value, torch.Tensor) else value
                for value in _prepare(baseline, inputs)
            )
            graph.replay()
            _assert_prequantized(baseline, captured, reference)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and has_humming()),
    reason="Requires CUDA and Humming",
)
@pytest.mark.skipif(current_platform.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("dp_size,pcp_size", [(2, 1), (1, 2)], ids=["dp2", "pcp2"])
def test_humming_per_token_fp8_dispatch_preserves_rows_and_graph(
    dtype, dp_size, pcp_size
):
    """The real producer/AgRs path preserves row scales and rank-distinct routes."""
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=dp_size,
            tensor_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
    )
    parallel_launch_with_config(2, _dispatch_worker, config, None, dtype=dtype)


def _standard_method_worker(pgi: ProcessGroupInfo, config: VllmConfig, cpu_group):
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.quantization.humming import (
        HummingConfig,
        HummingMoEMethod,
    )

    init_workspace_manager(pgi.device)
    parallel = config.parallel_config
    num_experts = 2 * pgi.world_size
    layers = []
    for enabled in (False, True):
        construction = VllmConfig(
            parallel_config=config.parallel_config,
            kernel_config=KernelConfig(
                moe_backend="humming", humming_fp8_dispatch=enabled
            ),
        )
        with set_current_vllm_config(construction):
            runner = FusedMoEFactory(
                num_experts=num_experts,
                top_k=2,
                hidden_size=256,
                intermediate_size=128,
                params_dtype=torch.bfloat16,
                quant_config=HummingConfig(),
                tp_size=parallel.tensor_parallel_size,
                dp_size=parallel.data_parallel_size,
                pcp_size=parallel.prefill_context_parallel_size,
                is_sequence_parallel=parallel.use_sequence_parallel_moe,
                prefix=f"humming_dispatch_{enabled}",
            )
            layer = runner.routed_experts
            assert type(layer.quant_method) is HummingMoEMethod
            for expert_id in range(num_experts):
                for index, shard in enumerate(("w1", "w3", "w2")):
                    name = "w2_weight" if shard == "w2" else "w13_weight"
                    shape = (256, 128) if shard == "w2" else (128, 256)
                    generator = torch.Generator(device=pgi.device).manual_seed(
                        100 + 3 * expert_id + index
                    )
                    weight = (
                        torch.randn(
                            shape,
                            dtype=torch.bfloat16,
                            device=pgi.device,
                            generator=generator,
                        )
                        / 16
                    )
                    param = layer.get_parameter(name)
                    param.weight_loader(
                        param, weight, "weight", shard_id=shard, expert_id=expert_id
                    )
        # The construction config is gone; the outer config has feature=False.
        layer.quant_method.process_weights_after_loading(layer)
        layers.append(layer)

    baseline, enabled_kernel = (layer.quant_method.moe_kernel for layer in layers)
    assert baseline is not None and enabled_kernel is not None
    baseline_prepare = baseline.prepare_finalize
    enabled_prepare = enabled_kernel.prepare_finalize
    assert isinstance(baseline_prepare, MoEPrepareAndFinalizeNaiveDPEPModular)
    assert isinstance(enabled_prepare, MoEPrepareAndFinalizeNaiveDPEPModular)
    assert baseline_prepare._input_quantizer is None
    assert enabled_prepare._input_quantizer is not None
    baseline_experts, enabled_experts = (
        baseline.fused_experts,
        enabled_kernel.fused_experts,
    )
    assert isinstance(baseline_experts, HummingIndexedExperts)
    assert isinstance(enabled_experts, HummingIndexedExperts)
    parallel = config.parallel_config
    sizes = (128, 129) if parallel.data_parallel_size == 2 else (128,)
    inputs = _inputs(
        _local_rows(config, sizes), pgi.rank, pgi.device, num_experts=num_experts
    )
    with _dispatch_context(config, sizes, CUDAGraphMode.NONE), _local_sizes(config):
        reference = tuple(
            value.clone() if isinstance(value, torch.Tensor) else value
            for value in _prepare(baseline, inputs)
        )
        actual = _prepare(enabled_kernel, inputs)
        _assert_prequantized(baseline, actual, reference)
        outputs = []
        for layer in layers:
            method = layer.quant_method
            assert isinstance(method, HummingMoEMethod)
            outputs.append(
                method.apply(layer, inputs[0], inputs[1], inputs[2], None, None).clone()
            )
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and has_humming()),
    reason="Requires CUDA and Humming",
)
@pytest.mark.skipif(current_platform.device_count() < 2, reason="Requires two GPUs")
def test_humming_fp8_dispatch_standard_method_post_load(monkeypatch):
    """The ordinary Humming factory retains dispatch policy through post-load."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "indexed")
    monkeypatch.setenv(
        "VLLM_HUMMING_ONLINE_QUANT_CONFIG", '{"dtype":"int4","group_size":128}'
    )
    monkeypatch.setenv("VLLM_HUMMING_INPUT_QUANT_CONFIG", '{"dtype":"float8e4m3"}')
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
        kernel_config=KernelConfig(moe_backend="humming"),
    )
    parallel_launch_with_config(2, _standard_method_worker, config, None)
