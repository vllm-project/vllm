# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.hip_custom_communicator import (
    AllReduceAddAttnResRmsNormConfig,
    AllReduceAddRmsNormConfig,
    AllReduceConfig,
    HipCustomCommunicator,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only HipCustomCommunicator tests"
)

HIDDEN = 7168
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
MAX_TOKENS = 2048
EPS = 1e-5
ATTN_RES_EPS = 1e-6
INT4_ONE_SHOT_RELATIVE_RMS = 0.15
INT4_TWO_SHOT_RELATIVE_RMS = 0.25


def int4_bar(config) -> float:
    two_shot = config.quantization == "int4" and config.algorithm.startswith("two_shot")
    return INT4_TWO_SHOT_RELATIVE_RMS if two_shot else INT4_ONE_SHOT_RELATIVE_RMS


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    want = x.clone()
                    dist.all_reduce(want, group=group.device_group)
                    assert comm.should_allreduce(x)

                    out = comm.all_reduce(x)
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned("all_reduce", num_tokens, HIDDEN, None)
                        ), (num_tokens, dtype)
                    else:
                        assert torch.equal(out, want), (num_tokens, dtype)

                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce(x)
                        with torch.cuda.graph(graph):
                            out = comm.all_reduce(x)
                    x.copy_(x + 1)
                    want = x.clone()
                    dist.all_reduce(want, group=group.device_group)
                    graph.replay()
                    torch.accelerator.synchronize()
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned("all_reduce", num_tokens, HIDDEN, None)
                        ), (num_tokens, dtype)
                    else:
                        assert torch.equal(out, want), (num_tokens, dtype)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce(monkeypatch: pytest.MonkeyPatch, tp_size, quantization):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, all_reduce_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_rms_norm_no_residual_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.ir.ops import rms_norm

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g.manual_seed(0))
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want = rms_norm(summed, weight, EPS)
                    assert comm.should_allreduce_add_rms_norm(x, weight)

                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddRmsNormConfig.algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        out = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, config=config
                        )
                        if comm.arguments.quantization == "int4":
                            err = (out.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                config
                                if config is not None
                                else comm.tuned(
                                    "all_reduce_add_rms_norm", num_tokens, HIDDEN, False
                                )
                            ), (
                                num_tokens,
                                dtype,
                                config,
                            )
                        else:
                            torch.testing.assert_close(
                                out, want, atol=2e-2, rtol=2e-2, msg=str(config)
                            )

                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_rms_norm(x, weight, EPS)
                        with torch.cuda.graph(graph):
                            out = comm.all_reduce_add_rms_norm(x, weight, EPS)
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want = rms_norm(summed, weight, EPS)
                    graph.replay()
                    torch.accelerator.synchronize()
                    if comm.arguments.quantization == "int4":
                        err = (out.float() - want.float()).square().mean().sqrt()
                        rms = want.float().square().mean().sqrt()
                        assert err / rms <= int4_bar(
                            comm.tuned(
                                "all_reduce_add_rms_norm", num_tokens, HIDDEN, False
                            )
                        ), (num_tokens, dtype)
                    else:
                        torch.testing.assert_close(out, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_rms_norm_no_residual(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(
        monkeypatch, tp_size, 1, all_reduce_add_rms_norm_no_residual_ranks
    )


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_rms_norm_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.ir.ops import fused_add_rms_norm

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        try:
            for num_tokens in NUM_TOKENS:
                for dtype in [torch.bfloat16, torch.float16]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    g.manual_seed(0)
                    residual = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_out, want_residual = fused_add_rms_norm(
                        summed, residual.clone(), weight, EPS
                    )
                    assert comm.should_allreduce_add_rms_norm(x, weight, residual)

                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddRmsNormConfig.algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        out, residual_out = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, residual=residual, config=config
                        )
                        for got, want in [
                            (out, want_out),
                            (residual_out, want_residual),
                        ]:
                            if comm.arguments.quantization == "int4":
                                err = (
                                    (got.float() - want.float()).square().mean().sqrt()
                                )
                                rms = want.float().square().mean().sqrt()
                                assert err / rms <= int4_bar(
                                    config
                                    if config is not None
                                    else comm.tuned(
                                        "all_reduce_add_rms_norm",
                                        num_tokens,
                                        HIDDEN,
                                        True,
                                    )
                                ), (
                                    num_tokens,
                                    dtype,
                                    config,
                                )
                            else:
                                torch.testing.assert_close(
                                    got, want, atol=2e-2, rtol=2e-2, msg=str(config)
                                )

                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_rms_norm(x, weight, EPS, residual=residual)
                        with torch.cuda.graph(graph):
                            out, residual_out = comm.all_reduce_add_rms_norm(
                                x, weight, EPS, residual=residual
                            )
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_out, want_residual = fused_add_rms_norm(
                        summed, residual.clone(), weight, EPS
                    )
                    graph.replay()
                    torch.accelerator.synchronize()
                    for got, want in [(out, want_out), (residual_out, want_residual)]:
                        if comm.arguments.quantization == "int4":
                            err = (got.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                comm.tuned(
                                    "all_reduce_add_rms_norm", num_tokens, HIDDEN, True
                                )
                            ), (num_tokens, dtype)
                        else:
                            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_rms_norm(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, all_reduce_add_rms_norm_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_add_attn_res_rms_norm_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        dtype = torch.bfloat16
        try:
            for num_tokens in NUM_TOKENS:
                for has_prefix, num_blocks, write_index in [
                    (True, 4, -1),
                    (False, 4, 4),
                ]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randint(
                        -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    g.manual_seed(0)
                    prefix = (
                        torch.randint(
                            -8, 8, (num_tokens, HIDDEN), generator=g, device=device
                        ).to(dtype)
                        if has_prefix
                        else None
                    )
                    blocks = torch.randint(
                        -8, 8, (num_tokens, 10, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    norm_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    norm_weight.normal_(mean=1.0, std=0.1, generator=g)
                    qk_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    qk_weight.normal_(mean=0.0, std=0.1, generator=g)
                    output_norm_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    output_norm_weight.normal_(mean=1.0, std=0.1, generator=g)

                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_prefix = summed.clone() if prefix is None else prefix.clone()
                    want_blocks = blocks.clone()
                    want_out = attn_res(
                        want_prefix,
                        None if prefix is None else summed,
                        want_blocks,
                        norm_weight,
                        qk_weight,
                        output_norm_weight,
                        num_blocks,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    assert comm.should_allreduce_add_attn_res_rms_norm(
                        x,
                        prefix,
                        blocks,
                        norm_weight,
                        qk_weight,
                        output_norm_weight,
                        num_blocks,
                        write_index,
                    )

                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_attn_res_rms_norm",
                            algorithm,
                            num_tokens,
                            HIDDEN,
                            comm.arguments.quantization,
                        )
                        for algorithm in AllReduceAddAttnResRmsNormConfig.algorithms
                    ]
                    original_blocks = blocks.clone()
                    for config in [None, *(c for c in forced if c is not None)]:
                        blocks.copy_(original_blocks)
                        out_prefix, out = comm.all_reduce_add_attn_res_rms_norm(
                            x,
                            prefix,
                            blocks,
                            norm_weight,
                            qk_weight,
                            output_norm_weight,
                            num_blocks,
                            write_index,
                            ATTN_RES_EPS,
                            EPS,
                            config=config,
                        )
                        for got, want in [
                            (out, want_out),
                            (out_prefix, want_prefix),
                            (blocks, want_blocks),
                        ]:
                            if comm.arguments.quantization == "int4":
                                err = (
                                    (got.float() - want.float()).square().mean().sqrt()
                                )
                                rms = want.float().square().mean().sqrt()
                                assert err / rms <= int4_bar(
                                    config
                                    if config is not None
                                    else comm.tuned(
                                        "all_reduce_add_attn_res_rms_norm",
                                        num_tokens,
                                        HIDDEN,
                                        None,
                                    )
                                ), (
                                    num_tokens,
                                    dtype,
                                    config,
                                )
                            else:
                                torch.testing.assert_close(
                                    got, want, atol=2e-2, rtol=2e-2, msg=str(config)
                                )
                    blocks.copy_(original_blocks)

                    graph = torch.cuda.CUDAGraph()
                    with comm.capture():
                        comm.all_reduce_add_attn_res_rms_norm(
                            x,
                            prefix,
                            blocks,
                            norm_weight,
                            qk_weight,
                            output_norm_weight,
                            num_blocks,
                            write_index,
                            ATTN_RES_EPS,
                            EPS,
                        )
                        with torch.cuda.graph(graph):
                            out_prefix, out = comm.all_reduce_add_attn_res_rms_norm(
                                x,
                                prefix,
                                blocks,
                                norm_weight,
                                qk_weight,
                                output_norm_weight,
                                num_blocks,
                                write_index,
                                ATTN_RES_EPS,
                                EPS,
                            )
                    x.copy_(x + 1)
                    summed = x.clone()
                    dist.all_reduce(summed, group=group.device_group)
                    want_prefix = summed.clone() if prefix is None else prefix.clone()
                    want_blocks = blocks.clone()
                    want_out = attn_res(
                        want_prefix,
                        None if prefix is None else summed,
                        want_blocks,
                        norm_weight,
                        qk_weight,
                        output_norm_weight,
                        num_blocks,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    graph.replay()
                    torch.accelerator.synchronize()
                    for got, want in [
                        (out, want_out),
                        (out_prefix, want_prefix),
                        (blocks, want_blocks),
                    ]:
                        if comm.arguments.quantization == "int4":
                            err = (got.float() - want.float()).square().mean().sqrt()
                            rms = want.float().square().mean().sqrt()
                            assert err / rms <= int4_bar(
                                comm.tuned(
                                    "all_reduce_add_attn_res_rms_norm",
                                    num_tokens,
                                    HIDDEN,
                                    None,
                                )
                            ), (num_tokens, dtype)
                        else:
                            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_all_reduce_add_attn_res_rms_norm(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(
        monkeypatch, tp_size, 1, all_reduce_add_attn_res_rms_norm_ranks
    )


@ray.remote(num_gpus=1, max_calls=1)
def outputs_independent_of_memory_ranks(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port
):
    """Outputs must not depend on what their memory held before."""
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        dtype = torch.bfloat16
        refused_at_size = (
            "staging_too_small",
            "partials_too_small",
            "unfenced_too_small",
            "grid_not_resident",
            "launch_out_of_range",
        )
        failures = []
        try:
            for cols in [HIDDEN, HIDDEN // 2]:
                for num_tokens in [8, 256, MAX_TOKENS]:
                    g = torch.Generator(device=device).manual_seed(rank)
                    x = torch.randn((num_tokens, cols), generator=g, device=device)
                    g.manual_seed(0)
                    x[:, torch.randperm(cols, generator=g, device=device)[:8]] *= 50
                    x = x.to(dtype)
                    residual = torch.randn(
                        (num_tokens, cols), generator=g, device=device
                    ).to(dtype)
                    prefix = (
                        torch.randn((num_tokens, cols), generator=g, device=device) * 4
                    ).to(dtype)
                    blocks = torch.randn(
                        (num_tokens, 10, cols), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(cols, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g)
                    qk_weight = torch.empty(cols, dtype=dtype, device=device)
                    qk_weight.normal_(mean=0.0, std=0.1, generator=g)

                    calls = [
                        ("all_reduce", "all_reduce"),
                        ("rms_norm", "all_reduce_add_rms_norm"),
                        ("add_rms_norm", "all_reduce_add_rms_norm"),
                        ("attn_res", "all_reduce_add_attn_res_rms_norm"),
                        ("attn_res_block_write", "all_reduce_add_attn_res_rms_norm"),
                    ]
                    for variant, op in calls:
                        algorithms = {
                            "all_reduce": AllReduceConfig,
                            "all_reduce_add_rms_norm": AllReduceAddRmsNormConfig,
                            "all_reduce_add_attn_res_rms_norm": (
                                AllReduceAddAttnResRmsNormConfig
                            ),
                        }[op].algorithms
                        forced = [
                            comm.kernel_config(
                                op,
                                algorithm,
                                num_tokens,
                                cols,
                                comm.arguments.quantization,
                            )
                            for algorithm in algorithms
                        ]
                        for config in [None, *(c for c in forced if c is not None)]:
                            answers = []
                            where = (variant, config, num_tokens, cols)
                            for fill in [float("nan"), 1e4]:
                                poison = [torch.full_like(x, fill) for _ in range(4)]
                                del poison
                                try:
                                    outputs = []
                                    if variant == "all_reduce":
                                        outputs = [comm.all_reduce(x, config=config)]
                                    elif variant == "rms_norm":
                                        outputs = [
                                            comm.all_reduce_add_rms_norm(
                                                x, weight, EPS, config=config
                                            )
                                        ]
                                    elif variant == "add_rms_norm":
                                        outputs = list(
                                            comm.all_reduce_add_rms_norm(
                                                x,
                                                weight,
                                                EPS,
                                                residual=residual,
                                                config=config,
                                            )
                                        )
                                    else:
                                        has_prefix = variant == "attn_res"
                                        written = blocks.clone()
                                        outputs = [
                                            written,
                                            *comm.all_reduce_add_attn_res_rms_norm(
                                                x,
                                                prefix if has_prefix else None,
                                                written,
                                                weight,
                                                qk_weight,
                                                weight,
                                                4,
                                                -1 if has_prefix else 4,
                                                ATTN_RES_EPS,
                                                EPS,
                                                config=config,
                                            ),
                                        ]
                                except RuntimeError as e:
                                    if not str(e).startswith(refused_at_size):
                                        raise
                                    break
                                answers.append([t.clone() for t in outputs])
                            torch.accelerator.synchronize()
                            if len(answers) < 2:
                                continue
                            for poisoned_nan, poisoned_finite in zip(*answers):
                                if not torch.isfinite(poisoned_nan).all():
                                    failures.append(("not finite", *where))
                                elif not torch.equal(poisoned_nan, poisoned_finite):
                                    failures.append(("differs", *where))
            assert not failures, "\n".join(map(str, failures))
        finally:
            comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("quantization", ["NONE", "INT4"])
def test_outputs_independent_of_memory(
    monkeypatch: pytest.MonkeyPatch, tp_size, quantization
):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", quantization)
    multi_process_parallel(monkeypatch, tp_size, 1, outputs_independent_of_memory_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def refused_calls_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        comm = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
        )
        assert not comm.disabled, comm.lifecycle
        try:
            past = torch.ones(
                16 * MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device
            )
            assert not comm.should_allreduce(past)

            with pytest.raises(RuntimeError, match="^not_two_d:"):
                comm.all_reduce(
                    torch.ones(2, 4, HIDDEN, dtype=torch.bfloat16, device=device)
                )

            x = torch.ones(4, 2 * HIDDEN, dtype=torch.bfloat16, device=device)[
                :, :HIDDEN
            ]
            want = x.clone()
            dist.all_reduce(want, group=group.device_group)
            if comm.should_allreduce(x):
                assert torch.equal(comm.all_reduce(x), want)
        finally:
            comm.close()


def test_refused_calls(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", "NONE")
    multi_process_parallel(monkeypatch, 2, 1, refused_calls_ranks)


@ray.remote(num_gpus=1, max_calls=1)
def tuned_table_ranks(monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, port):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group()
        dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
        torch.accelerator.synchronize()
        for quantization in ["exact", "int4"]:
            comm = HipCustomCommunicator(
                group.cpu_group,
                group.device_group,
                device,
                max_rows=MAX_TOKENS,
                max_cols=HIDDEN,
                quantization=quantization,
            )
            assert not comm.disabled, comm.lifecycle
            try:
                for op, residual in [
                    ("all_reduce", None),
                    ("all_reduce_add_rms_norm", False),
                    ("all_reduce_add_rms_norm", True),
                    ("all_reduce_add_attn_res_rms_norm", None),
                ]:
                    for num_tokens in [1, MAX_TOKENS]:
                        config = comm.tuned(op, num_tokens, HIDDEN, residual)
                        assert config is not None, (
                            op,
                            residual,
                            quantization,
                            num_tokens,
                        )
            finally:
                comm.close()


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_tuned_table(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, tuned_table_ranks)
