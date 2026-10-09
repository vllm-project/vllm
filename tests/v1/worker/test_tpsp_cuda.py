# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.v1.worker.tpsp_cuda import CudaTPSPOps
from vllm.v1.worker.tpsp_profile import (
    ChunkConfig,
    TPSPProjectionContext,
    get_tpsp_backend,
)


def _check_cuda_tpsp(rank: int, rendezvous: str, world_size: int) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    cpu_group = dist.new_group(backend="gloo")
    comm = PyNcclCommunicator(cpu_group, device)
    tp_group = SimpleNamespace(
        device_group=dist.group.WORLD,
        device_communicator=SimpleNamespace(pynccl_comm=comm),
    )
    try:
        assert CudaTPSPOps.open(dist.group.WORLD.group_name, device, 0, 4096) is None
        if world_size == 2:
            with (
                patch(
                    "vllm.distributed.parallel_state.get_tp_group",
                    return_value=tp_group,
                ),
                patch("torch.cuda.can_device_access_peer", return_value=False),
            ):
                unavailable = CudaTPSPOps.open(
                    dist.group.WORLD.group_name, device, 129, 4096
                )
            assert unavailable is not None
            assert unavailable.modes == {"nccl"}
            CudaTPSPOps.close_tpsp(unavailable)
            with (
                patch(
                    "vllm.distributed.parallel_state.get_tp_group",
                    return_value=tp_group,
                ),
                patch("torch.cuda.can_device_access_peer", return_value=False),
                patch.object(comm, "disabled", True),
            ):
                assert (
                    CudaTPSPOps.open(dist.group.WORLD.group_name, device, 129, 4096)
                    is None
                )
        backend = get_tpsp_backend(dist.group.WORLD.group_name, device)
        assert backend is not None
        with patch(
            "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
        ):
            tp_size = dist.get_world_size()
            first = backend.open(
                dtype=torch.bfloat16,
                tp_size=tp_size,
                hidden_size=4096,
                max_batched_tokens=129,
                group_name=dist.group.WORLD.group_name,
                device=device,
            )
        assert first is not None
        assert first.modes in ({"nccl"}, {"nccl", "p2p"})
        with (
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
            patch.object(comm, "disabled", True),
        ):
            p2p_only = backend.open(
                dtype=torch.bfloat16,
                tp_size=tp_size,
                hidden_size=4096,
                max_batched_tokens=129,
                group_name=dist.group.WORLD.group_name,
                device=device,
            )
        if "p2p" in first.modes:
            assert p2p_only is not None and p2p_only.modes == {"p2p"}
        else:
            assert p2p_only is None
        second = None
        if tp_size == 4:
            with patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ):
                second = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=tp_size,
                    hidden_size=4096,
                    max_batched_tokens=129,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
            assert second is not None
            assert second is not first
        for tokens, width in ((1, 64), (5, 2048), (128, 2048), (129, 7168)):
            generator = torch.Generator(device=device).manual_seed(123 + rank)
            a = torch.randn(
                (tokens + 1, width),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )[1:]
            b = torch.randn(
                (width + 1, 4096),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )[1:] / math.sqrt(width)
            weight = torch.ones(4096, device=device, dtype=torch.bfloat16)
            residual = torch.randn(
                (tokens, 4096),
                dtype=torch.bfloat16,
                device=device,
                generator=torch.Generator(device=device).manual_seed(407),
            )
            rows = (tokens + tp_size - 1) // tp_size
            local_residual = torch.zeros(
                (rows, 4096), device=device, dtype=torch.bfloat16
            )
            start = rank * rows
            count = min(rows, max(tokens - start, 0))
            local_residual[:count] = residual[start : start + count]

            expected = a @ b
            dist.all_reduce(expected)
            expected_residual = residual.clone()
            torch.ops._C.fused_add_rms_norm(expected, expected_residual, weight, 1e-5)
            modes = sorted((second or first).modes)
            for mode in modes:
                for chunk in (2, 64, 65536):
                    config = TPSPProjectionContext(
                        second or first, ChunkConfig(chunk, mode)
                    )
                    reduced, _, gathered = backend.fused(
                        a, b, weight, local_residual, 1e-5, config
                    )
                    torch.testing.assert_close(
                        gathered,
                        expected,
                        rtol=0.02,
                        atol=0.05,
                        msg=f"rank={rank} tokens={tokens} chunk={chunk} mode={mode}",
                    )
                    torch.testing.assert_close(
                        reduced[:count],
                        expected_residual[start : start + count],
                        rtol=0.02,
                        atol=0.05,
                    )
            torch.testing.assert_close(
                local_residual[:count],
                residual[start : start + count],
            )
            if tokens == 5 and p2p_only is not None:
                _, _, p2p_without_nccl = backend.fused(
                    a,
                    b,
                    weight,
                    local_residual,
                    1e-5,
                    TPSPProjectionContext(p2p_only, ChunkConfig(2, "p2p")),
                )
                torch.testing.assert_close(
                    p2p_without_nccl, expected, rtol=0.02, atol=0.05
                )
            if tokens == 129 and "p2p" in (second or first).modes:
                torch.accelerator.synchronize(device)
                dist.barrier(group=cpu_group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _, _, captured = backend.fused(
                        a,
                        b,
                        weight,
                        local_residual,
                        1e-5,
                        TPSPProjectionContext(second or first, ChunkConfig(2, "p2p")),
                    )
                dist.barrier(group=cpu_group)
                for _ in range(2):
                    graph.replay()
                    torch.accelerator.synchronize(device)
                    torch.testing.assert_close(
                        captured,
                        expected,
                        rtol=0.02,
                        atol=0.05,
                    )
        if tp_size == 2:
            profile = backend.profile(
                tp_size=tp_size,
                hidden_size=4096,
                input_width=64,
                max_batched_tokens=128,
                norm_eps=1e-5,
                sharded_residual=False,
                time_budget_s=60,
                context=first,
            )
            expected_modes = first.modes
            assert {config.comm_mode for config, _ in profile.candidates} == (
                expected_modes
            )
        else:
            assert second is not None
            profile = backend.profile(
                tp_size=tp_size,
                hidden_size=4096,
                input_width=64,
                max_batched_tokens=128,
                norm_eps=1e-5,
                sharded_residual=False,
                time_budget_s=60,
                context=second,
            )
            assert {config.comm_mode for config, _ in profile.candidates} == (
                second.modes
            )
        with pytest.raises(ValueError, match="context belongs"):
            backend.fused(
                a,
                b,
                weight,
                local_residual,
                1e-5,
                TPSPProjectionContext(object(), ChunkConfig(64)),
            )
        for context in (first, second, p2p_only):
            if context is not None:
                backend.close(context)
        backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.fused(a, b, weight, local_residual, 1e-5, ChunkConfig(64))
    finally:
        comm.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", (2, 4))
def test_cuda_tpsp_matches_llama_bf16_projection(tp_size: int):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} CUDA GPUs")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _check_cuda_tpsp,
            args=(os.path.join(directory, "rendezvous"), tp_size),
            nprocs=tp_size,
        )


def _check_cuda_tpsp_bias_and_layer_norm(rank: int, rendezvous: str) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    cpu_group = dist.new_group(backend="gloo")
    comm = PyNcclCommunicator(cpu_group, device)
    tp_group = SimpleNamespace(
        device_group=dist.group.WORLD,
        device_communicator=SimpleNamespace(pynccl_comm=comm),
    )
    try:
        backend = get_tpsp_backend(dist.group.WORLD.group_name, device)
        assert backend is not None
        for hidden in (4096, 4100):
            with patch(
                "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
            ):
                context = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=2,
                    hidden_size=hidden,
                    max_batched_tokens=33,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
            assert context is not None
            generator = torch.Generator(device=device).manual_seed(913 + rank)
            tokens, width, rows = 33, 128, 17
            a = torch.randn(
                (tokens, width),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            b = torch.randn(
                (width, hidden),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ) / math.sqrt(width)
            weight = (
                torch.randn(
                    hidden,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=torch.Generator(device=device).manual_seed(509),
                )
                * 0.05
                + 1
            )
            bias = torch.full((hidden,), 0.125, device=device, dtype=a.dtype)
            norm_bias = torch.full((hidden,), 0.25, device=device, dtype=a.dtype)
            residual = torch.randn(
                (tokens, hidden),
                device=device,
                dtype=a.dtype,
                generator=torch.Generator(device=device).manual_seed(409),
            )
            local_residual = torch.zeros((rows, hidden), device=device, dtype=a.dtype)
            start = rank * rows
            count = min(rows, tokens - start)
            local_residual[:count] = residual[start : start + count]

            for norm_type, projection_bias, affine_bias in (
                ("rms_norm", bias, None),
                ("layer_norm", bias, norm_bias),
                ("layer_norm", None, None),
            ):
                expected = a @ b
                dist.all_reduce(expected)
                if projection_bias is not None:
                    expected = expected + projection_bias
                expected_residual = residual.clone()
                if norm_type == "rms_norm":
                    torch.ops._C.fused_add_rms_norm(
                        expected, expected_residual, weight, 1e-5
                    )
                else:
                    expected_residual += expected
                    expected = torch.nn.functional.layer_norm(
                        expected_residual, (hidden,), weight, affine_bias, 1e-5
                    )
                modes = sorted(context.modes)
                for mode in modes:
                    for chunk in (2, 64):
                        reduced, _, gathered = backend.fused(
                            a,
                            b,
                            weight,
                            local_residual,
                            1e-5,
                            ChunkConfig(chunk, mode),
                            norm_type=norm_type,
                            projection_bias=projection_bias,
                            norm_bias=affine_bias,
                            context=context,
                        )
                        torch.testing.assert_close(
                            reduced[:count],
                            expected_residual[start : start + count],
                            rtol=0.02,
                            atol=0.05,
                        )
                        torch.testing.assert_close(
                            gathered,
                            expected,
                            rtol=0.02,
                            atol=0.05,
                        )
            backend.close(context)
        backend.close()
    finally:
        comm.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Requires two CUDA GPUs"
)
def test_cuda_tpsp_bias_and_layer_norm_match_pytorch():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _check_cuda_tpsp_bias_and_layer_norm,
            args=(os.path.join(directory, "rendezvous"),),
            nprocs=2,
        )
