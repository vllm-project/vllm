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
from torch import nn

from vllm.model_executor import tpsp as tpsp_utils
from vllm.model_executor.models import llama
from vllm.model_executor.tpsp import (
    TPSPBackend,
    TPSPContext,
    TPSPOpsGroup,
    TPSPProfile,
    TPSPScanResult,
)
from vllm.platforms import current_platform


def test_projection_profile_restores_residual_between_baseline_trials():
    class Projection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1, None

    class Norm(nn.Module):
        def forward(self, hidden_states, residual):
            residual.add_(hidden_states)
            return hidden_states, residual

    inputs = tpsp_utils._projection_inputs(
        3, SimpleNamespace(input_size_per_partition=2), 2, 2, 0, torch.device("cpu")
    )
    original_residual = inputs.residual.clone()
    projection, norm = Projection(), Norm()

    first = tpsp_utils._run_conventional_projection(projection, norm, inputs)
    assert not torch.equal(inputs.residual, original_residual)
    inputs.restore_residual()
    second = tpsp_utils._run_conventional_projection(projection, norm, inputs)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(inputs.residual, original_residual + first)


def test_compiled_ops_group_passes_projection_output_to_norm():
    class Projection(nn.Module):
        input_size_per_partition = 2

        def forward(self, x):
            return x + 2, None

    class Norm(nn.Module):
        def forward(self, x, residual):
            return x * residual, residual

    norm = Norm()
    norm.weight = nn.Parameter(torch.ones(2))
    norm.variance_epsilon = 1e-5
    compiled = tpsp_utils._compile_ops_groups(
        [TPSPOpsGroup("pair", Projection(), norm)]
    )[0]
    inputs = SimpleNamespace(
        hidden_states=torch.ones(1, 2), residual=torch.full((1, 2), 4.0)
    )
    torch.testing.assert_close(compiled.conventional(inputs), torch.full((1, 2), 12.0))


def _check_tpsp_backend(
    rank: int, rendezvous: str, world_size: int, availability: mp.SimpleQueue
) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device(current_platform.device_type, rank)
    dist.init_process_group(
        current_platform.dist_backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    comm = None
    if device.type == "cuda":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator(dist.new_group(backend="gloo"), device)
    tp_group = SimpleNamespace(
        device_group=dist.group.WORLD,
        device_communicator=SimpleNamespace(pynccl_comm=comm),
    )
    try:
        backend_cls = current_platform.get_tpsp_backend_cls()
        backend = (
            backend_cls(dist.group.WORLD.group_name, device)
            if backend_cls is not None
            else None
        )
        context = None
        if backend is not None:
            with patch(
                "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
            ):
                context = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=world_size,
                    hidden_size=4096,
                    max_batched_tokens=513,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
        available = torch.tensor(int(context is not None), device=device)
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if rank == 0:
            availability.put(bool(available.item()))
        if not available.item():
            if context is not None:
                assert backend is not None
                backend.close(context)
            if backend is not None:
                backend.close()
            return
        assert backend is not None and context is not None
        if os.environ.get("VLLM_TEST_TPSP_REQUIRE_P2P") == "1":
            assert context.p2p_handle, "TPSP P2P initialization fell back to NCCL"
        with patch(
            "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
        ):
            large_context = backend.open(
                dtype=torch.bfloat16,
                tp_size=world_size,
                hidden_size=64,
                max_batched_tokens=1_000_000,
                group_name=dist.group.WORLD.group_name,
                device=device,
            )
        assert large_context is not None
        large_projection = nn.Linear(
            64, 64, bias=False, device=device, dtype=torch.bfloat16
        )
        large_norm = nn.Module()
        large_norm.weight = torch.ones(64, dtype=torch.bfloat16, device=device)
        large_norm.variance_epsilon = 1e-5
        capped_rows = backend.tpsp_max_microchunk_rows
        assert large_context.max_chunk_rows == capped_rows
        with pytest.raises(RuntimeError, match="residual shard"):
            backend.fused_gemm_rs_norm_ag(
                large_context,
                torch.empty((world_size + 1, 64), dtype=torch.bfloat16, device=device),
                large_projection,
                torch.empty((world_size + 1, 64), dtype=torch.bfloat16, device=device),
                large_norm,
            )
        if large_context.p2p_handle:
            with pytest.raises(ValueError, match="P2P workspace capacity"):
                backend.fused_gemm_rs_norm_ag(
                    large_context,
                    torch.empty(
                        (world_size * (capped_rows + 1), 64),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    large_projection,
                    torch.empty(
                        (capped_rows + 1, 64), dtype=torch.bfloat16, device=device
                    ),
                    large_norm,
                    config=capped_rows + 1,
                )
        tokens = 65_538
        generator = torch.Generator(device=device).manual_seed(123 + rank)
        a = torch.randn(
            (tokens, 64),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        rows = (tokens + world_size - 1) // world_size
        residual = torch.randn(
            (tokens, 64),
            dtype=torch.bfloat16,
            device=device,
            generator=torch.Generator(device=device).manual_seed(407),
        )
        local_residual = torch.zeros((rows, 64), device=device, dtype=torch.bfloat16)
        start = rank * rows
        count = min(rows, max(tokens - start, 0))
        local_residual[:count] = residual[start : start + count]
        expected = a @ large_projection.weight.T
        dist.all_reduce(expected)
        expected_residual = residual.clone()
        torch.ops._C.fused_add_rms_norm(
            expected, expected_residual, large_norm.weight, 1e-5
        )
        for config in (1024, 2048):
            gathered, reduced = backend.fused_gemm_rs_norm_ag(
                large_context,
                a,
                large_projection,
                local_residual,
                large_norm,
                config=config,
            )
            torch.testing.assert_close(gathered, expected, rtol=0.02, atol=0.05)
            torch.testing.assert_close(
                reduced[:count],
                expected_residual[start : start + count],
                rtol=0.02,
                atol=0.05,
            )
        old_handle = large_context.p2p_handle
        backend.close(large_context)
        if old_handle:
            with pytest.raises(RuntimeError, match="Invalid TPSP P2P context"):
                torch.ops._C.destroy_tpsp_p2p(old_handle)
        for tokens, width in (
            (1, 64),
            (5, 2048),
            (128, 2048),
            (129, 7168),
            (512, 1024),
            (513, 1024),
        ):
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
            rows = (tokens + world_size - 1) // world_size
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
            with patch.object(
                torch.ops._C,
                "tpsp_fused_matmul_reduce_scatter_norm_all_gather",
                wraps=torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather,
            ) as fused_op:
                projection = nn.Module()
                projection.weight = nn.Parameter(b.T.contiguous())
                projection.bias = None
                norm = nn.Module()
                norm.weight = weight
                norm.variance_epsilon = 1e-5
                gathered, reduced = backend.fused_gemm_rs_norm_ag(
                    context,
                    a,
                    projection,
                    local_residual,
                    norm,
                    config=64,
                )
            assert fused_op.call_args.args[11] == context.p2p_handle
            torch.testing.assert_close(
                gathered,
                expected,
                rtol=0.02,
                atol=0.05,
                msg=f"rank={rank} tokens={tokens}",
            )
            torch.testing.assert_close(
                reduced[:count],
                expected_residual[start : start + count],
                rtol=0.02,
                atol=0.05,
            )
            if tokens in (512, 513):
                for config in backend.get_tuning_configs(context, 32):
                    tuned, tuned_residual = backend.fused_gemm_rs_norm_ag(
                        context, a, projection, local_residual, norm, config=config
                    )
                    torch.testing.assert_close(tuned, expected, rtol=0.02, atol=0.05)
                    torch.testing.assert_close(
                        tuned_residual[:count],
                        expected_residual[start : start + count],
                        rtol=0.02,
                        atol=0.05,
                    )
            if tokens == 1 and context.p2p_handle:
                p2p_handle = context.p2p_handle
                context.p2p_handle = 0
                try:
                    nccl_gathered, nccl_reduced = backend.fused_gemm_rs_norm_ag(
                        context, a, projection, local_residual, norm, config=64
                    )
                finally:
                    context.p2p_handle = p2p_handle
                torch.testing.assert_close(
                    nccl_gathered, expected, rtol=0.02, atol=0.05
                )
                torch.testing.assert_close(
                    nccl_reduced[:count],
                    expected_residual[start : start + count],
                    rtol=0.02,
                    atol=0.05,
                )
            if tokens == 129:
                projection_bias = torch.full_like(weight, 0.125)
                norm_bias = torch.full_like(weight, 0.25)
                expected_with_bias = a @ b
                dist.all_reduce(expected_with_bias)
                expected_with_bias += projection_bias
                expected_residual_with_bias = residual + expected_with_bias
                expected_normalized = torch.nn.functional.layer_norm(
                    expected_residual_with_bias, (4096,), weight, norm_bias, 1e-5
                )
                projection.bias = nn.Parameter(projection_bias)
                norm.bias = nn.Parameter(norm_bias)
                norm.eps = 1e-5
                gathered, reduced = backend.fused_gemm_rs_norm_ag(
                    context,
                    a,
                    projection,
                    local_residual,
                    norm,
                    config=64,
                    norm_type="layer_norm",
                )
                torch.testing.assert_close(
                    reduced[:count],
                    expected_residual_with_bias[start : start + count],
                    rtol=0.02,
                    atol=0.05,
                )
                torch.testing.assert_close(
                    gathered, expected_normalized, rtol=0.02, atol=0.05
                )

        class Projection(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.input_size_per_partition = width
                self.weight = nn.Parameter(
                    torch.randn(
                        4096,
                        width,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(
                            2026 + rank + width
                        ),
                    )
                    / math.sqrt(width)
                )
                self.bias = None

            def forward(self, x):
                result = torch.nn.functional.linear(x, self.weight)
                dist.all_reduce(result)
                return result, None

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.ones(4096, device=device, dtype=torch.bfloat16)
                )
                self.variance_epsilon = 1e-5

            def forward(self, x, residual):
                torch.ops._C.fused_add_rms_norm(
                    x, residual, self.weight, self.variance_epsilon
                )
                return x, residual

        o_proj, down_proj = Projection(64), Projection(4096)
        o_norm, down_norm = Norm(), Norm()
        profile = tpsp_utils._scan_chunk(
            backend,
            context,
            projection=o_proj,
            norm=o_norm,
            tp_size=world_size,
            max_batched_tokens=128,
            group_name="o_proj",
        )
        assert profile.tp_size == world_size
        assert profile.hidden_size == 4096
        assert profile.max_batched_tokens == 128
        assert profile.status == "candidate"
        assert profile.config is not None
        backend.set_config(context, profile.config)
        assert profile.threshold_tokens is None

        handles = []
        with (
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=SimpleNamespace(
                    world_size=world_size,
                    rank_in_group=rank,
                    device_group=dist.group.WORLD,
                    device_communicator=tp_group.device_communicator,
                ),
            ),
        ):
            for _ in range(2):
                handle = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=world_size,
                    hidden_size=4096,
                    max_batched_tokens=128,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
                assert handle is not None
                backend.set_config(handle, 64)
                handles.append(handle)
            pair = tpsp_utils._scan_threshold(
                backend,
                tpsp_utils._compile_ops_groups(
                    [
                        TPSPOpsGroup("o_proj", o_proj, o_norm),
                        TPSPOpsGroup("down_proj", down_proj, down_norm),
                    ]
                ),
                {"o_proj": handles[0], "down_proj": handles[1]},
                128,
            )
            assert pair.measurements and pair.measurements[0].tokens == 128
            for handle in handles:
                backend.close(handle)
        backend.close(context)
        backend.close()
    finally:
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", (2, 4))
def test_tpsp_backend(tp_size: int):
    if current_platform.get_tpsp_backend_cls() is None:
        pytest.skip("TPSP is not supported on this platform")
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} devices")
    with tempfile.TemporaryDirectory() as directory:
        availability = mp.get_context("spawn").SimpleQueue()
        mp.spawn(
            _check_tpsp_backend,
            args=(os.path.join(directory, "rendezvous"), tp_size, availability),
            nprocs=tp_size,
        )
        if not availability.get():
            pytest.skip("TPSP backend is unavailable for this configuration")


@pytest.mark.parametrize("p2p_handle", (0, 7))
def test_cuda_tpsp_refines_only_available_transports(monkeypatch, p2p_handle):
    from vllm.platforms.tpsp_cuda import (
        CudaTPSPBackend,
        CudaTPSPConfig,
        CudaTPSPContext,
    )

    device = torch.device("cuda", 0)
    backend = CudaTPSPBackend("test", device)
    context = CudaTPSPContext(device, 4, 0, 1, 4096, p2p_handle)
    backend._open_context_ids.add(id(context))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=78),
    )
    expected = tuple(
        CudaTPSPConfig(512, bool(p2p_handle), target)
        for target in (0, 58, 64, 70, 76, 78)
    )
    assert backend.get_tuning_configs(context, 512, group_name="o_proj") == expected
    backend.set_config(context, CudaTPSPConfig(512, False, 66))
    assert context.config == CudaTPSPConfig(512, False, 66)
    assert context.p2p_handle == p2p_handle
    if not p2p_handle:
        with pytest.raises(ValueError, match="P2P transport is unavailable"):
            backend.set_config(context, CudaTPSPConfig(512, True))


def test_cuda_tpsp_scans_group_sm_ranges_and_transport(monkeypatch):
    from vllm.platforms.tpsp_cuda import (
        CudaTPSPBackend,
        CudaTPSPConfig,
        CudaTPSPContext,
    )

    device = torch.device("cuda", 0)
    backend = CudaTPSPBackend("test", device)
    context = CudaTPSPContext(device, 4, 0, 1, 4096, 7)
    backend._open_context_ids.add(id(context))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=78),
    )
    assert backend._chunk_rows_range(65_536, 4) == (64, 4096)
    assert backend.get_tuning_configs(context, 64, "down_proj") == tuple(
        CudaTPSPConfig(64, True, target) for target in (0, 68, 74, 78)
    )
    assert backend.get_tuning_configs(
        context,
        64,
        "o_proj",
        compute_engine_counts=(49, 50, 51),
        compute_engine_step=1,
    ) == tuple(CudaTPSPConfig(64, True, target) for target in (49, 50, 51))


def test_tpsp_joint_refinement_uses_upper_midpoint_probe_and_default_fallback(
    monkeypatch, caplog
):
    from dataclasses import dataclass

    caplog.set_level("INFO", logger=tpsp_utils.__name__)
    clock = [0.0]
    coarse_grid: set[tuple[int, int]] = set()
    fine_grid: set[tuple[int, int]] = set()

    @dataclass(frozen=True)
    class Config:
        chunk_rows: int
        compute_engine_count: int

    class Projection(nn.Module):
        input_size_per_partition = 2

        def forward(self, x):
            return x, None

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.variance_epsilon = 1e-5

        def forward(self, x, residual):
            return x, residual

    class Backend(TPSPBackend):
        tpsp_chunk_granularity = 64
        fixed_count = None

        def open(self, **kwargs):
            return None

        def set_config(self, handle, config):
            raise NotImplementedError

        def close(self, context=None):
            pass

        def get_default_compute_engine_count(self):
            return 0

        def get_tuning_configs(
            self,
            handle,
            chunk,
            group_name=None,
            *,
            compute_engine_counts=None,
            compute_engine_step=3,
        ):
            if self.fixed_count is not None:
                counts = (self.fixed_count,)
            else:
                counts = (
                    (0, 3) if compute_engine_counts is None else compute_engine_counts
                )
            if compute_engine_step > 1:
                coarse_grid.update((chunk, count) for count in counts)
            elif compute_engine_step == 1:
                fine_grid.update((chunk, count) for count in counts)
            return tuple(Config(chunk, count) for count in counts)

        def fused_gemm_rs_norm_ag(
            self,
            handle,
            x,
            projection,
            residual,
            norm,
            *,
            config=None,
            norm_type="rms_norm",
        ):
            costs = {
                (64, 0): 3,
                (128, 0): 2,
                (192, 0): 0.8,
                (256, 0): 2.5,
                (64, 3): 4,
                (128, 3): 3,
                (192, 3): 2,
                (256, 3): 4,
            }
            cost = costs.get(
                (config.chunk_rows, config.compute_engine_count),
                1.5
                + abs(config.chunk_rows - 192) / 256
                + abs(config.compute_engine_count - 2) / 10,
            )
            if config == Config(192, 2):
                cost = 0.5
            clock[0] += cost / 1000
            output = x + 1 if config == Config(192, 2) else x
            return output, residual

    monkeypatch.setattr(tpsp_utils.c10d, "_resolve_process_group", lambda name: None)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)
    monkeypatch.setattr(tpsp_utils.time, "perf_counter", lambda: clock[0])
    backend = Backend("test", torch.device("cpu"))

    def run_scan():
        return tpsp_utils._scan_chunk(
            backend,
            object(),
            Projection(),
            Norm(),
            tp_size=2,
            max_batched_tokens=512,
            group_name="o_proj",
            compute_engine_count_min=1,
            compute_engine_count_max=5,
            compute_engine_step_coarse=4,
            compute_engine_step_fine=1,
            chunk_min_max=(64, 256),
            chunk_step_coarse=128,
            chunk_step_fine=64,
        )

    result = run_scan()
    assert result.config == Config(192, 0)
    assert len(result.candidates) == 3
    assert len(result.finalists) == 2
    assert result.selected_seed_rank == 2
    assert tuple(rank for rank, _, _ in result.refined_candidates) == (
        1,
        2,
    )
    assert result.refined_candidates[0][1] == Config(192, 2)
    assert result.refined_candidates[1][1] == Config(192, 0)
    assert "stage=coarse [####################] 100%" in caplog.text
    assert "stage=fine [####################] 100%" in caplog.text
    assert "stage=fine [--------------------]   0% 0/21" in caplog.text
    assert "14% 3/21" in caplog.text
    assert coarse_grid == {
        *((chunk, 5) for chunk in (64, 128, 192, 256)),
        *((chunk, 0) for chunk in (64, 192, 256)),
    }
    assert fine_grid == {
        *((chunk, count) for chunk in (64, 128, 192, 256) for count in range(1, 6)),
        (192, 0),
    }

    backend.fixed_count = 4
    coarse_grid.clear()
    fine_grid.clear()
    result = run_scan()
    assert result.status == "candidate"
    assert result.config == Config(192, 4)
    assert "TPSP midpoint probe unavailable" in caplog.text


def test_tpsp_threshold_scan_coarsely_finds_and_finely_refines_boundary(
    monkeypatch, caplog
):
    caplog.set_level("INFO", logger=tpsp_utils.__name__)
    clock = [0.0]
    measured_tokens = []
    baseline_calls: dict[int, int] = {}
    fused_calls: dict[int, int] = {}

    class Projection(nn.Module):
        input_size_per_partition = 2

        def forward(self, x):
            tokens = x.size(0)
            baseline_calls[tokens] = baseline_calls.get(tokens, 0) + 1
            clock[0] += 0.001
            return x, None

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.variance_epsilon = 1e-5

        def forward(self, x, residual):
            return x, residual

    class Backend:
        group_name = "test"
        device = torch.device("cpu")

    def make_inputs(tokens, projection, hidden_size, tp_size, rank, device):
        measured_tokens.append(tokens)
        hidden_states = torch.zeros(tokens, hidden_size)
        residual = torch.zeros_like(hidden_states)
        return tpsp_utils._ProjectionInputs(
            hidden_states, residual, residual, residual.clone()
        )

    def run_fused(backend, handle, projection, norm, inputs):
        tokens = inputs.hidden_states.size(0)
        fused_calls[tokens] = fused_calls.get(tokens, 0) + 1
        clock[0] += 0.002 if tokens < 300 else 0.0005
        return inputs.hidden_states

    monkeypatch.setattr(tpsp_utils.c10d, "_resolve_process_group", lambda name: None)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: None)
    monkeypatch.setattr(tpsp_utils.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(tpsp_utils, "_projection_inputs", make_inputs)
    monkeypatch.setattr(tpsp_utils, "_run_fused_projection", run_fused)
    ops_groups = tpsp_utils._compile_ops_groups(
        [TPSPOpsGroup("o_proj", Projection(), Norm())]
    )

    result = tpsp_utils._scan_threshold(
        Backend(),
        ops_groups,
        {"o_proj": object()},
        max_batched_tokens=2048,
    )

    assert result.threshold_tokens == 384
    assert set(measured_tokens) == {
        128,
        256,
        384,
        512,
        640,
        1152,
        1664,
        2048,
    }
    assert all(baseline_calls[tokens] == 2 for tokens in measured_tokens)
    assert all(fused_calls[tokens] == 4 for tokens in measured_tokens)
    assert "stage=threshold [####################] 100%" in caplog.text


def test_tpsp_benefit_threshold_is_one_percent():
    measurement = tpsp_utils.TPSPMeasurement
    assert tpsp_utils._beneficial(measurement(1, 100, 98.9, 1.1))
    assert not tpsp_utils._beneficial(measurement(1, 100, 99, 1))
    assert tpsp_utils._beneficial(measurement(1, 2, 1.97, 0.03))
    assert not tpsp_utils._beneficial(measurement(1, 2, 1.98, 0.02))


@pytest.mark.parametrize(
    "tokens,rank,expected_alias,expected",
    [
        (4, 1, True, [[4, 5], [6, 7]]),
        (3, 1, False, [[4, 5], [0, 0]]),
        (1, 1, False, [[0, 0]]),
        (3, 0, True, [[0, 1], [2, 3]]),
    ],
)
def test_tpsp_shard_residual_only_allocates_for_padding(
    monkeypatch, tokens, rank, expected_alias, expected
):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=rank),
    )
    residual = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
    shard = tpsp_utils.tpsp_shard_residual(residual)
    assert (shard.data_ptr() == residual[rank * 2 :].data_ptr()) is expected_alias
    torch.testing.assert_close(shard, torch.tensor(expected, dtype=torch.bfloat16))


def test_llama_tpsp_disables_compile_on_cpu(monkeypatch):
    model = nn.Module()
    model.make_empty_intermediate_tensors = lambda: None
    monkeypatch.setattr(
        llama.LlamaForCausalLM, "_init_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=2), enable_tpsp=True
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        quant_config=None,
        lora_config=None,
    )

    llama.LlamaForCausalLM(vllm_config=config)

    assert model.tpsp_requested
    assert model.do_not_compile


def test_llama_tpsp_forward(monkeypatch):
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        device_group=SimpleNamespace(group_name="test"),
    )
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True),
    )

    class Projection(nn.Module):
        input_size_per_partition = 2

        def __init__(self, bias=False):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = (
                nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)) if bias else None
            )
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            result = x @ self.weight
            if self.bias is not None:
                result += self.bias
            return result, None

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-5

        def forward(self, x, residual):
            result = x + residual
            return result, result

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = Projection(bias=True)

        def forward(self, positions, hidden_states, tpsp_active=False):
            result = hidden_states + 1
            return result if tpsp_active else self.o_proj(result)[0]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = Projection()

        def forward(self, hidden_states, tpsp_active=False):
            result = hidden_states + 1
            return result if tpsp_active else self.down_proj(result)[0]

    class Backend:
        def __init__(self):
            self.calls = 0
            self.biases = []
            self.configs = []

        def fused_gemm_rs_norm_ag(
            self,
            projection_context,
            x,
            projection,
            residual,
            norm,
            *,
            config=None,
            norm_type="rms_norm",
        ):
            self.calls += 1
            self.biases.append(projection.bias)
            self.configs.append(projection_context.config if config is None else config)
            assert projection_context is o_plan or projection_context is down_plan
            reduced = x @ projection.weight.T + residual
            if projection.bias is not None:
                reduced += projection.bias
            return reduced, reduced

    layer = llama.LlamaDecoderLayer.__new__(llama.LlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.hidden_size = 2
    layer.input_layernorm = nn.Identity()
    layer.self_attn = Attention()
    layer.post_attention_layernorm = Norm()
    layer.mlp = MLP()

    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList([layer])
    model.norm = Norm()
    model.start_layer, model.end_layer = 0, 1
    model.aux_hidden_state_layers = ()
    model._aux_upstream_total_cached = 0
    model.tpsp_requested = False
    model.max_tpsp_batched_tokens = 8
    backend = Backend()
    monkeypatch.setattr(
        llama,
        "current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: Backend),
    )
    o_plan, down_plan = (SimpleNamespace(config=64) for _ in range(2))
    down_plan.config = 32

    def set_profile(threshold: int | None) -> None:
        model.tpsp_context = TPSPContext(
            TPSPProfile(threshold is not None, threshold, 8),
            backend,
            {"o_proj": o_plan, "down_proj": down_plan},
        )
        if threshold is not None:
            layer.tpsp_context = model.tpsp_context

    set_profile(1)
    assert model.tpsp_context.profile.is_active(2)
    with pytest.raises(ValueError, match="profiled range"):
        model.tpsp_context.profile.is_active(9)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        model.forward(
            None,
            None,
            None,
            inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16),
            unexpected=1,
        )
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 2
    assert backend.configs == [64, 32]
    assert backend.biases[0] is layer.self_attn.o_proj.bias
    assert backend.biases[1] is None
    assert layer.self_attn.o_proj.calls == 0
    assert layer.mlp.down_proj.calls == 0

    set_profile(3)
    assert not model.tpsp_context.profile.is_active(2)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 2
    assert layer.self_attn.o_proj.calls == 1
    assert layer.mlp.down_proj.calls == 1

    set_profile(1)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 1
    assert layer.mlp.down_proj.calls == 1
    set_profile(3)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 2
    assert layer.mlp.down_proj.calls == 2

    set_profile(None)
    assert not model.tpsp_context.profile.enabled
    assert not model.tpsp_context.profile.is_active(2)
    with pytest.raises(RuntimeError, match="invalid enabled profile"):
        TPSPProfile(True, None, 8).is_active(2)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 3
    assert layer.mlp.down_proj.calls == 3

    set_profile(1)
    first_hidden, first_residual = layer(
        None,
        torch.ones(2, 2, dtype=torch.bfloat16),
        None,
        next_norm=model.norm,
    )
    hidden, residual = layer(
        None,
        first_hidden,
        first_residual,
        next_norm=model.norm,
    )
    torch.testing.assert_close(hidden, torch.full((2, 2), 31, dtype=torch.bfloat16))
    torch.testing.assert_close(residual, hidden)

    set_profile(None)
    model.tpsp_context = None
    model.tpsp_requested = True
    next_layer = nn.Module()
    next_layer.input_layernorm = Norm()
    model.layers.append(next_layer)
    calls = []

    def profile_once(*args):
        calls.append(args)
        return None

    monkeypatch.setattr(
        llama,
        "current_platform",
        SimpleNamespace(
            get_tpsp_backend_cls=lambda: (
                lambda *args: SimpleNamespace(profile=profile_once)
            )
        ),
    )
    for _ in range(2):
        result = model.forward(
            None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert calls == [
        (
            [
                TPSPOpsGroup(
                    "o_proj", layer.self_attn.o_proj, layer.post_attention_layernorm
                ),
                TPSPOpsGroup(
                    "down_proj", layer.mlp.down_proj, next_layer.input_layernorm
                ),
            ],
            8,
        )
    ]
    assert model.tpsp_context is None
    assert not model.tpsp_requested

    enabled_profile = TPSPContext(
        TPSPProfile(True, 1, 8),
        backend,
        {"o_proj": o_plan, "down_proj": down_plan},
    )
    model.tpsp_requested = True

    class ProfileBackend(Backend):
        def __init__(self, *args):
            super().__init__()

        def profile(self, *args):
            return enabled_profile

    monkeypatch.setattr(
        llama,
        "current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: ProfileBackend),
    )
    model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    assert layer.tpsp_context is enabled_profile


def test_qwen3_tpsp_layer_uses_fused_projection_path(monkeypatch):
    from vllm.model_executor.models import qwen3

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x, None

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = Projection()

        def forward(self, positions, hidden_states, tpsp_active=False):
            assert tpsp_active
            return hidden_states + 1

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = Projection()

        def forward(self, hidden_states, tpsp_active=False):
            assert tpsp_active
            return hidden_states + 1

    class Backend:
        def __init__(self):
            self.calls = []

        def fused_gemm_rs_norm_ag(
            self, handle, x, projection, residual, norm, **kwargs
        ):
            self.calls.append((handle, projection, norm))
            return x + 1, residual + 1

    monkeypatch.setattr(qwen3, "tpsp_shard_residual", lambda residual: residual)
    layer = qwen3.Qwen3DecoderLayer.__new__(qwen3.Qwen3DecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = nn.Identity()
    layer.self_attn = Attention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = MLP()

    o_handle, down_handle = object(), object()
    backend = Backend()
    layer.tpsp_context = SimpleNamespace(
        backend=backend,
        handles={"o_proj": o_handle, "down_proj": down_handle},
    )
    next_norm = nn.Identity()
    hidden_states = torch.zeros(1, 2)
    output, residual = layer(None, hidden_states, None, next_norm=next_norm)

    torch.testing.assert_close(output, torch.full_like(hidden_states, 4))
    torch.testing.assert_close(residual, torch.full_like(hidden_states, 2))
    assert backend.calls == [
        (o_handle, layer.self_attn.o_proj, layer.post_attention_layernorm),
        (down_handle, layer.mlp.down_proj, next_norm),
    ]
    assert layer.self_attn.o_proj.calls == 0
    assert layer.mlp.down_proj.calls == 0


def test_llama_tpsp_profiles_next_layer_norm(monkeypatch):
    first = nn.Module()
    first.self_attn = nn.Module()
    first.self_attn.o_proj = nn.Module()
    first.self_attn.o_proj.weight = nn.Parameter(torch.ones(2, 2))
    first.post_attention_layernorm = nn.Module()
    first.mlp = nn.Module()
    first.mlp.down_proj = nn.Module()
    second = nn.Module()
    second.input_layernorm = nn.Module()

    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([first, second])
    model.norm = nn.Module()
    model.tpsp_requested = True
    model.tpsp_context = None
    model.max_tpsp_batched_tokens = 128
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(group_name="test")),
    )

    def check_profile(*args):
        assert args[0][1].norm is second.input_layernorm
        raise RuntimeError("profile called")

    monkeypatch.setattr(
        llama,
        "current_platform",
        SimpleNamespace(
            get_tpsp_backend_cls=lambda: (
                lambda *args: SimpleNamespace(profile=check_profile)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="profile called"):
        model.forward(None, None, None)


def test_llama_tpsp_without_backend_skips_profiling(monkeypatch):
    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.tpsp_requested = True
    model.tpsp_context = None
    calls: list[None] = []

    def get_backend_cls():
        calls.append(None)
        return None

    monkeypatch.setattr(
        llama,
        "current_platform",
        SimpleNamespace(get_tpsp_backend_cls=get_backend_cls),
    )

    model._maybe_profile_tpsp()
    model._maybe_profile_tpsp()
    assert model.tpsp_context is None
    assert not model.tpsp_requested
    assert len(calls) == 1


def test_tpsp_projection_profile_keeps_distinct_configs(monkeypatch):
    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.input_size_per_partition = width
            self.weight = nn.Parameter(torch.ones(2, width))

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = Projection(2)
    layer.post_attention_layernorm = nn.Module()
    layer.post_attention_layernorm.weight = nn.Parameter(torch.ones(2))
    layer.post_attention_layernorm.variance_epsilon = 1e-5
    layer.mlp = nn.Module()
    layer.mlp.down_proj = Projection(4)

    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    model.norm = nn.Module()
    model.norm.weight = nn.Parameter(torch.ones(2))
    model.norm.variance_epsilon = 1e-5
    model.config = SimpleNamespace(hidden_size=2)

    class Backend(TPSPBackend):
        tpsp_chunk_granularity = 64

        def __init__(self, group_name, device):
            super().__init__(group_name, device)
            self.closed = []

        def open(self, **kwargs):
            return SimpleNamespace(config=None)

        def set_config(self, handle, config):
            handle.config = config

        def fused_gemm_rs_norm_ag(self, *args, **kwargs):
            raise NotImplementedError

        def close(self, context=None):
            self.closed.append(context)

    backend = Backend("test", torch.device("cpu"))
    monkeypatch.setattr(
        tpsp_utils,
        "_scan_chunk",
        lambda selected_backend, handle, *, projection, **kwargs: TPSPScanResult(
            2,
            2,
            8,
            "candidate",
            "",
            input_width=projection.input_size_per_partition,
            config=projection.input_size_per_partition * 8,
        ),
    )

    def enabled_threshold(*args):
        return TPSPScanResult(2, 2, 8, "enabled", "", threshold_tokens=3)

    monkeypatch.setattr(tpsp_utils, "_scan_threshold", enabled_threshold)
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            device_group=SimpleNamespace(group_name="test"),
        ),
    )

    groups = [
        TPSPOpsGroup("o_proj", layer.self_attn.o_proj, layer.post_attention_layernorm),
        TPSPOpsGroup("down_proj", layer.mlp.down_proj, model.norm),
    ]
    profile = backend.profile(groups, 8)
    assert profile.profile.enabled and profile.profile.threshold_tokens == 3
    assert profile.backend is backend
    assert (
        profile.handles["o_proj"].config,
        profile.handles["down_proj"].config,
    ) == (16, 32)
    assert profile.handles["o_proj"] is not profile.handles["down_proj"]

    monkeypatch.setattr(
        tpsp_utils,
        "_scan_threshold",
        lambda *args: TPSPScanResult(2, 2, 8, "disabled", "no benefit"),
    )
    disabled = backend.profile(groups, 8)
    assert disabled is None
    assert len(backend.closed) == 3
    assert backend.closed[-1] is None
    assert backend.closed[-2] is not backend.closed[-3]


def test_tpsp_profile_accepts_multiple_named_groups(monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            device_group=SimpleNamespace(group_name="test"),
        ),
    )

    class Backend(TPSPBackend):
        tpsp_chunk_granularity = 64

        def __init__(self):
            super().__init__("test", torch.device("cpu"))
            self.closed = []
            self.profiled = []
            self.configs = []

        def open(self, **kwargs):
            return object()

        def set_config(self, handle, config):
            self.configs.append((handle, config))

        def fused_gemm_rs_norm_ag(self, *args, **kwargs):
            raise NotImplementedError

        def close(self, context=None):
            self.closed.append(context)

    backend = Backend()

    def scan_chunk(selected_backend, handle, *, projection, **kwargs):
        assert selected_backend is backend
        assert kwargs["max_batched_tokens"] == 10
        backend.profiled.append(projection)
        return TPSPScanResult(4, 4, 10, "candidate", "", config=1)

    monkeypatch.setattr(tpsp_utils, "_scan_chunk", scan_chunk)
    groups = []
    for name in ("first", "second", "third"):
        projection = nn.Module()
        projection.weight = nn.Parameter(torch.ones(4, 4))
        projection.input_size_per_partition = 4
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.ones(4))
        norm.variance_epsilon = 1e-5
        groups.append(TPSPOpsGroup(name, projection, norm))

    def check_threshold(selected_backend, selected_groups, handles, max_tokens):
        assert selected_backend is backend
        assert [entry.ops for entry in selected_groups] == groups
        assert list(handles) == [group.name for group in groups]
        assert max_tokens == 8
        return TPSPScanResult(4, 4, 8, "enabled", "", threshold_tokens=3)

    monkeypatch.setattr(tpsp_utils, "_scan_threshold", check_threshold)
    context = backend.profile(groups, 10)
    assert context is not None
    assert context.profile.max_batched_tokens == 10
    assert backend.profiled == [group.projection for group in groups]
    assert list(context.handles) == [group.name for group in groups]
    assert len({id(handle) for handle in context.handles.values()}) == 3
    assert backend.configs == [(handle, 1) for handle in context.handles.values()]

    with pytest.raises(ValueError, match="unique"):
        backend.profile([groups[0], groups[0]], 8)
    with pytest.raises(ValueError, match="named ops groups"):
        backend.profile([], 8)
    assert not backend.closed

    groups[2].norm.variance_epsilon = 0
    with pytest.raises(ValueError, match="norm_eps"):
        backend.profile(groups, 8)
    assert not backend.closed
    assert len(backend.configs) == 3
