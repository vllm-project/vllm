# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for sequence-parallel comm+GEMM kernel correctness tests.

Builds a real ``ColumnParallelLinear``/``RowParallelLinear``, patches it with
``mark_sp_region`` (the same entry point the SP models use), and checks that
its ``forward`` under an active SP forward context reproduces what its
plain (all-gather/all-reduce) ``forward`` outside of one would, against a
high-precision reference. ``RowParallelLinear``s reduce-scatter instead of
all-reducing, so they're checked against this rank's token shard of the
baseline's (full-token) output; ``ColumnParallelLinear``s all-gather the
input shard before the baseline's own (already-full-token) forward.
"""

from __future__ import annotations

import functools
import queue
from collections.abc import Callable

import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.kernels.sequence_parallel import init_fused_comm_kernel
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.models.common.ops.sequence_parallel import mark_sp_region
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

# Keeps the generator-based `set_current_vllm_config` context manager(s)
# created by `init_worker_distributed` alive for the life of the worker
# process -- otherwise they'd be garbage-collected immediately, which would
# run their `finally` clause and revert the "current" vLLM config.
_held_vllm_config_ctxs: list = []

K = 1024
N = 512
M_PER_RANK_CASES = (1, 3, 16, 24)
BIAS_MODES = ("none", "bias", "skip_bias_add")


def run_workers(worker: Callable[[int, int, mp.Queue], None], world_size: int) -> None:
    """Spawn ``world_size`` copies of ``worker`` and fail the test if any
    reported an error (or request a skip if none are usable)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(worker, args=(world_size, q), nprocs=world_size, join=True)
    results = []
    try:
        while True:
            results.append(q.get(timeout=1))
    except queue.Empty:
        pass
    finally:
        cleanup_dist_env_and_memory()

    assert len(results) == world_size, (
        f"expected {world_size} worker results, got {len(results)}: {results}"
    )
    skips = [r for r in results if isinstance(r, str) and r.startswith("SKIP:")]
    if skips:
        import pytest

        pytest.skip(skips[0][len("SKIP:") :])
    failures = [r for r in results if r != "OK"]
    assert not failures, f"worker failure(s): {failures}"


def init_worker_distributed(
    local_rank: int, world_size: int, port: int, fused: bool
) -> tuple[torch.device, VllmConfig]:
    """Set this process's accelerator device and bring up vLLM's
    distributed/tensor-parallel environment. Returns the device to run on
    and a ``VllmConfig`` with SP enabled (``enable_async_tp=fused``,
    ``sequence_parallel_min_tokens=0`` so every case below activates it),
    the same object a model would get via ``self.parallel_config``.

    Also activates this ``VllmConfig`` as "current" for the remaining
    lifetime of this worker process, since the SP kernels' constructors
    (``get_tp_group()`` et al.) rely on one being set.
    """
    device = torch.device(f"{current_platform.device_type}:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            enable_sequence_parallel=True,
            enable_async_tp=fused,
            sequence_parallel_min_tokens=0,
        )
    )
    ctx = set_current_vllm_config(config)
    ctx.__enter__()
    _held_vllm_config_ctxs.append(ctx)

    init_distributed_environment(backend=current_platform.dist_backend)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return device, config


def _build_linear(
    row_parallel: bool,
    w_full: torch.Tensor,
    bias_full: torch.Tensor | None,
    skip_bias_add: bool,
) -> ColumnParallelLinear | RowParallelLinear:
    """Build a real Column/RowParallelLinear and load this rank's shard of
    ``w_full``/``bias_full``, the way a checkpoint load would."""
    rank = get_tensor_model_parallel_rank()
    ws = get_tensor_model_parallel_world_size()
    common = dict(
        input_size=K,
        output_size=N,
        bias=bias_full is not None,
        skip_bias_add=skip_bias_add,
        params_dtype=w_full.dtype,
        return_bias=skip_bias_add,
    )
    if row_parallel:
        layer: ColumnParallelLinear | RowParallelLinear = RowParallelLinear(
            input_is_parallel=True, **common
        )
        layer = layer.to(w_full.device)
        k = K // ws
        layer.weight.data.copy_(w_full[:, rank * k : (rank + 1) * k])
        if bias_full is not None:
            # RowParallelLinear's bias isn't partitioned: it's added once,
            # after the all-reduce/reduce-scatter.
            layer.bias.data.copy_(bias_full)
    else:
        layer = ColumnParallelLinear(gather_output=False, **common)
        layer = layer.to(w_full.device)
        n = N // ws
        layer.weight.data.copy_(w_full[rank * n : (rank + 1) * n])
        if bias_full is not None:
            layer.bias.data.copy_(bias_full[rank * n : (rank + 1) * n])
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _check_case(
    layer: ColumnParallelLinear | RowParallelLinear,
    vllm_config: VllmConfig,
    w_full: torch.Tensor,
    m: int,
    rank: int,
    ws: int,
) -> str | None:
    """Compare ``layer``'s forward under an active SP forward context (the
    comm+GEMM kernel, via ``mark_sp_region``) to its plain forward outside
    one (the baseline), against a high-precision reference. ``layer`` must
    already be ``mark_sp_region``-patched."""
    row_parallel = isinstance(layer, RowParallelLinear)
    gen = torch.Generator(device=w_full.device).manual_seed(100 + m)
    M = m * ws
    x_full = torch.randn(M, K, generator=gen, device=w_full.device, dtype=w_full.dtype)
    # Distinct per-shard magnitudes so a per-shard scale applied to the wrong
    # shard shows up.
    x_full *= (1 + 3 * torch.arange(ws, device=x_full.device)).repeat_interleave(m)[
        :, None
    ]
    tokens = slice(rank * m, (rank + 1) * m)
    # `M` is the un-sharded token count a model would report for the whole
    # batch -- matches the real `sp_shard`/`sp_all_gather` boundary.
    batch_descriptor = BatchDescriptor(num_tokens=M)

    if row_parallel:
        k = K // ws
        x = x_full[:, rank * k : (rank + 1) * k].contiguous()
        y_ref = layer(x)  # no forward context active: plain all-reduce
        with set_forward_context(None, vllm_config, batch_descriptor=batch_descriptor):
            y_kernel = layer(x)  # SP active: fused/unfused reduce-scatter
        y_hp = x_full.float() @ w_full.float().t()
    else:
        n = N // ws
        x = x_full[tokens]
        y_ref = layer(x_full)  # no forward context active: plain all-gather
        with set_forward_context(None, vllm_config, batch_descriptor=batch_descriptor):
            y_kernel = layer(x)  # SP active: fused/unfused all-gather
        y_hp = x_full.float() @ w_full[rank * n : (rank + 1) * n].float().t()

    if layer.return_bias:
        (y_ref, b_ref), (y_kernel, b_kernel) = y_ref, y_kernel
        assert b_kernel is b_ref, "returned bias differs"
    elif layer.bias is not None:
        y_hp = y_hp + layer.bias.float()
    if row_parallel:
        y_ref, y_hp = y_ref[tokens], y_hp[tokens]

    if y_kernel.shape != y_ref.shape:
        return f"shape {tuple(y_kernel.shape)} != {tuple(y_ref.shape)}"
    ref_err, kernel_err = _rel_err(y_ref, y_hp), _rel_err(y_kernel, y_hp)
    if not kernel_err <= 1.5 * ref_err + 1e-2:
        return f"rel_err {kernel_err:.4f} vs baseline {ref_err:.4f}"
    return None


def sp_matrix_worker(
    local_rank: int,
    world_size: int,
    q: mp.Queue,
    *,
    port: int,
    row_parallel: bool,
    fused: bool,
) -> None:
    failures = []
    try:
        device, vllm_config = init_worker_distributed(
            local_rank, world_size, port, fused
        )
        torch.manual_seed(0)
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias_full = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1

        for bias_mode in BIAS_MODES:
            layer = _build_linear(
                row_parallel,
                w_full,
                None if bias_mode == "none" else bias_full,
                skip_bias_add=bias_mode == "skip_bias_add",
            )
            if fused and init_fused_comm_kernel(layer) is None:
                q.put("SKIP: fused kernel not supported on this platform")
                return
            row_parallel = isinstance(layer, RowParallelLinear)
            mark_sp_region(
                entry_after=layer if row_parallel else None,
                exit_before=None if row_parallel else layer,
                parallel_config=vllm_config.parallel_config,
            )
            for m in M_PER_RANK_CASES:
                try:
                    err = _check_case(
                        layer, vllm_config, w_full, m, local_rank, world_size
                    )
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                if err:
                    failures.append(f"[{bias_mode}, m/rank={m}] {err}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"{type(e).__name__}: {e}")
    q.put("OK" if not failures else f"rank{local_rank}: " + " | ".join(failures))


def run_sp_matrix(row_parallel: bool, fused: bool, world_size: int) -> None:
    """Run the fused or unfused kernel (via ``mark_sp_region``) for a
    Row/ColumnParallelLinear over the token-count/bias matrix on
    ``world_size`` ranks; fails listing every mismatching case."""
    import pytest

    if torch.accelerator.device_count() < world_size:
        pytest.skip("Not enough accelerators to run the test.")
    worker = functools.partial(
        sp_matrix_worker,
        port=get_open_port(),
        row_parallel=row_parallel,
        fused=fused,
    )
    run_workers(worker, world_size)
