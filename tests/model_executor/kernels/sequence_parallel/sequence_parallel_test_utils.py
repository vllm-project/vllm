# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the fused-comm (eager sequence-parallel AllGather+GEMM
and GEMM+ReduceScatter) kernel correctness tests.

``sp_matrix_worker`` runs a kernel the way ``mark_sp_region`` does -- via
``kernel.apply`` wrapping a ``FakeLinear`` that mimics the Column/Row
parallel ``forward`` -- over several token counts and bias modes. AllGather
kernels get this rank's token shard; ReduceScatter kernels get all tokens.
The baseline is the same layer's plain TP ``forward`` on all tokens; both
are checked against a bf16/fp32 reference of the full computation.
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
    tensor_model_parallel_all_reduce,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

# Keeps the generator-based `set_current_vllm_config` context manager(s)
# created by `init_worker_distributed` alive for the life of the worker
# process -- otherwise they'd be garbage-collected immediately, which would
# run their `finally` clause and revert the "current" vLLM config.
_held_vllm_config_ctxs: list = []


class FakeQuantMethod:
    """Stand-in for a linear quant method: ``apply`` runs the FP8 kernel
    (``fp8_linear``), the MXFP8 kernel (``kernel``) or a plain GEMM."""

    def __init__(self, fp8_linear=None, kernel=None) -> None:
        self.fp8_linear = fp8_linear
        self.kernel = kernel

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (k := self.fp8_linear or self.kernel) is not None:
            k.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (k := self.fp8_linear or self.kernel) is not None:
            return k.apply_weights(layer, x, bias)
        return torch.nn.functional.linear(x, layer.weight, bias)


class FakeLinear(torch.nn.Module):
    """Minimal ``ColumnParallelLinear``/``RowParallelLinear`` whose
    ``forward`` follows the real one: a row-parallel layer passes the bias
    on rank 0 only and all-reduces when ``reduce_results``."""

    def __init__(
        self,
        quant_method: FakeQuantMethod,
        row_parallel: bool = False,
        bias: torch.Tensor | None = None,
        skip_bias_add: bool = False,
        **tensors: torch.Tensor,
    ) -> None:
        super().__init__()
        self.quant_method = quant_method
        self.row_parallel = row_parallel
        self.reduce_results = True
        self.input_is_parallel = True
        self.skip_bias_add = skip_bias_add
        self.return_bias = skip_bias_add
        self.bias = bias
        for name, value in tensors.items():
            self.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )

    def forward(self, x: torch.Tensor):
        bias = None if self.skip_bias_add else self.bias
        if self.row_parallel and get_tensor_model_parallel_rank() > 0:
            bias = None
        output = self.quant_method.apply(self, x, bias)
        if (
            self.row_parallel
            and self.reduce_results
            and get_tensor_model_parallel_world_size() > 1
        ):
            output = tensor_model_parallel_all_reduce(output)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


def quantize_per_tensor_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor FP8 quantization: ``w ~= q * scale``."""
    scale = w.abs().max().float() / FP8_MAX
    scale = scale.clamp(min=1e-12)
    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale.reshape(1)


def quantize_per_channel_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel FP8 quantization of a ``[N, K]`` weight;
    the scale is ``[N, 1]`` like ``ChannelQuantScaleParameter``."""
    scale = (w.abs().amax(dim=1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


def quantize_block_fp8(
    w: torch.Tensor, block_n: int = 128, block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D block FP8 quantization of a ``[N, K]`` weight.

    Returns the quantized weight (``[N, K]``) and a per-block scale
    (``[N // block_n, K // block_k]``), matching the checkpoint layout
    ``Fp8BlockScaledMMLinearKernel``/``XPUFp8BlockScaledMMKernel`` expect
    (``N`` and ``K`` must already be block-aligned).
    """
    N, K = w.shape
    assert N % block_n == 0 and K % block_k == 0
    n_tiles, k_tiles = N // block_n, K // block_k
    blocks = w.float().view(n_tiles, block_n, k_tiles, block_k).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(-2, -1)).clamp(min=1e-12) / FP8_MAX
    q_blocks = (blocks / scale[..., None, None]).clamp(-FP8_MAX, FP8_MAX)
    q = q_blocks.permute(0, 2, 1, 3).reshape(N, K).to(FP8_DTYPE)
    return q, scale.contiguous()


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
    local_rank: int, world_size: int, port: int
) -> torch.device:
    """Set this process's accelerator device and bring up vLLM's
    distributed/tensor-parallel environment. Returns the device to run on.

    Also activates a (dummy) current ``VllmConfig`` for the remaining
    lifetime of this worker process, since the fused-comm kernels'
    constructors (``get_tp_group()`` et al.) rely on one being set.
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
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    ctx = set_current_vllm_config(config)
    ctx.__enter__()
    _held_vllm_config_ctxs.append(ctx)

    init_distributed_environment(backend=current_platform.dist_backend)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return device


K = 1024
N = 512
M_PER_RANK_CASES = (1, 3, 16, 24)
BIAS_MODES = ("none", "bias", "skip_bias_add")
# A static activation scale covering every rank's token magnitude.
STATIC_INPUT_SCALE = 64.0 / FP8_MAX

Build = Callable[[torch.Tensor], tuple[FakeQuantMethod, dict[str, torch.Tensor]]]


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _check_case(
    kernel, layer: FakeLinear, w_full: torch.Tensor, m: int, rank: int, ws: int
) -> str | None:
    gen = torch.Generator(device=w_full.device).manual_seed(100 + m)
    M = m * ws
    x_full = torch.randn(M, K, generator=gen, device=w_full.device, dtype=w_full.dtype)
    # Distinct per-shard magnitudes so a per-shard scale applied to the wrong
    # shard shows up.
    x_full *= (1 + 3 * torch.arange(ws, device=x_full.device)).repeat_interleave(m)[
        :, None
    ]
    tokens = slice(rank * m, (rank + 1) * m)

    if layer.row_parallel:
        k = K // ws
        x = x_full[:, rank * k : (rank + 1) * k].contiguous()
        y_ref = layer(x)
        y_fused = kernel.apply(x)
        y_hp = x_full.float() @ w_full.float().t()
    else:
        n = N // ws
        x = x_full[tokens]
        y_ref = layer(x_full)
        y_fused = kernel.apply(x)
        y_hp = x_full.float() @ w_full[rank * n : (rank + 1) * n].float().t()
    assert layer.reduce_results

    if layer.return_bias:
        (y_ref, b_ref), (y_fused, b_fused) = y_ref, y_fused
        assert b_fused is b_ref, "returned bias differs"
    elif layer.bias is not None:
        y_hp = y_hp + layer.bias.float()
    if layer.row_parallel:
        y_ref, y_hp = y_ref[tokens], y_hp[tokens]

    if y_fused.shape != y_ref.shape:
        return f"shape {tuple(y_fused.shape)} != {tuple(y_ref.shape)}"
    ref_err, fused_err = _rel_err(y_ref, y_hp), _rel_err(y_fused, y_hp)
    if not fused_err <= 1.5 * ref_err + 1e-2:
        return f"rel_err {fused_err:.4f} vs baseline {ref_err:.4f}"
    return None


def sp_matrix_worker(
    local_rank: int,
    world_size: int,
    q: mp.Queue,
    *,
    port: int,
    build: Build,
    kernel_cls: type,
    row_parallel: bool,
) -> None:
    failures = []
    try:
        device = init_worker_distributed(local_rank, world_size, port)
        torch.manual_seed(0)
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias_full = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1
        if row_parallel:
            k = K // world_size
            w = w_full[:, local_rank * k : (local_rank + 1) * k]
            bias = bias_full
        else:
            n = N // world_size
            w = w_full[local_rank * n : (local_rank + 1) * n]
            bias = bias_full[local_rank * n : (local_rank + 1) * n]

        for bias_mode in BIAS_MODES:
            quant_method, tensors = build(w.contiguous())
            layer = FakeLinear(
                quant_method,
                row_parallel=row_parallel,
                bias=None if bias_mode == "none" else bias,
                skip_bias_add=bias_mode == "skip_bias_add",
                **tensors,
            )
            quant_method.process_weights_after_loading(layer)
            kernel = kernel_cls(layer)
            for m in M_PER_RANK_CASES:
                try:
                    err = _check_case(kernel, layer, w_full, m, local_rank, world_size)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                if err:
                    failures.append(f"[{bias_mode}, m/rank={m}] {err}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"{type(e).__name__}: {e}")
    q.put("OK" if not failures else f"rank{local_rank}: " + " | ".join(failures))


def run_sp_matrix(
    build: Build, kernel_cls: type, row_parallel: bool, world_size: int
) -> None:
    """Run ``kernel_cls`` over the token-count/bias matrix on ``world_size``
    ranks; fails listing every mismatching case."""
    import pytest

    if torch.accelerator.device_count() < world_size:
        pytest.skip("Not enough accelerators to run the test.")
    worker = functools.partial(
        sp_matrix_worker,
        port=get_open_port(),
        build=build,
        kernel_cls=kernel_cls,
        row_parallel=row_parallel,
    )
    run_workers(worker, world_size)
