# SPDX-License-Identifier: Apache-2.0
"""Correctness test for SymmMem fused allreduce + RMSNorm (+ quant).

Minimal multi-GPU test using mp.spawn (matches vllm distributed test style).
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

import pytest
from vllm.platforms import current_platform

# Import from the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from vllm.distributed.device_communicators.symm_mem_fused_norm import (
    EPILOGUE_NONE,
    EPILOGUE_STATIC_FP8,
    EPILOGUE_DYNAMIC_TOKEN_FP8,
)


SHAPES = [(32, 4096), (128, 8192), (256, 16384)]
DTYPES = [torch.bfloat16]
EPS = 1e-5


def _ref_allreduce_rmsnorm(x, weight, eps, residual=None):
    """Eager reference: NCCL allreduce -> (optional residual add) -> RMSNorm."""
    out = x.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    if residual is not None:
        out = out + residual
    f = out.to(torch.float32)
    var = f.pow(2).mean(dim=-1, keepdim=True)
    normed = (f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)
    return normed, out  # normed output, updated residual


def _worker(rank, world_size, results_queue):
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29557",
                       "RANK": str(rank), "WORLD_SIZE": str(world_size)})
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            device_id=torch.device(f"cuda:{rank}"))
    enable_symm_mem_for_group(dist.group.WORLD.group_name)
    group_name = dist.group.WORLD.group_name

    # Lazy import (avoids importing triton before CUDA is set up)
    from vllm.distributed.device_communicators.symm_mem_fused_norm import (
        _impl, EPILOGUE_NONE, EPILOGUE_STATIC_FP8, EPILOGUE_DYNAMIC_TOKEN_FP8,
        _FP8_MIN, _FP8_MAX,
    )

    all_pass = True
    for shape in SHAPES:
        for dtype in DTYPES:
            n_rows, n_cols = shape
            torch.manual_seed(42 + rank)
            local = torch.randn(shape, dtype=dtype, device="cuda") * 0.5
            torch.manual_seed(0)
            weight = torch.randn(n_cols, dtype=dtype, device="cuda") * 0.1

            # --- Test 1: basic (no residual, no quant) ---
            sym = _SymmetricMemory.empty_strided_p2p(
                shape, (n_cols, 1), dtype, torch.device(f"cuda:{rank}"), group_name)
            sym.copy_(local)
            _SymmetricMemory.rendezvous(sym)
            norm_out = torch.empty_like(sym)

            # Can't use _impl directly (needs vllm parallel_state), use kernel directly
            # Instead test via the standalone path
            from vllm.distributed.device_communicators.symm_mem_fused_norm import (
                _fused_ar_rmsnorm_kernel, _pick_config,
            )
            import triton

            peer_ptrs = torch.empty(world_size, dtype=torch.int64, device="cuda")
            hdl = _SymmetricMemory.rendezvous(sym)
            for r in range(world_size):
                buf = hdl.get_buffer(r, shape, dtype)
                peer_ptrs[r] = buf.data_ptr()

            bs = triton.next_power_of_2(n_cols)
            nrpb, nw, ns = _pick_config(n_cols)
            grid = (triton.cdiv(n_rows, nrpb),)

            hdl.barrier(channel=0)
            _fused_ar_rmsnorm_kernel[grid](
                peer_ptrs, norm_out, sym, sym, sym, weight, sym,
                n_rows, n_cols, EPS,
                BLOCK_SIZE=bs, NUM_ROWS_PER_BLOCK=nrpb, WORLD_SIZE=world_size,
                HAS_RESIDUAL=False, EPILOGUE=0, GROUP_SIZE=128,
                fp8_min=_FP8_MIN, fp8_max=_FP8_MAX,
                num_warps=nw, num_stages=ns,
            )
            hdl.barrier(channel=1)
            torch.cuda.synchronize()
            dist.barrier()

            ref, _ = _ref_allreduce_rmsnorm(local, weight, EPS)
            tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
            if (norm_out.float() - ref.float()).abs().max().item() > tol:
                all_pass = False

            # --- Test 2: with residual ---
            sym.copy_(local)
            _SymmetricMemory.rendezvous(sym)
            residual_in = torch.randn(shape, dtype=dtype, device="cuda") * 0.3
            residual_fused = residual_in.clone()
            norm_out2 = torch.empty_like(sym)

            hdl = _SymmetricMemory.rendezvous(sym)
            for r in range(world_size):
                buf = hdl.get_buffer(r, shape, dtype)
                peer_ptrs[r] = buf.data_ptr()

            hdl.barrier(channel=0)
            _fused_ar_rmsnorm_kernel[grid](
                peer_ptrs, norm_out2, sym, sym, residual_fused, weight, sym,
                n_rows, n_cols, EPS,
                BLOCK_SIZE=bs, NUM_ROWS_PER_BLOCK=nrpb, WORLD_SIZE=world_size,
                HAS_RESIDUAL=True, EPILOGUE=0, GROUP_SIZE=128,
                fp8_min=_FP8_MIN, fp8_max=_FP8_MAX,
                num_warps=nw, num_stages=ns,
            )
            hdl.barrier(channel=1)
            torch.cuda.synchronize()
            dist.barrier()

            ref2, ref_res = _ref_allreduce_rmsnorm(local, weight, EPS, residual_in)
            if (norm_out2.float() - ref2.float()).abs().max().item() > tol:
                all_pass = False

            # --- Test 3: static FP8 quant ---
            sym.copy_(local)
            _SymmetricMemory.rendezvous(sym)
            scale_in = torch.tensor([1.0 / 0.5], dtype=torch.float32, device="cuda")
            quant_out = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
            norm_out3 = torch.empty_like(sym)

            hdl = _SymmetricMemory.rendezvous(sym)
            for r in range(world_size):
                buf = hdl.get_buffer(r, shape, dtype)
                peer_ptrs[r] = buf.data_ptr()

            hdl.barrier(channel=0)
            _fused_ar_rmsnorm_kernel[grid](
                peer_ptrs, norm_out3, quant_out, sym, sym, weight, scale_in,
                n_rows, n_cols, EPS,
                BLOCK_SIZE=bs, NUM_ROWS_PER_BLOCK=nrpb, WORLD_SIZE=world_size,
                HAS_RESIDUAL=False, EPILOGUE=1, GROUP_SIZE=128,
                fp8_min=_FP8_MIN, fp8_max=_FP8_MAX,
                num_warps=nw, num_stages=ns,
            )
            hdl.barrier(channel=1)
            torch.cuda.synchronize()
            dist.barrier()

            ref3, _ = _ref_allreduce_rmsnorm(local, weight, EPS)
            ref_q = (ref3.float() * scale_in.item()).to(torch.float8_e4m3fn)
            if (quant_out.float() - ref_q.float()).abs().max().item() > 0.5:
                all_pass = False

            # --- Test 4: dynamic per-token FP8 ---
            sym.copy_(local)
            _SymmetricMemory.rendezvous(sym)
            quant_out4 = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
            scale_out4 = torch.empty(n_rows, dtype=torch.float32, device="cuda")
            norm_out4 = torch.empty_like(sym)

            hdl = _SymmetricMemory.rendezvous(sym)
            for r in range(world_size):
                buf = hdl.get_buffer(r, shape, dtype)
                peer_ptrs[r] = buf.data_ptr()

            hdl.barrier(channel=0)
            _fused_ar_rmsnorm_kernel[grid](
                peer_ptrs, norm_out4, quant_out4, scale_out4, sym, weight, sym,
                n_rows, n_cols, EPS,
                BLOCK_SIZE=bs, NUM_ROWS_PER_BLOCK=nrpb, WORLD_SIZE=world_size,
                HAS_RESIDUAL=False, EPILOGUE=2, GROUP_SIZE=128,
                fp8_min=_FP8_MIN, fp8_max=_FP8_MAX,
                num_warps=nw, num_stages=ns,
            )
            hdl.barrier(channel=1)
            torch.cuda.synchronize()
            dist.barrier()

            # Verify: dequant(quant_out4, scale_out4) ≈ norm_out4
            deq = quant_out4.float() * scale_out4.unsqueeze(1)
            if (deq - norm_out4.float()).abs().max().item() > 0.5:
                all_pass = False

    if rank == 0:
        results_queue.put("ALL_PASS" if all_pass else "FAILED")
    dist.destroy_process_group()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("tp_size", [2, 4])
def test_symm_mem_fused_allreduce_rmsnorm(tp_size: int):
    if tp_size > torch.cuda.device_count():
        pytest.skip(f"Need {tp_size} GPUs")
    q = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(tp_size, q), nprocs=tp_size)
    result = q.get(timeout=60)
    assert result == "ALL_PASS", f"Test failed: {result}"
