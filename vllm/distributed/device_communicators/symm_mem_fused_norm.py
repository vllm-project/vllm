# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SymmetricMemory-backed fused all_reduce + RMSNorm (+ quant epilogue).

Portable Triton kernel. Supports:
    * Arbitrary world_size
    * Optional residual add
    * Quant epilogues: static FP8, dynamic per-token FP8, dynamic group FP8
    * Multiple platforms via current_platform abstraction

Refs: https://github.com/vllm-project/vllm/issues/25179
"""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
    _symm_mem_available = True
except ImportError:
    torch_symm_mem = None  # type: ignore[assignment]
    _symm_mem_available = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPILOGUE_NONE = 0
EPILOGUE_STATIC_FP8 = 1
EPILOGUE_DYNAMIC_TOKEN_FP8 = 2
EPILOGUE_DYNAMIC_GROUP_FP8 = 3

# Per-GPU symmetric memory buffer capacity (hard limit on input tensor size).
_SYMM_MEM_BUFFER_BYTES: int = 64 << 20  # 64 MiB

# FP8 quantization bounds (platform-aware).
_FP8_MIN, _FP8_MAX = get_fp8_min_max()

# ---------------------------------------------------------------------------
# Capability checks
# ---------------------------------------------------------------------------


def is_supported() -> bool:
    return _symm_mem_available and HAS_TRITON


def supported_max_input_bytes(world_size: int) -> int | None:
    if not is_supported() or world_size < 2:
        return None
    return _SYMM_MEM_BUFFER_BYTES


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _fused_ar_rmsnorm_kernel(
        peer_ptrs,
        out_ptr,
        quant_out_ptr,
        scale_out_ptr,
        residual_ptr,
        weight_ptr,
        scale_in_ptr,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        NUM_ROWS_PER_BLOCK: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        EPILOGUE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        fp8_min: tl.constexpr,
        fp8_max: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_start = pid * NUM_ROWS_PER_BLOCK
        cols = tl.arange(0, BLOCK_SIZE)
        col_mask = cols < n_cols

        for row_offset in tl.static_range(NUM_ROWS_PER_BLOCK):
            row = row_start + row_offset
            if row < n_rows:
                offset = row * n_cols + cols

                # AllReduce: P2P load + sum
                x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for i in tl.static_range(WORLD_SIZE):
                    peer_base = tl.load(peer_ptrs + i).to(
                        tl.pointer_type(out_ptr.dtype.element_ty)
                    )
                    xi = tl.load(peer_base + offset, mask=col_mask, other=0.0)
                    x += xi.to(tl.float32)

                # Residual add (in-place)
                if HAS_RESIDUAL:
                    res = tl.load(residual_ptr + offset,
                                  mask=col_mask, other=0.0)
                    x += res.to(tl.float32)
                    tl.store(residual_ptr + offset,
                             x.to(residual_ptr.dtype.element_ty),
                             mask=col_mask)

                # RMSNorm
                var = tl.sum(x * x, axis=0) / n_cols
                rstd = 1.0 / tl.sqrt(var + eps)
                w = tl.load(weight_ptr + cols,
                            mask=col_mask, other=0.0).to(tl.float32)
                y = x * rstd * w

                # Store norm output (bf16/fp16)
                tl.store(out_ptr + offset,
                         y.to(out_ptr.dtype.element_ty), mask=col_mask)

                # Epilogue: Static FP8
                if EPILOGUE == 1:
                    s = tl.load(scale_in_ptr)
                    q = tl.clamp(y * s, fp8_min, fp8_max)
                    tl.store(quant_out_ptr + offset,
                             q.to(quant_out_ptr.dtype.element_ty),
                             mask=col_mask)

                # Epilogue: Dynamic per-token FP8
                if EPILOGUE == 2:
                    amax = tl.max(tl.abs(y), axis=0)
                    scale = tl.where(amax > 0, amax * (1.0 / fp8_max), 1.0)
                    q = tl.clamp(y / scale, fp8_min, fp8_max)
                    tl.store(quant_out_ptr + offset,
                             q.to(quant_out_ptr.dtype.element_ty),
                             mask=col_mask)
                    tl.store(scale_out_ptr + row, scale)

                # Epilogue: Dynamic group FP8
                if EPILOGUE == 3:
                    n_groups: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
                    y_grp = tl.reshape(y, [n_groups, GROUP_SIZE])
                    amax_g = tl.max(tl.abs(y_grp), axis=1)
                    scale_g = tl.where(
                        amax_g > 0, amax_g * (1.0 / fp8_max), 1.0)
                    scale_2d = tl.broadcast_to(
                        tl.reshape(scale_g, [n_groups, 1]),
                        [n_groups, GROUP_SIZE],
                    )
                    q = tl.clamp(y_grp / scale_2d, fp8_min, fp8_max)
                    tl.store(
                        quant_out_ptr + offset,
                        tl.reshape(q, [BLOCK_SIZE]).to(
                            quant_out_ptr.dtype.element_ty),
                        mask=col_mask,
                    )
                    n_groups_actual = (n_cols + GROUP_SIZE - 1) // GROUP_SIZE
                    group_offsets = (row * n_groups_actual
                                    + tl.arange(0, n_groups))
                    group_mask = tl.arange(0, n_groups) < n_groups_actual
                    tl.store(scale_out_ptr + group_offsets,
                             scale_g, mask=group_mask)


# ---------------------------------------------------------------------------
# Kernel launch helper
# ---------------------------------------------------------------------------

def _pick_config(n_cols: int) -> tuple[int, int, int]:
    """Returns (num_rows_per_block, num_warps, num_stages)."""
    if n_cols <= 4096:
        return 4, 4, 2
    if n_cols <= 8192:
        return 2, 8, 2
    return 1, 16, 2


def _impl(
    allreduce_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    norm_out: torch.Tensor,
    residual: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_in: torch.Tensor | None = None,
    epilogue: int = 0,
    group_size: int = 128,
) -> None:
    assert is_supported()
    assert world_size >= 2 and allreduce_in.dim() == 2
    assert allreduce_in.dtype in (torch.float16, torch.bfloat16)
    assert allreduce_in.is_contiguous()

    n_rows, n_cols = allreduce_in.shape

    # Symmetric memory setup (rendezvous is internally cached for the same
    # buffer — repeated calls return the existing handle without overhead)
    group = get_tp_group().device_group
    hdl = torch_symm_mem.rendezvous(allreduce_in, group.group_name)
    ptrs = [hdl.get_buffer(r, allreduce_in.shape, allreduce_in.dtype).data_ptr()
            for r in range(world_size)]
    peer_data_ptrs = torch.tensor(ptrs, dtype=torch.int64,
                                  device=allreduce_in.device)

    # Grid config
    block_size = triton.next_power_of_2(n_cols)
    if epilogue == EPILOGUE_DYNAMIC_GROUP_FP8:
        block_size = max(block_size, group_size)
    num_rows_per_block, num_warps, num_stages = _pick_config(n_cols)
    grid = (triton.cdiv(n_rows, num_rows_per_block),)

    # Dummy pointer for unused optional args (Triton doesn't accept None)
    dummy = allreduce_in
    _quant_out = quant_out if quant_out is not None else dummy
    _scale_out = scale_out if scale_out is not None else dummy
    _residual = residual if residual is not None else dummy
    _scale_in = scale_in if scale_in is not None else dummy

    # Launch
    hdl.barrier(channel=0)
    _fused_ar_rmsnorm_kernel[grid](
        peer_data_ptrs, norm_out, _quant_out, _scale_out,
        _residual, rms_gamma, _scale_in,
        n_rows, n_cols, rms_eps,
        BLOCK_SIZE=block_size,
        NUM_ROWS_PER_BLOCK=num_rows_per_block,
        WORLD_SIZE=world_size,
        HAS_RESIDUAL=(residual is not None),
        EPILOGUE=epilogue,
        GROUP_SIZE=group_size,
        fp8_min=_FP8_MIN,
        fp8_max=_FP8_MAX,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    hdl.barrier(channel=1)


def _fake(
    allreduce_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    norm_out: torch.Tensor,
    residual: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_in: torch.Tensor | None = None,
    epilogue: int = 0,
    group_size: int = 128,
) -> None:
    return None


# ---------------------------------------------------------------------------
# Op registration
# ---------------------------------------------------------------------------

direct_register_custom_op(
    op_name="symm_mem_fused_allreduce_rmsnorm",
    op_func=_impl,
    mutates_args=["allreduce_in", "norm_out", "residual", "quant_out",
                  "scale_out"],
    fake_impl=_fake,
)

symm_mem_fused_allreduce_rmsnorm = (
    torch.ops.vllm.symm_mem_fused_allreduce_rmsnorm.default
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_allreduce_rmsnorm(
    x_symm: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    world_size: int,
    out: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_in: torch.Tensor | None = None,
    epilogue: int = EPILOGUE_NONE,
    group_size: int = 128,
) -> torch.Tensor:
    """Fused all_reduce + RMSNorm with optional FP8 quantization."""
    if out is None:
        out = torch.empty_like(x_symm)
    symm_mem_fused_allreduce_rmsnorm(
        allreduce_in=x_symm, rms_gamma=weight, rms_eps=eps,
        world_size=world_size, norm_out=out, residual=residual,
        quant_out=quant_out, scale_out=scale_out,
        scale_in=scale_in, epilogue=epilogue, group_size=group_size,
    )
    return out
