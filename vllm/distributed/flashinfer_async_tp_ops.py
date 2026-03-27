# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed._functional_collectives as funcol

from vllm.utils.torch_utils import direct_register_custom_op

_FLASHINFER_ASYNC_TP_OPS_REGISTERED = False


def _flashinfer_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.vllm.bmm_fp8(
        A.unsqueeze(0),
        B.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).squeeze(0)


def _flashinfer_scaled_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor,
    **_kwargs: Any,
) -> torch.Tensor:
    """Adapt FlashInfer bmm_fp8 to mm_out_op(A, B, ..., out=...) shape."""
    result = _flashinfer_scaled_mm(
        A,
        B,
        scale_a,
        scale_b,
        out_dtype or out.dtype,
    )
    out.copy_(result)
    return out


def fused_flashinfer_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    from torch.distributed._symmetric_memory import (
        _fused_scaled_matmul_reduce_scatter_impl,
    )

    return _fused_scaled_matmul_reduce_scatter_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A=A,
        B=B,
        A_scale=A_scale,
        kwargs={"scale_b": B_scale, "out_dtype": out_dtype},
        out_dtype=out_dtype,
        reduce_op=reduce_op,
        orig_scatter_dim=orig_scatter_dim,
        scatter_dim_after_maybe_reshape=scatter_dim_after_maybe_reshape,
        group_name=group_name,
        output_shape=output_shape,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = out_dtype or A.dtype
    C = _flashinfer_scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        out_dtype,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        orig_scatter_dim,
        group_name,
    )
    return funcol.wait_tensor(res)


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed._symmetric_memory import _fused_all_gather_matmul_impl

    A, res_list = _fused_all_gather_matmul_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A_shard=A_shard,
        Bs=[B],
        A_scale=A_scale,
        kwargs_list=[{"scale_b": B_scale, "out_dtype": out_dtype}],
        out_dtypes=[out_dtype],
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=True,
    )
    return A, res_list[0]


def fused_all_gather_flashinfer_scaled_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dtype = out_dtype or A_shard.dtype
    A = funcol.all_gather_tensor(A_shard, gather_dim, group_name)
    A = funcol.wait_tensor(A)
    mm_out = _flashinfer_scaled_mm(A, B, A_scale, B_scale, out_dtype)
    return A, mm_out


def register_flashinfer_async_tp_ops() -> None:
    global _FLASHINFER_ASYNC_TP_OPS_REGISTERED

    if _FLASHINFER_ASYNC_TP_OPS_REGISTERED:
        return

    direct_register_custom_op(
        op_name="fused_flashinfer_scaled_matmul_reduce_scatter",
        op_func=fused_flashinfer_scaled_matmul_reduce_scatter,
        fake_impl=fused_flashinfer_scaled_matmul_reduce_scatter_fake,
    )

    direct_register_custom_op(
        op_name="fused_all_gather_flashinfer_scaled_matmul",
        op_func=fused_all_gather_flashinfer_scaled_matmul,
        fake_impl=fused_all_gather_flashinfer_scaled_matmul_fake,
    )

    _FLASHINFER_ASYNC_TP_OPS_REGISTERED = True
