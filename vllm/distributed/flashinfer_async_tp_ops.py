# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Protocol

import torch

from vllm.utils.torch_utils import direct_register_custom_op


class _TPGroup(Protocol):
    unique_name: str

    def _all_gather_out_place(self, tensor: torch.Tensor, dim: int) -> torch.Tensor: ...

    def _reduce_scatter_out_place(
        self, tensor: torch.Tensor, dim: int
    ) -> torch.Tensor: ...


_GET_TP_GROUP: Callable[[], _TPGroup] | None = None
_GET_TP_WORLD_SIZE: Callable[[], int] | None = None
_FLASHINFER_ASYNC_TP_OPS_REGISTERED = False
_FLASHINFER_ASYNC_TP_DECOMPS_REGISTERED = False


def _require_tp_group() -> _TPGroup:
    assert _GET_TP_GROUP is not None, "FlashInfer AsyncTP ops are not registered"
    return _GET_TP_GROUP()


def _require_tp_world_size() -> int:
    assert _GET_TP_WORLD_SIZE is not None, "FlashInfer AsyncTP ops are not registered"
    return _GET_TP_WORLD_SIZE()


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
    from flashinfer import bmm_fp8 as flashinfer_bmm_fp8

    if reduce_op != "sum":
        raise NotImplementedError(
            "Only sum reduce_op is supported for FlashInfer AsyncTP RS"
        )

    mm_out = torch.empty(
        output_shape, dtype=out_dtype or torch.bfloat16, device=A.device
    )
    flashinfer_bmm_fp8(
        A.unsqueeze(0),
        B.unsqueeze(0),
        A_scale,
        B_scale,
        out_dtype or mm_out.dtype,
        mm_out.unsqueeze(0),
        "auto",
    )
    return _require_tp_group()._reduce_scatter_out_place(mm_out, orig_scatter_dim)


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
    rs_shape = list(output_shape)
    group_size = _require_tp_world_size()
    rs_shape[orig_scatter_dim] = rs_shape[orig_scatter_dim] // max(group_size, 1)
    return torch.empty(rs_shape, dtype=out_dtype, device=A.device)


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import bmm_fp8 as flashinfer_bmm_fp8

    A = _require_tp_group()._all_gather_out_place(A_shard, gather_dim)
    mm_shape = (*A.shape[:-1], B.shape[1])
    mm_out = torch.empty(mm_shape, dtype=out_dtype or torch.bfloat16, device=A.device)
    flashinfer_bmm_fp8(
        A.unsqueeze(0),
        B.unsqueeze(0),
        A_scale,
        B_scale,
        out_dtype or mm_out.dtype,
        mm_out.unsqueeze(0),
        "auto",
    )
    return A, mm_out


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
    gathered_shape = list(A_shard.shape)
    group_size = _require_tp_world_size()
    gathered_shape[gather_dim] = gathered_shape[gather_dim] * max(group_size, 1)
    gathered = torch.empty(gathered_shape, dtype=A_shard.dtype, device=A_shard.device)
    mm_out = torch.empty(
        (gathered_shape[0], B.shape[1]),
        dtype=out_dtype,
        device=A_shard.device,
    )
    return gathered, mm_out


def register_flashinfer_async_tp_ops(
    get_tp_group: Callable[[], _TPGroup],
    get_tp_world_size: Callable[[], int],
) -> None:
    global _GET_TP_GROUP, _GET_TP_WORLD_SIZE, _FLASHINFER_ASYNC_TP_OPS_REGISTERED

    _GET_TP_GROUP = get_tp_group
    _GET_TP_WORLD_SIZE = get_tp_world_size
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


def _register_inductor_lowering_for_flashinfer_collective_fp8_ops() -> None:
    """Register decompositions for FlashInfer AsyncTP custom ops.

    Expanding the fused wrappers back into the visible collective+GEMM sequence
    lets Inductor schedule the inner vLLM ops directly instead of treating the
    entire wrapper as a fallback black box.
    """
    global _FLASHINFER_ASYNC_TP_DECOMPS_REGISTERED

    if _FLASHINFER_ASYNC_TP_DECOMPS_REGISTERED:
        return

    from torch._decomp import register_decomposition

    @register_decomposition(
        torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default
    )
    def _decompose_fused_flashinfer_scaled_matmul_reduce_scatter(
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
        del scatter_dim_after_maybe_reshape, group_name
        if reduce_op != "sum":
            raise NotImplementedError(
                "Only sum reduce_op is supported for FlashInfer AsyncTP RS"
            )

        bmm_result = torch.ops.vllm.bmm_fp8.default(
            A.unsqueeze(0),
            B.unsqueeze(0),
            A_scale,
            B_scale,
            out_dtype or torch.bfloat16,
            "auto",
        )
        mm_result = bmm_result.squeeze(0)
        if list(mm_result.shape) != output_shape:
            mm_result = mm_result.view(output_shape)

        return torch.ops.vllm.reduce_scatter.default(
            mm_result,
            orig_scatter_dim,
            _require_tp_world_size(),
            _require_tp_group().unique_name,
        )

    @register_decomposition(
        torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default
    )
    def _decompose_fused_all_gather_flashinfer_scaled_matmul(
        A_shard: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        gather_dim: int,
        group_name: str,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del group_name

        gathered = torch.ops.vllm.all_gather.default(
            A_shard,
            gather_dim,
            _require_tp_world_size(),
            _require_tp_group().unique_name,
        )
        bmm_result = torch.ops.vllm.bmm_fp8.default(
            gathered.unsqueeze(0),
            B.unsqueeze(0),
            A_scale,
            B_scale,
            out_dtype or torch.bfloat16,
            "auto",
        )
        return gathered, bmm_result.squeeze(0)

    _FLASHINFER_ASYNC_TP_DECOMPS_REGISTERED = True
