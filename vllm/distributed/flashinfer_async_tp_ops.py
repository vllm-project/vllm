# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Protocol

import torch

from vllm.utils.torch_utils import direct_register_custom_op


class _TPGroup(Protocol):
    def _all_gather_out_place(self, tensor: torch.Tensor, dim: int) -> torch.Tensor: ...

    def _reduce_scatter_out_place(
        self, tensor: torch.Tensor, dim: int
    ) -> torch.Tensor: ...


_GET_TP_GROUP: Callable[[], _TPGroup] | None = None
_GET_TP_WORLD_SIZE: Callable[[], int] | None = None
_FLASHINFER_ASYNC_TP_OPS_REGISTERED = False


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
    """Register explicit Inductor lowerings for FlashInfer AsyncTP custom ops."""
    import torch._inductor.lowering as _lowering

    ops = (
        torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default,
        torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
    )
    for op in ops:
        if op not in _lowering.lowerings:
            _lowering.make_fallback(op)
