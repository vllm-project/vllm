# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import suppress
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.utils.torch_utils import direct_register_custom_op

_FLASHINFER_ASYNC_TP_OPS_REGISTERED = False
_AG_OVERLAP_LOCAL_ROWS = 256
_MIN_AG_OVERLAP_LOCAL_ROWS = 2048
_RS_OVERLAP_OUTPUT_ROWS = 256
_MIN_RS_OVERLAP_OUTPUT_ROWS = 512


@dataclass(frozen=True)
class _ChunkSlices:
    local: slice
    gathered: slice


@dataclass(frozen=True)
class _ReduceScatterChunkSlices:
    matmul: slice
    scattered: slice


@dataclass
class _OverlapStreams:
    compute: torch.Stream
    comm: torch.Stream


_OVERLAP_STREAMS = threading.local()


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
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.ops.vllm.bmm_fp8_out(
        A.unsqueeze(0),
        B.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
        out.unsqueeze(0),
    )
    return out


def _flashinfer_mm_output_shape(
    A: torch.Tensor,
    B: torch.Tensor,
) -> list[int]:
    return [*A.shape[:-1], B.shape[1]]


def _has_symbolic_shape(tensor: torch.Tensor) -> bool:
    from torch.fx.experimental.symbolic_shapes import is_symbolic

    return any(is_symbolic(dim) for dim in tensor.shape)


def _is_fake_tensor(tensor: torch.Tensor) -> bool:
    return isinstance(tensor, FakeTensor)


def _is_compile_time_context() -> bool:
    return (
        torch.compiler.is_compiling()
        or torch._dynamo.is_compiling()
        or torch.jit.is_tracing()
    )


def _should_use_compile_safe_path(*tensors: torch.Tensor) -> bool:
    if _is_compile_time_context():
        return True

    return any(
        _is_fake_tensor(tensor) or _has_symbolic_shape(tensor) for tensor in tensors
    )


def _get_group(group_name: str):
    with suppress(Exception):
        from vllm.distributed.parallel_state import _groups

        if (group_ref := _groups.get(group_name)) is not None:
            return group_ref()

    return None


def _require_group(group_name: str):
    group = _get_group(group_name)
    if group is None:
        raise RuntimeError(f"Group {group_name} is not found.")
    return group


def _resolve_out_dtype(
    tensor: torch.Tensor, out_dtype: torch.dtype | None
) -> torch.dtype:
    return out_dtype or tensor.dtype


def _resolve_world_size(group_name: str, group=None) -> int:
    if group is not None:
        return group.world_size

    resolved_group = _get_group(group_name)
    if resolved_group is not None:
        return resolved_group.world_size

    from torch.distributed.distributed_c10d import _get_group_size_by_name

    return _get_group_size_by_name(group_name)


def _allocate_mm_out(
    A: torch.Tensor, B: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        _flashinfer_mm_output_shape(A, B),
        dtype=out_dtype,
        device=A.device,
    )


def _allocate_all_gather_outputs(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    gather_dim: int,
    world_size: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered_shape = list(A_shard.shape)
    gathered_shape[gather_dim] *= world_size
    gathered = torch.empty(
        gathered_shape,
        dtype=A_shard.dtype,
        device=A_shard.device,
    )
    return gathered, _allocate_mm_out(gathered, B, out_dtype)


def _allocate_reduce_scatter_output(
    A: torch.Tensor,
    B: torch.Tensor,
    world_size: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    mm_output_shape = _flashinfer_mm_output_shape(A, B)
    output_shape = list(mm_output_shape)
    output_shape[0] //= world_size
    return (
        torch.empty(output_shape, dtype=out_dtype, device=A.device),
        mm_output_shape,
    )


def _get_overlap_streams(device: torch.device) -> _OverlapStreams:
    by_device = getattr(_OVERLAP_STREAMS, "by_device", None)
    if by_device is None:
        by_device = {}
        _OVERLAP_STREAMS.by_device = by_device

    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()

    streams = by_device.get(device_index)
    if streams is None:
        streams = _OverlapStreams(
            compute=torch.Stream(device=device),
            comm=torch.Stream(device=device),
        )
        by_device[device_index] = streams
    return streams


def _fork_streams(caller_stream: torch.Stream, *streams: torch.Stream) -> None:
    fork_event = torch.Event(enable_timing=False)
    caller_stream.record_event(fork_event)
    for stream in streams:
        stream.wait_event(fork_event)


def _join_streams(caller_stream: torch.Stream, *streams: torch.Stream) -> None:
    for stream in streams:
        done_event = torch.Event(enable_timing=False)
        done_event.record(stream)
        caller_stream.wait_event(done_event)


def _iter_ag_chunk_slices(
    local_rows: int, world_size: int, chunk_local_rows: int
) -> tuple[_ChunkSlices, ...]:
    gathered_start = 0
    chunks: list[_ChunkSlices] = []
    for local_start in range(0, local_rows, chunk_local_rows):
        local_end = min(local_start + chunk_local_rows, local_rows)
        gathered_rows = (local_end - local_start) * world_size
        chunks.append(
            _ChunkSlices(
                local=slice(local_start, local_end),
                gathered=slice(gathered_start, gathered_start + gathered_rows),
            )
        )
        gathered_start += gathered_rows
    return tuple(chunks)


def _is_overlap_eligible(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or not tensor.is_contiguous():
        return False
    if _should_use_compile_safe_path(tensor):
        return False
    return not torch.cuda.is_current_stream_capturing()


def _should_use_ag_overlap(A_shard: torch.Tensor, gather_dim: int) -> bool:
    if gather_dim != 0:
        return False
    if not _is_overlap_eligible(A_shard):
        return False
    return A_shard.shape[0] >= _MIN_AG_OVERLAP_LOCAL_ROWS


def _iter_rs_chunk_slices(
    matmul_rows: int, world_size: int, chunk_output_rows: int
) -> tuple[_ReduceScatterChunkSlices, ...]:
    chunk_matmul_rows = chunk_output_rows * world_size
    chunks: list[_ReduceScatterChunkSlices] = []
    scattered_start = 0
    for matmul_start in range(0, matmul_rows, chunk_matmul_rows):
        matmul_end = min(matmul_start + chunk_matmul_rows, matmul_rows)
        scattered_rows = (matmul_end - matmul_start) // world_size
        chunks.append(
            _ReduceScatterChunkSlices(
                matmul=slice(matmul_start, matmul_end),
                scattered=slice(scattered_start, scattered_start + scattered_rows),
            )
        )
        scattered_start += scattered_rows
    return tuple(chunks)


def _should_use_rs_overlap(A: torch.Tensor, group_world_size: int) -> bool:
    if not _is_overlap_eligible(A):
        return False
    if A.shape[0] % group_world_size != 0:
        return False
    output_rows = A.shape[0] // group_world_size
    return output_rows >= _MIN_RS_OVERLAP_OUTPUT_ROWS


def _fused_all_gather_flashinfer_scaled_matmul_compile(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = _resolve_world_size(group_name)
    gathered = torch.ops.vllm.all_gather(
        A_shard,
        gather_dim,
        world_size,
        group_name,
    )
    mm_out = _flashinfer_scaled_mm(gathered, B, A_scale, B_scale, out_dtype)
    return gathered, mm_out


def _fused_flashinfer_scaled_matmul_reduce_scatter_compile(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    group_name: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    mm_out = _flashinfer_scaled_mm(A, B, A_scale, B_scale, out_dtype)
    return torch.ops.vllm.reduce_scatter(
        mm_out,
        0,
        _resolve_world_size(group_name),
        group_name,
    )


def _fused_all_gather_flashinfer_scaled_matmul_one_shot(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    A, mm_out = _allocate_all_gather_outputs(
        A_shard,
        B,
        gather_dim,
        group.world_size,
        out_dtype,
    )
    group._all_gather_into(A, A_shard, gather_dim)
    _flashinfer_scaled_mm_out(
        A,
        B,
        A_scale,
        B_scale,
        out_dtype,
        mm_out,
    )
    return A, mm_out


def _fused_all_gather_flashinfer_scaled_matmul_overlap(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    group,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_local_rows = min(_AG_OVERLAP_LOCAL_ROWS, A_shard.shape[0])
    chunks = _iter_ag_chunk_slices(
        A_shard.shape[0],
        group.world_size,
        chunk_local_rows,
    )
    if len(chunks) < 2:
        return _fused_all_gather_flashinfer_scaled_matmul_one_shot(
            A_shard,
            B,
            A_scale,
            B_scale,
            0,
            group,
            out_dtype,
        )

    A, mm_out = _allocate_all_gather_outputs(
        A_shard,
        B,
        0,
        group.world_size,
        out_dtype,
    )
    streams = _get_overlap_streams(A_shard.device)
    caller_stream = torch.accelerator.current_stream(A_shard.device)
    _fork_streams(caller_stream, streams.comm, streams.compute)

    chunk_ready = [torch.Event(enable_timing=False) for _ in chunks]
    for idx, chunk in enumerate(chunks):
        with streams.comm:
            group._all_gather_into(A[chunk.gathered], A_shard[chunk.local], 0)
            chunk_ready[idx].record(streams.comm)

        if idx == 0:
            continue

        prev_chunk = chunks[idx - 1]
        with streams.compute:
            streams.compute.wait_event(chunk_ready[idx - 1])
            _flashinfer_scaled_mm_out(
                A[prev_chunk.gathered],
                B,
                A_scale,
                B_scale,
                out_dtype,
                mm_out[prev_chunk.gathered],
            )

    with streams.compute:
        streams.compute.wait_event(chunk_ready[-1])
        _flashinfer_scaled_mm_out(
            A[chunks[-1].gathered],
            B,
            A_scale,
            B_scale,
            out_dtype,
            mm_out[chunks[-1].gathered],
        )

    _join_streams(caller_stream, streams.comm, streams.compute)
    return A, mm_out


def _fused_flashinfer_scaled_matmul_reduce_scatter_one_shot(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    group,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    output, mm_output_shape = _allocate_reduce_scatter_output(
        A,
        B,
        group.world_size,
        out_dtype,
    )
    mm_out = torch.empty(mm_output_shape, dtype=out_dtype, device=A.device)
    _flashinfer_scaled_mm_out(
        A,
        B,
        A_scale,
        B_scale,
        out_dtype,
        mm_out,
    )
    group._reduce_scatter_into(output, mm_out, 0)
    return output


def _fused_flashinfer_scaled_matmul_reduce_scatter_overlap(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    group,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    chunk_output_rows = min(_RS_OVERLAP_OUTPUT_ROWS, A.shape[0] // group.world_size)
    chunks = _iter_rs_chunk_slices(A.shape[0], group.world_size, chunk_output_rows)
    if len(chunks) < 2:
        return _fused_flashinfer_scaled_matmul_reduce_scatter_one_shot(
            A,
            B,
            A_scale,
            B_scale,
            group,
            out_dtype,
        )

    n = B.shape[1]
    max_chunk_rows = max(chunk.matmul.stop - chunk.matmul.start for chunk in chunks)
    staging = [
        torch.empty((max_chunk_rows, n), dtype=out_dtype, device=A.device)
        for _ in range(2)
    ]
    output, _ = _allocate_reduce_scatter_output(
        A,
        B,
        group.world_size,
        out_dtype,
    )

    streams = _get_overlap_streams(A.device)
    caller_stream = torch.accelerator.current_stream(A.device)
    _fork_streams(caller_stream, streams.comm, streams.compute)

    gemm_done = [torch.Event(enable_timing=False) for _ in range(2)]
    comm_done = [torch.Event(enable_timing=False) for _ in range(2)]
    for event in comm_done:
        caller_stream.record_event(event)

    for idx, chunk in enumerate(chunks):
        slot = idx % 2
        chunk_rows = chunk.matmul.stop - chunk.matmul.start

        with streams.compute:
            streams.compute.wait_event(comm_done[slot])
            _flashinfer_scaled_mm_out(
                A[chunk.matmul],
                B,
                A_scale,
                B_scale,
                out_dtype,
                staging[slot][:chunk_rows],
            )
            gemm_done[slot].record(streams.compute)

        with streams.comm:
            streams.comm.wait_event(gemm_done[slot])
            group._reduce_scatter_into(
                output[chunk.scattered],
                staging[slot][:chunk_rows],
                0,
            )
            comm_done[slot].record(streams.comm)

    _join_streams(caller_stream, streams.comm, streams.compute)
    return output


def fused_flashinfer_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if reduce_op != "sum":
        raise NotImplementedError(
            "FlashInfer fused matmul+reduce_scatter only supports sum."
        )

    if orig_scatter_dim != 0 or scatter_dim_after_maybe_reshape != 0:
        raise NotImplementedError(
            "FlashInfer fused matmul+reduce_scatter currently expects scatter dim 0."
        )

    out_dtype = _resolve_out_dtype(A, out_dtype)
    if _should_use_compile_safe_path(A, B):
        return _fused_flashinfer_scaled_matmul_reduce_scatter_compile(
            A,
            B,
            A_scale,
            B_scale,
            group_name,
            out_dtype,
        )
    group = _require_group(group_name)
    if _should_use_rs_overlap(A, group.world_size):
        return _fused_flashinfer_scaled_matmul_reduce_scatter_overlap(
            A,
            B,
            A_scale,
            B_scale,
            group,
            out_dtype,
        )
    return _fused_flashinfer_scaled_matmul_reduce_scatter_one_shot(
        A,
        B,
        A_scale,
        B_scale,
        group,
        out_dtype,
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
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_out_dtype = _resolve_out_dtype(A, out_dtype)
    world_size = _resolve_world_size(group_name)
    output_shape = _flashinfer_mm_output_shape(A, B)
    output_shape[0] //= world_size
    return torch.empty(
        output_shape,
        dtype=resolved_out_dtype,
        device=A.device,
    )


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gather_dim != 0:
        raise NotImplementedError(
            "FlashInfer fused all_gather+matmul currently expects gather dim 0."
        )

    out_dtype = _resolve_out_dtype(A_shard, out_dtype)
    if _should_use_compile_safe_path(A_shard, B):
        return _fused_all_gather_flashinfer_scaled_matmul_compile(
            A_shard,
            B,
            A_scale,
            B_scale,
            gather_dim,
            group_name,
            out_dtype,
        )
    group = _require_group(group_name)
    if _should_use_ag_overlap(A_shard, gather_dim):
        return _fused_all_gather_flashinfer_scaled_matmul_overlap(
            A_shard,
            B,
            A_scale,
            B_scale,
            group,
            out_dtype,
        )
    return _fused_all_gather_flashinfer_scaled_matmul_one_shot(
        A_shard,
        B,
        A_scale,
        B_scale,
        gather_dim,
        group,
        out_dtype,
    )


def fused_all_gather_flashinfer_scaled_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_out_dtype = _resolve_out_dtype(A_shard, out_dtype)
    world_size = _resolve_world_size(group_name)
    gathered_shape = list(A_shard.shape)
    gathered_shape[gather_dim] *= world_size
    gathered = torch.empty(
        gathered_shape,
        dtype=A_shard.dtype,
        device=A_shard.device,
    )
    mm_out = torch.empty(
        _flashinfer_mm_output_shape(gathered, B),
        dtype=resolved_out_dtype,
        device=A_shard.device,
    )
    return gathered, mm_out


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
