# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from contextlib import contextmanager
from functools import cache

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.v1.worker.ubatching import dbo_enabled
from vllm.v1.worker.workspace import WorkspaceManager, current_workspace_manager


@cache
def _make_comm_stream(device: torch.device) -> torch.cuda.ExternalStream:
    """Keep one worker-lifetime HIP queue per device with the compute CU mask."""
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    runtime = CudaRTLibrary()
    lib = runtime.lib
    pointer = ctypes.c_void_p
    words = ctypes.POINTER(ctypes.c_uint32)
    get_mask = lib.hipExtStreamGetCUMask
    get_mask.argtypes = [pointer, ctypes.c_uint32, words]
    create = lib.hipExtStreamCreateWithCUMask
    create.argtypes = [ctypes.POINTER(pointer), ctypes.c_uint32, words]
    destroy = lib.hipStreamDestroy
    destroy.argtypes = [pointer]
    with torch.cuda.device(device):
        compute = torch.cuda.current_stream(device)
        count = (
            torch.cuda.get_device_properties(device).multi_processor_count + 31
        ) // 32
        mask = (ctypes.c_uint32 * count)()
        runtime.CUDART_CHECK(get_mask(compute.cuda_stream, count, mask))
        handle = pointer()
        runtime.CUDART_CHECK(create(ctypes.byref(handle), count, mask))
        try:
            # The caching allocator can keep blocks associated with this handle
            # after tensors and temporary Python stream references go away.
            # Let HIP release the queue at worker exit together with the context.
            return torch.cuda.ExternalStream(handle.value, device=device)
        except BaseException:
            destroy(handle)
            raise


@contextmanager
def _piece_sizes(metadata, sizes):
    previous = metadata.local_sizes
    metadata.local_sizes = sizes
    try:
        yield
    finally:
        metadata.local_sizes = previous


def _split(
    kernel,
    hidden,
    w1,
    w2,
    ids,
    weights,
    activation,
    experts,
    expert_map,
    rank,
    sizes,
    comm_stream,
):
    metadata = get_forward_context().dp_metadata
    assert min(sizes) >= 2 and sizes[rank] == hidden.shape[0]
    first = [n // 2 for n in sizes]
    parts = (first, [n - n0 for n, n0 in zip(sizes, first)])
    rows = (slice(0, first[rank]), slice(first[rank], None))
    compute = torch.cuda.current_stream(hidden.device)
    comm = comm_stream
    output = hidden.new_empty(
        (hidden.shape[0], kernel.fused_experts.output_hidden_dim(hidden.shape[1]))
    )

    def gather(i):
        with torch.cuda.stream(comm), _piece_sizes(metadata, parts[i]):
            return kernel._prepare(
                hidden[rows[i]],
                weights[rows[i]],
                ids[rows[i]],
                experts,
                expert_map,
                False,
            )

    def expert(i, prepared):
        q, scale, token_meta, gathered_ids, gathered_weights = prepared
        private = (
            hidden.new_empty((sum(parts[0]), output.shape[-1])) if i == 0 else None
        )
        with _piece_sizes(metadata, parts[i]):
            y = kernel._fused_experts(
                in_dtype=hidden.dtype,
                a1q=q,
                a1q_scale=scale,
                w1=w1,
                w2=w2,
                topk_weights=gathered_weights,
                topk_ids=gathered_ids,
                activation=activation,
                global_num_experts=experts,
                local_num_experts=w1.shape[0],
                expert_map=expert_map,
                apply_router_weight_on_input=False,
                expert_tokens_meta=token_meta,
                output_alias=private,
            )
            # C1 may overwrite an internal workspace while R0 still reads it.
            # R1 is joined on the compute stream before the caller can reuse it.
            if private is not None and y.data_ptr() != private.data_ptr():
                return y.clone()
            return y

    def scatter(i, prepared, result):
        with torch.cuda.stream(comm), _piece_sizes(metadata, parts[i]):
            kernel._finalize(
                output[rows[i]],
                result,
                hidden[rows[i]],
                prepared[4],
                prepared[3],
                False,
                None,
                None,
            )

    comm.wait_stream(compute)
    try:
        prepared0 = gather(0)
        compute.wait_stream(comm)
        prepared1 = gather(1)
        result0 = expert(0, prepared0)
        # G1 was submitted before this wait. C1 only needs G1, not R0.
        compute.wait_stream(comm)
        comm.wait_stream(compute)
        scatter(0, prepared0, result0)
        result1 = expert(1, prepared1)
        comm.wait_stream(compute)
        scatter(1, prepared1, result1)
    except BaseException:
        # A partially submitted expert may still read comm-allocated inputs.
        comm.wait_stream(compute)
        raise
    finally:
        # A partially submitted collective may still use compute-owned storage.
        compute.wait_stream(comm)
    return output


def _reserve_unsplit_workspace(
    manager: WorkspaceManager, rows: int, width: int, dtype: torch.dtype
) -> None:
    # AITER manages its own scratch. The modular kernel reserves one output
    # of this shape even when the caller supplies an alias. Preserve that
    # high-water mark for whole calls after graph capture locks the workspace.
    if not manager.is_locked():
        manager.get_simultaneous(((rows, width), dtype))


def _supports_layout(sizes, width, element_bytes):
    if len(sizes) != 4 or min(sizes) < 2:
        return False
    # Below 320 MiB the extra expert launches have not been validated.
    return sum(sizes) * width * element_bytes >= 320 * 1024**2


def try_split(
    kernel, hidden, w1, w2, ids, weights, activation, experts, expert_map, apply_weight
):
    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    if (
        ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or ctx.dp_metadata is None
        or ctx.dp_metadata.local_sizes is None
        or torch.cuda.is_current_stream_capturing()
        or dbo_enabled()
        or apply_weight
        or hidden.dtype != torch.bfloat16
    ):
        return None
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return None
    pf = kernel.prepare_finalize
    if (
        not isinstance(pf, MoEPrepareAndFinalizeNaiveDPEPModular)
        or pf.is_sequence_parallel
        or not isinstance(kernel.fused_experts, AiterMxfp4Experts)
        or pf.supports_async()
        or getattr(kernel.fused_experts, "_lora_context", None) is not None
    ):
        return None
    sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()
    if not _supports_layout(sizes, hidden.shape[-1], hidden.element_size()):
        return None
    ep = get_ep_group()
    if not isinstance(
        getattr(ep.device_communicator, "all2all_manager", None), AgRsAll2AllManager
    ):
        return None
    dp = get_dp_group()
    if ep.world_size != dp.world_size or dp.world_size != len(sizes):
        return None
    # All ranks derive the same decision from this vector.
    assert hidden.shape[0] == sizes[dp.rank_in_group]
    _reserve_unsplit_workspace(
        current_workspace_manager(),
        sum(sizes),
        kernel.fused_experts.output_hidden_dim(hidden.shape[-1]),
        kernel.fused_experts.workspace_dtype(hidden.dtype),
    )
    return _split(
        kernel,
        hidden,
        w1,
        w2,
        ids,
        weights,
        activation,
        w1.shape[0] if experts == -1 else experts,
        expert_map,
        dp.rank_in_group,
        sizes,
        _make_comm_stream(hidden.device),
    )
