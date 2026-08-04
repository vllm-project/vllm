# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse


def _get_sparse_mla_offload_context(layer_name: LayerNameType):
    from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLALayerView

    name = _resolve_layer_name(layer_name)
    context = get_forward_context()
    layer_view = context.no_compile_layers.get(name)
    if not isinstance(layer_view, SparseMLALayerView):
        raise RuntimeError(f"missing sparse MLA offload view for {name!r}")
    if not isinstance(context.attn_metadata, dict):
        raise RuntimeError("sparse MLA offload does not support microbatch metadata")
    metadata = context.attn_metadata.get(name)
    if metadata is None or not hasattr(metadata, "req_id_per_token"):
        raise RuntimeError(f"missing sparse MLA metadata for {name!r}")
    return layer_view, metadata.req_id_per_token


def sparse_mla_cache_plan(
    current_main_kv: torch.Tensor,
    topk_logical_ids: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_view, req_id_per_token = _get_sparse_mla_offload_context(layer_name)
    main_host_kv_uva = layer_view.main_host_kv_uva
    if main_host_kv_uva is None:
        raise RuntimeError("sparse MLA offload requires a CUDA Host view")
    buffers = layer_view.local_buffers
    ops.sparse_mla_cache_plan(
        current_main_kv,
        buffers["request_block_ids"],
        buffers["request_num_blocks"],
        buffers["request_num_tokens"],
        buffers["request_generation"],
        buffers["request_active"],
        req_id_per_token,
        topk_logical_ids,
        buffers["resident_main_kv"],
        buffers["resident_logical_ids"],
        buffers["resident_last_access"],
        buffers["resident_generation"],
        buffers["newest_main_kv"],
        buffers["newest_logical_ids"],
        buffers["newest_generation"],
        buffers["topk_physical_ids"],
        buffers["topk_hit_mask"],
        buffers["miss_logical_ids"],
        buffers["miss_victim_slots"],
        buffers["miss_counts"],
        buffers["accepted_counts"],
        main_host_kv_uva.shape[0],
    )
    return current_main_kv.new_empty(0)


def sparse_mla_cache_plan_fake(
    current_main_kv: torch.Tensor,
    topk_logical_ids: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return current_main_kv.new_empty(0)


direct_register_custom_op(
    op_name="sparse_mla_cache_plan",
    op_func=sparse_mla_cache_plan,
    fake_impl=sparse_mla_cache_plan_fake,
)


def _sparse_mla_partial_attention(
    query: torch.Tensor,
    resident_main_kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> None:
    num_heads = query.shape[1]
    padded_heads = 64 if num_heads < 64 else num_heads
    if padded_heads % 64 != 0:
        raise ValueError("sparse MLA offload requires 32 or a multiple of 64 heads")
    padded_query = query
    if padded_heads != num_heads:
        padded_query = F.pad(query, (0, 0, 0, padded_heads - num_heads))
    kwargs = {
        "topk_length": lengths,
    }
    if padded_heads == num_heads:
        kwargs["out"] = partial_output
    attention_output, _, attention_lse = flash_mla_sparse_fwd(
        padded_query,
        resident_main_kv.view(-1, 1, resident_main_kv.shape[-1]),
        indices,
        query.shape[-1] ** -0.5,
        **kwargs,
    )
    if padded_heads != num_heads:
        partial_output.copy_(attention_output[:, :num_heads])
    partial_lse.copy_(attention_lse[:, :num_heads])


@eager_break_during_capture
def sparse_mla_offload_attention(
    query: torch.Tensor,
    current_main_kv: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    cache_plan_dep: torch.Tensor,
) -> None:
    del current_main_kv, cache_plan_dep
    layer_view, req_id_per_token = _get_sparse_mla_offload_context(layer_name)
    main_host_kv_uva = layer_view.main_host_kv_uva
    side_stream = layer_view.side_stream
    if main_host_kv_uva is None or side_stream is None:
        raise RuntimeError("sparse MLA offload CUDA resources are not initialized")
    buffers = layer_view.local_buffers
    token_rows = query.shape[0]
    hit_output = buffers["hit_output"].view(-1, query.shape[1], output.shape[-1])[
        :token_rows
    ]
    hit_lse = buffers["hit_lse"].view(-1, query.shape[1])[:token_rows]
    miss_output = buffers["miss_output"].view(-1, query.shape[1], output.shape[-1])[
        :token_rows
    ]
    miss_lse = buffers["miss_lse"].view(-1, query.shape[1])[:token_rows]
    hit_indices = buffers["topk_physical_ids"].view(
        -1, 1, buffers["topk_physical_ids"].shape[-1]
    )[:token_rows]
    miss_indices = buffers["miss_victim_slots"].view(
        -1, 1, buffers["miss_victim_slots"].shape[-1]
    )[:token_rows]
    hit_counts = buffers["accepted_counts"].view(-1)[:token_rows]
    miss_counts = buffers["miss_counts"].view(-1)[:token_rows]

    current_stream = torch.cuda.current_stream(query.device)
    fork_event = layer_view.fork_ready_events[0]
    ready_event = layer_view.miss_ready_events[0]
    if fork_event is None or ready_event is None:
        raise RuntimeError("sparse MLA offload CUDA events are not initialized")
    fork_event.record(current_stream)
    side_stream.wait_event(fork_event)
    with torch.cuda.stream(side_stream):
        ops.sparse_mla_offload_transfer(
            main_host_kv_uva,
            buffers["request_block_ids"],
            buffers["request_num_blocks"],
            buffers["request_num_tokens"],
            buffers["request_generation"],
            buffers["request_active"],
            req_id_per_token,
            buffers["newest_main_kv"],
            buffers["newest_logical_ids"],
            buffers["miss_logical_ids"],
            buffers["miss_victim_slots"],
            buffers["miss_counts"],
            buffers["accepted_counts"],
            buffers["resident_main_kv"],
            buffers["resident_logical_ids"],
            buffers["resident_last_access"],
            buffers["resident_generation"],
            layer_view.is_host_writer,
            main_host_kv_uva.shape[1],
        )
        ready_event.record(side_stream)

    _sparse_mla_partial_attention(
        query,
        buffers["resident_main_kv"],
        hit_indices,
        hit_counts,
        hit_output,
        hit_lse,
    )
    current_stream.wait_event(ready_event)
    _sparse_mla_partial_attention(
        query,
        buffers["resident_main_kv"],
        miss_indices,
        miss_counts,
        miss_output,
        miss_lse,
    )
    merge_attn_states(
        output,
        hit_output,
        hit_lse.transpose(0, 1),
        miss_output,
        miss_lse.transpose(0, 1),
    )
    output.masked_fill_(req_id_per_token[:token_rows].view(-1, 1, 1) < 0, 0)


def sparse_mla_offload_attention_fake(
    query: torch.Tensor,
    current_main_kv: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    cache_plan_dep: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="sparse_mla_offload_attention",
    op_func=sparse_mla_offload_attention,
    mutates_args=["output"],
    fake_impl=sparse_mla_offload_attention_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.flexible_layout,),
)
