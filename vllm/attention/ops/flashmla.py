# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

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
    is_availble, maybe_reason = _is_flashmla_available()
    if not is_availble:
        return False, maybe_reason
    if current_platform.get_device_capability()[0] != 9:
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_availble, maybe_reason = _is_flashmla_available()
    if not is_availble:
        return False, maybe_reason
    if current_platform.get_device_capability()[0] not in (9, 10):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: int | None = None,
    is_fp8_kvcache: bool = False,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
    - cache_seqlens: (batch_size), dtype torch.int32.
    - num_q_tokens_per_head_k:
            Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
    - num_heads_k: The number of k heads.
    - num_heads_q:
            The number of q heads.
            This argument is optional when sparse attention is not enabled
    - is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
    - topk: If not None, sparse attention will be enabled,
            and only tokens in the `indices` array
            passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns:
    - tile_scheduler_metadata:
            (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    - num_splits: (batch_size + 1), dtype torch.int32.
    """
    if is_fp8_kvcache and topk is None:
        return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
        )
    return torch.ops._flashmla_C.get_mla_decoding_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
    )


def flash_mla_with_kvcache(
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
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
    - q: (batch_size, seq_len_q, num_heads_q, head_dim).
    - k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
    - block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
    - cache_seqlens: (batch_size), torch.int32.
    - head_dim_v: Head dimension of v.
    - tile_scheduler_metadata:
        (num_sm_parts, TileSchedulerMetaDataSize), torch.int32,
        returned by get_mla_metadata.
    - num_splits:
        (batch_size + 1), torch.int32, returned by get_mla_metadata.
    - softmax_scale: float.
        The scale of QK^T before applying softmax.
        Default to 1 / sqrt(head_dim).
    - causal: bool. Whether to apply causal attention mask.
    - descale_q: (batch_size),
        torch.float32. Descaling factors for Q, used for fp8 quantization.
    - descale_k: (batch_size),
        torch.float32. Descaling factors for K, used for fp8 quantization.
    - is_fp8_kvcache: bool.
        Whether the k_cache and v_cache are in fp8 format.
        For the format of FP8 KV cache, please refer to README.md
    - indices: (batch_size, seq_len_q, topk), torch.int32.
        If not None, sparse attention will be enabled,
        and only tokens in the `indices` array will be attended to.
        Invalid indices should be set to -1 or numbers >= total_seq_len_kv.
        For details about how to set up `indices`, please refer to README.md.

    Returns:
    - out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
    - softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        # NOTE (zyongye): sparse attention is also causal
        # since it only attend to the tokens before
        # but here `causal` should not be specified
        assert not causal, "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (descale_k is None), (
        "descale_q and descale_k should be both None or both not None"
    )

    if indices is None and q.element_size() == 1:
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
    else:
        out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits,
            is_fp8_kvcache,
            indices,
        )
    return out, softmax_lse


def flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
    - q: [s_q, h_q, d_qk], bfloat16
    - kv: [s_kv, h_kv, d_qk], bfloat16
    - indices: [s_q, h_kv, topk], int32.
        Invalid indices should be set to -1 or numbers >= s_kv
    - sm_scale: float
    - d_v: The dimension of value vectors. Can only be 512

    Returns:
    - (output, max_logits, lse)
        About the definition of output,
        max_logits and lse, please refer to README.md
    - output: [s_q, h_q, d_v], bfloat16
    - max_logits:  [s_q, h_q], float
    - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    results = torch.ops._flashmla_C.sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
    return results


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
