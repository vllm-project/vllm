# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch
# Required to register custom ops.
import torch_xla.experimental.custom_kernel  # noqa: F401

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * 2, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")


@dataclass
class PallasMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the PallasAttentionBackendImpl
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: int


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("Paged attention Pallas kernel does "
                             "not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        tpu_version = torch_xla.tpu.version()
        if tpu_version < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PallasMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            if output is None:
                output = torch.ones_like(query)
            return output

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)

        if kv_cache.numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            write_to_kv_cache(key, value, kv_cache, slot_mapping)

        if envs.VLLM_TPU_VALIDATE_DYNAMIC_INPUTS:
            validate_dynamic_inputs(query, kv_cache,
                                    attn_metadata.context_lens,
                                    attn_metadata.block_tables,
                                    attn_metadata.query_start_loc,
                                    attn_metadata.num_seqs,
                                    self.sliding_window, self.logits_soft_cap)

        output = torch.ops.xla.ragged_paged_attention(
            query,
            kv_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            # By default, the system utilizes optimized block size and
            # vmem_limit_bytes parameters from the kernel repository. However,
            # these can be manually adjusted for debugging if necessary.
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            vmem_limit_bytes=None,
            use_kernel=True,
            sm_scale=self.scale,
            sliding_window=self.sliding_window,
            soft_cap=self.logits_soft_cap,
        )

        return output.reshape(num_tokens, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads *  head_size]
        kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]

    """
    _, _, num_combined_kv_heads, head_size = kv_cache.shape
    num_kv_heads = num_combined_kv_heads // 2

    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    kv = torch.cat([key, value], axis=-1).reshape(-1, num_combined_kv_heads,
                                                  head_size)

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)

    kv_cache = kv_cache.flatten(0, 1)
    kv_cache.index_copy_(0, slot_mapping, kv)


def validate_static_inputs(
    q: torch.Tensor,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: torch.
    Tensor,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: torch.Tensor,  # i32[max_num_seqs]
    page_indices: torch.Tensor,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: torch.Tensor,  # i32[max_num_seqs + 1]
    num_seqs: torch.Tensor,  # i32[1]
    sliding_window: int | None = None,
    soft_cap: float | None = None,
):
    """Validates the static inputs for the attention mechanism.

    Args:
        q: Query tensor.
        kv_pages: Key/value pages tensor.
        kv_lens: Key/value lengths tensor.
        page_indices: Page indices tensor.
        cu_q_lens: Cumulative query lengths tensor.
        num_seqs: Number of sequences tensor.
        sliding_window: Sliding window size.
        soft_cap: Soft cap value.

    Raises:
        ValueError: If any of the input constraints are violated.
    """
    _, num_q_heads, head_dim = q.shape
    _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    max_num_seqs, _ = page_indices.shape

    if num_seqs.shape != (1, ):
        raise ValueError(f"{num_seqs.shape=} must be (1,)")
    if head_dim_k != head_dim:
        raise ValueError(f"Q head_dim {head_dim} must be the same as"
                         " that of K/V {head_dim_k}.")
    if kv_lens.shape != (max_num_seqs, ):
        raise ValueError(
            f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
            " `max_num_seqs` is `page_indices.shape[0]`.")
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
            " `max_num_seqs` is `page_indices.shape[0]`.")
    if (kv_lens.dtype != torch.int32 or page_indices.dtype != torch.int32
            or cu_q_lens.dtype != torch.int32):
        raise ValueError(
            "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
            f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}.")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def validate_dynamic_inputs(
    q: torch.Tensor,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: torch.
    Tensor,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: torch.Tensor,  # i32[max_num_seqs]
    page_indices: torch.Tensor,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: torch.Tensor,  # i32[max_num_seqs + 1]
    num_seqs: torch.Tensor,  # i32[1]
    sliding_window: int | None = None,
    soft_cap: float | None = None,
):
    """Validates the dynamic inputs for the attention mechanism.

    Args:
        q: Query tensor.
        kv_pages: Key/value pages tensor.
        kv_lens: Key/value lengths tensor.
        page_indices: Page indices tensor.
        cu_q_lens: Cumulative query lengths tensor.
        num_seqs: Number of sequences tensor.
        sliding_window: Sliding window size.
        soft_cap: Soft cap value.

    Raises:
        ValueError: If any of the input constraints are violated.
    """
    validate_static_inputs(q, kv_pages, kv_lens, page_indices, cu_q_lens,
                           num_seqs, sliding_window, soft_cap)
    max_num_batched_tokens = q.shape[0]
    page_size = kv_pages.shape[1]
    max_num_seqs, pages_per_seq = page_indices.shape

    if num_seqs[0] > max_num_seqs:
        raise ValueError(
            f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")

    max_kv_len = torch.max(kv_lens)
    min_pages_per_seq = cdiv(max_kv_len, page_size)

    if pages_per_seq < min_pages_per_seq:
        raise ValueError(
            f"{pages_per_seq=} must be greater or equal to"
            f" {min_pages_per_seq=} given {max_kv_len=} and {page_size=}.")

    if cu_q_lens[num_seqs[0]] > max_num_batched_tokens:
        raise ValueError(
            f"Total q tokens {cu_q_lens[num_seqs[0]]} must be less or equal to"
            f" {max_num_batched_tokens=}.")

    for i in range(num_seqs[0]):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_len = kv_lens[i]
        if q_len > kv_len:
            raise ValueError(
                f"{q_len=} must be less or equal to {kv_len=} at sequence {i}."
            )
