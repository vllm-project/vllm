# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch_xla.core.xla_builder as xb
import torch_xla.experimental.custom_kernel  # noqa: F401
# Required to register custom ops.
from torch.library import impl
from torch_xla._internal.jax_workarounds import requires_jax
from torch_xla.experimental.custom_kernel import XLA_LIB

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import cdiv, next_power_of_2

logger = init_logger(__name__)

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128


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
        padded_head_size = cdiv(
            head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
        return (num_blocks, block_size, num_kv_heads * 2, padded_head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the PallasMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        max_num_page_per_req = (1024 * 1024 // 2 //
                                vllm_config.scheduler_config.max_num_seqs // 4)
        min_page_size = cdiv(vllm_config.model_config.max_model_len,
                             max_num_page_per_req)
        min_page_size = 1 << (min_page_size - 1).bit_length()
        return min_page_size

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        page_size = next_power_of_2(
            vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


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
    num_seqs: torch.Tensor
    num_slices_per_kv_cache_update_block: int


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
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
    ) -> None:
        if use_irope:
            logger.warning_once(
                "Using irope in Pallas is not supported yet, it will fall back "
                "to global attention for long context.")
        if blocksparse_params is not None:
            raise ValueError("Paged attention Pallas kernel does "
                             "not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
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
        output_scale: Optional[torch.Tensor] = None,
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
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for PallasAttentionBackendImpl")

        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            if output is None:
                output = torch.ones_like(query)
            return output

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        if self.head_size % TPU_HEAD_SIZE_ALIGNMENT != 0:
            padded_head_size = cdiv(
                self.head_size,
                TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
            query = torch.nn.functional.pad(
                query, (0, padded_head_size - self.head_size), value=0.0)
            key = torch.nn.functional.pad(
                key, (0, padded_head_size - self.head_size), value=0.0)
            value = torch.nn.functional.pad(
                value, (0, padded_head_size - self.head_size), value=0.0)

        if self.kv_sharing_target_layer_name is None and kv_cache.numel() > 0:
            # Write input keys and values to the KV cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            slot_mapping = attn_metadata.slot_mapping
            write_to_kv_cache(
                key, value, kv_cache, slot_mapping,
                attn_metadata.num_slices_per_kv_cache_update_block)

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

        if self.head_size % TPU_HEAD_SIZE_ALIGNMENT != 0:
            output = output[:, :, :self.head_size]

        return output.reshape(num_tokens, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_slices_per_kv_cache_update_block: int,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads *  head_size]
        kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]
        num_slices_per_kv_cache_update_block: int
    """
    _, page_size, num_combined_kv_heads, head_size = kv_cache.shape
    head_size = cdiv(head_size,
                     TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
    kv = torch.cat([key, value], axis=-1).reshape(-1, num_combined_kv_heads,
                                                  head_size)

    torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)

    kv_cache = kv_cache.flatten(0, 1)
    new_kv_cache = torch.ops.xla.kv_cache_update_op(
        kv, slot_mapping, kv_cache, page_size,
        num_slices_per_kv_cache_update_block)
    # NOTE: the in-place copy will be optimized away by XLA compiler.
    kv_cache.copy_(new_kv_cache)


@requires_jax
def kv_cache_update_op_impl(kv: torch.Tensor, slot_mapping: torch.Tensor,
                            kv_cache: torch.Tensor, page_size: int,
                            num_slices_per_block: int):
    from vllm.attention.ops.pallas_kv_cache_update import kv_cache_update
    new_kv_cache = xb.call_jax(kv_cache_update, (kv, slot_mapping, kv_cache), {
        "page_size": page_size,
        "num_slices_per_block": num_slices_per_block
    })
    return new_kv_cache


XLA_LIB.define(
    "kv_cache_update_op(Tensor kv, Tensor slot_mapping, Tensor kv_cache, "
    "int page_size, int num_slices_per_block) -> Tensor", )


@impl(XLA_LIB, "kv_cache_update_op", "XLA")
def kv_cache_update_op_xla(kv: torch.Tensor, slot_mapping: torch.Tensor,
                           kv_cache: torch.Tensor, page_size: int,
                           num_slices_per_block: int) -> torch.Tensor:
    new_kv_cache = kv_cache_update_op_impl(kv, slot_mapping, kv_cache,
                                           page_size, num_slices_per_block)
    return new_kv_cache


@impl(XLA_LIB, "kv_cache_update_op", "CompositeExplicitAutograd")
def kv_cache_update_op_non_xla(kv: torch.Tensor, slot_mapping: torch.Tensor,
                               kv_cache: torch.Tensor, page_size: int,
                               num_slices_per_block: int) -> torch.Tensor:
    return kv_cache
