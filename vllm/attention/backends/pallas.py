from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.
import torch_xla.experimental.dynamo_set_buffer_donor

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    @torch.compile(backend="openxla")
    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]


@dataclass
class PallasMetadata(AttentionMetadata):

    # Currently, input sequences can only contain all prefills
    # or all decoding.
    block_tables: Optional[torch.Tensor]
    context_lens: Optional[torch.Tensor]

    @property
    def prefill_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
        assert self.block_tables is None
        assert self.context_lens is None
        return self

    @property
    def decode_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.block_tables is not None
        assert self.context_lens is not None
        return self


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")

        if torch_xla.tpu.version() < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")

        self.megacore_mode = None
        tpu_type = torch_xla.tpu.get_tpu_env()["TYPE"].lower()
        if not tpu_type.endswith("lite"):
            if self.num_kv_heads % 2 == 0:
                self.megacore_mode = "kv_head"
            else:
                # NOTE(woosuk): If the batch size is not a multiple of 2, the
                # megacore mode will be None.
                self.megacore_mode = "batch"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        attn_metadata: PallasMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache = [num_kv_heads, num_blocks, block_size, head_size]
            value_cache = [num_kv_heads, num_blocks, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        assert kv_scale == 1.0
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)

        if kv_cache[0] is not None:
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        if attn_metadata.num_prefills > 0:
            assert seq_len % 16 == 0, (
                "Pallas FlashAttention kernel requires seq_len to be a "
                f"multiple of 16 but got {seq_len}")

            # Handle GQA/MQA.
            if self.num_kv_heads != self.num_heads:
                key = key.repeat_interleave(self.num_queries_per_kv, dim=-2)
                key = key.view(batch_size, seq_len, self.num_heads,
                               self.head_size)
                value = value.repeat_interleave(self.num_queries_per_kv,
                                                dim=-2)
                value = value.view(batch_size, seq_len, self.num_heads,
                                   self.head_size)
            # FlashAttention requires [batch_size, num_heads, seq_len, d_model]
            # while the input is [batch_size, seq_len, num_heads, d_model].
            # Permute the input to match the required format.
            output = torch.ops.xla.flash_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                True,
            )
            output = output.permute(0, 2, 1, 3)
        else:
            # Decoding run.
            assert kv_cache is not None

            pages_per_compute_block = 16  # TODO(woosuk): Tune this value.
            if self.megacore_mode == "batch" and batch_size % 2 != 0:
                megacore_mode = None
            else:
                megacore_mode = self.megacore_mode

            # NOTE(woosuk): A temporary workaround to avoid the error:
            # "xla::paged_attention() Expected a value of type 'str' for
            # argument 'megacore_mode' but instead found type 'NoneType'."
            if megacore_mode is not None:
                output = torch.ops.xla.paged_attention(
                    query.squeeze(dim=1),
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    pages_per_compute_block,
                    megacore_mode=megacore_mode,
                )
            else:
                output = torch.ops.xla.paged_attention(
                    query.squeeze(dim=1),
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    pages_per_compute_block,
                )

        # Reshape the output tensor.
        return output.reshape(batch_size, seq_len, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)
