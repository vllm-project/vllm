from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

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
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    effective_query_lens: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
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
        logits_soft_cap: Optional[float] = None,
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
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Attention logits soft-capping is not supported.")

        if torch_xla.tpu.version() < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")

        self.megacore_mode = None
        tpu_env = torch_xla.tpu.get_tpu_env()
        tpu_type = (tpu_env.get("ACCELERATOR_TYPE", None)
                    or tpu_env.get("TYPE", None)
                    or tpu_env.get("TPU_ACCELERATOR_TYPE", None))
        assert tpu_type is not None
        tpu_type = tpu_type.lower()

        if (("lite" not in tpu_type) and ("v6" not in tpu_type)):
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
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: PallasMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache[0] = [num_kv_heads, num_blocks, block_size, head_size]
            kv_cache[1] = [num_kv_heads, num_blocks, block_size, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor 
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)

        if kv_cache[0].numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        if attn_metadata.num_prefills > 0:
            if attn_metadata.block_tables is None:
                # Prefill without paged KV cache.
                assert seq_len % 16 == 0, (
                    "Pallas FlashAttention kernel requires seq_len to be a "
                    f"multiple of 16 but got {seq_len}")

                # Handle GQA/MQA.
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv,
                                                dim=-2)
                    key = key.view(batch_size, seq_len, self.num_heads,
                                   self.head_size)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=-2)
                    value = value.view(batch_size, seq_len, self.num_heads,
                                       self.head_size)
                # FlashAttention kernel requires the input shape to be
                # [batch_size, num_heads, seq_len, d_model]
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
                # Prefill with paged KV cache.
                # TODO(woosuk): Tune the below knobs.
                num_kv_pages_per_compute_block = 16
                num_queries_per_compute_block = 16
                assert seq_len % num_queries_per_compute_block == 0
                output = torch.ops.xla.multi_queries_paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    attn_metadata.effective_query_lens,
                    num_kv_pages_per_compute_block,
                    num_queries_per_compute_block,
                    use_kernel=True,
                )
        else:
            # Decoding run.
            assert kv_cache[0].numel() > 0
            query = query.squeeze(dim=1)
            pages_per_compute_block = 16  # TODO(woosuk): Tune this value.

            assert attn_metadata.block_tables is not None
            assert attn_metadata.context_lens is not None
            # NOTE(woosuk): The PagedAttention Pallas kernel stores the entire
            # block table in SMEM. Therefore, if the block table is too large,
            # the kernel compilation will fail. To avoid this, we split the
            # batch dimension into smaller chunks and run the kernel multiple
            # times.
            MAX_SMEM_USAGE = 512 * 1024
            size_per_seq = 4 * attn_metadata.block_tables.shape[1]
            max_num_seq = MAX_SMEM_USAGE // size_per_seq

            if batch_size <= max_num_seq:
                output = paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    pages_per_compute_block,
                    self.megacore_mode,
                )
            else:
                chunk_size = max_num_seq
                # Make sure the chunk size is a multiple of 2.
                chunk_size = chunk_size // 2 * 2
                num_chunks = (batch_size + chunk_size - 1) // chunk_size

                output = torch.empty_like(query)
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = chunk_start + chunk_size
                    # NOTE(woosuk): We skip this line because it causes Dynamo
                    # compilation error. Instead, we rely on the slice operation
                    # to handle the out-of-bound case.
                    # chunk_end = min(chunk_end, batch_size)
                    chunk_output = paged_attention(
                        query[chunk_start:chunk_end],
                        key_cache,
                        value_cache,
                        attn_metadata.context_lens[chunk_start:chunk_end],
                        attn_metadata.block_tables[chunk_start:chunk_end],
                        pages_per_compute_block,
                        self.megacore_mode,
                    )
                    output[chunk_start:chunk_end] = chunk_output

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


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    pages_per_compute_block: int,
    megacore_mode: Optional[str],
) -> torch.Tensor:
    batch_size = query.shape[0]
    if megacore_mode == "batch" and batch_size % 2 != 0:
        megacore_mode = None
    else:
        megacore_mode = megacore_mode

    # NOTE(woosuk): A temporary workaround to avoid the error:
    # "xla::paged_attention() Expected a value of type 'str' for
    # argument 'megacore_mode' but instead found type 'NoneType'."
    if megacore_mode is not None:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
            megacore_mode=megacore_mode,
        )
    else:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
        )
    return output
