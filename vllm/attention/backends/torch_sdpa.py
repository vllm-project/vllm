""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.utils import is_cpu

if is_cpu():
    try:
        from vllm.attention.ops.ipex_attn import PagedAttention
    except ImportError:
        from vllm.attention.ops.paged_attn import PagedAttention
else:
    from vllm.attention.ops.paged_attn import PagedAttention


class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "torch-sdpa"

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TorchSDPAMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TorchSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchSDPABackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    slot_mapping: torch.Tensor
    seq_lens: Optional[List[int]]

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def prefill_metadata(self) -> Optional["TorchSDPAMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_decode_tokens == 0:
            assert self.num_prefills > 0
            return self

        return None

    @property
    def decode_metadata(self) -> Optional["TorchSDPAMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_prefills > 0:
            assert self.num_decode_tokens == 0
            return None

        return self


class TorchSDPABackendImpl(AttentionImpl[TorchSDPAMetadata]):

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
        assert blocksparse_params is None, ValueError(
            "Torch SPDA does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: TorchSDPAMetadata,  # type: ignore
        kv_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert kv_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TorchSDPABackendImpl")
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                self.kv_cache_dtype, kv_scale)

        if attn_metadata.is_prompt:
            assert attn_metadata.seq_lens is not None
            if (kv_cache is None or attn_metadata.block_tables.numel() == 0):
                if self.num_kv_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=1)

                if attn_metadata.attn_bias is None:
                    if self.alibi_slopes is not None:
                        att_masks = _make_alibi_bias(
                            self.alibi_slopes, query.dtype,
                            attn_metadata.seq_lens)  # type: ignore
                    elif self.sliding_window is not None:
                        att_masks = _make_sliding_window_bias(
                            attn_metadata.seq_lens, self.sliding_window,
                            query.dtype)  # type: ignore
                    else:
                        att_masks = [None] * len(attn_metadata.seq_lens)
                    attn_metadata.attn_bias = att_masks

                query = query.movedim(0, query.dim() - 2)
                key = key.movedim(0, key.dim() - 2)
                value = value.movedim(0, value.dim() - 2)

                start = 0
                output = torch.empty(
                    (num_tokens, self.num_heads, self.head_size),
                    dtype=query.dtype)
                for seq_len, mask in zip(attn_metadata.seq_lens,
                                         attn_metadata.attn_bias):
                    end = start + seq_len
                    sub_out = scaled_dot_product_attention(
                        query[None, :, start:end, :],
                        key[None, :, start:end, :],
                        value[None, :, start:end, :],
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=not self.need_mask,
                        scale=self.scale).squeeze(0).movedim(
                            query.dim() - 2, 0)
                    output[start:end, :, :] = sub_out
                    start = end
            else:
                # prefix-enabled attention
                raise RuntimeError(
                    "Torch SDPA backend doesn't support prefix decoding.")

        else:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_tables,
                attn_metadata.seq_lens_tensor,
                attn_metadata.max_decode_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases: List[torch.Tensor] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None]).unsqueeze_(0)
        inf_mask = torch.empty(
            (1, seq_len, seq_len),
            dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases: List[torch.Tensor] = []
    for seq_len in seq_lens:
        tensor = torch.full(
            (1, seq_len, seq_len),
            dtype=dtype,
            fill_value=1,
        )
        shift = 0
        mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
        if window_size is not None:
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
        mask = torch.log(mask)
        attn_biases.append(mask.to(dtype))

    return attn_biases
