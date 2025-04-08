# SPDX-License-Identifier: Apache-2.0
""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

# yapf conflicts with isort for this block
# yapf: disable
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)
# yapf: enable
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.ipex_attn import PagedAttention, _use_ipex
from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.logger import init_logger
from vllm.utils import make_tensor_with_pad
from vllm.worker.cpu_model_runner import ModelInputForCPUBuilder

logger = init_logger(__name__)


class TorchSDPABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TorchSDPAMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> Type["TorchSDPAMetadataBuilder"]:
        return TorchSDPAMetadataBuilder

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
    chunked_prefill: bool
    seq_lens: Optional[List[int]] = None  # For non-chunked prefill

    # For chunked prefill only
    max_query_len: Optional[int] = None
    max_kv_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    kv_start_loc: Optional[torch.Tensor] = None
    prefill_block_tables: Optional[torch.Tensor] = None

    # Begin encoder attn & enc/dec cross-attn fields...
    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None
        self.encoder_attn_bias: Optional[List[torch.Tensor]] = None
        self.cross_attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))

    @property
    def prefill_metadata(self) -> Optional["TorchSDPAMetadata"]:
        if self.num_prefill_tokens == 0:
            return None
        return self

    @property
    def decode_metadata(self) -> Optional["TorchSDPAMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        return self

    def get_seq_lens(
        self,
        attn_type: str,
    ):
        '''
        Extract appropriate sequence lengths from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate sequence lengths tensor for query
        * Appropriate sequence lengths tensor for key & value
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.seq_lens
        elif attn_type == AttentionType.ENCODER:
            seq_lens_q = self.encoder_seq_lens
            seq_lens_kv = self.encoder_seq_lens
        elif attn_type == AttentionType.ENCODER_DECODER:
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.encoder_seq_lens
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        return seq_lens_q, seq_lens_kv

    def get_attn_bias(
        self,
        attn_type: str,
    ) -> Optional[List[torch.Tensor]]:
        '''
        Extract appropriate attention bias from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate attention bias value given the attention type
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            return self.attn_bias
        elif attn_type == AttentionType.ENCODER:
            return self.encoder_attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            return self.cross_attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def set_attn_bias(
        self,
        attn_bias: List[torch.Tensor],
        attn_type: str,
    ) -> None:
        '''
        Update appropriate attention bias field of attention metadata,
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_bias: The desired attention bias value
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            self.attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER:
            self.encoder_attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            self.cross_attn_bias = attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def get_seq_len_block_table_args(
        self,
        attn_type: str,
    ) -> tuple:
        '''
        The particular choice of sequence-length- and block-table-related
        attributes which should be extracted from attn_metadata is dependent
        on the type of attention operation.

        Decoder attn -> select entirely decoder self-attention-related fields
        Encoder/decoder cross-attn -> select encoder sequence lengths &
                                    cross-attn block-tables fields
        Encoder attn -> select encoder sequence lengths fields & no block tables

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * is_prompt: True if prefill, False otherwise
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:

        * Appropriate sequence-lengths tensor
        * Appropriate max sequence-length scalar
        * Appropriate block tables (or None)
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            # Decoder self-attention
            # Choose max_seq_len based on whether we are in prompt_run
            return (self.seq_lens_tensor, self.max_decode_seq_len,
                    self.block_tables)
        elif attn_type == AttentionType.ENCODER_DECODER:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    self.cross_block_tables)
        elif attn_type == AttentionType.ENCODER:
            # No block tables associated with encoder attention
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    None)
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")


class TorchSDPAMetadataBuilder(AttentionMetadataBuilder[TorchSDPAMetadata]):

    def __init__(self, input_builder: ModelInputForCPUBuilder) -> None:
        self.chunked_prefill = input_builder.chunked_prefill
        self.input_builder = input_builder

    def prepare(self):
        self.input_data = self.input_builder.input_data

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int) -> TorchSDPAMetadata:
        input_data = self.input_data
        prefill_seq_lens = seq_lens[0:input_data.num_prefills]
        prefill_query_lens = query_lens[0:input_data.num_prefills]
        slot_mapping = torch.tensor(input_data.slot_mapping,
                                    dtype=torch.long,
                                    device="cpu")

        # For chunked-prefill
        if self.chunked_prefill and input_data.num_prefill_tokens != 0:
            prefill_block_tables = make_tensor_with_pad(
                self.input_data.prefill_block_tables,
                pad=0,
                dtype=torch.int32,
                device="cpu",
            )
            query_lens_tensor = torch.tensor(prefill_query_lens,
                                             dtype=torch.int32,
                                             device="cpu")
            kv_lens_tensor = torch.tensor(prefill_seq_lens,
                                          dtype=torch.int32,
                                          device="cpu")
            query_start_loc = torch.zeros(input_data.num_prefills + 1,
                                          dtype=torch.int32,
                                          device="cpu")
            kv_start_loc = torch.zeros(input_data.num_prefills + 1,
                                       dtype=torch.int32,
                                       device="cpu")
            torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=query_start_loc[1:])
            torch.cumsum(kv_lens_tensor,
                         dim=0,
                         dtype=torch.int32,
                         out=kv_start_loc[1:])
            max_query_len = max(prefill_query_lens)
            max_kv_len = max(prefill_seq_lens)
        else:
            prefill_block_tables = None
            query_start_loc = None
            kv_start_loc = None
            max_query_len = None
            max_kv_len = None

        # For paged attention
        if input_data.num_decode_tokens != 0:
            seq_lens_tensor = torch.tensor(
                input_data.seq_lens[input_data.num_prefills:],
                dtype=torch.int32,
                device="cpu",
            )
            block_tables = make_tensor_with_pad(
                self.input_data.decode_block_tables,
                pad=0,
                dtype=torch.int32,
                device="cpu",
            )
        else:
            block_tables = torch.tensor([])
            seq_lens_tensor = torch.tensor(
                input_data.seq_lens[:input_data.num_prefills],
                dtype=torch.int32,
                device="cpu",
            )

        # For multi-modal models
        placeholder_index_maps = None
        if len(input_data.multi_modal_inputs_list) != 0:
            placeholder_index_maps = {
                modality: placeholder_map.index_map()
                for modality, placeholder_map in
                input_data.multi_modal_placeholder_maps.items()
            }

        attn_metadata = TorchSDPAMetadata(
            chunked_prefill=self.chunked_prefill,
            seq_lens=prefill_seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_kv_len=max_kv_len,
            query_start_loc=query_start_loc,
            kv_start_loc=kv_start_loc,
            max_decode_seq_len=input_data.max_decode_seq_len,
            num_prefills=input_data.num_prefills,
            num_prefill_tokens=input_data.num_prefill_tokens,
            num_decode_tokens=input_data.num_decode_tokens,
            block_tables=block_tables,
            prefill_block_tables=prefill_block_tables,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
        )

        return attn_metadata


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
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "Torch SPDA does not support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once("Torch SPDA does not support logits soft cap. "
                                "Outputs may be slightly off.")
        if use_irope:
            logger.warning_once(
                "Using irope in Torch SPDA is not supported yet, it will fall"
                " back to global attention for long context.")
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

        if is_quantized_kv_cache(kv_cache_dtype) and not _use_ipex:
            raise NotImplementedError(
                "Torch SDPA backend FP8 KV cache requires "
                "intel_extension_for_pytorch support.")
        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchSDPAMetadata,  # type: ignore
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                PagedAttention.write_to_paged_cache(
                    key, value, key_cache, value_cache, updated_slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)

        if attn_type != AttentionType.ENCODER:
            # Decoder self-attention supports chunked prefill.
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
        else:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0

        if attn_type == AttentionType.DECODER:
            # Only enforce this shape-constraint for decoder
            # self-attention
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        if prefill_meta := attn_metadata.prefill_metadata:
            assert attn_metadata.seq_lens is not None
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                self._run_sdpa_forward(output,
                                       query,
                                       key,
                                       value,
                                       prefill_meta,
                                       attn_type=attn_type)
            else:
                # prefix-enabled attention
                assert not self.need_mask
                import intel_extension_for_pytorch.llm.modules as ipex_modules
                output = torch.empty_like(query)
                ipex_modules.PagedAttention.flash_attn_varlen_func(
                    output[:prefill_meta.num_prefill_tokens, :, :],
                    query[:prefill_meta.num_prefill_tokens, :, :],
                    key_cache,
                    value_cache,
                    prefill_meta.query_start_loc,
                    prefill_meta.kv_start_loc,
                    prefill_meta.max_query_len,
                    prefill_meta.max_kv_len,
                    self.scale,
                    True,
                    prefill_meta.prefill_block_tables,
                    self.alibi_slopes,
                )

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")
            # Decoding run.
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)

            PagedAttention.forward_decode(
                output[attn_metadata.num_prefill_tokens:, :, :],
                query[attn_metadata.num_prefill_tokens:, :, :],
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_sdpa_forward(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TorchSDPAMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_masks = attn_metadata.get_attn_bias(attn_type)
        if attn_masks is None:
            if self.alibi_slopes is not None:
                attn_masks = _make_alibi_bias(
                    self.alibi_slopes, query.dtype,
                    attn_metadata.seq_lens)  # type: ignore
            elif self.sliding_window is not None:
                assert attn_metadata.seq_lens is not None
                attn_masks = _make_sliding_window_bias(
                    attn_metadata.seq_lens, self.sliding_window,
                    query.dtype)  # type: ignore
            else:
                seq_lens, _ = attn_metadata.get_seq_lens(attn_type)
                attn_masks = [None] * len(seq_lens)
            attn_metadata.set_attn_bias(attn_masks, attn_type)

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        causal_attn = (attn_type == AttentionType.DECODER)

        seq_lens_q, seq_lens_kv = attn_metadata.get_seq_lens(attn_type)
        start_q, start_kv = 0, 0
        for seq_len_q, seq_len_kv, mask in zip(seq_lens_q, seq_lens_kv,
                                               attn_masks):
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            sub_out = scaled_dot_product_attention(
                query[None, :, start_q:end_q, :],
                key[None, :, start_kv:end_kv, :],
                value[None, :, start_kv:end_kv, :],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=causal_attn and mask is None,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = sub_out
            start_q, start_kv = end_q, end_kv


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
