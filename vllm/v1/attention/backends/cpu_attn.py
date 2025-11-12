# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

try:
    import intel_extension_for_pytorch.llm.modules as ipex_modules

    _use_ipex = True
# AttributeError is to handle a bug in ipex
# https://github.com/intel/intel-extension-for-pytorch/pull/813
except (ImportError, AttributeError):
    _use_ipex = False

from vllm import _custom_ops as ops

logger = init_logger(__name__)


class TorchSDPABackend(AttentionBackend):
    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        attn_impl = _get_paged_attn_impl()
        return attn_impl.get_supported_head_sizes()

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def get_builder_cls() -> type["TorchSDPAMetadataBuilderV1"]:
        return TorchSDPAMetadataBuilderV1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return _get_paged_attn_impl().get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TorchSDPAMetadata(AttentionMetadata):
    """Attention metadata for prefill and decode batched together."""

    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    decode_seq_lens_tensor: torch.Tensor | None
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    decode_max_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    decode_block_tables: torch.Tensor | None
    """Metadata for TorchSDPABackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    chunked_prefill: bool
    seq_lens: list[int] | None = None  # For non-chunked prefill

    # For chunked prefill only
    max_query_len: int | None = None
    prefill_max_seq_len: int | None = None
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_seq_start_loc: torch.Tensor | None = None
    prefill_block_tables: torch.Tensor | None = None

    # For V1 logits index only
    query_start_loc: torch.Tensor | None = None

    # Begin encoder attn & enc/dec cross-attn fields...
    # Encoder sequence lengths representation
    encoder_seq_lens: list[int] | None = None
    encoder_seq_lens_tensor: torch.Tensor | None = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: int | None = None

    # Number of tokens input to encoder
    num_encoder_tokens: int | None = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: torch.Tensor | None = None
    cross_block_tables: torch.Tensor | None = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: list[torch.Tensor] | None = None
        self.encoder_attn_bias: list[torch.Tensor] | None = None
        self.cross_attn_bias: list[torch.Tensor] | None = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        """
        All attention metadata required for encoder attention is set.
        """
        return (
            (self.encoder_seq_lens is not None)
            and (self.encoder_seq_lens_tensor is not None)
            and (self.max_encoder_seq_len is not None)
        )

    @property
    def is_all_cross_attn_metadata_set(self):
        """
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        """
        return (
            self.is_all_encoder_attn_metadata_set
            and (self.cross_slot_mapping is not None)
            and (self.cross_block_tables is not None)
        )

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
        """
        Extract appropriate sequence lengths from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate sequence lengths tensor for query
        * Appropriate sequence lengths tensor for key & value
        """

        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
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
    ) -> list[torch.Tensor] | None:
        """
        Extract appropriate attention bias from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate attention bias value given the attention type
        """

        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
            return self.attn_bias
        elif attn_type == AttentionType.ENCODER:
            return self.encoder_attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            return self.cross_attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def set_attn_bias(
        self,
        attn_bias: list[torch.Tensor],
        attn_type: str,
    ) -> None:
        """
        Update appropriate attention bias field of attention metadata,
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_bias: The desired attention bias value
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention
        """

        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
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
        """
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
        """

        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
            # Decoder self-attention
            # Choose max_seq_len based on whether we are in prompt_run
            return (
                self.decode_seq_lens_tensor,
                self.decode_max_seq_len,
                self.decode_block_tables,
            )
        elif attn_type == AttentionType.ENCODER_DECODER:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            return (
                self.encoder_seq_lens_tensor,
                self.max_encoder_seq_len,
                self.cross_block_tables,
            )
        elif attn_type == AttentionType.ENCODER:
            # No block tables associated with encoder attention
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len, None)
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")


class TorchSDPAMetadataBuilderV1(AttentionMetadataBuilder[TorchSDPAMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.scheduler_config = vllm_config.scheduler_config
        self._init_reorder_batch_threshold(1, False)

        self.seq_start_loc_cpu = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TorchSDPAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        max_query_len = common_attn_metadata.max_query_len

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        query_start_loc_np = query_start_loc_cpu.numpy()

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        max_prefill_seq_len = (
            seq_lens_np[num_decodes:num_reqs].max().item() if num_prefills > 0 else 0
        )
        max_decode_seq_len = (
            seq_lens_np[:num_decodes].max().item() if num_prefills < num_reqs else 0
        )
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=self.seq_start_loc_np[1 : num_reqs + 1])

        slot_mapping = common_attn_metadata.slot_mapping.long()
        block_table_tensor = common_attn_metadata.block_table_tensor
        query_start_loc_np = query_start_loc_cpu.numpy()
        query_start_loc_np[num_decodes : num_reqs + 1] -= num_decode_tokens

        attn_metadata = TorchSDPAMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            # to ensure inference when chunked_prefill is disabled
            seq_lens=seq_lens_cpu.tolist()[num_decodes:],  # prefill
            decode_seq_lens_tensor=seq_lens_cpu[:num_decodes],  # decode
            decode_max_seq_len=max_decode_seq_len,  # decode
            decode_block_tables=block_table_tensor[:num_decodes],  # decode
            chunked_prefill=self.scheduler_config.chunked_prefill_enabled,
            max_query_len=max_query_len,
            prefill_max_seq_len=max_prefill_seq_len,
            prefill_query_start_loc=query_start_loc_cpu[
                num_decodes : num_reqs + 1
            ],  # prefill
            prefill_seq_start_loc=self.seq_start_loc_cpu[
                num_decodes : num_reqs + 1
            ],  # prefill
            prefill_block_tables=block_table_tensor[num_decodes:num_reqs],  # prefill
            query_start_loc=query_start_loc_cpu[: num_reqs + 1],  # for logits index
        )

        return attn_metadata


class TorchSDPABackendImpl(AttentionImpl[TorchSDPAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if logits_soft_cap is not None:
            logger.warning_once(
                "Torch SPDA does not support logits soft cap. "
                "Outputs may be slightly off."
            )
        self.paged_attn_impl = _get_paged_attn_impl()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (
            self.alibi_slopes is not None or self.sliding_window is not None
        )

        if is_quantized_kv_cache(kv_cache_dtype) and not _use_ipex:
            raise NotImplementedError(
                "Torch SDPA backend FP8 KV cache requires "
                "intel_extension_for_pytorch support."
            )
        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchSDPAMetadata,  # type: ignore
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape =
                [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for TorchSDPABackendImpl"
            )

        # For warming-up
        if attn_metadata is None:
            return query

        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (
            not attn_metadata.is_all_encoder_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder attention requires setting encoder metadata attributes."
            )
        elif attn_type == AttentionType.ENCODER_DECODER and (
            not attn_metadata.is_all_cross_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder/decoder cross-attention "
                "requires setting cross-attention "
                "metadata attributes."
            )

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        if attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = self.paged_attn_impl.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                self.paged_attn_impl.write_to_paged_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    updated_slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

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
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                assert attn_metadata.seq_lens is not None
                self._run_sdpa_forward(
                    output, query, key, value, prefill_meta, attn_type=attn_type
                )
            else:
                # prefix-enabled attention
                assert not self.need_mask
                import intel_extension_for_pytorch.llm.modules as ipex_modules

                output = torch.empty_like(query)
                ipex_modules.PagedAttention.flash_attn_varlen_func(
                    output[prefill_meta.num_decode_tokens :, :, :],
                    query[prefill_meta.num_decode_tokens :, :, :],
                    key_cache,
                    value_cache,
                    prefill_meta.prefill_query_start_loc,
                    prefill_meta.prefill_seq_start_loc,
                    prefill_meta.max_query_len,
                    prefill_meta.prefill_max_seq_len,
                    self.scale,
                    True,
                    prefill_meta.prefill_block_tables,
                    self.alibi_slopes,
                )
        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata."
            )
            # Decoding run.
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)

            self.paged_attn_impl.forward_decode(
                output[: attn_metadata.num_decode_tokens, :, :],
                query[: attn_metadata.num_decode_tokens, :, :],
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
        attn_masks = attn_metadata.get_attn_bias(attn_type)
        if attn_masks is None:
            if self.alibi_slopes is not None:
                attn_masks = _make_alibi_bias(
                    self.alibi_slopes,
                    query.dtype,
                    attn_metadata.seq_lens,  # type: ignore
                )
            elif self.sliding_window is not None:
                assert attn_metadata.seq_lens is not None
                attn_masks = _make_sliding_window_bias(
                    attn_metadata.seq_lens, self.sliding_window, query.dtype
                )
            else:
                seq_lens, _ = attn_metadata.get_seq_lens(attn_type)
                attn_masks = [None] * len(seq_lens)
            attn_metadata.set_attn_bias(attn_masks, attn_type)

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=-3)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=-3)

        causal_attn = attn_type == AttentionType.DECODER

        seq_lens_q, seq_lens_kv = attn_metadata.get_seq_lens(attn_type)
        # Incoming Q and KV contain decoded tokens as well, hence start at an offset
        # equal to num_decode_tokens since decode requests appear first
        start_q, start_kv = (
            attn_metadata.num_decode_tokens,
            attn_metadata.num_decode_tokens,
        )
        for seq_len_q, seq_len_kv, mask in zip(seq_lens_q, seq_lens_kv, attn_masks):
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            sub_out = (
                scaled_dot_product_attention(
                    query[None, :, start_q:end_q, :],
                    key[None, :, start_kv:end_kv, :],
                    value[None, :, start_kv:end_kv, :],
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=causal_attn and mask is None,
                    scale=self.scale,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = sub_out
            start_q, start_kv = end_q, end_kv


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: list[int],
) -> list[torch.Tensor]:
    attn_biases: list[torch.Tensor] = []
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
        inf_mask = (
            torch.empty((1, seq_len, seq_len), dtype=bias.dtype)
            .fill_(-torch.inf)
            .triu_(diagonal=1)
        )
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: list[int],
    window_size: int | None,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    attn_biases: list[torch.Tensor] = []
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


class _PagedAttention:
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 80, 96, 112, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> tuple[int, ...]:
        return 2, num_blocks, block_size * num_kv_heads * head_size

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        *args,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def forward_decode(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: torch.Tensor | None,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        *args,
    ) -> None:
        tp_rank: int = 0
        blocksparse_local_blocks: int = 0
        blocksparse_vert_stride: int = 0
        blocksparse_block_size: int = 64
        blocksparse_head_sliding_step: int = 0
        block_size = value_cache.shape[3]

        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank,
            blocksparse_local_blocks,
            blocksparse_vert_stride,
            blocksparse_block_size,
            blocksparse_head_sliding_step,
        )


class _IPEXPagedAttention(_PagedAttention):
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        *args,
    ) -> None:
        ipex_modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping.flatten().int()
        )

    @staticmethod
    def forward_decode(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: torch.Tensor | None,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        *args,
    ) -> None:
        block_size = value_cache.shape[2]
        head_mapping = (
            torch.arange(
                0,
                num_kv_heads,
                device="cpu",
                dtype=torch.int32,
            )
            .view(num_kv_heads, 1)
            .repeat_interleave(query.size(1) // num_kv_heads)
            .flatten()
        )
        ipex_modules.PagedAttention.single_query_cached_kv_attention(
            output,
            query.contiguous(),
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )


def _get_paged_attn_impl():
    if _use_ipex:
        return _IPEXPagedAttention
    else:
        return _PagedAttention
