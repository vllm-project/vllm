# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu_input_batch import InputBatch

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

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        attn_impl = _get_paged_attn_impl()
        is_valid, supported_head_sizes = attn_impl.validate_head_size(
            head_size)
        if not is_valid:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TorchSDPAMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["TorchSDPAMetadataBuilderV1"]:
        return TorchSDPAMetadataBuilderV1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return _get_paged_attn_impl().get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TorchSDPAMetadata(AttentionMetadata):
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]
    """Metadata for TorchSDPABackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    chunked_prefill: bool
    seq_lens: Optional[list[int]] = None  # For non-chunked prefill

    # For chunked prefill only
    max_query_len: Optional[int] = None
    max_kv_len: Optional[int] = None
    prefill_query_start_loc: Optional[torch.Tensor] = None
    kv_start_loc: Optional[torch.Tensor] = None
    prefill_block_tables: Optional[torch.Tensor] = None

    # For V1 logits index only
    query_start_loc: Optional[torch.Tensor] = None

    # Begin encoder attn & enc/dec cross-attn fields...
    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[list[int]] = None
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
        self.attn_bias: Optional[list[torch.Tensor]] = None
        self.encoder_attn_bias: Optional[list[torch.Tensor]] = None
        self.cross_attn_bias: Optional[list[torch.Tensor]] = None

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
    ) -> Optional[list[torch.Tensor]]:
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
        attn_bias: list[torch.Tensor],
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


class TorchSDPAMetadataBuilderV1(AttentionMetadataBuilder[TorchSDPAMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config

        # For reorder
        self.reorder_prompt_req_index_list = np.empty(
            vllm_config.scheduler_config.max_num_seqs, dtype=np.int64)
        self.reorder_decode_req_index_list = np.empty(
            vllm_config.scheduler_config.max_num_seqs, dtype=np.int64)
        self.num_prompt_req: int = 0

        self.seq_start_loc_cpu = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        prompt_list_idx = 0
        decode_list_idx = 0
        for req_index in range(input_batch.num_reqs):
            if input_batch.num_computed_tokens_cpu[
                    req_index] < input_batch.num_prompt_tokens[req_index]:
                # prompt stage
                self.reorder_prompt_req_index_list[prompt_list_idx] = req_index
                prompt_list_idx += 1
            else:
                # decode stage
                self.reorder_decode_req_index_list[decode_list_idx] = req_index
                decode_list_idx += 1
        assert decode_list_idx + prompt_list_idx == input_batch.num_reqs

        # Update prompt requests number
        self.num_prompt_req = prompt_list_idx

        reorder_req_num = 0
        for req_index in range(decode_list_idx):
            if self.reorder_decode_req_index_list[req_index] < prompt_list_idx:
                reorder_req_num += 1
            else:
                break

        if reorder_req_num == 0:
            return False

        reorder_prompt_list = (
            self.reorder_prompt_req_index_list[:prompt_list_idx]
            [-reorder_req_num:])
        reorder_decode_list = (
            self.reorder_decode_req_index_list[:decode_list_idx]
            [:reorder_req_num])
        assert reorder_decode_list.size == reorder_prompt_list.size

        for idx in range(reorder_req_num):
            prompt_req_index = reorder_prompt_list[idx].item()
            decode_req_index = reorder_decode_list[idx].item()
            input_batch.swap_states(prompt_req_index, decode_req_index)

        return True

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> TorchSDPAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        max_query_len = common_attn_metadata.max_query_len

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()
        num_prompt_req = self.num_prompt_req
        max_prefill_seq_len = seq_lens_np[:num_prompt_req].max().item(
        ) if num_prompt_req > 0 else 0
        max_decode_seq_len = seq_lens_np[num_prompt_req:num_reqs].max().item(
        ) if num_prompt_req < num_reqs else 0
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=self.seq_start_loc_np[1:num_reqs + 1])

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_prefill_tokens = int(query_start_loc_cpu[num_prompt_req].item())
        num_decode_tokens = int(query_start_loc_cpu[num_reqs].item() -
                                num_prefill_tokens)

        slot_mapping = common_attn_metadata.slot_mapping.long()
        block_table_tensor = common_attn_metadata.block_table_tensor

        attn_metadata = TorchSDPAMetadata(
            num_prefills=num_prompt_req,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            # to ensure inference when chunked_prefill is disabled
            seq_lens=seq_lens_cpu.tolist(),
            seq_lens_tensor=seq_lens_cpu[num_prompt_req:num_reqs],  # decode
            max_decode_seq_len=max_decode_seq_len,  # decode
            block_tables=block_table_tensor[num_prompt_req:num_reqs],  # decode
            chunked_prefill=self.scheduler_config.chunked_prefill_enabled,
            max_query_len=max_query_len,
            max_kv_len=max_prefill_seq_len,
            prefill_query_start_loc=query_start_loc_cpu[:num_prompt_req +
                                                        1],  # prefill
            kv_start_loc=self.seq_start_loc_cpu[:num_prompt_req +
                                                1],  # prefill
            prefill_block_tables=block_table_tensor[:
                                                    num_prompt_req],  # prefill
            query_start_loc=query_start_loc_cpu[:num_reqs +
                                                1],  # for logits index
            multi_modal_placeholder_index_maps=None,
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
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if logits_soft_cap is not None:
            logger.warning_once("Torch SPDA does not support logits soft cap. "
                                "Outputs may be slightly off.")
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
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

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
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
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
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for TorchSDPABackendImpl")

        # For warming-up
        if attn_metadata is None:
            return query

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
            key_cache, value_cache = self.paged_attn_impl.split_kv_cache(
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

                self.paged_attn_impl.write_to_paged_cache(
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
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                assert attn_metadata.seq_lens is not None
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
                    prefill_meta.prefill_query_start_loc,
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

            self.paged_attn_impl.forward_decode(
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
        inf_mask = torch.empty(
            (1, seq_len, seq_len),
            dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: list[int],
    window_size: Optional[int],
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
    def validate_head_size(head_size: int) -> tuple[bool, list[int]]:
        SUPPORT_HS = [32, 64, 80, 96, 112, 128, 192, 256]
        return head_size in SUPPORT_HS, SUPPORT_HS

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
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
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
        alibi_slopes: Optional[torch.Tensor],
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

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
        *args,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


class _IPEXPagedAttention(_PagedAttention):

    @staticmethod
    def validate_head_size(head_size: int) -> tuple[bool, list[int]]:
        return True, []

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
            key, value, key_cache, value_cache,
            slot_mapping.flatten().int())

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
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        *args,
    ) -> None:
        block_size = value_cache.shape[2]
        head_mapping = torch.arange(
            0,
            num_kv_heads,
            device="cpu",
            dtype=torch.int32,
        ).view(num_kv_heads,
               1).repeat_interleave(query.size(1) // num_kv_heads).flatten()
        ipex_modules.PagedAttention.single_query_cached_kv_attention(
            output, query.contiguous(), key_cache, value_cache, head_mapping,
            scale, block_tables, context_lens, block_size, max_context_len,
            alibi_slopes)


def _get_paged_attn_impl():
    if _use_ipex:
        return _IPEXPagedAttention
    else:
        return _PagedAttention
