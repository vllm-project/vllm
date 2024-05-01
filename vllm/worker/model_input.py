import torch
from dataclasses import dataclass
from typing import List, Set, Optional, Type

from vllm.vllm.sequence import SequenceGroupMetadata
from vllm.attention.backends.abstract import AttentionMetadata, AttentionBackend
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import SchedulerConfig, LoRAConfig, VisionLanguageConfig
from vllm.utils import make_tensor_with_pad


_PAD_SLOT_ID = -1


@dataclass
class GpuModelInput:
    """Input to run a model.

    Input tensors include inputs across multiple sequence groups.
    It assumes inputs are ordered by prefill -> decode sequences.
    """
    # (num_tokens,) 1D Flattened input token IDs.
    input_tokens: torch.Tensor
    # (num_tokens,) Positions of a token in its sequence. Used for RoPE.
    input_positions: torch.Tensor
    # Attention metadata to run attention kernels.
    attn_metadata: AttentionMetadata
    # (batch_size,) A sequence length for each sequence group in a batch.
    seq_lens: List[int]
    # (batch_size,) A query length for eaach sequence group in a batch.
    query_lens: List[int]
    # Set of lora requests.
    lora_requests: Set[LoRARequest]
    # Inputs used for multi modality.
    multi_modal_input: Optional[torch.Tensor]
    # (num_tokens,) A page index per token. Each slot index is flattened. For
    # example, if slot mapping is 15 and block size is 8, it means block index
    # 1 and offset 3.
    slot_mapping: torch.Tensor
    # Lora mapping. None if lora is not used.
    lora_mapping: Optional[LoRAMapping]

    @classmethod
    def from_sequence_groups(
            cls,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduler_config: SchedulerConfig,
            lora_config: Optional[LoRAConfig],
            vision_language_config: Optional[VisionLanguageConfig],
            block_size: int,
            device: str,
            attn_backend: Type[AttentionBackend],
            sliding_window: Optional[int]) -> "GpuModelInput":
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        multi_modal_input_list: List[torch.Tensor] = []

        is_prompt = False
        for seq_group_metadata in seq_group_metadata_list:
            # assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            # assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            is_prompt = seq_group_metadata.is_prompt

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (scheduler_config is not None
                    and scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            computed_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            prefill_end = min(seq_data.get_len(),
                              computed_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[computed_len:prefill_end]
            seqlen = prefill_end
            seq_lens.append(seqlen)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and sliding_window is None:
                # Prefix is not supported with sliding_window
                computed_len = len(computed_block_nums) * block_size
                prompt_tokens = prompt_tokens[computed_len:]
                prefix_block_tables.append(computed_block_nums)
            elif scheduler_config.chunked_prefill_enabled or not is_prompt:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.block_tables[seq_id]
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert computed_len == 0

            # actual prompt lens
            context_lens.append(computed_len)
            query_lens.append(seqlen - computed_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prefill_end)))
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (seqlen - computed_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (seqlen - computed_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seqlen)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seqlen - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if sliding_window is not None:
                assert computed_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, seqlen - sliding_window)

            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)

        max_query_len = max(query_lens)
        max_seqlen = max(seq_lens)
        assert max_query_len > 0

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=device)

        if multi_modal_input_list:
            assert vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(device)
        else:
            multi_modal_input = None

        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=device,
        )

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        query_lens_tensor = torch.tensor(query_lens,
                                            dtype=torch.long,
                                            device=device)
        subquery_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                          dtype=torch.int,
                                          device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = attn_backend.make_metadata(
            is_prompt=is_prompt,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_context_len=max(context_lens),
            max_seqlen=max_seqlen,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
        )

        # Decode
        # attn_metadata = self.attn_backend.make_metadata(
        #     is_prompt=False,
        #     seq_lens=None,
        #     seq_lens_tensor=None,
        #     max_query_len=None,
        #     max_context_len=max_context_len,
        #     max_seqlen=None,
        #     subquery_start_loc=None,
        #     seq_start_loc=None,
        #     context_lens=context_lens_tensor,
        #     block_tables=block_tables,
        #     use_cuda_graph=use_captured_graph,
        # )

        input_tokens_tensor = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=device)
        input_positions_tensor = torch.tensor(input_positions,
                                        dtype=torch.long,
                                        device=device)
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=device)

        if lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=attn_metadata,
            decode_metadata=attn_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return ModelInput(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping=slot_mapping_tensor,
            lora_mapping=lora_mapping,
        )
