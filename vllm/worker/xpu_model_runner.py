from typing import List, Set

import torch

from vllm.lora.request import LoRARequest
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner import ModelRunner, PreparePromptMetadata

_PAD_SLOT_ID = -1


class XPUModelRunner(ModelRunner):

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PreparePromptMetadata:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        subquery_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        multi_modal_input_list: List[torch.Tensor] = []

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and computed_block_nums is not None):
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
            # TODO(sang): Rename it after chunked prefill is introduced.
            prompt_tokens = seq_data.get_token_ids()[computed_len:prefill_end]
            prompt_len = len(prompt_tokens)
            # Right now, the prefill_end is always same as the length of
            # sequence. However, once chunked prefill is introduced, this
            # assumption can be changed.
            assert prefill_end == seq_data.get_len()
            prompt_lens.append(prompt_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                computed_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[computed_len:]
                prefix_block_tables.append(computed_block_nums)
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert computed_len == 0

            # actual prompt lens
            context_lens.append(computed_len)
            subquery_lens.append(prompt_len - computed_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prefill_end)))

            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (prompt_len - computed_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (prompt_len - computed_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert computed_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, prompt_len - self.sliding_window)

            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # padding to 8 bytes for XeTLA
        padding_num = (8 - len(input_tokens) % 8) % 8
        input_tokens.extend([0] * padding_num)
        input_positions.extend([0] * padding_num)
        slot_mapping.extend([_PAD_SLOT_ID] * padding_num)
        prompt_lens[-1] += padding_num

        max_subquery_len = max(subquery_lens)
        max_prompt_len = max(prompt_lens)
        assert max_subquery_len > 0

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        subquery_lens_tensor = torch.tensor(subquery_lens,
                                            dtype=torch.long,
                                            device=self.device)
        subquery_start_loc = torch.zeros(subquery_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)

        prompt_lens_tensor = torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device=self.device)
        seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(subquery_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(prompt_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            # slot_mapping=slot_mapping,
            prompt_lens=prompt_lens,
            prompt_lens_tensor=prompt_lens_tensor,
            # num_prompt_tokens=num_prompt_tokens,
            # num_generation_tokens=0,
            max_subquery_len=max_subquery_len,
            max_context_len=None,
            max_prompt_len=max_prompt_len,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # kv_cache_dtype=self.kv_cache_dtype,
        )
        return PreparePromptMetadata(input_tokens,
                                     input_positions,
                                     attn_metadata,
                                     prompt_lens,
                                     subquery_lens,
                                     lora_index_mapping,
                                     lora_prompt_mapping,
                                     lora_requests,
                                     multi_modal_input,
                                     slot_mapping=slot_mapping)
