import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.sequence import (SamplerOutput,
                           SequenceGroupMetadata)
from vllm.utils import (get_kv_cache_torch_dtype, make_tensor_with_pad)

from vllm.worker.model_runner import ModelRunner, PreparePromptMetadata, PrepareDecodeMetadata, BatchType, _PAD_SLOT_ID, _BATCH_SIZES_TO_CAPTURE, _get_graph_batch_size, _is_block_tables_empty

class EncoderDecoderModelRunner(ModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        # Pass all arguments to superclass constructor
        #kwargs = locals()
        #kwargs.pop('self')
        # Call the superclass constructor with all collected arguments
        super().__init__(model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        cache_config=cache_config,
        load_config=load_config,
        lora_config=lora_config,
        kv_cache_dtype=kv_cache_dtype,
        is_driver_worker=is_driver_worker,
        vision_language_config=vision_language_config,)

        # Assert that this is an encoder/decoder model
        assert (self.model_config is not None) and (getattr(self.model_config.hf_config, "is_encoder_decoder", False))

    def _prepare_cross_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> PreparePromptMetadata:
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

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.encoder_seq_data
            context_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            seq_len = min(seq_data.get_len(), context_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[context_len:seq_len]
            seq_lens.append(seq_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                context_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[context_len:]
                prefix_block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                if seq_group_metadata.block_tables is not None:
                    # Prefill has chunked before.
                    block_table = seq_group_metadata.encoder_block_table
                    prefix_block_tables.append(block_table)
                else:
                    # The first prefill.
                    prefix_block_tables.append([])
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert context_len == 0

            # actual prompt lens
            context_lens.append(context_len)
            query_lens.append(seq_len - context_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(context_len, seq_len)))
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (seq_len - context_len)
            lora_prompt_mapping.extend([lora_id] * (
                seq_len - context_len if seq_group_metadata.sampling_params
                and seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if _is_block_tables_empty(seq_group_metadata.block_tables):
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.encoder_block_table

            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert context_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(context_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_query_len = max(query_lens)
        max_seq_len = max(seq_lens)
        assert max_query_len > 0

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
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        subquery_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        if self.attn_backend.get_name() == "flashinfer":
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=True,
                use_cuda_graph=False,
                seq_start_loc=seq_start_loc,
                max_seq_len=max_seq_len,
                block_tables=block_tables)
        else:
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=True,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                subquery_start_loc=subquery_start_loc,
                seq_start_loc=seq_start_loc,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=False,
            )

        return PreparePromptMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping=slot_mapping,
        )

    def _prepare_cross_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> PrepareDecodeMetadata:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        # The following fields are only for flashinfer
        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        paged_kv_last_page_len: List[int] = []

        if len(seq_group_metadata_list) == 0:
            return PrepareDecodeMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            seq_data = seq_group_metadata.encoder_seq_data
            generation_token = seq_data.get_last_token_id()
            input_tokens.append(generation_token)

            seq_len = seq_data.get_len()
            position = seq_len - 1
            input_positions.append(position)

            seq_len = seq_len if self.sliding_window is None else min(
                seq_len, self.sliding_window)
            seq_lens.append(seq_len)

            block_table = seq_group_metadata.encoder_block_table
            block_number = block_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
            lora_index_mapping.append(lora_id)
            lora_prompt_mapping.append(lora_id)

            if self.sliding_window is not None:
                sliding_window_blocks = (self.sliding_window //
                                            self.block_size)
                block_table = block_table[-sliding_window_blocks:]
            block_tables.append(block_table)

            paged_kv_indices.extend(block_table)
            paged_kv_indptr.append(paged_kv_indptr[-1] + len(block_table))
            last_page_len = seq_data.get_len() % self.block_size
            if last_page_len == 0:
                last_page_len = self.block_size
            paged_kv_last_page_len.append(last_page_len)

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        # For decoding requests, batch_size == input_tokens.
        batch_size = len(input_tokens)
        max_seq_len = max(seq_lens)
        use_captured_graph = (not self.model_config.enforce_eager
                              and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
                              and max_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                seq_lens.append(1)
                block_tables.append([])
                lora_index_mapping.append(0)
            batch_size = graph_batch_size

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            assert seq_lens_tensor.shape[0] == len(input_tokens)
            assert seq_lens_tensor.shape[0] == len(input_positions)
            assert seq_lens_tensor.shape[0] == len(slot_mapping)

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )

        if self.attn_backend.get_name() == "flashinfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            paged_kv_indptr = torch.tensor(paged_kv_indptr,
                                           dtype=torch.int,
                                           device=self.device)
            paged_kv_indices = torch.tensor(paged_kv_indices,
                                            dtype=torch.int,
                                            device=self.device)
            paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len,
                                                  dtype=torch.int,
                                                  device=self.device)
            kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                      self.model_config.dtype)

            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=False,
                use_cuda_graph=False,
                workspace_buffer=self.flashinfer_workspace_buffer,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config),
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_dim=self.model_config.get_head_size(),
                page_size=self.block_size,
                data_type=kv_cache_dtype)
        else:
            attn_metadata = self.attn_backend.make_metadata(
                is_prompt=False,
                seq_lens=None,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=None,
                max_seq_len=max_seq_len,
                subquery_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=block_tables,
                use_cuda_graph=use_captured_graph,
            )
        return PrepareDecodeMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
            lora_index_mapping=lora_index_mapping,
            lora_prompt_mapping=lora_prompt_mapping,
            lora_requests=lora_requests,
            slot_mapping=slot_mapping,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, torch.Tensor]:
        if self.is_driver_worker:
            prefill_reqs = []
            decode_reqs = []
            for seq_group_meta in seq_group_metadata_list:
                if seq_group_meta.is_prompt:
                    prefill_reqs.append(seq_group_meta)
                else:
                    decode_reqs.append(seq_group_meta)

            # Prepare self-attention input tensors.
            (
                input_tokens,
                input_positions,
                prefill_attn_metadata,
                seq_lens,
                query_lens,
                lora_index_mapping,
                lora_prompt_mapping,
                lora_requests,
                multi_modal_input,
                slot_mapping,
            ) = self._prepare_prompt(prefill_reqs)
            (
                decode_input_tokens,
                decode_input_positions,
                decode_attn_metadata,
                decode_lora_index_mapping,
                decode_lora_prompt_mapping,
                decode_lora_requests,
                decode_slot_mapping,
            ) = self._prepare_decode(decode_reqs)

            # Prepare cross-attention input tensor
            (
                cross_input_tokens,
                cross_input_positions,
                cross_prefill_attn_metadata,
                cross_seq_lens,
                cross_query_lens,
                cross_lora_index_mapping,
                cross_lora_prompt_mapping,
                cross_lora_requests,
                cross_multi_modal_input,
                cross_slot_mapping,
            ) = self._prepare_cross_prompt(prefill_reqs)
            (
                cross_decode_input_tokens,
                cross_decode_input_positions,
                cross_decode_attn_metadata,
                cross_decode_lora_index_mapping,
                cross_decode_lora_prompt_mapping,
                cross_decode_lora_requests,
                cross_decode_slot_mapping,
            ) = self._prepare_cross_decode(decode_reqs)

            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            if not self.scheduler_config.chunked_prefill_enabled:
                assert (len(prefill_reqs) and len(decode_reqs)) == 0

            num_prefills = len(seq_lens)
            num_prefill_tokens = len(input_tokens)
            num_decode_tokens = len(decode_input_tokens)

            # Coalesce tensors. Note that attn_metadata is currently not
            # coalesced for simplicity.
            input_tokens.extend(decode_input_tokens)
            input_positions.extend(decode_input_positions)
            slot_mapping.extend(decode_slot_mapping)
            lora_index_mapping.extend(decode_lora_index_mapping)
            lora_prompt_mapping.extend(decode_lora_prompt_mapping)
            lora_requests.update(decode_lora_requests)

            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device=self.device)
            input_positions = torch.tensor(input_positions,
                                           dtype=torch.long,
                                           device=self.device)
            slot_mapping = torch.tensor(slot_mapping,
                                        dtype=torch.long,
                                        device=self.device)

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None

            # Broadcast the metadata.
            # If batch contains both prefill and decode, it sends 2 broadcasts.
            # If it only contains 1 type, it triggers a single broadcast.
            if (prefill_attn_metadata is not None
                    and decode_attn_metadata is not None):
                batch_type = BatchType.MIXED
            elif prefill_attn_metadata is not None:
                batch_type = BatchType.PREFILL
            else:
                batch_type = BatchType.DECODE

            cross_metadata_dict = {
                "cross_input_tokens": cross_input_tokens,
                "cross_input_positions": cross_input_positions,
                "cross_prefill_attn_metadata": cross_prefill_attn_metadata,
                "cross_seq_lens": cross_seq_lens,
                "cross_query_lens": cross_query_lens,
                "cross_multi_modal_input": cross_multi_modal_input,
                "cross_slot_mapping": cross_slot_mapping,
                "cross_decode_input_tokens": cross_decode_input_tokens,
                "cross_decode_input_positions": cross_decode_input_positions,
                "cross_decode_attn_metadata": cross_decode_attn_metadata,
                "cross_decode_slot_mapping": cross_decode_slot_mapping,
            }

            if cross_prefill_attn_metadata is not None:
                cross_metadata_dict.update(cross_prefill_attn_metadata.asdict_zerocopy())
            else:
                assert cross_decode_attn_metadata is not None
                cross_metadata_dict.update(cross_decode_attn_metadata.asdict_zerocopy())

            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_input": multi_modal_input,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
                "batch_type": batch_type,
                "cross_metadata_dict": cross_metadata_dict,
            }
            
            if prefill_attn_metadata is not None:
                metadata_dict.update(prefill_attn_metadata.asdict_zerocopy())
            else:
                assert decode_attn_metadata is not None
                metadata_dict.update(decode_attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)

            # Broadcast decode attn metadata for mixed batch type.
            # The additional broadcast costs 300us overhead on 4 A10 GPUs.
            # We can potentially reduce the overhead by coelescing tensors.
            if batch_type == BatchType.MIXED:
                assert decode_attn_metadata is not None
                metadata_dict = decode_attn_metadata.asdict_zerocopy()
                broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            slot_mapping = metadata_dict.pop("slot_mapping")
            num_prefills = metadata_dict.pop("num_prefills")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            num_prefill_tokens = metadata_dict.pop("num_prefill_tokens")
            num_decode_tokens = metadata_dict.pop("num_decode_tokens")
            batch_type = metadata_dict.pop("batch_type")

            cross_metadata_dict = metadata_dict.pop("cross_metadata_dict")
            cross_input_tokens = cross_metadata_dict.pop("cross_input_tokens")
            cross_input_positions = cross_metadata_dict.pop("cross_input_positions")
            #cross_prefill_attn_metadata = metadata_dict.pop("cross_prefill_attn_metadata")
            cross_seq_lens = cross_metadata_dict.pop("cross_seq_lens")
            cross_query_lens = cross_metadata_dict.pop("cross_query_lens")
            cross_multi_modal_input = cross_metadata_dict.pop("cross_multi_modal_input")
            cross_slot_mapping = cross_metadata_dict.pop("cross_slot_mapping")
            cross_decode_input_tokens = cross_metadata_dict.pop("cross_decode_input_tokens")
            cross_decode_input_positions = cross_metadata_dict.pop("cross_decode_input_positions")
            #cross_decode_attn_metadata = metadata_dict.pop("cross_decode_attn_metadata")
            cross_decode_slot_mapping = cross_metadata_dict.pop("cross_decode_slot_mapping")               

            # Create an attention metadata.
            prefill_attn_metadata = None
            decode_attn_metadata = None
            if batch_type == BatchType.PREFILL or batch_type == BatchType.MIXED:
                prefill_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

            # if it is a mixed batch, decode attn_metadata is broadcasted
            # separately.
            if batch_type == BatchType.MIXED:
                metadata_dict = broadcast_tensor_dict(src=0)
                decode_attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_attn_metadata,
            decode_metadata=decode_attn_metadata,
        )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output