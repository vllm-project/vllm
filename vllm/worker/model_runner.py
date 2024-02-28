import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
                         BaseSchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.input_metadata import InputType
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    with_cupy_nccl_for_all_reduce)
from vllm.model_executor.kv_buffer import KVBuffer
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = None
        self.block_size = None  # Set after initial profiling.
        self.lora_manager = None

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()
        self.kv_cache_dtype = kv_cache_dtype

    def load_model(self) -> None:
        self.model = get_model(self.model_config, self.device_config,
                               self.lora_config)

        vocab_size = self.model.config.vocab_size

        if self.lora_config:
            assert hasattr(
                self.model, "supported_lora_modules"
            ) and self.model.supported_lora_modules, "Model does not support LoRA"
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens +
                self.scheduler_config.max_paddings, vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

        # Initialize the KV buffer
        # When chunking is involved, the buffer size is max_modeL_len + max_num_batched_tokens
        # i.e. [almost_finished_big_request] + [last_piece_big_request, small prefills, very_first_piece_big_request]
        # Otherwise (say vllm, orca), this buffer is not actually used in the codepath
        self.kv_buffers = [
            KVBuffer(
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_size=self.model_config.get_head_size(),
                dtype=self.model_config.dtype,
                device=self.device,
            ) for _ in range(
                self.model_config.get_num_layers(self.parallel_config))
        ]

    def reset_kv_buffers(self) -> None:
        for buffer in self.kv_buffers:
            buffer.reset()

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (self.max_context_len_to_capture + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_mixed_batch(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int],
               List[int], List[int], Set[LoRARequest]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        prefix_plus_current_prompt_tokens_slot_mapping: List[int] = []
        current_tokens_slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()
        processed_prompt_lens: List[int] = []
        current_prompt_chunk_lens: List[int] = []
        total_prompt_lens: List[int] = []
        prompt_seq_ids: List[int] = []
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        max_num_blocks_per_seq = 0
        max_context_len = 0
        is_profiling_iteration = False

        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_chunk_size = seq_group_metadata.prompt_chunk_size
            current_prompt_chunk_tokens = seq_data.get_next_prompt_chunk_token_ids(
                prompt_chunk_size)
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = seq_data.prompt_tokens_processed
            processed_prompt_lens.append(processed_prompt_len)
            total_prompt_len = seq_data.get_prompt_len()
            total_prompt_lens.append(total_prompt_len)
            prompt_seq_ids.append(seq_id)
            
            prefix = seq_group_metadata.prefix
            assert(prefix is None or not prefix.computed)
            
            input_tokens.extend(current_prompt_chunk_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(
                list(range(processed_prompt_len, processed_prompt_len + current_prompt_chunk_len)))

            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping.append([lora_id] * current_prompt_chunk_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (current_prompt_chunk_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            # ONLY used for profiling
            if seq_group_metadata.block_tables is None:
                is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                prefix_plus_current_prompt_tokens_slot_mapping.extend(
                    [_PAD_SLOT_ID] * (processed_prompt_len + current_prompt_chunk_len))
                current_tokens_slot_mapping.extend([_PAD_SLOT_ID] *
                                                   current_prompt_chunk_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            context_end = processed_prompt_len + current_prompt_chunk_len
            context_start = 0
            if self.sliding_window is not None:
                context_start = max(0, context_end - self.sliding_window)
            for i in range(context_end):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                if i < context_start:
                    prefix_plus_current_prompt_tokens_slot_mapping.append(_PAD_SLOT_ID)
                else:
                    prefix_plus_current_prompt_tokens_slot_mapping.append(slot)
                    if i >= processed_prompt_len:
                        current_tokens_slot_mapping.append(slot)

        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                max_context_len = max(max_context_len, context_len)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                current_tokens_slot_mapping.append([slot])
                lora_index_mapping.append([lora_id])
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)
        
        input_tokens = _make_tensor_with_pad_to_align(input_tokens,
                                             multiple_of=8,
                                             pad=0,
                                             dtype=torch.long,
                                             device=self.device)
        input_positions = _make_tensor_with_pad_to_align(input_positions,
                                                multiple_of=8,
                                                pad=0,
                                                dtype=torch.long,
                                                device=self.device)
        prefix_plus_current_prompt_tokens_slot_mapping = _make_tensor_with_pad_to_align(
            prefix_plus_current_prompt_tokens_slot_mapping,
            multiple_of=1,
            pad=_PAD_SLOT_ID,
            dtype=torch.long,
            device=self.device
        )
        current_tokens_slot_mapping = _make_tensor_with_pad_to_align(current_tokens_slot_mapping,
                                             multiple_of=8,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=self.device)
        # No change is done to `lora_index_mapping`
        context_lens = _make_tensor_with_pad_to_align(context_lens,
                                             multiple_of=1,
                                             pad=0,
                                             dtype=torch.long,
                                             device=self.device)
        # Prepare prefix block tables
        generation_block_tables = _make_tensor_with_pad_to_max(
            generation_block_tables,
            max_len=max_num_blocks_per_seq,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        input_metadata = InputMetadata(
            input_type=InputType.MIXED,
            prompt_seq_ids = prompt_seq_ids,
            processed_prompt_lens = processed_prompt_lens,
            current_prompt_chunk_lens = current_prompt_chunk_lens,
            total_prompt_lens = total_prompt_lens,
            prefix_plus_current_prompt_tokens_slot_mapping = prefix_plus_current_prompt_tokens_slot_mapping,
            current_tokens_slot_mapping = current_tokens_slot_mapping,
            max_context_len = max_context_len,
            context_lens = context_lens,
            block_tables = generation_block_tables,
            use_cuda_graph = False,
            kv_cache_dtype = self.kv_cache_dtype,
            is_profiling_iteration = is_profiling_iteration
        )
        return (input_tokens, input_positions, input_metadata,
                lora_index_mapping, lora_prompt_mapping, lora_requests)


    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        generators: List[torch.Generator] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        max_subquery_len = max(subquery_lens) if subquery_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                assert subquery_lens is not None
                subquery_len = subquery_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += subquery_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + subquery_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              subquery_len - 1)
                selected_token_start_idx += max_subquery_len

                if sampling_params.seed is not None:
                    seq_group_metadata.state.generator = torch.Generator(
                        device="cuda").manual_seed(sampling_params.seed)
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

            if sampling_params.seed is not None:
                generators.append(seq_group_metadata.state.generator)

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            target_device=self.device,
                                            pin_memory=not self.in_wsl)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids,
                          dtype=torch.int,
                          target_device=self.device,
                          pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            generators=generators,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata,
               Set[int], LoRAMapping]:
        if self.is_driver_worker:
            (input_tokens, input_positions, input_metadata,
            lora_index_mapping, lora_prompt_mapping, lora_requests) = self._prepare_mixed_batch(seq_group_metadata_list)
            # Prompt lens that are passed to prepare_sample should be consistent with
            # prompt chunk sizes in this iteration
            sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                     input_metadata.current_prompt_chunk_lens,
                                                     input_metadata.current_prompt_chunk_lens)

            if self.lora_config:
                flat_lora_index_mapping = [
                    item for sublist in lora_index_mapping for item in sublist
                ]
                lora_mapping = LoRAMapping(
                    flat_lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None

            # Broadcast the metadata.
            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "input_type": input_metadata.input_type,
                "prompt_seq_ids": input_metadata.prompt_seq_ids,
                "processed_prompt_lens": input_metadata.processed_prompt_lens,
                "current_prompt_chunk_lens": input_metadata.current_prompt_chunk_lens,
                "total_prompt_lens": input_metadata.total_prompt_lens,
                "prefix_plus_current_prompt_tokens_slot_mapping": input_metadata.prefix_plus_current_prompt_tokens_slot_mapping,
                "current_tokens_slot_mapping": input_metadata.current_tokens_slot_mapping,
                "max_context_len": input_metadata.max_context_len,
                "context_lens": input_metadata.context_lens,
                "block_tables": input_metadata.block_tables,
                "use_cuda_graph": input_metadata.use_cuda_graph,
                "kv_cache_dtype": input_metadata.kv_cache_dtype,
                "is_profiling_iteration": input_metadata.is_profiling_iteration,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
            }
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict["input_tokens"]
            input_positions = metadata_dict["input_positions"]
            lora_mapping = metadata_dict["lora_mapping"]
            lora_requests = metadata_dict["lora_requests"]
            input_metadata = InputMetadata(
                input_type=metadata_dict["input_type"],
                prompt_seq_ids=metadata_dict["prompt_seq_ids"],
                processed_prompt_lens=metadata_dict["processed_prompt_lens"],
                current_prompt_chunk_lens=metadata_dict["current_prompt_chunk_lens"],
                total_prompt_lens=metadata_dict["total_prompt_lens"],
                prefix_plus_current_prompt_tokens_slot_mapping=metadata_dict["prefix_plus_current_prompt_tokens_slot_mapping"],
                current_tokens_slot_mapping=metadata_dict["current_tokens_slot_mapping"],
                max_context_len=metadata_dict["max_context_len"],
                context_lens=metadata_dict["context_lens"],
                block_tables=metadata_dict["block_tables"],
                use_cuda_graph=metadata_dict["use_cuda_graph"],
                kv_cache_dtype=metadata_dict["kv_cache_dtype"],
                is_profiling_iteration=metadata_dict["is_profiling_iteration"],
            )
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=metadata_dict["selected_token_indices"],
                categorized_sample_indices=None,
                generators=None,
                perform_sampling=False,
            )

        return (input_tokens, input_positions, input_metadata,
                sampling_metadata, lora_requests, lora_mapping)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, input_metadata, sampling_metadata,
         lora_requests,
         lora_mapping) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Execute the model.
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            kv_buffers=self.kv_buffers,
            input_metadata=input_metadata,
        )

        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            for idx in range(self.lora_config.max_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_local_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)]
                for idx in range(max_num_seqs)
            ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        if self.scheduler_config.type_name in ["sarathi", "dsarathi"]:
            # Profile memory usage with a single `chunk_size` chunk
            # which is the last chunk in the longest supported sequence
            chunk_size = self.scheduler_config.chunk_size
            seq_len = self.model_config.max_model_len
            seq_data = SequenceData(prompt_token_ids=[0] * seq_len,
                                    prompt_tokens_processed=max(
                                        seq_len - chunk_size, 0))
            seq = SequenceGroupMetadata(
                request_id="0",
                is_prompt=True,
                seq_data={0: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                prompt_chunk_size=chunk_size,
            )
            seqs.append(seq)
            # We need to consider only dangling prefills (prefills which dont complete in one iteration)
            # as they use KV buffers
            # max_running_prefills = 1 + self.parallel_config.pipeline_parallel_size + self.scheduler_config.max_pre_queue_batches
            max_running_prefills = 1 + self.scheduler_config.max_pre_queue_batches

            for i in range(1, max_running_prefills + 1):
                for kv_buffer in self.kv_buffers:
                    kv_buffer.add_request(-i, seq_len)

        else:
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                seq_data = SequenceData([0] * seq_len)
                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    prompt_chunk_size=seq_len,
                )
                seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_all_loras()

    def set_active_loras(self, lora_requests: List[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_loras()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        # NOTE(woosuk): This is a hack to ensure that the NCCL backend is never
        # deleted before the CUDA graphs.
        self.cupy_nccl_backend = cupy_utils.get_nccl_backend()

        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, 1,
                                      dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
        # kernel, CuPy NCCL, and PyTorch NCCL. When using CUDA graph, we use
        # either custom all-reduce kernel or CuPy NCCL. When not using CUDA
        # graph, we use either custom all-reduce kernel or PyTorch NCCL.
        # We always prioritize using custom all-reduce kernel but fall back
        # to PyTorch or CuPy NCCL if it is disabled or not supported.
        with custom_all_reduce.capture():
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                # Create dummy input_metadata.
                input_metadata = InputMetadata(
                    is_prompt=False,
                    slot_mapping=slot_mapping[:batch_size],
                    prompt_lens=None,
                    max_seq_len=None,
                    start_loc=None,
                    max_context_len=self.max_context_len_to_capture,
                    context_lens=context_lens[:batch_size],
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                    kv_cache_dtype=self.kv_cache_dtype,
                )

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size,
                        [0] * batch_size,
                    )
                    self.set_active_loras(set(), lora_mapping)

                graph_runner = CUDAGraphRunner(self.model)
                graph_runner.capture(
                    input_tokens[:batch_size],
                    input_positions[:batch_size],
                    kv_caches,
                    input_metadata,
                    memory_pool=self.graph_memory_pool,
                )
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the CuPy NCCL communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        self.graph_runners.clear()
        self.cupy_nccl_backend = None


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        with _maybe_cupy_nccl():
            self.model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        # NOTE(woosuk): Python 3.8 does not support multi-line with statements.
        # https://stackoverflow.com/questions/31039022/python-multi-line-with-statement
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):  # noqa: SIM117
            with _maybe_cupy_nccl():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    kv_caches,
                    input_metadata,
                )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(input_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(input_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(input_metadata.block_tables,
                                                 non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@contextlib.contextmanager
def _maybe_cupy_nccl():
    if cupy_utils.is_initialized() and not custom_all_reduce.is_initialized():
        with with_cupy_nccl_for_all_reduce():
            yield
    else:
        yield


def _pad_to_align(x: List[int], multiple_of: int, pad: int) -> List[int]:
    return x + [pad] * ((-len(x)) % multiple_of)

def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return x + [pad] * (max_len - len(x))

def _make_tensor_with_pad_to_align(
    x: List[int],
    multiple_of: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> List[int]:
    padded_x = _pad_to_align(x, multiple_of, pad)
    return torch.tensor(padded_x, dtype=dtype, device=device)

def _make_tensor_with_pad_to_max(
    x: List[int],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> List[int]:
    padded_x = _pad_to_max(x, max_len, pad)
    return torch.tensor(padded_x, dtype=dtype, device=device)

def _get_graph_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8


def _async_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)
