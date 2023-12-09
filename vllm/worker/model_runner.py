import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_object_list, broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    with_cupy_nccl_for_all_reduce)
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (SamplerOutput, SequenceData, SequenceGroupMetadata,
                           SpeculateOutput, SpeculateSequenceGroupOutput)
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
from vllm.model_executor.parallel_utils.parallel_state import (MarkActiveModel,
                                                               ActiveModel)

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
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        rank: int = 0,
        draft_model_config: Optional[ModelConfig] = None,
        speculate_length: Optional[int] = None,
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

        # Variables used in speculative decoding
        self.rank = rank
        self.draft_model = None
        self.draft_model_config = draft_model_config
        self.use_speculate = draft_model_config is not None
        self.speculate_length = speculate_length
        self.d_graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.d_graph_memory_pool = None  # Set during graph capture

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

        # Load draft model when enabling speculative decoding
        if self.use_speculate and self.rank < self.parallel_config.draft_model_tp_size:
            with MarkActiveModel(ActiveModel.DRAFT):
                # NOTE: We use global variable to control parallel state (world size, rank)
                # of draft model used in speculative decoding.
                self.draft_model = get_model(self.draft_model_config,
                                             self.device_config)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (self.max_context_len_to_capture + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_draft_model: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int],
               List[int], List[int], Set[LoRARequest]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        subquery_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)
            prefix_len = 0
            prefix = seq_group_metadata.prefix
            if prefix is not None and prefix.computed:
                prefix_len = prefix.get_length()
                prompt_tokens = prompt_tokens[prefix_len:]
                prefix_block_tables.append(prefix.get_block_numbers())
            else:
                prefix_block_tables.append([])
            # actual prompt lens
            context_lens.append(prefix_len)
            subquery_lens.append(prompt_len - prefix_len)

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(
                list(range(prefix_len, prefix_len + len(prompt_tokens))))

            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping.append([lora_id] * (prompt_len - prefix_len))
            lora_prompt_mapping.extend(
                [lora_id] *
                (prompt_len - prefix_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.d_block_tables[seq_id] if is_draft_model \
                else seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert prefix_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prefix_len, prompt_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        max_prompt_len = max(subquery_lens)
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_prompt_len,
                                             pad=0,
                                             dtype=torch.long,
                                             device=self.device)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_prompt_len,
                                                pad=0,
                                                dtype=torch.long,
                                                device=self.device)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_prompt_len,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=self.device)
        lora_index_mapping = [
            _pad_to_max(mapping, max_prompt_len, pad=0)
            for mapping in lora_index_mapping
        ]
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)
        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = _make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )
        start_loc_tensor = torch.arange(0,
                                        len(prompt_lens) * max_prompt_len,
                                        max_prompt_len,
                                        dtype=torch.long,
                                        device=self.device)
        prompt_lens_tensor = torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device=self.device)

        input_metadata = InputMetadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            prompt_lens=prompt_lens_tensor,
            max_seq_len=max_prompt_len,
            start_loc=start_loc_tensor,
            max_context_len=None,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        return (input_tokens, input_positions, input_metadata, prompt_lens,
                subquery_lens, lora_index_mapping, lora_prompt_mapping,
                lora_requests)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_draft_model: bool = False,
        is_multi_query_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int],
               Set[LoRARequest]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]

                if is_multi_query_mode:
                    assert not is_draft_model, \
                        "Only target model is allowed to run multi-query mode"
                    num_generation_tokens = seq_data.get_num_draft_tokens() + 1
                    generation_tokens = [seq_data.get_last_token_id()
                                         ] + seq_data.get_draft_token_ids()
                else:
                    num_generation_tokens = 1
                    token_id = seq_data.get_last_draft_token_id() if \
                        seq_data.get_num_draft_tokens() > 0 else seq_data.get_last_token_id()
                    generation_tokens = [token_id]

                input_tokens.append(generation_tokens)
                seq_len = seq_data.get_len() + seq_data.get_num_draft_tokens()

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                first_position = seq_len - num_generation_tokens
                positions = [
                    first_position + offset
                    for offset in range(num_generation_tokens)
                ]
                input_positions.append(positions)

                block_table = seq_group_metadata.d_block_tables[seq_id] if is_draft_model \
                    else seq_group_metadata.block_tables[seq_id]

                slots = []
                for position in positions:
                    block_number = block_table[position // self.block_size]
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slots.append(slot)
                slot_mapping.append(slots)

                lora_index_mapping.append([lora_id])
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        batch_size = len(input_tokens)
        # record the real batch size for cudagraph in case of padding
        real_batch_size = batch_size
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(1)
                block_tables.append([])
            batch_size = graph_batch_size

        max_len = max([len(t) for t in input_tokens])
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_len=max_len,
                                             pad=0,
                                             dtype=torch.long,
                                             device=self.device)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_len=max_len,
                                                pad=0,
                                                dtype=torch.long,
                                                device=self.device)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_len=max_len,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        if use_captured_graph:
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
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )

        lora_index_mapping = [
            _pad_to_max(mapping, 1, pad=0) for mapping in lora_index_mapping
        ]

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            max_seq_len=None,
            start_loc=None,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            kv_cache_dtype=self.kv_cache_dtype,
            use_speculate=self.use_speculate,
            is_multi_query_mode=is_multi_query_mode,
            batch_size=real_batch_size,
        )
        return (input_tokens, input_positions, input_metadata,
                lora_index_mapping, lora_prompt_mapping, lora_requests)

    def _prepare_draft_token_probs(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs_list = []
        token_ids_list = []
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(
                seq_ids
            ) == 1, "Only one seq is allowed per seq group when using speculative decoding."
            seq_id = seq_ids[0]
            draft_tokens = seq_group_metadata.seq_data[seq_id].draft_tokens
            probs_builder = []
            token_ids_builder = []
            for draft_token_data in draft_tokens:
                token_id, parent_probs = draft_token_data.token_id, draft_token_data.parent_probs
                # parent_probs should have shape [1, vocab_size]
                probs_builder.append(parent_probs)
                token_ids_builder.append(token_id)
            probs_list.append(probs_builder)
            token_ids_list.append(token_ids_builder)
        # draft_token_probs should have shape [bsz, speculate_length, vocab_size]
        draft_token_probs = torch.stack(
            [torch.cat(inner_list) for inner_list in probs_list])
        # NOTE: Pad an extra draft token for the convenience of handling the case when all tokens
        # accepted. For example, draft_token_probs will have shape [32, 7, 32000] after padding if its
        # original shape is [32, 6, 32000].
        draft_token_probs = torch.nn.functional.pad(draft_token_probs,
                                                    (0, 0, 0, 1),
                                                    value=0)
        # draft_token_ids should have shape [bsz, speculate_length]
        draft_token_ids = torch.tensor(token_ids_list,
                                       dtype=torch.long,
                                       device="cuda")
        return (draft_token_ids, draft_token_probs)

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
        is_multi_query_mode: bool = False,
        input_tokens_ids: Optional[torch.Tensor] = None,
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

        # Prepare draft tokens and their probs during draft token evaluation stage
        draft_token_ids, draft_token_probs = None, None
        if is_multi_query_mode:
            draft_token_ids, draft_token_probs = self._prepare_draft_token_probs(
                seq_group_metadata_list)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            generators=generators,
            use_speculate=self.use_speculate,
            is_multi_query_mode=is_multi_query_mode,
            speculate_length=self.speculate_length,
            input_token_ids=input_tokens_ids,
            draft_token_ids=draft_token_ids,
            draft_token_probs=draft_token_probs,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        is_draft_model: bool = False,
        is_multi_query_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata,
               Set[int], LoRAMapping]:
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, input_metadata, prompt_lens,
                 subquery_lens, lora_index_mapping, lora_prompt_mapping,
                 lora_requests) = self._prepare_prompt(
                     seq_group_metadata_list, is_draft_model=is_draft_model)
            else:
                (input_tokens, input_positions, input_metadata,
                 lora_index_mapping, lora_prompt_mapping,
                 lora_requests) = self._prepare_decode(
                     seq_group_metadata_list,
                     is_draft_model=is_draft_model,
                     is_multi_query_mode=is_multi_query_mode)
                prompt_lens = []
                subquery_lens = None
            sampling_metadata = self._prepare_sample(
                seq_group_metadata_list,
                prompt_lens,
                subquery_lens,
                is_multi_query_mode=is_multi_query_mode,
                input_tokens_ids=input_tokens[:input_metadata.batch_size]
                if is_multi_query_mode else None)

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
                "is_prompt": input_metadata.is_prompt,
                "slot_mapping": input_metadata.slot_mapping,
                "prompt_lens": input_metadata.prompt_lens,
                "max_seq_len": input_metadata.max_seq_len,
                "start_loc": input_metadata.start_loc,
                "max_context_len": input_metadata.max_context_len,
                "context_lens": input_metadata.context_lens,
                "block_tables": input_metadata.block_tables,
                "use_cuda_graph": input_metadata.use_cuda_graph,
                "kv_cache_dtype": input_metadata.kv_cache_dtype,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "use_speculate": input_metadata.use_speculate,
                "is_multi_query_mode": input_metadata.is_multi_query_mode,
                "batch_size": input_metadata.batch_size,
            }
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict["input_tokens"]
            input_positions = metadata_dict["input_positions"]
            lora_mapping = metadata_dict["lora_mapping"]
            lora_requests = metadata_dict["lora_requests"]
            input_metadata = InputMetadata(
                is_prompt=metadata_dict["is_prompt"],
                slot_mapping=metadata_dict["slot_mapping"],
                prompt_lens=metadata_dict["prompt_lens"],
                max_seq_len=metadata_dict["max_seq_len"],
                start_loc=metadata_dict["start_loc"],
                max_context_len=metadata_dict["max_context_len"],
                context_lens=metadata_dict["context_lens"],
                block_tables=metadata_dict["block_tables"],
                use_cuda_graph=metadata_dict["use_cuda_graph"],
                kv_cache_dtype=metadata_dict["kv_cache_dtype"],
                use_speculate=metadata_dict["use_speculate"],
                is_multi_query_mode=metadata_dict["is_multi_query_mode"],
                batch_size=metadata_dict["batch_size"],
            )
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=metadata_dict["selected_token_indices"],
                categorized_sample_indices=None,
                generators=None,
                perform_sampling=False,
                use_speculate=metadata_dict["use_speculate"],
                is_multi_query_mode=metadata_dict["is_multi_query_mode"],
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
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
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
        self.log_cudagraph_warning()
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

    def log_cudagraph_warning(self) -> None:
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the CuPy NCCL communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        self.graph_runners.clear()
        if self.use_speculate:
            self.d_graph_runners.clear()
        self.cupy_nccl_backend = None

    @torch.inference_mode()
    def speculate_capture_model(self, kv_caches: List[KVCache],
                                draft_kv_caches: List[KVCache]) -> None:
        # NOTE(woosuk): This is a hack to ensure that the NCCL backend is never
        # deleted before the CUDA graphs.
        self.cupy_nccl_backend = cupy_utils.get_nccl_backend()

        assert not self.model_config.enforce_eager
        self.log_cudagraph_warning()
        start_time = time.perf_counter()

        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        def capture_graph_inner(is_draft_model: bool = False,
                                caches: List[KVCache] = None) -> None:
            if is_draft_model:
                num_tokens = 1
                is_multi_query_mode = False
                active_model = ActiveModel.DRAFT
                graph_runners = self.d_graph_runners
                graph_memory_pool = self.d_graph_memory_pool
            else:
                num_tokens = self.speculate_length + 1
                is_multi_query_mode = True
                active_model = ActiveModel.TARGET
                graph_runners = self.graph_runners
                graph_memory_pool = self.graph_memory_pool

            # Prepare dummy inputs. These will be reused for all batch sizes.
            input_tokens = torch.zeros(max_batch_size,
                                       num_tokens,
                                       dtype=torch.long).cuda()
            input_positions = torch.zeros(max_batch_size,
                                          num_tokens,
                                          dtype=torch.long).cuda()
            slot_mapping = torch.empty(max_batch_size,
                                       num_tokens,
                                       dtype=torch.long).cuda()
            slot_mapping.fill_(_PAD_SLOT_ID)
            context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
            block_tables = torch.from_numpy(self.graph_block_tables).cuda()

            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
            # kernel, CuPy NCCL, and PyTorch NCCL. When using CUDA graph, we use
            # either custom all-reduce kernel or CuPy NCCL. When not using CUDA
            # graph, we use either custom all-reduce kernel or PyTorch NCCL.
            # We always prioritize using custom all-reduce kernel but fall back
            # to PyTorch or CuPy NCCL if it is disabled or not supported.
            with custom_all_reduce.capture():
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
                        is_multi_query_mode=is_multi_query_mode,
                    )
                    graph_runner = CUDAGraphRunner(self.draft_model) if is_draft_model \
                        else CUDAGraphRunner(self.model)
                    with MarkActiveModel(active_model):
                        graph_runner.capture(
                            input_tokens[:batch_size],
                            input_positions[:batch_size],
                            caches,
                            input_metadata,
                            memory_pool=graph_memory_pool,
                        )
                    graph_memory_pool = graph_runner.graph.pool()
                    graph_runners[batch_size] = graph_runner
                if is_draft_model:
                    self.d_graph_memory_pool = graph_memory_pool
                else:
                    self.graph_memory_pool = graph_memory_pool

        # 1. Capture draft model
        if self.rank < self.parallel_config.draft_model_tp_size:
            capture_graph_inner(is_draft_model=True, caches=draft_kv_caches)
        # 2. Capture target model
        capture_graph_inner(is_draft_model=False, caches=kv_caches)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")

    def speculate_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        is_draft_model: bool = False,
        is_prompt: bool = False,
    ) -> None:
        is_multi_query_mode = (not is_draft_model) and (not is_prompt)
        input_tokens, input_positions, input_metadata, sampling_metadata, _, _ = (
            self.prepare_input_tensors(seq_group_metadata_list, is_draft_model,
                                       is_multi_query_mode))

        if is_multi_query_mode:
            # The target model evaluates (speculate_length+1) tokens during the evaluation stage.
            assert input_tokens.shape[1] == self.speculate_length + 1
            assert not is_draft_model

        # Unused devices in tensor parallelism will skip this step.
        tp_size = self.parallel_config.draft_model_tp_size if is_draft_model \
            else self.parallel_config.tensor_parallel_size
        if self.rank >= tp_size:
            return None

        active_model = ActiveModel.DRAFT if is_draft_model else ActiveModel.TARGET
        with MarkActiveModel(active_model):
            graph_batch_size = input_tokens.shape[0]
            model = self.draft_model if is_draft_model else self.model
            model_executable = model
            if input_metadata.use_cuda_graph:
                graph_runners = self.d_graph_runners if is_draft_model else self.graph_runners
                # For speculative decoding, cudagraph is only used in the evaluation stage
                # for target model.
                if is_draft_model or input_metadata.is_multi_query_mode:
                    model_executable = graph_runners[graph_batch_size]
            # Execute the model.
            hidden_states = model_executable(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=kv_caches,
                input_metadata=input_metadata,
            )
            if input_metadata.is_multi_query_mode:
                hidden_states = hidden_states[:input_metadata.batch_size]
            # Sample the next token.
            output = model.sample(
                hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )

        if not self.is_driver_worker:
            return None

        speculate_outputs: SpeculateOutput = []

        for seq_group_metadata, seq_group_output in zip(
                seq_group_metadata_list, output):
            # 1. Handling the case when multiple tokens can be generated.
            if sampling_metadata.is_multi_query_mode:
                # Here we drop the last sampled token when all draft tokens were accepted.
                # When all draft tokens were accepted, the last 2 generated tokens (the last
                # accepted draft token plus the token sampled from its logits) will miss
                # their kv caches in the draft model and requires multi-query attention.
                # Mixing single query and multi-query attention is currently not supported.
                seq_id = seq_group_output.parent_seq_id
                num_accepted = seq_group_output.num_accepted_tokens
                max_num_generated_tokens = min(
                    num_accepted + 1, sampling_metadata.speculate_length)
                output_tokens = seq_group_output.output_tokens[:
                                                               max_num_generated_tokens]
                output_token_logprobs = seq_group_output.logprobs_list[:
                                                                       max_num_generated_tokens]
                speculate_outputs.append(
                    SpeculateSequenceGroupOutput(seq_id,
                                                 output_tokens,
                                                 output_token_logprobs,
                                                 num_accepted,
                                                 prompt_logprobs=None))
            # 2. Handling the case when only 1 token can be generated.
            else:
                samples = seq_group_output.samples
                assert len(
                    samples
                ) == 1, "Speculative decoding only allows one seq per seq group."
                sample = samples[0]
                seq_id = sample.parent_seq_id
                parent_probs = sample.parent_probs
                token_id = sample.output_token
                logprobs_dict = sample.logprobs
                if is_draft_model:
                    seq_data = seq_group_metadata.seq_data[seq_id]
                    seq_data.append_draft_token(token_id,
                                                logprobs_dict[token_id],
                                                parent_probs)
                else:
                    speculate_outputs.append(
                        SpeculateSequenceGroupOutput(
                            seq_id, [token_id], [logprobs_dict], 0,
                            seq_group_output.prompt_logprobs))
        return speculate_outputs

    def clear_draft_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ):
        """Clear draft tokens stored in seq data.

        Args:
            seq_group_metadata_list (List[SequenceGroupMetadata]): input metadata.
        """
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(list(seq_group_metadata.seq_data.keys()))
            assert len(
                seq_ids
            ) == 1, "Speculative decoding only allows one seq per seq group."
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            seq_data.clear_draft_tokens()

    @torch.inference_mode()
    def speculate_execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        d_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculateOutput:
        if self.is_driver_worker:
            self.clear_draft_tokens(seq_group_metadata_list)
            is_prompt = seq_group_metadata_list[0].is_prompt
            broadcast_object_list([is_prompt], src=0)
        else:
            obj_list = [None]
            is_prompt = broadcast_object_list(obj_list, src=0)[0]
        # Prompt evaluation
        if is_prompt:
            # Step 1: Run the target model
            output = self.speculate_step(seq_group_metadata_list,
                                         kv_caches,
                                         is_draft_model=False,
                                         is_prompt=True)
            # Step 2: Run the draft model
            self.speculate_step(seq_group_metadata_list,
                                d_kv_caches,
                                is_draft_model=True,
                                is_prompt=True)
            return output
        # Token generation
        # Step 1: Generate draft tokens
        for _ in range(self.speculate_length):
            self.speculate_step(seq_group_metadata_list,
                                d_kv_caches,
                                is_draft_model=True)
        # Step 2: Run target model to evaluate draft tokens
        output = self.speculate_step(seq_group_metadata_list,
                                     kv_caches,
                                     is_draft_model=False)
        if self.is_driver_worker:
            self.clear_draft_tokens(seq_group_metadata_list)
        return output


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


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
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
