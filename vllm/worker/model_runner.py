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
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    with_cupy_nccl_for_all_reduce)
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl, measure_cuda_memory

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]


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

        # Set enforce_eager to True for Neuron backend, to avoid capturing graph
        if self.device_config.is_neuron:
            self.model_config.enforce_eager = True

    def load_model(self) -> None:
        with measure_cuda_memory() as m:
            self.model = get_model(self.model_config,
                                   self.device_config,
                                   lora_config=self.lora_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config)

        self.model_memory_usage = m.consumed_memory
        logger.info(
            f"Loading model weights took {self.model_memory_usage / float(2**30):.4f} GB"
        )

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
                self.scheduler_config.max_num_batched_tokens, vocab_size,
                self.lora_config, self.device, self.model.embedding_modules,
                self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_context_len_to_capture + block_size - 1) // block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int],
               List[int], List[int], Set[LoRARequest]]:
        """Process prompt requests and prepare data structure
        that is converted to tensors.

        All tokens are coalesced into a single dimension instead of
        having a batch dimension. It is to avoid unnecessary paddings.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        context_lens: List[int] = []
        subquery_lens: List[int] = []
        block_tables: List[List[int]] = []
        num_chunked_prefill = 0

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prefill_start, prefill_end = seq_data.get_prefill_range()
            prompt_tokens = seq_data.get_token_ids()[prefill_start:prefill_end]
            seqlen = prefill_end
            seq_lens.append(seqlen)

            computed_len = prefill_start
            # NOTE: This only works for oooooooxxx style attention.
            computed_block_nums = seq_group_metadata.computed_block_nums
            prefix_caching_enabled = (computed_block_nums is not None
                                      and len(computed_block_nums) > 0
                                      and self.sliding_window is None)

            if seq_group_metadata.is_chunked_prefill:
                num_chunked_prefill += 1
                # TODO(sang): Support prefix caching.
                if prefix_caching_enabled:
                    raise RuntimeError(
                        "chunked prefill cannot be used with prefix caching now."
                    )
            if prefix_caching_enabled:
                # Prefix is not supported with sliding_window
                computed_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[computed_len:]
                block_tables.append(computed_block_nums)
            elif self.scheduler_config.chunked_prefill_enabled:
                # Update the block table so that KV cache location
                # can be found. TODO(sang): Make it work with
                # prefix caching.
                # This only happens when it is profiling run.
                if seq_group_metadata.block_tables is None:
                    block_tables.append([])
                else:
                    block_table = seq_group_metadata.block_tables[seq_id]
                    block_tables.append(block_table)
                computed_len = prefill_start
            else:
                # Normal prefill. Don't update the block table because
                # we don't need to use KV cache when running attentions.
                block_tables.append([])
                assert prefill_start == 0
                computed_len = prefill_start
            context_lens.append(computed_len)
            subquery_lens.append(prefill_end - computed_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            # NOTE(sang): prefill_end is always # of prompts if chunked
            # prefill is not enabled. Prefix caching is not working with
            # chunked prefill now.
            input_positions.extend(range(computed_len, prefill_end))
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            lora_index_mapping += [lora_id] * (seqlen - computed_len)
            lora_prompt_mapping.extend(
                [lora_id] *
                (seqlen - computed_len
                 if seq_group_metadata.sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seqlen)
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
                start_idx = max(0, seqlen - self.sliding_window)

            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        return (input_tokens, input_positions, slot_mapping,
                lora_index_mapping, lora_prompt_mapping, lora_requests,
                seq_lens, context_lens, subquery_lens, block_tables,
                num_chunked_prefill)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int],
               Set[LoRARequest]]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
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
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
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

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        # For decoding requests, batch_size == input_tokens.
        batch_size = len(input_tokens)
        max_context_len = max(context_lens, default=0)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                context_lens.append(1)
                block_tables.append([])
                lora_index_mapping.append(0)

        return (input_tokens, input_positions, slot_mapping, context_lens,
                block_tables, lora_index_mapping, lora_prompt_mapping,
                lora_requests, use_captured_graph, max_context_len)

    def _prepare_mixed_batch(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, torch.Tensor,
               torch.Tensor, torch.Tensor, Set[LoRARequest]]:
        """Create input tensors and metadata that's needed for
        running prefill and decode requests. The returned tensor
        includes input tensors and metadata of prefill
        and decoding requests. By default, the returned tensor only
        inlucdes EITHER prefill or decode requests, but if chunked
        prefill is enabled, both requests can coalesced to the same tensors.
        Each tensor is always ordered. I.e., prefill inputs are
        coming first, and decoding inputs are coming later based on
        the order of a given seq_group_metadata_list.

        It automatically adds padding that's required to run cuda graph.
        NOTE: cuda graph is not used if prompt requests are included
        in return tensors.

        If the input tensors contain prompt tokens, the layout is as follows:	
        |<---------------------- num_valid_tokens ---------------------->|	
        |<--------------- num_prompt_tokens -------------->|	
        |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|	

        Otherwise, the layout is as follows:	
        |<------------------ num_valid_tokens ------------------->|	
        |<------- num_generation_tokens (M) ------->|	
        |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|	

        The prompts might have different lengths, while the generation tokens always	
        have length 1. The paddings are appended to make the input length a multiple	
        of 8, which is desirable for Tensor Cores.

        If chunked prefill is enabled, the input will include both prompt tokens
        and generation tokens. The layout is as follows:
        |<---------------------- num_valid_tokens -------------------------->|
        |<--------- num_prompt_tokens ----->|<--- num_generation_tokens----->|
        |<-prompt_0->|<-prompt_1->|...|<pad>||<-gen_0->|<-gen_1->|.....|<pad>|


        Arguments:
            seq_group_metadata_list: The list of requests.

        Returns:
            input_tokens: [batch_size]. Token ID of each token
                coalesced to a flattened tensor. Padded.
            input_positions: [batch_size]. Index from the beginning
                of the sequence to the end of the sequence. Note that
                if prefix caching is used or chunked prefill is enabled,
                it could be shorter than actual prompt lengths. Padded.
            input_metadata: The metadata that's needed to run a model.
            subquery_lens: [batch_size]. The length of the sequence that's
                going to processed. The total number of tokens processed
                with this run is context_lens[i] + subquery_lens[i].
                Padded.
            TODO(sang): Fill it up.
            lora_index_mapping:
            lora_prompt_mapping:
            lora_requests:
        """
        prefill_reqs = []
        decode_reqs = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                prefill_reqs.append(seq_group_metadata)
            else:
                decode_reqs.append(seq_group_metadata)

        # Prepare inputs for both prefill and decoding phases
        # and concatenate. If chunked prefill is enabled,
        # requests are batched together. Otherwise,
        # only prefill or decoding requests should be received.
        if not self.scheduler_config.chunked_prefill_enabled:
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        (input_tokens, input_positions, slot_mapping, lora_index_mapping,
         lora_prompt_mapping, lora_requests, seq_lens, context_lens,
         subquery_lens, block_tables,
         num_chunked_prefill) = self._prepare_prompt(prefill_reqs)

        (input_tokens_decode, input_positions_decode, slot_mapping_decode,
         context_lens_decode, block_tables_decode, lora_index_mapping_decode,
         lora_prompt_mapping_decode, lora_requests_decode, use_captured_graph,
         max_context_len) = self._prepare_decode(decode_reqs)

        # Cuda graph is used only when there's decoding only
        # requests. Note that chunked prefill also doesn't use
        # cuda graph.
        if len(prefill_reqs) > 0:
            use_captured_graph = False

        max_subquery_len = max(subquery_lens, default=0)
        max_seq_len = max(seq_lens, default=0)
        num_prompt_tokens = len(input_tokens)
        num_generation_tokens = len(input_tokens_decode)

        # Concatenate inputs.
        input_tokens.extend(input_tokens_decode)
        input_positions.extend(input_positions_decode)
        slot_mapping.extend(slot_mapping_decode)
        lora_index_mapping.extend(lora_index_mapping_decode)
        lora_prompt_mapping.extend(lora_prompt_mapping_decode)
        lora_requests.update(lora_requests_decode)
        context_lens.extend(context_lens_decode)
        block_tables.extend(block_tables_decode)

        # Create tensors.
        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        # Prepare block table tensors.
        if use_captured_graph:
            # NOTE: cuda-graph is not used only when all requests
            # are decoding requests.
            batch_size = _get_graph_batch_size(len(seq_group_metadata_list))
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

        # Prepare tensors to index into query and key.
        subquery_lens_tensor = torch.tensor(subquery_lens,
                                            dtype=torch.long,
                                            device=self.device)
        subquery_start_loc = torch.zeros(subquery_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.long,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(subquery_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        # Create an input metadata that contains both prefill
        # and decoding data. We concatenate the data to avoid
        # calling broadcast 2X.
        input_metadata = InputMetadata(
            slot_mapping=slot_mapping,
            prompt_lens=seq_lens,
            prompt_lens_tensor=seq_lens_tensor,
            num_chunked_prefill=num_chunked_prefill,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
            max_subquery_len=max_subquery_len,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return (input_tokens, input_positions, input_metadata, subquery_lens,
                seq_lens, lora_index_mapping, lora_prompt_mapping,
                lora_requests)

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
        pin_memory = not self.in_wsl and not self.device_config.is_neuron

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
                selected_token_start_idx += subquery_len

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
                                            pin_memory=pin_memory)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids,
                          dtype=torch.int,
                          target_device=self.device,
                          pin_memory=pin_memory)
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
            (input_tokens, input_positions, input_metadata, subquery_lens,
             prompt_lens, lora_index_mapping, lora_prompt_mapping,
             lora_requests
             ) = self._prepare_mixed_batch(seq_group_metadata_list)
            sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens)

            if self.lora_config:
                lora_mapping = LoRAMapping(
                    lora_index_mapping,
                    lora_prompt_mapping,
                )
            else:
                lora_mapping = None

            # Broadcast the metadata.
            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
            }
            metadata_dict.update(input_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            input_metadata = InputMetadata(**metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=selected_token_indices,
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
            input_metadata=input_metadata,
        )

        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        # print("SANG-TODO output: ", output)
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
            seq_data.advance_prefill_range(seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                is_chunked_prefill=False,
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
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
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
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
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
                    slot_mapping=slot_mapping[:batch_size],
                    prompt_lens=None,
                    prompt_lens_tensor=None,
                    num_chunked_prefill=0,
                    num_prompt_tokens=0,
                    num_generation_tokens=batch_size,
                    max_subquery_len=None,
                    max_context_len=self.max_context_len_to_capture,
                    max_seq_len=None,
                    subquery_start_loc=None,
                    seq_start_loc=None,
                    context_lens=context_lens[:batch_size],
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                    kv_cache_dtype=self.kv_cache_dtype)

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
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _async_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)
