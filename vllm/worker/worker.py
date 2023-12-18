"""A GPU worker class."""
import gc
import os

from typing import Dict, List, Tuple, Optional, Set, Union

import msgspec
import torch
import torch.backends
import torch.distributed
import traceback

from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
from vllm.worker.base_worker import BaseLoraWorker
from vllm.anyscale.profiler_utils import TorchProfiler, Profilable
from vllm.anyscale.cuda_graph import CudaGraphCapturedModel
from vllm.anyscale.lora.utils import LoRARequest
from vllm.anyscale.lora.worker_manager import (
    DisabledWorkerLoRAManager,
    LRUCacheWorkerLoRAManager,
)
from vllm.config import (
    CacheConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    LoRAConfig,
)
from vllm.logger import init_logger
from vllm.model_executor import InputMetadata, MultiStepInputMetadata, get_model, set_random_seed
from vllm.model_executor.layers.sampler import pythonize_sampler_output
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, model_parallel_is_initialized,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size, get_tensor_model_parallel_group)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata, ExecuteModelData, SequenceGroupMetadataDelta
from vllm.worker.cache_engine import CacheEngine
from vllm.anyscale.lora.layers import LoRAMapping
from vllm.engine.ray_utils import ray

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8
MAX_INT_32 = 2**31 - 1


class Worker(Profilable, BaseLoraWorker):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        load_config: Optional[LoadConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.load_config = load_config
        self.lora_config = lora_config

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        # Stats, updated every iteration
        self.num_input_tokens = 0
        self.num_seq_groups = 0

        self.lora_manager = None

        self.seq_metadata_cache = None
        self.input_padding_size = self.scheduler_config.input_padding_size

        # Enable small batch padding optimization for chunked prefill.
        self.optimize_small_batch_padding = \
            self.scheduler_config.max_chunked_prefill_len > 0

        self._profiler = TorchProfiler()

    def init_model(self, should_init_distributed_env: bool = True):
        """Initialize the model.

        If should_init_distributed_env is False, do not initialize torch
        distributed or other collective utilities.
        """
        # Torch default: False
        torch.backends.cuda.matmul.allow_tf32 = True
        # Torch default: True
        torch.backends.cudnn.allow_tf32 = True

        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = (self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1")))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        _check_if_gpu_supports_dtype(self.model_config.dtype)

        if should_init_distributed_env:
            # Initialize the distributed environment.
            _init_distributed_environment(self.parallel_config, self.rank,
                                          self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config, self.load_config,
                               self.lora_config)

        vocab_size = self.model.config.vocab_size

        if self.lora_config:
            logger.info("Creating LoRA adapter...")
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, vocab_size,
                self.lora_config, self.device)
            self.model = self.lora_manager.create_lora_adapter(self.model)
        else:
            self.lora_manager = DisabledWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, vocab_size,
                self.lora_config, self.device)

        if self.scheduler_config.use_deltas:
            self.seq_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
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
                    lora_id=f"warmup_{lora_id}",
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

        def run_prefill_max_tokens():
            """Run the prefill step with the maximum number of sequences.

            Attempt to fill the batch (total num tokens ==
            max_num_batched_tokens). This may not be possible if `max_num_seqs
            * max_model_len < max_num_batched_tokens`.
            This is to mimic running
            the largest possible prefill step

            Apply the maximum number of loras if necessary (1 for every
                sequence)

            """
            seqs = []
            input_tokens = []
            for group_id in range(max_num_seqs):
                seq_len = min(
                    max_num_batched_tokens // max_num_seqs +
                    (group_id < max_num_batched_tokens % max_num_seqs),
                    self.model_config.max_model_len)
                prompt_tokens = [0] * seq_len
                seq_data = SequenceData(prompt_tokens)
                seq_data.advance_prefill_range(seq_len)
                input_tokens.extend(prompt_tokens)
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

            (
                input_tokens,
                input_positions,
                input_metadata,
                lora_mapping,
                prepared_lora_requests,
            ) = self._prepare_inputs(seqs)

            if self.lora_config:
                self.apply_loras(prepared_lora_requests, lora_mapping)

            # Execute the model.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=[(None, None)] * num_layers,
                input_metadata=input_metadata,
                cache_events=None,
            )

        def run_generation_max_seqs():
            """Run the generation step with maximum number of sequences.

            This is to mimic running the largest possible generation step.
            each sequences has a length of 1 to mimic the generation step.

            Apply the maximum number of loras if necessary (1 for every
                sequence)

            """
            seqs = []
            input_tokens = []
            for group_id in range(max_num_seqs):
                # setting sequence length to 1 to mimic the generation/decode
                # step, where we only are operating on sequences of 1 token at
                # a time.
                seq_len = 1
                prompt_tokens = [0] * seq_len
                seq_data = SequenceData(prompt_tokens)
                seq_data.advance_prefill_range(seq_len)
                input_tokens.extend(prompt_tokens)
                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    # though this is not meant to be a prompt, we set this to
                    # true because we don't have block tables / kv caches
                    # initialized, and we still want to mimic the generation
                    # with lora.
                    is_prompt=True,
                    is_chunked_prefill=False,
                    seq_data={group_id: seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    lora_request=dummy_lora_requests_per_seq[group_id]
                    if dummy_lora_requests_per_seq else None,
                )
                seqs.append(seq)

            (input_tokens, input_positions, input_metadata, lora_mapping,
             prepared_lora_requests) = (self._prepare_inputs(seqs))

            if self.lora_config:
                self.apply_loras(prepared_lora_requests, lora_mapping)

            # Execute the model.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=[(None, None)] * num_layers,
                input_metadata=input_metadata,
                cache_events=None,
            )

        # Run both prefill with the maximum number of tokens and generation
        # with the maximum number of sequences. Apply any loras if necessary.
        # If there are no loras applied then prefill will use the more memory
        # than during generation. However, if loras are applied then it is
        # possible for generation to use more memory than prefill.
        # This is because when applying loras during prefill, the loras are
        # applied iteratively on the batch for each sequence/lora, however
        # during generation the loras are stacked and then 1 forward pass is
        # done. While is is more efficient in terms of computation, it is less
        # memory efficient since all the loras need to be loaded in GPU memory
        # at the same time.

        run_prefill_max_tokens()
        # Since memory consumption for generation is only potentially larger
        # than prefill when loras are applied, we only run the generation step
        # when loras are applied to save time.

        if dummy_lora_requests:
            run_generation_max_seqs()
        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        self.lora_manager.remove_all_loras()
        if self.seq_metadata_cache:
            self.seq_metadata_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.sliding_window = cache_config.sliding_window

        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.captured_model = CudaGraphCapturedModel(self.model,
                                                     self.gpu_cache,
                                                     self.model_config,
                                                     self.scheduler_config,
                                                     self.block_size)

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                            SequenceGroupMetadataDelta]],
        return_logits: bool = False,
        num_steps: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, LoRAMapping,
               Set[LoRARequest]]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        lora_requests: Set[LoRARequest] = set()

        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        is_multi_step = num_steps > 0
        num_steps = 1 if not is_multi_step else num_steps

        # Add prompt tokens.
        prompt_lens: List[int] = []
        block_tables: List[List[List[int]]] = [[] for _ in range(num_steps)]

        max_num_blocks_per_seq = 0
        slot_mapping: List[List[int]] = [[] for _ in range(num_steps)]
        context_lens: List[int] = []
        num_chunked_prefill = 0

        for seq_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            if not seq_group_metadata.is_prompt:
                continue

            assert num_steps == 1

            if seq_group_metadata.is_chunked_prefill:
                num_chunked_prefill += 1

            if self.seq_metadata_cache is not None:
                self.seq_metadata_cache[
                    seq_group_metadata.request_id] = seq_group_metadata

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))
            lora_id = seq_group_metadata.lora_int_id

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prefill_start, prefill_end = seq_data.get_prefill_range()
            prompt_tokens = seq_data.get_token_ids()[prefill_start:prefill_end]
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)
            context_lens.append(prefill_end)

            input_tokens.extend(prompt_tokens)

            # Set the right input_position for positional encoding.
            input_positions.extend(range(prefill_start, prefill_end))

            assert len(seq_ids) == 1, "Prompt input should have only one seq."
            if sampling_params.prompt_logprobs is not None:
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + prompt_len - 1))

            selected_token_indices.append(selected_token_start_idx +
                                          prompt_len - 1)
            selected_token_start_idx += prompt_len

            if lora_id > 0:
                # if we are preparing inputs for the warmup step, we want the
                # lora computation to take up the maximum possible amount of
                # memory that way we can get a tighter upper bound on the
                # amount of memory we can use and therefore not oom. If
                # for_warmup is true, we add the lora lora mapping that is used
                # during generation.
                lora_requests.add(seq_group_metadata.lora_request)
            lora_index_mapping += [lora_id] * prompt_len
            lora_prompt_mapping.extend(
                [lora_id] *
                (prompt_len if sampling_params.prompt_logprobs else 1))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping[0].extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prefill_start, prefill_end):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[0].append(slot)

            block_tables[0].append(block_table)

        # pad prompt tokens. This is required for cuda-graph.
        input_tokens = self._pad_to_alignment(input_tokens)
        input_positions = self._pad_to_alignment(input_positions)
        slot_mapping[0] = self._pad_to_alignment(slot_mapping[0],
                                                 padded_value=-1)
        num_prompt_tokens = len(input_tokens)
        selected_token_start_idx = len(input_tokens)

        # Add generation tokens.
        num_generation_tokens = 0

        for seq_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_metadata.is_prompt or \
                    seq_group_metadata.is_chunked_prefill:
                continue

            if (self.seq_metadata_cache is not None and
                    seq_group_metadata.request_id in self.seq_metadata_cache):
                seq_group_metadata = self.seq_metadata_cache[
                    seq_group_metadata.request_id].update_from_delta(
                        seq_group_metadata)
                seq_group_metadata_list[seq_idx] = seq_group_metadata

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))
            lora_id = seq_group_metadata.lora_int_id

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                seq_block_table = seq_group_metadata.block_tables[seq_id]

                generation_token_positions = (
                    seq_data.get_unprocessed_token_positions())
                generation_token_ids = seq_data.get_unprocessed_token_ids()

                # Only the output from the last token needs to be sampled from.
                token_position_to_sample = generation_token_positions[-1]

                for input_token, input_position in zip(
                        generation_token_ids, generation_token_positions):
                    # Calculate metadata of generation token.
                    context_len = input_position + 1

                    block_table = seq_block_table
                    block_number = block_table[input_position //
                                               self.block_size]
                    block_offset = input_position % self.block_size
                    slot = block_number * self.block_size + block_offset

                    # If sliding window is enabled, truncate the context len and
                    # block table.
                    if self.sliding_window is not None:
                        context_len = min(context_len, self.sliding_window)

                        sliding_window_blocks = (self.sliding_window //
                                                 self.block_size)
                        block_table = block_table[-sliding_window_blocks:]

                    # Append metadata of generation token to input lists.
                    input_positions.append(input_position)
                    input_tokens.append(input_token)
                    lora_index_mapping.append(lora_id)
                    slot_mapping[0].append(slot)
                    context_lens.append(context_len)
                    block_tables[0].append(block_table)

                    # If we should sample a token from the output, append
                    # sampling metadata to input lists.
                    if input_position == token_position_to_sample:
                        selected_token_indices.append(selected_token_start_idx)
                        selected_token_start_idx += 1
                    else:
                        # Do not select this token for sampling.
                        selected_token_start_idx += 1
                    num_generation_tokens += 1

                # Update LoRA mapping.
                if lora_id > 0:
                    lora_requests.add(seq_group_metadata.lora_request)
                lora_prompt_mapping.append(lora_id)

        # This couples concerns between the multi step worker and the worker.
        # TODO(cade,antoni) Clean up when making multi step worker cuda
        # graphable.
        for step in range(1, num_steps):
            for seq_idx, seq_group_metadata in enumerate(
                    seq_group_metadata_list):
                if seq_group_metadata.is_prompt or \
                        seq_group_metadata.is_chunked_prefill:
                    continue

                seq_ids = list(seq_group_metadata.seq_data.keys())
                for seq_id in seq_ids:
                    seq_data = seq_group_metadata.seq_data[seq_id]
                    seq_block_table = seq_group_metadata.block_tables[seq_id]

                    generation_token_positions = (
                        seq_data.get_unprocessed_token_positions())
                    token_position_to_sample = generation_token_positions[-1]
                    for input_position in generation_token_positions:
                        # Calculate metadata of generation token.
                        if input_position != token_position_to_sample:
                            continue
                        input_position = input_position + step
                        context_len = input_position + 1

                        block_table = seq_block_table
                        block_number = block_table[input_position //
                                                   self.block_size]
                        block_offset = input_position % self.block_size
                        slot = block_number * self.block_size + block_offset

                        # If sliding window is enabled, truncate the context len
                        # and block table.
                        if self.sliding_window is not None:
                            context_len = min(context_len, self.sliding_window)

                            sliding_window_blocks = (self.sliding_window //
                                                     self.block_size)
                            block_table = block_table[-sliding_window_blocks:]

                        # Append metadata of generation token to input lists.
                        slot_mapping[step].append(slot)
                        block_tables[step].append(block_table)

        max_num_blocks_per_seq = [
            max((len(b) for b in block_table), default=0)
            for block_table in block_tables
        ]
        max_context_len = max(context_lens, default=0)

        self.num_input_tokens = len(input_tokens)
        self.num_seq_groups = len(seq_groups)

        # Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = self._pad_to_alignment(
            input_tokens, num_generation_tokens=num_generation_tokens)
        input_positions = self._pad_to_alignment(
            input_positions, num_generation_tokens=num_generation_tokens)
        slot_mapping = [
            self._pad_to_alignment(s,
                                   padded_value=-1,
                                   num_generation_tokens=num_generation_tokens)
            for s in slot_mapping
        ]

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensors = [
            torch.tensor(sm, dtype=torch.long, device="cuda")
            for sm in slot_mapping
        ]
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cuda")
        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="cuda")
        padded_block_tables = [[
            _pad_to_max(b_inner, max_num_blocks_per_seq[i])
            for b_inner in b_outer
        ] for i, b_outer in enumerate(block_tables)]
        block_tables_tensors = [
            torch.tensor(bt, dtype=torch.int, device="cuda")
            for bt in padded_block_tables
        ]

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        lora_mapping = LoRAMapping(
            self._pad_to_alignment(lora_index_mapping),
            lora_prompt_mapping,
        )

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            num_chunked_prefill=num_chunked_prefill,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
            slot_mapping=slot_mapping_tensors[0],
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensors[0],
            selected_token_indices=selected_token_indices,
            sliding_window=self.sliding_window,
            return_logits=return_logits,
            flash_style=self.scheduler_config.flash_style,
        )
        if is_multi_step:
            input_metadata = MultiStepInputMetadata(
                num_steps,
                input_metadata,
                extra_slot_mapping=slot_mapping_tensors[1:],
                extra_block_tables=block_tables_tensors[1:])
        return (
            tokens_tensor,
            positions_tensor,
            input_metadata,
            lora_mapping,
            lora_requests,
        )

    def execute_model_shared_memory(
            self,
            shared_memory_input: SharedMsgspecBufferWithEvent,
            shared_memory_output: SharedMsgspecBufferWithEvent,
            participant_id: int  # pylint: disable=unused-argument
    ):
        shared_memory_input.decoder = msgspec.msgpack.Decoder(ExecuteModelData)
        logger.info("Worker shared memory input buffer id: "
                    f"{shared_memory_input.participant_id}")
        logger.info("Worker shared memory output buffer id: "
                    f"{shared_memory_input.participant_id}")
        parallel_group = get_tensor_model_parallel_group()
        try:
            while True:
                logger.debug("Waiting for incoming data...")
                shared_memory_input.wait_for_incoming_data()
                data = shared_memory_input.get_data()
                logger.debug(f"Received data {data}.")
                torch.distributed.barrier(group=parallel_group)
                shared_memory_input.clear()
                logger.debug("Executing model...")
                outputs = self.execute_model(data)
                logger.debug(f"Execute output {outputs}.")
                if self.rank < 1:
                    logger.debug("Setting output")
                    shared_memory_output.set_data(outputs)
        except Exception:
            traceback.print_exc()
            shared_memory_output.set_error()
            raise

    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_data: ExecuteModelData,
            *,
            return_python_output: bool = True) -> List[SamplerOutput]:
        (seq_group_metadata_list, finished_request_ids_list, blocks_to_swap_in,
         blocks_to_swap_out, blocks_to_copy,
         return_logits) = (execute_model_data.seq_group_metadata_list,
                           execute_model_data.finished_request_ids_list,
                           execute_model_data.blocks_to_swap_in,
                           execute_model_data.blocks_to_swap_out,
                           execute_model_data.blocks_to_copy,
                           execute_model_data.return_logits)

        # Clean up the cache
        if self.seq_metadata_cache:
            for finished_request_id in finished_request_ids_list:
                self.seq_metadata_cache.pop(finished_request_id, None)

        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return [SamplerOutput([])]

        seq_group_request_ids = [
            seq_group_metadata.request_id
            for seq_group_metadata in seq_group_metadata_list
        ]

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            input_metadata,
            lora_mapping,
            lora_requests,
        ) = self._prepare_inputs(seq_group_metadata_list,
                                 return_logits=return_logits)

        if self.lora_config:
            lora_requests = [
                seq_group_metadata.lora_request
                for seq_group_metadata in seq_group_metadata_list
            ]
            self.apply_loras(lora_requests, lora_mapping)

        output = self.captured_model.execute_if_capturable(
            input_ids=input_tokens,
            positions=input_positions,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        if return_python_output:
            output = pythonize_sampler_output(output, input_metadata)

            if self.seq_metadata_cache is not None:
                for request_id, sampler_output in zip(seq_group_request_ids,
                                                      output):
                    cached_seq_metadata = self.seq_metadata_cache[request_id]
                    for sample in sampler_output.samples:
                        cached_seq_metadata.seq_data[
                            sample.parent_seq_id].append_token_ids(
                                [sample.output_token], [0])
            output = [output]

        return output

    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        self.lora_manager.apply_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.lora_manager.list_loras()

    def get_metadata_cache_len(self) -> int:
        return len(self.seq_metadata_cache
                   ) if self.seq_metadata_cache is not None else -1

    def get_runtime_context(self) -> Optional[dict]:
        if ray:
            runtime_ctx = ray.get_runtime_context()
            return {
                "job_id": runtime_ctx.get_job_id(),
                "node_id": runtime_ctx.get_node_id(),
                "worker_id": runtime_ctx.get_worker_id(),
            }
        return None

    def start_profile(self, **kwargs) -> None:
        self._profiler.start_profile(**kwargs)

    def stop_profile(self) -> None:
        self._profiler.stop_profile()

    def _pad_to_alignment(self,
                          x: List[int],
                          padded_value: int = 0,
                          num_generation_tokens: int = 0) -> List[int]:
        """Pad the input to be a multiple of the alignment size.

        Args:
            x: The input list.
            padded_value: The value to pad with.
            num_generation_tokens: The number of generation tokens in the input.
                If this is between 1 and 8, and we enable small batch
                optimization, the input will padded to 8.
        Returns:
            The padded list.
        """
        pad_size = self.input_padding_size
        if self.optimize_small_batch_padding and 0 < num_generation_tokens < 32:
            pad_size = 8
        return x + [padded_value] * ((-len(x)) % pad_size)


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())

    if model_parallel_is_initialized():
        assert get_tensor_model_parallel_world_size(
        ) == parallel_config.tensor_parallel_size
        assert get_pipeline_model_parallel_world_size(
        ) == parallel_config.pipeline_parallel_size
    else:
        initialize_model_parallel(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )


def _pad_to_max(x: List[int],
                max_len: int,
                padded_value: int = 0) -> List[int]:
    return x + [padded_value] * (max_len - len(x))


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
