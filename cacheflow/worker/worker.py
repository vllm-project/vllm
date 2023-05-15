"""A GPU worker class."""
from typing import Dict, List, Optional, Tuple

import torch

from cacheflow.model_executor import get_model, InputMetadata, set_random_seed
from cacheflow.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel,
    initialize_all_reduce_launcher,
    get_tensor_model_parallel_world_size)
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import (SequenceData, SequenceGroupMetadata,
                                SequenceOutputs)
from cacheflow.worker.cache_engine import CacheEngine


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_name: str,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: str,
        seed: int,
        distributed_init_method: str,
        rank: int,
        world_size: int,
        cache_dir: Optional[str],
        use_dummy_weights: bool,
        use_np_cache: bool,
        max_num_batched_tokens: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> None:
        self.init_distributed_environment(distributed_init_method,
                                          rank,
                                          world_size,
                                          tensor_parallel_size,
                                          pipeline_parallel_size)
        self.worker_id = rank
        self.block_size = block_size
        set_random_seed(seed)

        # Initialize the model.
        self.model, self.dtype = get_model(
            model_name, dtype=dtype, cache_dir=cache_dir,
            use_dummy_weights=use_dummy_weights, use_np_cache=use_np_cache)
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        initialize_all_reduce_launcher(
            max_num_batched_tokens, self.model.config.hidden_size, self.dtype)
        self.num_layers = self.model.config.num_hidden_layers
        assert self.model.config.num_attention_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.model.config.num_attention_heads // tensor_model_parallel_world_size
        self.head_size = self.model.config.hidden_size // (self.num_heads * tensor_model_parallel_world_size)

        # We reset the seed after initializing the model to ensure that
        # the random state is not affected by the model initialization.
        set_random_seed(seed)

        self.cache_engine = CacheEngine(
            worker_id=self.worker_id,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            dtype=self.dtype,
        )
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def init_distributed_environment(self,
                                     distributed_init_method: str,
                                     rank: int,
                                     world_size: int,
                                     tensor_parallel_size: int = 1,
                                     pipeline_parallel_size: int = 1) -> None:
        """Initialize the distributed environment."""
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        initialize_model_parallel(tensor_parallel_size,
                                  pipeline_parallel_size)

    def prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(
                    max_num_blocks_per_seq, len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(
            input_tokens, dtype=torch.long, device='cuda')
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device='cuda')
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.int, device='cuda')
        context_lens_tensor = torch.tensor(
            context_lens, dtype=torch.int, device='cuda')
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables]
        block_tables_tensor = torch.tensor(
            padded_block_tables, dtype=torch.int, device='cuda')

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_stage(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
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
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self.prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
