from typing import Dict, List, Tuple, Union

import torch

from cacheflow.models import get_model
from cacheflow.models import InputMetadata
from cacheflow.worker.cache_engine import CacheEngine
from cacheflow.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed
from cacheflow.parallel_utils.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_world_size)
from cacheflow.utils import set_random_seed
class Worker:

    def __init__(
        self,
        model_name: str,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: str,
        distributed_init_method: str,
        rank: int,
        world_size: int,
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

        # Initialize the model.
        self.model, self.dtype = get_model(model_name, dtype=dtype)
        self.model = self.model.cuda()
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.num_layers = self.model.config.num_hidden_layers
        assert self.model.config.num_attention_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.model.config.num_attention_heads // tensor_model_parallel_world_size
        self.head_size = self.model.config.hidden_size // self.num_heads

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
                                     pipeline_parallel_size: int = 1,
        ):
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
        set_random_seed(0)
        model_parallel_cuda_manual_seed(0)


    def prepare_inputs(
        self,
        prompt_tokens: Dict[int, List[int]],    # Seq id -> List of input token ids.
        generation_tokens: Dict[int, int],      # Seq id -> Input token id.
        context_lens: Dict[int, int],           # Seq id -> Number of tokens participating in attention.
        block_tables: Dict[int, List[int]],     # Seq id -> List of physical block numbers.
    ) -> Tuple[torch.LongTensor, torch.LongTensor, InputMetadata]:
        # TODO(woosuk): Support interactive generation.
        # Add the prompt tokens.
        prompt_lens: List[int] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_seq_ids = sorted(prompt_tokens.keys())
        for seq_id in prompt_seq_ids:
            prompt_len = len(prompt_tokens[seq_id])
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens[seq_id])
            input_positions.extend(range(len(prompt_tokens[seq_id])))

            block_table = block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add the generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        generation_block_tables: List[List[int]] = []

        generation_seq_ids = sorted(generation_tokens.keys())
        for seq_id in generation_seq_ids:
            input_tokens.append(generation_tokens[seq_id])
            position_id = context_lens[seq_id] - 1
            input_positions.append(position_id)

            block_table = block_tables[seq_id]
            generation_block_tables.append(block_table)

            max_context_len = max(max_context_len, context_lens[seq_id])
            max_num_blocks_per_seq = max(
                max_num_blocks_per_seq, len(block_table))

            block_number = block_table[position_id // self.block_size]
            block_offset = position_id % self.block_size
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
            [context_lens[seq_id] for seq_id in generation_seq_ids],
            dtype=torch.int, device='cuda')
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables]
        block_tables_tensor = torch.tensor(
            padded_block_tables, dtype=int, device='cuda')

        input_metadata = InputMetadata(
            seq_ids=prompt_seq_ids + generation_seq_ids,
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
        prompt_tokens: Dict[int, List[int]],    # Seq id -> List of input token ids.
        generation_tokens: Dict[int, int],      # Seq id -> Input token id.
        context_lens: Dict[int, int],           # Seq id -> Number of tokens participating in attention.
        block_tables: Dict[int, List[int]],     # Seq id -> List of physical block numbers.
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, int],
    ) -> Union[torch.Tensor, Dict[int, Tuple[int, int]]]:
        # Issue cache operations.
        command_issued = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            command_issued = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            command_issued = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            command_issued = True

        if command_issued:
            cache_events = self.cache_events
        else:
            cache_events = None

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self.prepare_inputs(
            prompt_tokens, generation_tokens, context_lens, block_tables)

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
