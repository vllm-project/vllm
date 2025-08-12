import torch
from typing import List, Optional, Dict

from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields


@dataclass
class ExecuteModelData:
    """Helper data structure which facilitates cleaner tests.
    """
    seq_group_metadata_list: List[SequenceGroupMetadata]
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]

    def to_dict(self):
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))


def round_up_to_next_block(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def create_execute_model_data(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    blocks_to_swap_in: Optional[Dict[int, int]] = None,
    blocks_to_swap_out: Optional[Dict[int, int]] = None,
    blocks_to_copy: Optional[Dict[int, int]] = None,
) -> ExecuteModelData:
    if blocks_to_swap_in is None:
        blocks_to_swap_in = {}
    if blocks_to_swap_out is None:
        blocks_to_swap_out = {}
    if blocks_to_copy is None:
        blocks_to_copy = {}

    return ExecuteModelData(
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
    )


def patch_execute_model_with_seeds(worker: Worker, rand_seeds: List[int]):
    seed_iter = iter(rand_seeds)
    original_execute_model = worker.execute_model

    def new_execute_model(*args, **kwargs):
        result = original_execute_model(*args, **kwargs)
        set_random_seed(next(seed_iter))
        return result

    return new_execute_model


def zero_kv_cache(cache_engine: CacheEngine):
    assert cache_engine.gpu_cache
    for key_blocks, value_blocks in cache_engine.gpu_cache:
        key_blocks.zero_()
        value_blocks.zero_()


def create_worker(cls: type,
                  model_name: str,
                  block_size: int,
                  num_gpu_blocks: int,
                  seed: int,
                  is_driver_worker: bool = True,
                  enforce_eager: bool = True):
    engine_args = EngineArgs(
        model=model_name,
        seed=seed,
        block_size=block_size,
        enforce_eager=enforce_eager,
    )

    (model_config, cache_config, parallel_config, scheduler_config,
     device_config, _) = engine_args.create_engine_configs()

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())

    worker = cls(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )

    worker.init_model()
    worker.load_model()

    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0
    worker.init_cache_engine(cache_config)
    worker.warm_up_model()

    return worker


def create_seq_group_metadata_from_prompts(
    prompts: List[List[int]],
    num_gpu_blocks: int,
    block_size: int,
    final_seq_lens: List[int],
    continuations: Optional[List[List[int]]] = None,
    num_tokens_processed: Optional[List[int]] = None,
    seq_ids: Optional[List[int]] = None,
) -> List[SequenceGroupMetadata]:

    if continuations is None:
        continuations = [[] for _ in prompts]

    if num_tokens_processed is None:
        # Default to 1 token missing from kv cache for generation sequences.
        num_tokens_processed = []
        for continuation, prompt in zip(continuations, prompts):
            # If prefill, then default to zero tokens processed.
            if not continuation:
                num_tokens_processed.append(0)
            else:
                # If generation, then default to all but one tokens processed.
                num_tokens_processed.append(
                    len(continuation) + len(prompt) - 1)

    if seq_ids is None:
        seq_ids = list(i for i, _ in enumerate(prompts))

    free_gpu_blocks = list(range(num_gpu_blocks))

    block_allocations = {
        i: [
            free_gpu_blocks.pop()
            for _ in range(round_up_to_next_block(final_len, block_size))
        ]
        for i, final_len in enumerate(final_seq_lens)
    }

    return [
        SequenceGroupMetadata(
            request_id=str(i),
            is_prompt=len(cont_token_ids) == 0,
            seq_data={
                i:
                SequenceData(prompt_token_ids=prompt_token_ids[:] +
                             cont_token_ids[:])
            },
            sampling_params=SamplingParams(temperature=0.0, ),
            block_tables={i: block_allocations[i][:]},
        ) for i, (prompt_token_ids, cont_token_ids, num_tokens_saved) in
        enumerate(zip(prompts, continuations, num_tokens_processed))
    ]


def assert_logprobs_dict_allclose(
        actual_logprobs: List[Dict[int, float]],
        expected_logprobs: List[Dict[int, float]]) -> None:
    for single_step_actual_logprobs, single_step_expected_logprobs in zip(
            actual_logprobs, expected_logprobs):
        assert set(single_step_actual_logprobs.keys()) == set(
            single_step_expected_logprobs.keys())
        for token_id in single_step_actual_logprobs:
            actual = torch.tensor(single_step_actual_logprobs[token_id])
            expected = torch.tensor(single_step_expected_logprobs[token_id])
            assert torch.allclose(actual, expected)
