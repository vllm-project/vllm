from itertools import count
from typing import Dict, Iterable, List, Optional, Union
from unittest.mock import MagicMock

import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.utils import set_random_seed
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, SamplerOutput, SequenceData,
                           SequenceGroupMetadata, SequenceGroupOutput,
                           SequenceOutput)
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker


def round_up_to_next_block(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def mock_worker(cls=None,
                vocab_size: int = 30_000,
                max_model_len: int = 2048,
                rank: int = 0,
                use_spec: bool = True) -> MagicMock:
    if cls is None:
        cls = Worker

    spec = cls if use_spec else None

    worker = MagicMock(spec=spec)
    worker.vocab_size = vocab_size
    worker.max_model_len = max_model_len
    worker.rank = rank
    worker.device = 'cuda:0'
    return worker


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
    engine_config = engine_args.create_engine_config()

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())

    worker = cls(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )

    worker.init_device()
    worker.load_model()

    engine_config.cache_config.num_gpu_blocks = num_gpu_blocks
    engine_config.cache_config.num_cpu_blocks = 0
    worker.initialize_cache(
        num_gpu_blocks=engine_config.cache_config.num_gpu_blocks,
        num_cpu_blocks=engine_config.cache_config.num_cpu_blocks)

    return worker


def create_seq_group_metadata_from_prompts(
    prompts: List[List[int]],
    num_gpu_blocks: int,
    block_size: int,
    final_prompt_lens: List[int],
    continuations: Optional[List[List[int]]] = None,
    seq_ids: Optional[List[int]] = None,
) -> List[SequenceGroupMetadata]:

    if continuations is None:
        continuations = [[] for _ in prompts]

    if seq_ids is None:
        seq_ids = list(i for i, _ in enumerate(prompts))

    free_gpu_blocks = list(range(num_gpu_blocks))

    block_allocations = {
        i: [
            free_gpu_blocks.pop()
            for _ in range(round_up_to_next_block(final_len, block_size))
        ]
        for i, final_len in enumerate(final_prompt_lens)
    }

    return [
        SequenceGroupMetadata(
            request_id=str(i),
            is_prompt=len(cont_token_ids) == 0,
            seq_data={
                i:
                SequenceData(
                    prompt_token_ids=prompt_token_ids[:],
                    output_token_ids=cont_token_ids[:],
                ),
            },
            sampling_params=SamplingParams(temperature=0.0, ),
            block_tables={i: block_allocations[i][:]},
        ) for i, (prompt_token_ids,
                  cont_token_ids) in enumerate(zip(prompts, continuations))
    ]


def assert_logprobs_dict_allclose(
        actual_logprobs: List[Dict[int, Logprob]],
        expected_logprobs: List[Dict[int, Logprob]]) -> None:
    for single_step_actual_logprobs, single_step_expected_logprobs in zip(
            actual_logprobs, expected_logprobs):
        assert set(single_step_actual_logprobs.keys()) == set(
            single_step_expected_logprobs.keys())
        for token_id in single_step_actual_logprobs:
            actual = torch.tensor(
                single_step_actual_logprobs[token_id].logprob)
            expected = torch.tensor(
                single_step_expected_logprobs[token_id].logprob)
            assert torch.allclose(actual, expected)


def create_sampler_output_list(
        token_ids: torch.Tensor,
        probs: Iterable[Optional[torch.Tensor]],
        logprobs: Iterable[Optional[torch.Tensor]],
        seq_ids: Optional[List[int]] = None) -> List[SamplerOutput]:
    num_steps, batch_size = token_ids.shape
    token_ids_by_step = token_ids.tolist()

    if seq_ids is None:
        seq_ids = list(range(batch_size))

    return [
        SamplerOutput(outputs=[
            SequenceGroupOutput(
                samples=[
                    SequenceOutput(
                        output_token=token_id,
                        parent_seq_id=seq_ids[seq_index],
                        logprobs={token_id: Logprob(0)},
                    )
                ],
                prompt_logprobs=None,
            ) for seq_index, token_id in enumerate(token_ids_by_step[step])
        ],
                      sampled_token_probs=probs[step],
                      logprobs=logprobs[step],
                      sampled_token_ids=token_ids[step])
        for step in range(num_steps)
    ]


def create_batch(batch_size,
                 k,
                 prompt_len: Union[int, List[int]] = 10,
                 prev_output_token_len: int = 10,
                 seq_ids: Optional[List[int]] = None,
                 num_gpu_blocks: Optional[int] = None,
                 block_size: Optional[int] = None):
    if block_size is None:
        block_size = 8

    if num_gpu_blocks is None:
        num_gpu_blocks = 2048 // block_size

    iterator = count()

    if isinstance(prompt_len, int):
        prompt_lens = [prompt_len for _ in range(batch_size)]
    else:
        prompt_lens = prompt_len

    prompts = [[next(iterator) for _ in range(p_len)] for p_len in prompt_lens]
    prev_output_tokens = [[
        next(iterator) for _ in range(prev_output_token_len)
    ] for _ in range(batch_size)]
    final_prompt_lens = [
        len(prompt) + len(prev_output_token) + k + 1
        for prompt, prev_output_token in zip(prompts, prev_output_tokens)
    ]

    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts, num_gpu_blocks, block_size, final_prompt_lens,
        prev_output_tokens, seq_ids)
    return seq_group_metadata_list, prompts, prev_output_tokens
