
import torch
from typing import List, Optional, Dict, Iterable
from unittest.mock import MagicMock
from itertools import count

from vllm.worker.worker import Worker
from vllm.worker.base_worker import BaseWorker
from vllm.engine.ray_utils import initialize_cluster
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelData, SequenceGroupMetadata, SequenceData, SamplerOutput, SequenceGroupOutputs, SequenceOutputs
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from tests.utils import round_up_to_next_block


def create_execute_model_data(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    finished_request_ids_list: Optional[List[str]] = None,
    blocks_to_swap_in: Optional[Dict[int, int]] = None,
    blocks_to_swap_out: Optional[Dict[int, int]] = None,
    blocks_to_copy: Optional[Dict[int, int]] = None,
    num_preallocated_slots: Optional[int] = None,
) -> ExecuteModelData:

    if finished_request_ids_list is None:
        finished_request_ids_list = []
    if blocks_to_swap_in is None:
        blocks_to_swap_in = {}
    if blocks_to_swap_out is None:
        blocks_to_swap_out = {}
    if blocks_to_copy is None:
        blocks_to_copy = {}
    if num_preallocated_slots is None:
        num_preallocated_slots = 0

    return ExecuteModelData(
        seq_group_metadata_list=seq_group_metadata_list,
        finished_request_ids_list=finished_request_ids_list,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        num_preallocated_slots=num_preallocated_slots,
    )


def mock_worker(vocab_size: int = 30_000,
                max_model_len: int = 2048) -> MagicMock:
    worker = MagicMock()
    worker.model.config.vocab_size = vocab_size
    worker.model_config.max_model_len = max_model_len
    return worker


def patch_execute_model_with_seeds(worker: BaseWorker, rand_seeds: List[int]):
    seed_iter = iter(rand_seeds)
    original_execute_model = worker.execute_model

    def new_execute_model(execute_model_data):
        result = original_execute_model(execute_model_data)
        set_random_seed(next(seed_iter))
        return result

    return new_execute_model


def zero_kv_cache(cache_engine: CacheEngine):
    assert cache_engine.gpu_cache
    for key_blocks, value_blocks in cache_engine.gpu_cache:
        key_blocks.zero_()
        value_blocks.zero_()


def create_worker(cls: type, model_name: str, block_size: int,
                  num_gpu_blocks: int, seed: int):
    engine_args = EngineArgs(
        model=model_name,
        seed=seed,
        block_size=block_size,
    )

    (model_config, cache_config, parallel_config, scheduler_config, _, _,
     _) = engine_args.create_engine_configs()

    distributed_init_method, _ = initialize_cluster(parallel_config)

    worker = cls(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        rank=0,
        distributed_init_method=distributed_init_method,
    )

    worker.init_model()

    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0
    worker.init_cache_engine(cache_config)

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
            is_chunked_prefill=False,
            seq_data={
                i:
                SequenceData(token_ids=prompt_token_ids[:] + cont_token_ids[:],
                             num_prompt_tokens=len(prompt_token_ids[:]),
                             num_processed_token_ids=num_tokens_saved,
                             prefill_start=0,
                             prefill_end=len(prompt_token_ids)),
            },
            sampling_params=SamplingParams(temperature=0.0, ),
            block_tables={i: block_allocations[i][:]},
            lora_request=None,
        ) for i, (prompt_token_ids, cont_token_ids, num_tokens_saved) in
        enumerate(zip(prompts, continuations, num_tokens_processed))
    ]


def create_workers(test_type: type,
                   reference_type: type = Worker,
                   seed: int = 100,
                   block_size: int = 32,
                   num_gpu_blocks: int = 2048 // 32,
                   model_name: str = 'JackFram/llama-68m'):
    test_worker = create_worker(
        test_type,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    reference_worker = create_worker(
        reference_type,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    return test_worker, reference_worker


def get_output_tokens(outputs):
    return [output[0].output_token for output in outputs.outputs]


def get_output_logprobs(outputs):
    return [output[0].logprobs for output in outputs.outputs]


def assert_logprobs_dict_allclose(
        actual_logprobs: List[Dict[int, float]],
        expected_logprobs: List[Dict[int, float]]) -> None:
    for single_step_actual_logprobs, single_step_expected_logprobs in zip(
            actual_logprobs, expected_logprobs):
        assert set(single_step_actual_logprobs.keys()) == set(
            single_step_expected_logprobs.keys())
        for token_id in single_step_actual_logprobs.keys():
            actual = torch.tensor(single_step_actual_logprobs[token_id])
            expected = torch.tensor(single_step_expected_logprobs[token_id])
            assert torch.allclose(actual, expected)


def create_sampler_output_list(
        token_ids: torch.Tensor,
        probs: Iterable[Optional[torch.Tensor]],
        seq_ids: Optional[List[int]] = None) -> List[SamplerOutput]:
    num_steps, batch_size = token_ids.shape
    token_ids_by_step = token_ids.tolist()

    if seq_ids is None:
        seq_ids = list(range(batch_size))

    return [
        SamplerOutput(outputs=[
            SequenceGroupOutputs(
                samples=[
                    SequenceOutputs(
                        output_token=token_id,
                        parent_seq_id=seq_ids[seq_index],
                        logprobs={token_id: 0},
                    )
                ],
                prompt_logprobs=None,
            ) for seq_index, token_id in enumerate(token_ids_by_step[step])
        ],
                      probs=probs[step],
                      sampled_tokens=token_ids[step])
        for step in range(num_steps)
    ]


def create_batch(batch_size,
                 k,
                 prompt_len: int = 10,
                 prev_output_token_len: int = 10,
                 seq_ids: Optional[List[int]] = None):
    block_size = 8
    num_gpu_blocks = 2048 // block_size
    iterator = count()
    prompts = [[next(iterator) for _ in range(prompt_len)]
               for _ in range(batch_size)]
    prev_output_tokens = [[
        next(iterator) for _ in range(prev_output_token_len)
    ] for _ in range(batch_size)]
    final_seq_lens = [
        len(prompt) + len(prev_output_token) + k + 1
        for prompt, prev_output_token in zip(prompts, prev_output_tokens)
    ]
    execute_model_data = create_execute_model_data(
        create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks,
                                               block_size, final_seq_lens,
                                               prev_output_tokens, seq_ids),
        num_preallocated_slots=k)
    return execute_model_data, prompts, prev_output_tokens
