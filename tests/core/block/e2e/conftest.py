import contextlib
import gc

import pytest
import ray
import torch

from vllm import LLM
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel)
from vllm.model_executor.utils import set_random_seed


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


@pytest.fixture
def baseline_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                           baseline_llm_kwargs, seed):
    return create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)


@pytest.fixture
def test_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                       test_llm_kwargs, seed):
    return create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                                test_llm_kwargs, seed)


def create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                         distinct_llm_kwargs, seed):
    kwargs = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **distinct_llm_kwargs,
    }

    def generator_inner():
        llm = LLM(**kwargs)

        set_random_seed(seed)

        yield llm
        del llm
        cleanup()

    for llm in generator_inner():
        yield llm
        del llm
