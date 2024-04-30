import asyncio
from typing import List, Optional, Tuple, Union

import pytest
import ray

from tests.conftest import cleanup
from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.model_executor.utils import set_random_seed
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, random_uuid


class AsyncLLM:
    """AsyncLLM

    Note: Current LLM class in vllm don't support async mode, for test purpose,
    we implement async one in here. Maybe we could move to
    vllm/entrypoints/llm.py in future.

    Below AsyncLLM is directly borrow from vllm/entrypoints/llm.py with changes
    to make to work in async mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        self.engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            engine_use_ray=True,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.request_counter = Counter()

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:

        llm_engine = AsyncLLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.LLM_CLASS)

        if prompts is None:
            raise ValueError("prompts must be provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        if prompts is not None:
            num_requests = len(prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and "
                             "sampling_params must be the same.")

        async def get_output(prompt, sampling_param) -> str:
            request_id = random_uuid()
            results_generator = llm_engine.generate(prompt, sampling_param,
                                                    request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            return final_output

        outputs = []
        try:
            for i in range(num_requests):
                prompt = prompts[i] if prompts is not None else None
                res = asyncio.run(get_output(prompt, sampling_params))
                outputs.append(res)
        finally:
            ray.shutdown()
        return outputs


@pytest.fixture
def baseline_llm_generator(request, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           seed):
    return create_llm_generator("baseline", request, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)


@pytest.fixture
def test_llm_generator(request, common_llm_kwargs, per_test_common_llm_kwargs,
                       test_llm_kwargs, seed):
    return create_llm_generator("test", request, common_llm_kwargs,
                                per_test_common_llm_kwargs, test_llm_kwargs,
                                seed)


def create_llm_generator(baseline_or_test, request, common_llm_kwargs,
                         per_test_common_llm_kwargs, distinct_llm_kwargs,
                         seed):
    kwargs = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **distinct_llm_kwargs,
    }
    test_name = request.node.name

    def generator_inner():
        print(f'Creating {baseline_or_test=} LLM for {test_name=}. {kwargs=}')

        use_async = False
        if "use_async" in kwargs:
            use_async = kwargs.pop("use_async")

        llm = AsyncLLM(**kwargs) if use_async else LLM(**kwargs)
        set_random_seed(seed)

        yield llm
        del llm
        cleanup()

    def generator_outer():
        for llm in generator_inner():
            yield llm
            del llm

    return generator_outer


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]]]:
    tokens = []
    token_ids = []
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]
        del llm

    return tokens, token_ids
