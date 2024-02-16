import pytest
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams

TENSOR_PARALLEL_SIZE = 2
MAX_GENERATION_TOKENS = 256


def llm_generate(result_queue, prompt_token_ids, worker_use_ray=False):
    try:
        llm = LLM(model="facebook/opt-350m",
                  tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                  worker_use_ray=worker_use_ray)

        output = llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(max_tokens=MAX_GENERATION_TOKENS))
    except BaseException as e:
        output = e

    result_queue.put(output)


def run_llm(prompt_token_ids, worker_use_ray=False):
    result_queue = mp.Queue()
    proc = mp.Process(target=llm_generate,
                      args=(result_queue, prompt_token_ids, worker_use_ray))
    proc.start()
    result = result_queue.get()
    proc.join()
    if isinstance(result, BaseException):
        raise result
    return result


def get_prompts():
    # https://github.com/vllm-project/vllm/issues/367#issuecomment-1629872996
    batch_size = 32
    dim = 120
    max_token_id = 32000
    torch.manual_seed(42)
    batch = torch.randint(max_token_id, (batch_size, dim))
    prompt_token_ids = [tokens.tolist() for tokens in batch]
    return prompt_token_ids


@pytest.mark.skip("Requires multiple GPUs")
def test_local_worker():
    # Similar to tests/lora/test_llama.py
    # Cannot use as it will initialize torch.cuda too early...
    # if torch.cuda.device_count() < 2:
    #     pytest.skip(f"Not enough GPUs for tensor parallelism {2}")

    prompt_token_ids = get_prompts()
    output1 = run_llm(prompt_token_ids, worker_use_ray=False)
    output2 = run_llm(prompt_token_ids, worker_use_ray=True)
    assert len(output1) == len(output2)

    completion_token_ids1 = [item.outputs[0].token_ids for item in output1]
    completion_token_ids2 = [item.outputs[0].token_ids for item in output2]
    assert completion_token_ids1 == completion_token_ids2


if __name__ == "__main__":
    test_local_worker()
