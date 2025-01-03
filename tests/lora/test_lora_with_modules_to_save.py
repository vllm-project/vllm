"""
Compare the outputs of HF and vLLM for Llama3 with/without 
LoRA with modules_to_save.
"""
import pytest

from tests.models.utils import check_logprobs_close
from vllm.lora.request import LoRARequest

MODELS = ["AnatoliiPotapov/T-lite-instruct-0.1"]

LORAS = ["SergeyKochetkovT/llama3-lora-with-modules-to-save"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [2])
def test_llama3_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("adapter_name", LORAS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_lora_with_modules_to_save(
    peft_runner,
    vllm_runner,
    example_prompts,
    model: str,
    adapter_name: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    with vllm_runner(model,
                     dtype=dtype,
                     enable_lora=True,
                     max_loras=4,
                     max_lora_rank=32,
                     enable_lora_modules_to_save=True,
                     gpu_memory_utilization=0.5) as vllm_lora_model:
        vllm_outputs = []
        lora_request = LoRARequest('lora', 1, lora_local_path=adapter_name)

        for i in range(len(example_prompts)):
            output = vllm_lora_model.generate_greedy_logprobs(
                [example_prompts[i]],
                max_tokens,
                num_logprobs,
                lora_requests=lora_request)

            vllm_outputs.extend(output)

    with peft_runner(model, adapter_name, dtype=dtype) as peft_model:
        peft_outputs = peft_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=peft_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="peft",
        name_1="vllm_lora",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("adapter_name", LORAS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_llama3_loras_switches(
    peft_runner,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    adapter_name: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    with vllm_runner(model,
                     dtype=dtype,
                     enable_lora=True,
                     max_loras=4,
                     max_lora_rank=32,
                     enable_lora_modules_to_save=True,
                     gpu_memory_utilization=0.5) as vllm_lora_model:
        vllm_outputs = []
        for i in range(len(example_prompts)):
            lora_request = None if i % 2 == 0 else LoRARequest(
                'lora', 1, lora_local_path=adapter_name)
            output = vllm_lora_model.generate_greedy_logprobs(
                [example_prompts[i]],
                max_tokens,
                num_logprobs,
                lora_requests=lora_request)

            vllm_outputs.extend(output)

    with peft_runner(model, adapter_name, dtype=dtype) as peft_model:
        peft_outputs = peft_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    peft_hf_outputs = [
        hf_outputs[i] if i % 2 == 0 else peft_outputs[i]
        for i in range(len(example_prompts))
    ]
    check_logprobs_close(
        outputs_0_lst=peft_hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="peft_hf",
        name_1="vllm_lora",
    )
