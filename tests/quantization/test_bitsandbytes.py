'''Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
'''

import gc

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported

from ..utils import fork_new_process_for_each_test

models_4bit_to_test = [
    ('huggyllama/llama-7b', 'quantize model inflight'),
]

models_pre_qaunt_4bit_to_test = [
    ('lllyasviel/omost-llama-3-8b-4bits',
     'read pre-quantized 4-bit NF4 model'),
    ('PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed',
     'read pre-quantized 4-bit FP4 model'),
]

models_pre_quant_8bit_to_test = [
    ('meta-llama/Llama-Guard-3-8B-INT8', 'read pre-quantized 8-bit model'),
]


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@fork_new_process_for_each_test
def test_load_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                             model_name, description) -> None:

    hf_model_kwargs = {"load_in_4bit": True}
    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name, hf_model_kwargs)


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_qaunt_4bit_to_test)
@fork_new_process_for_each_test
def test_load_pre_quant_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                                       model_name, description) -> None:

    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name)


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description",
                         models_pre_quant_8bit_to_test)
@fork_new_process_for_each_test
def test_load_8bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                             model_name, description) -> None:

    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name)


def log_generated_texts(prompts, outputs, runner_name):
    logged_texts = []
    for i, (_, generated_text) in enumerate(outputs):
        log_entry = {
            "prompt": prompts[i],
            "runner_name": runner_name,
            "generated_text": generated_text,
        }
        logged_texts.append(log_entry)
    return logged_texts


def validate_generated_texts(hf_runner,
                             vllm_runner,
                             prompts,
                             model_name,
                             hf_model_kwargs=None):

    # NOTE: run vLLM first, as it requires a clean process
    # when using distributed inference

    #Run with vLLM runner
    with vllm_runner(model_name,
                     quantization='bitsandbytes',
                     load_format='bitsandbytes',
                     enforce_eager=True,
                     gpu_memory_utilization=0.8) as llm:
        vllm_outputs = llm.generate_greedy(prompts, 8)
        vllm_logs = log_generated_texts(prompts, vllm_outputs, "VllmRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    if hf_model_kwargs is None:
        hf_model_kwargs = {}

    # Run with HF runner
    with hf_runner(model_name, model_kwargs=hf_model_kwargs) as llm:
        hf_outputs = llm.generate_greedy(prompts, 8)
        hf_logs = log_generated_texts(prompts, hf_outputs, "HfRunner")

    # Clean up the GPU memory for the next test
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Compare the generated strings
    for hf_log, vllm_log in zip(hf_logs, vllm_logs):
        hf_str = hf_log["generated_text"]
        vllm_str = vllm_log["generated_text"]
        prompt = hf_log["prompt"]
        assert hf_str == vllm_str, (f"Model: {model_name}"
                                    f"Mismatch between HF and vLLM outputs:\n"
                                    f"Prompt: {prompt}\n"
                                    f"HF Output: '{hf_str}'\n"
                                    f"vLLM Output: '{vllm_str}'")
