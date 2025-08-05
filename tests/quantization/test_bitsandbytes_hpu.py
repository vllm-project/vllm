# SPDX-License-Identifier: Apache-2.0
"""Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
"""

import gc

import pytest
import torch
from transformers import BitsAndBytesConfig

from tests.quantization.utils import is_quant_method_supported

from ..utils import compare_two_settings, create_new_process_for_each_test

models_4bit_to_test = [
    (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "quantize_inflight_model_with_both_HF_and_Mistral_format_weights",
    ),
    ("meta-llama/Llama-3.2-1B", "quantize_llama_model_inflight"),
]

models_pre_quant_4bit_to_test = [("unsloth/Llama-3.2-1B-bnb-4bit",
                                  "read_pre-quantized_4-bit_NF4_model")]


@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@create_new_process_for_each_test()
def test_load_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                             model_name, description) -> None:
    hf_model_kwargs = dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ))
    validate_generated_texts(
        hf_runner,
        vllm_runner,
        example_prompts[:1],
        model_name,
        False,
        hf_model_kwargs,
    )


@pytest.mark.parametrize("model_name, description",
                         models_pre_quant_4bit_to_test)
@create_new_process_for_each_test()
def test_load_pre_quant_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                                       model_name, description) -> None:
    validate_generated_texts(hf_runner, vllm_runner, example_prompts[:1],
                             model_name, True)


@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@create_new_process_for_each_test()
def test_load_tp_4bit_bnb_model(hf_runner, vllm_runner, example_prompts,
                                model_name, description) -> None:
    hf_model_kwargs = dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ))
    validate_generated_texts(
        hf_runner,
        vllm_runner,
        example_prompts[:1],
        model_name,
        False,
        hf_model_kwargs,
        vllm_tp_size=2,
    )


@pytest.mark.skipif(not is_quant_method_supported("bitsandbytes"),
                    reason='bitsandbytes is not supported on this GPU type.')
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@create_new_process_for_each_test()
def test_load_pp_4bit_bnb_model(model_name, description) -> None:
    common_args = [
        "--disable-log-stats",
        "--disable-log-requests",
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        "--quantization",
        "bitsandbytes",
        "--gpu-memory-utilization",
        "0.7",
    ]
    pp_args = [
        *common_args,
        "--pipeline-parallel-size",
        "2",
    ]
    compare_two_settings(model_name, common_args, pp_args)


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


def validate_generated_texts(
    hf_runner,
    vllm_runner,
    prompts,
    model_name,
    pre_quant=False,
    hf_model_kwargs=None,
    vllm_tp_size=1,
):
    # NOTE: run vLLM first, as it requires a clean process
    # when using distributed inference
    with vllm_runner(
            model_name,
            quantization=None if pre_quant else "bitsandbytes",
            tensor_parallel_size=vllm_tp_size,
            enforce_eager=False,
    ) as llm:
        vllm_outputs = llm.generate_greedy(prompts, 8)
        vllm_logs = log_generated_texts(prompts, vllm_outputs, "VllmRunner")

    # Clean up the GPU memory for the next test
    gc.collect()

    if hf_model_kwargs is None:
        hf_model_kwargs = {}

    # Run with HF runner
    with hf_runner(model_name, model_kwargs=hf_model_kwargs) as llm:
        hf_outputs = llm.generate_greedy(prompts, 8)
        hf_logs = log_generated_texts(prompts, hf_outputs, "HfRunner")

    # Clean up the GPU memory for the next test
    gc.collect()

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
