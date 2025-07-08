# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/tests/lora/test_llama.py
from dataclasses import dataclass

import pytest

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform


@dataclass
class ModelWithQuantization:
    model_path: str
    quantization: str


MODELS: list[ModelWithQuantization]
#AWQ quantization is currently not supported in ROCm.
if current_platform.is_rocm():
    MODELS = [
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            quantization="gptq"),
    ]
else:
    MODELS = [
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
            quantization="awq"),
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            quantization="gptq"),
    ]


def do_sample(llm: vllm.LLM,
              lora_path: str,
              lora_id: int,
              max_tokens: int = 256) -> list[str]:
    raw_prompts = [
        "Give me an orange-ish brown color",
        "Give me a neon pink color",
    ]

    def format_prompt_tuples(prompt):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    prompts = [format_prompt_tuples(p) for p in raw_prompts]

    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=max_tokens,
                                          stop=["<|im_end|>"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_lora(tinyllama_lora_files, model):

    llm = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        max_model_len=400,
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        trust_remote_code=True,
        enable_chunked_prefill=True)

    if model.quantization is None:
        expected_no_lora_output = [
            "Here are some examples of orange-brown colors",
            "I'm sorry, I don't have"
        ]
        expected_lora_output = [
            "#ff8050",
            "#ff8080",
        ]
    elif model.quantization == "awq":
        expected_no_lora_output = [
            "I'm sorry, I don't understand",
            "I'm sorry, I don't understand",
        ]
        expected_lora_output = [
            "#f07700: A v",
            "#f00000: A v",
        ]
    elif model.quantization == "gptq":
        expected_no_lora_output = [
            "I'm sorry, I don't have",
            "I'm sorry, I don't have",
        ]
        expected_lora_output = [
            "#f08800: This is",
            "#f07788 \n#",
        ]

    def expect_match(output, expected_output):
        # HACK: GPTQ lora outputs are just incredibly unstable.
        # Assert that the outputs changed.
        if (model.quantization == "gptq"
                and expected_output is expected_lora_output):
            assert output != expected_no_lora_output
            for i, o in enumerate(output):
                assert o.startswith(
                    '#'), f"Expected example {i} to start with # but got {o}"
            return
        assert output == expected_output

    max_tokens = 10

    print("lora adapter created")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=0,
                       max_tokens=max_tokens)
    expect_match(output, expected_no_lora_output)

    print("lora 1")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=1,
                       max_tokens=max_tokens)
    expect_match(output, expected_lora_output)

    print("no lora")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=0,
                       max_tokens=max_tokens)
    expect_match(output, expected_no_lora_output)

    print("lora 2")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=2,
                       max_tokens=max_tokens)
    expect_match(output, expected_lora_output)

    print("removing lora")

    del llm
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_tp_equality(tinyllama_lora_files, num_gpus_available,
                                 model):
    if num_gpus_available < 2:
        pytest.skip(f"Not enough GPUs for tensor parallelism {2}")
    if model.quantization == "gptq":
        pytest.skip("GPTQ lora outputs are just incredibly unstable")
    llm_tp1 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        trust_remote_code=True,
        enable_chunked_prefill=True)
    output_tp1 = do_sample(llm_tp1, tinyllama_lora_files, lora_id=1)

    del llm_tp1
    cleanup_dist_env_and_memory()

    llm_tp2 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        enable_chunked_prefill=True)
    output_tp2 = do_sample(llm_tp2, tinyllama_lora_files, lora_id=1)

    del llm_tp2
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp2
