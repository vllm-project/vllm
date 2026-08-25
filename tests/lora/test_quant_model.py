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

from ..models.utils import check_logprobs_close


@dataclass
class ModelWithQuantization:
    model_path: str
    quantization: str


MODELS: list[ModelWithQuantization]
# AWQ quantization is currently not supported in ROCm.
if current_platform.is_rocm():
    MODELS = [
        ModelWithQuantization(
            model_path="JunHowie/Qwen3-0.6B-GPTQ-Int4",
            quantization="gptq",
        ),
    ]
else:
    MODELS = [
        ModelWithQuantization(
            model_path="Orion-zhen/Qwen3-0.6B-AWQ", quantization="awq"
        ),
        ModelWithQuantization(
            model_path="JunHowie/Qwen3-0.6B-GPTQ-Int4",
            quantization="gptq",
        ),
    ]


def do_sample(
    llm: vllm.LLM, lora_path: str | None, lora_id: int, max_tokens: int = 256
) -> list[str]:
    messages = [
        [
            {
                "role": "system",
                "content": "Follow the instructions to make animal noises",
            },
            {"role": "user", "content": "Make your favorite animal noise."},
        ],
        [
            {
                "role": "system",
                "content": "You are a cat. Reply only with your sound.",
            },
            {"role": "user", "content": "What do you say?"},
        ],
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=(
            LoRARequest(str(lora_id), lora_id, lora_path) if lora_path else None
        ),
        use_tqdm=False,
    )
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_lora(qwen3_meowing_lora_files, qwen3_woofing_lora_files, model):
    llm = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        max_model_len=400,
        gpu_memory_utilization=0.2,  # avoid OOM
        quantization=model.quantization,
        enable_chunked_prefill=True,
    )

    loras = [
        (
            qwen3_meowing_lora_files,
            1,
            ["Meow Meow Meow Meow Meow"] * 2,
        ),
        (
            qwen3_woofing_lora_files,
            2,
            ["Woof Woof Woof Woof Woof"] * 2,
        ),
    ]
    max_tokens = 10
    try:
        base_output = do_sample(llm, None, lora_id=0, max_tokens=max_tokens)
        for lora_path, lora_id, expected_output in loras:
            output = do_sample(
                llm,
                lora_path,
                lora_id=lora_id,
                max_tokens=max_tokens,
            )
            assert output == expected_output
            assert output != base_output
    finally:
        del llm
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_tp_equality(qwen3_meowing_lora_files, num_gpus_available, model):
    if num_gpus_available < 2:
        pytest.skip(f"Not enough GPUs for tensor parallelism {2}")
    if model.quantization == "gptq":
        pytest.skip("GPTQ lora outputs are just incredibly unstable")
    num_logprobs = 5
    prompts = [
        "<|im_start|>user\nGive me an orange-ish brown color<|im_end|>\n"
        "<|im_start|>assistant\n",
        "<|im_start|>user\nGive me a neon pink color<|im_end|>\n"
        "<|im_start|>assistant\n",
    ]
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["<|im_end|>"],
        logprobs=num_logprobs,
    )
    lora_request = LoRARequest("1", 1, qwen3_meowing_lora_files)
    llm_tp1 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        max_model_len=400,
        gpu_memory_utilization=0.2,  # avoid OOM
        quantization=model.quantization,
        enable_chunked_prefill=True,
    )
    try:
        outputs_tp1 = llm_tp1.generate(
            prompts, sampling_params, lora_request=lora_request
        )
        output_tp1 = [
            (o.outputs[0].token_ids, o.outputs[0].text, o.outputs[0].logprobs)
            for o in outputs_tp1
        ]
    finally:
        del llm_tp1
        cleanup_dist_env_and_memory()

    llm_tp2 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=2,
        max_model_len=400,
        gpu_memory_utilization=0.2,  # avoid OOM
        quantization=model.quantization,
        enable_chunked_prefill=True,
    )
    try:
        outputs_tp2 = llm_tp2.generate(
            prompts, sampling_params, lora_request=lora_request
        )
        output_tp2 = [
            (o.outputs[0].token_ids, o.outputs[0].text, o.outputs[0].logprobs)
            for o in outputs_tp2
        ]
    finally:
        del llm_tp2
        cleanup_dist_env_and_memory()

    check_logprobs_close(
        outputs_0_lst=output_tp1,
        outputs_1_lst=output_tp2,
        name_0="tp1",
        name_1="tp2",
    )
