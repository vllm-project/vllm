# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether bitsandbytes computation is enabled correctly.

Run `pytest tests/quantization/test_bitsandbytes.py`.
"""

import pytest
from transformers import BitsAndBytesConfig

from tests.quantization.utils import is_quant_method_supported
from vllm.platforms import current_platform

from ...utils import compare_two_settings, multi_gpu_test
from ..utils import check_embeddings_close, check_logprobs_close

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx9

    pytestmark = pytest.mark.skipif(
        on_gfx9(),
        reason="bitsandbytes not supported on gfx9 (warp size 64 limitation)",
    )

models_4bit_to_test = [
    ("facebook/opt-125m", "quantize opt model inflight"),
    (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "quantize inflight model with both HF and Mistral format weights",
    ),
]

models_4bit_to_embedding_test = [
    ("intfloat/e5-mistral-7b-instruct", "quantize embedding model inflight"),
]

models_4bit_to_moe_test = [
    ("allenai/OLMoE-1B-7B-0125-Instruct", "quantize moe model inflight"),
]

models_pre_qaunt_4bit_to_test = [
    (
        "PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed",
        "read pre-quantized 4-bit FP4 model",
    ),
    ("poedator/opt-125m-bnb-4bit", "read pre-quantized 4-bit NF4 opt model"),
]

models_pre_quant_8bit_to_test = [
    ("meta-llama/Llama-Guard-3-8B-INT8", "read pre-quantized llama 8-bit model"),
    ("yec019/fbopt-350m-8bit", "read pre-quantized 8-bit opt model"),
]


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
def test_load_4bit_bnb_model(
    hf_runner, vllm_runner, example_prompts, model_name, description
) -> None:
    hf_model_kwargs = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    validate_generated_texts(
        hf_runner, vllm_runner, example_prompts[:1], model_name, False, hf_model_kwargs
    )


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_pre_qaunt_4bit_to_test)
def test_load_pre_quant_4bit_bnb_model(
    hf_runner, vllm_runner, example_prompts, model_name, description
) -> None:
    validate_generated_texts(
        hf_runner, vllm_runner, example_prompts[:1], model_name, True
    )


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_pre_quant_8bit_to_test)
def test_load_8bit_bnb_model(
    hf_runner, vllm_runner, example_prompts, model_name, description
) -> None:
    validate_generated_texts(
        hf_runner, vllm_runner, example_prompts[:1], model_name, True
    )


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@multi_gpu_test(num_gpus=2)
def test_load_tp_4bit_bnb_model(
    hf_runner, vllm_runner, example_prompts, model_name, description
) -> None:
    hf_model_kwargs = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    validate_generated_texts(
        hf_runner,
        vllm_runner,
        example_prompts[:1],
        model_name,
        False,
        hf_model_kwargs,
        vllm_tp_size=2,
    )


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_4bit_to_test)
@multi_gpu_test(num_gpus=2)
def test_load_pp_4bit_bnb_model(model_name, description) -> None:
    common_args = [
        "--disable-log-stats",
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


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_4bit_to_moe_test)
def test_4bit_bnb_moe_model(
    hf_runner, vllm_runner, example_prompts, model_name, description
) -> None:
    hf_model_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    )
    with vllm_runner(
        model_name,
        quantization="bitsandbytes",
        enforce_eager=False,
        default_torch_num_threads=1,
    ) as llm:
        vllm_outputs = llm.generate_greedy_logprobs(
            example_prompts, max_tokens=32, num_logprobs=5
        )

    with hf_runner(
        model_name, model_kwargs=hf_model_kwargs, default_torch_num_threads=1
    ) as llm:
        transformers_outputs = llm.generate_greedy_logprobs_limit(
            example_prompts, max_tokens=32, num_logprobs=5
        )
    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )


@pytest.mark.skipif(
    not is_quant_method_supported("bitsandbytes"),
    reason="bitsandbytes is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_name, description", models_4bit_to_embedding_test)
@pytest.mark.parametrize("dtype", ["half"])
def test_4bit_bnb_embedding_model(
    model_name,
    description,
    hf_runner,
    vllm_runner,
    example_prompts,
    dtype: str,
) -> None:
    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    # Inflight 4bit quantization
    with vllm_runner(
        model_name,
        runner="pooling",
        dtype=dtype,
        gpu_memory_utilization=0.5,
        quantization="bitsandbytes",
        default_torch_num_threads=1,
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(example_prompts)

    hf_model_kwargs = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    with hf_runner(
        model_name,
        dtype=dtype,
        model_kwargs=hf_model_kwargs,
        is_sentence_transformer=True,
        default_torch_num_threads=1,
    ) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=5e-2,
    )


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
    max_tokens=8,
):
    # NOTE: run vLLM first, as it requires a clean process
    # when using distributed inference
    with vllm_runner(
        model_name,
        quantization=None if pre_quant else "bitsandbytes",
        tensor_parallel_size=vllm_tp_size,
        enforce_eager=False,
        default_torch_num_threads=1,
        tokenizer_mode="hf",
        load_format="hf",
        config_format="hf",
    ) as llm:
        vllm_outputs = llm.generate_greedy(prompts, max_tokens)
        vllm_logs = log_generated_texts(prompts, vllm_outputs, "VllmRunner")

    if hf_model_kwargs is None:
        hf_model_kwargs = {}

    # Run with HF runner
    with hf_runner(
        model_name, model_kwargs=hf_model_kwargs, default_torch_num_threads=1
    ) as llm:
        hf_outputs = llm.generate_greedy(prompts, max_tokens)
        hf_logs = log_generated_texts(prompts, hf_outputs, "HfRunner")

    # Compare the generated strings
    for hf_log, vllm_log in zip(hf_logs, vllm_logs):
        hf_str = hf_log["generated_text"]
        vllm_str = vllm_log["generated_text"]
        prompt = hf_log["prompt"]
        assert hf_str == vllm_str, (
            f"Model: {model_name}"
            f"Mismatch between HF and vLLM outputs:\n"
            f"Prompt: {prompt}\n"
            f"HF Output: '{hf_str}'\n"
            f"vLLM Output: '{vllm_str}'"
        )
