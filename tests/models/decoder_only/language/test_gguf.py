"""
Tests gguf models against unquantized models generations
Note: To pass the test, quantization higher than Q4 should be used
"""

import os

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from tests.quantization.utils import is_quant_method_supported

from ...utils import check_logprobs_close

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 1024

# FIXME: Move this to confest
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     hf_hub_download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                     filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     hf_hub_download("duyntnet/TinyLlama-1.1B-Chat-v1.0-imatrix-GGUF",
                     filename="TinyLlama-1.1B-Chat-v1.0-IQ4_XS.gguf")),
    ("Qwen/Qwen2-1.5B-Instruct",
     hf_hub_download("Qwen/Qwen2-1.5B-Instruct-GGUF",
                     filename="qwen2-1_5b-instruct-q4_k_m.gguf")),
    ("Qwen/Qwen2-1.5B-Instruct",
     hf_hub_download("legraphista/Qwen2-1.5B-Instruct-IMat-GGUF",
                     filename="Qwen2-1.5B-Instruct.IQ4_XS.gguf")),
]


@pytest.mark.skipif(not is_quant_method_supported("gguf"),
                    reason="gguf is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_models(
    num_gpus_available,
    vllm_runner,
    example_prompts,
    model,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    original_model, gguf_model = model

    tokenizer = AutoTokenizer.from_pretrained(original_model)
    messages = [[{
        'role': 'user',
        'content': prompt
    }] for prompt in example_prompts]
    example_prompts = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)

    # Run unquantized model.
    with vllm_runner(model_name=original_model,
                     dtype=dtype,
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=tp_size) as original_model:

        original_outputs = original_model.generate_greedy_logprobs(
            example_prompts[:-1], max_tokens, num_logprobs)

    # Run gguf model.
    with vllm_runner(model_name=gguf_model,
                     dtype=dtype,
                     max_model_len=MAX_MODEL_LEN,
                     tensor_parallel_size=tp_size) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            example_prompts[:-1], max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
        name_1="gguf",
    )
