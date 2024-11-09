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


@pytest.mark.skipif(not is_quant_method_supported("gguf"),
                    reason="gguf is not supported on this GPU type.")
@pytest.mark.parametrize(("original_model", "gguf_id", "gguf_path"), [
    ("meta-llama/Llama-3.2-1B-Instruct",
     "bartowski/Llama-3.2-1B-Instruct-GGUF",
     "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("meta-llama/Llama-3.2-1B-Instruct",
     "bartowski/Llama-3.2-1B-Instruct-GGUF",
     "Llama-3.2-1B-Instruct-IQ4_XS.gguf"),
    ("Qwen/Qwen2-1.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct-GGUF",
     "qwen2-1_5b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen2-1.5B-Instruct", "legraphista/Qwen2-1.5B-Instruct-IMat-GGUF",
     "Qwen2-1.5B-Instruct.IQ4_XS.gguf"),
])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_models(
    num_gpus_available,
    vllm_runner,
    example_prompts,
    original_model,
    gguf_id,
    gguf_path,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tp_size: int,
) -> None:
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    gguf_model = hf_hub_download(gguf_id, filename=gguf_path)

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
