"""Compare the outputs of HF and vLLM for BART models using greedy sampling.

Run `pytest tests/models/test_bart.py`.
"""
import pytest

from tests.kernels.utils import override_backend_env_variable
from vllm.utils import STR_XFORMERS_ATTN_VAL

from .utils import check_logprobs_close, check_logprobs_close_encoder_decoder

MODELS = ["facebook/bart-base","facebook/bart-large-cnn"]

# Backends under test
#
# Currently only XFormers is supported
BACKEND_NAMES = [STR_XFORMERS_ATTN_VAL]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float","bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
def test_models(
    hf_runner,
    vllm_runner,
    example_encoder_decoder_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    backend_name: str,
    monkeypatch,
) -> None:
    # TODO(sang): Sliding window should be tested separately.

    # Force Attention wrapper backend
    override_backend_env_variable(monkeypatch, backend_name)

    with hf_runner(model, dtype=dtype,
                   is_encoder_decoder_model=True) as hf_model:
        hf_outputs = hf_model.generate_encoder_decoder_greedy_logprobs_limit(
            example_encoder_decoder_prompts, max_tokens, num_logprobs)

        decoder_input_ids_list = [hf_model.tokenizer(decoder_prompt,
                                    return_tensors="pt").input_ids 
                                        for decoder_prompt in example_encoder_decoder_prompts[1]]

    with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
            example_encoder_decoder_prompts, max_tokens, num_logprobs)

    # print(hf_outputs)
    # print("\n\n\n\n\n")
    # print(vllm_outputs)

    check_logprobs_close_encoder_decoder(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        decoder_input_ids_list=decoder_input_ids_list,
        name_0="hf",
        name_1="vllm"
    )
