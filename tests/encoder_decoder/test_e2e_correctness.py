"""E2E tests to verify the correctness of the encoder-decoder framework

Run `pytest tests/encoder_decoder/test_e2e_correctness.py`.
"""
from typing import List, Optional, Tuple

import pytest
from transformers import AutoModelForSeq2SeqLM

from vllm.attention.selector import (_Backend, _cached_get_attn_backend,
                                     global_force_attn_backend_context_manager)
from vllm.platforms import current_platform
from vllm.sequence import SampleLogprobs

from ..conftest import DecoderPromptType
from ..models.utils import check_logprobs_close

LIST_ENC_DEC_SUPPORTED_BACKENDS = [
    _Backend.XFORMERS, _Backend.FLASH_ATTN, None
]


def vllm_to_hf_output(
    vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
    decoder_prompt_type: DecoderPromptType,
):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "</s>"
    if decoder_prompt_type == DecoderPromptType.NONE:
        hf_output_str = "<s>" + hf_output_str

    return output_ids, hf_output_str, out_logprobs


@pytest.fixture(autouse=True)
def clear_cache():
    """Fixture to clear backend cache before each test."""
    _cached_get_attn_backend.cache_clear()  # Clear the cache
    yield  # This allows the test to run


@pytest.mark.parametrize("model", ["facebook/bart-large-cnn"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", list(DecoderPromptType))
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason="CPU backend is not currently supported with encoder/decoder models"
)
def test_encoder_decoder_e2e(
    hf_runner,
    vllm_runner,
    example_encoder_decoder_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    decoder_prompt_type: DecoderPromptType,
    enforce_eager: bool,
    attn_backend: _Backend,
) -> None:
    '''
    End-to-End (E2E) test for the encoder-decoder framework.
    This test evaluates the encoder-decoder functionality using the BART
    model. We compare the outputs of the Hugging Face and vLLM
    implementations to ensure that both implementations produce consistent
    and correct results.
    '''
    with global_force_attn_backend_context_manager(attn_backend):
        if attn_backend == _Backend.FLASH_ATTN:
            # Flash Attention works only with bfloat16 data-type
            dtype = 'bfloat16'
        test_case_prompts = example_encoder_decoder_prompts[
            decoder_prompt_type]

        # Configuration settings for HF baseline
        hf_kwargs = {
            "top_k": None,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "length_penalty": 1.0,
            "early_stopping": False,
            "no_repeat_ngram_size": None,
            "min_length": 0
        }

        with hf_runner(model, dtype=dtype,
                       auto_cls=AutoModelForSeq2SeqLM) as hf_model:
            hf_outputs = (
                hf_model.generate_encoder_decoder_greedy_logprobs_limit(
                    test_case_prompts,
                    max_tokens,
                    num_logprobs,
                    **hf_kwargs,
                ))
        with vllm_runner(model, dtype=dtype,
                         enforce_eager=enforce_eager) as vllm_model:
            vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
                test_case_prompts, max_tokens, num_logprobs)

        hf_skip_tokens = (1 if decoder_prompt_type == DecoderPromptType.NONE
                          else 0)

        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, decoder_prompt_type)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
            num_outputs_0_skip_tokens=hf_skip_tokens,
        )
