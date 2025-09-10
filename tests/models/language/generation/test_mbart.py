# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import pytest
from transformers import AutoModelForSeq2SeqLM

from vllm.sequence import SampleLogprobs

from ....conftest import DecoderPromptType, HfRunner, VllmRunner
from ...utils import check_logprobs_close


def vllm_to_hf_output(
    vllm_output: tuple[list[int], str, Optional[SampleLogprobs]],
    decoder_prompt_type: DecoderPromptType,
):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output
    hf_output_str = output_str + "</s>"
    return output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    prompts: list[dict[str, str]],
    decoder_prompt_type: DecoderPromptType,
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    '''
    Test the vLLM mBART model by validating it against HuggingFace (HF).
    (Docstring content is omitted for brevity)
    '''

    vllm_prompts = prompts
    if decoder_prompt_type == DecoderPromptType.NONE:
        vllm_prompts = [{
            "encoder_prompt": p['encoder_prompt'],
            "decoder_prompt": ""
        } for p in prompts]

    vllm_kwargs = {
        "hf_overrides": {
            "architectures": ["MBartForConditionalGeneration"]
        }
    }

    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     **vllm_kwargs) as vllm_model:  # type: ignore
        vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
            vllm_prompts, max_tokens, num_logprobs)

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
        hf_kwargs["decoder_start_token_id"] = (
            hf_model.tokenizer.lang_code_to_id["ro_RO"])

        hf_outputs = (
            hf_model.generate_encoder_decoder_greedy_logprobs_limit(
                prompts,  # HF runner still uses the original prompts
                max_tokens,
                num_logprobs,
                **hf_kwargs,
            ))

    hf_skip_tokens = 0

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


@pytest.mark.parametrize(
    "model",
    [pytest.param("facebook/mbart-large-en-ro")],
)
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", list(DecoderPromptType))
def test_models(hf_runner, vllm_runner, example_encoder_decoder_prompts, model,
                dtype, max_tokens, num_logprobs, decoder_prompt_type) -> None:

    run_test(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts[decoder_prompt_type],
        decoder_prompt_type,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
