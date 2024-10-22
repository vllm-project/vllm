"""Compare the outputs of HF and vLLM for BART models using greedy sampling.

Run `pytest tests/models/encoder_decoder/language/test_bart.py`.
"""
from typing import List, Optional, Tuple, Type

import pytest

from vllm.inputs.data import ExplicitEncoderDecoderPrompt
from vllm.sequence import SampleLogprobs
from PIL import Image

from ....conftest import (HfRunner, VllmRunner)
from ...utils import check_logprobs_close


MODELS = ["/data/LLM-model/Florence-2-base/"]
TOKENIZER = "/data/LLM-model/bart-base"
PROMPTS = [
    ExplicitEncoderDecoderPrompt(encoder_prompt="<CAPTION>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<DETAILED_CAPTION>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<MORE_DETAILED_CAPTION>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<CAPTION_TO_PHRASE_GROUNDING>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<DENSE_REGION_CAPTION>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<REGION_PROPOSAL>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<OCR_WITH_REGION>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<OCR>", decoder_prompt=None),
    ExplicitEncoderDecoderPrompt(encoder_prompt="<OD>", decoder_prompt=None),
]

def vllm_to_hf_output(
    vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = "</s><s>" + output_str + "</s>"

    return output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    prompts: List[str],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    with vllm_runner(model,
                     tokenizer_name=TOKENIZER,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
            prompts, max_tokens, num_logprobs)

    # Configuration settings for HF baseline
    hf_kwargs = {}
    image = Image.new(mode="RGB", size=(2,2))

    with hf_runner(model, dtype=dtype, skip_tokenizer_init=True) as hf_model:
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.lm_head
        hf_outputs = (hf_model.generate_encoder_decoder_greedy_logprobs_limit(
            prompts,
            max_tokens,
            num_logprobs,
            **hf_kwargs,
            images=[image] * len(prompts),
        ))

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=[
            vllm_to_hf_output(vllm_output)
            for vllm_output in vllm_outputs
        ],
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, model,
                dtype, max_tokens, num_logprobs) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        PROMPTS,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
