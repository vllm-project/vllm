# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional, Tuple, Type

import pytest
from PIL import Image

from vllm.inputs.data import ExplicitEncoderDecoderPrompt
from vllm.sequence import SampleLogprobs

from ....conftest import HfRunner, VllmRunner
from ...utils import check_logprobs_close

Florence2Prompt = partial(ExplicitEncoderDecoderPrompt,
                          decoder_prompt=None,
                          mm_processor_kwargs=None)

MODELS = ["microsoft/Florence-2-base"]
# Florence-2 uses BartFastTokenizer which can't be loaded from AutoTokenizer
# Therefore, we borrow the BartTokenizer from the original Bart model
TOKENIZER = "facebook/bart-base"
PROMPTS = [
    Florence2Prompt(encoder_prompt="<CAPTION>"),
    Florence2Prompt(encoder_prompt="<DETAILED_CAPTION>"),
    Florence2Prompt(encoder_prompt="<MORE_DETAILED_CAPTION>"),
    Florence2Prompt(encoder_prompt="<CAPTION_TO_PHRASE_GROUNDING>"),
    Florence2Prompt(encoder_prompt="<DENSE_REGION_CAPTION>"),
    Florence2Prompt(encoder_prompt="<REGION_PROPOSAL>"),
    Florence2Prompt(encoder_prompt="<OCR_WITH_REGION>"),
    Florence2Prompt(encoder_prompt="<OCR>"),
    Florence2Prompt(encoder_prompt="<OD>"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]], ):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = "</s><s>" + output_str + "</s>"

    return output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    prompts: List[ExplicitEncoderDecoderPrompt],
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

    # Florence-2 processors require image inputs
    dummy_image = Image.new(mode="RGB", size=(2, 2))
    with hf_runner(model, dtype=dtype, skip_tokenizer_init=True) as hf_model:
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.lm_head
        hf_outputs = (hf_model.generate_encoder_decoder_greedy_logprobs_limit(
            prompts,
            max_tokens,
            num_logprobs,
            images=[dummy_image] * len(prompts),
        ))

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=[
            vllm_to_hf_output(vllm_output) for vllm_output in vllm_outputs
        ],
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, model, dtype, max_tokens,
                num_logprobs) -> None:
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
