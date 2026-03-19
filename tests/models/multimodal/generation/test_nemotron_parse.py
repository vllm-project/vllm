# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence

import pytest
import regex as re
from transformers import AutoModel

from tests.models.utils import check_logprobs_close
from vllm.assets.image import ImageAsset
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.tokenizers import TokenizerLike

from ....conftest import HfRunner, PromptImageInput, VllmRunner

IMAGE = ImageAsset("paper-11").pil_image_ext(ext="png").convert("RGB")
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


class DummyLogprobs(dict[int, float]):
    def __init__(self, vocab_ids: Iterable[int]):
        super().__init__(dict.fromkeys(vocab_ids, 0.0))

    def __repr__(self):
        return "DummyLogprobs()"


def mask_bbox_tokens(
    output: tuple[list[int], str, SampleLogprobs | None],
    tokenizer: TokenizerLike,
) -> tuple[list[int], str, SampleLogprobs | None]:
    """
    Always pass check_logprobs_close check for bounding box tokens
    because it is reasonable for them to differ slightly.
    """
    ignore_pattern = r"<[xy]_[\d.]+>"
    vocab = tokenizer.get_vocab()

    output_ids, output_str, out_logprobs = output

    masked_logprobs = list[dict[int, Logprob]]()
    for token, logprobs in zip(output_ids, out_logprobs):
        if re.match(ignore_pattern, tokenizer.decode(token)):
            masked_logprobs.append(DummyLogprobs(vocab.values()))
        else:
            masked_logprobs.append(logprobs)

    return output_ids, output_str, masked_logprobs


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Verify that the inference result is the same between hf and vllm."""
    with vllm_runner(
        model,
        dtype=dtype,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs
        ]

        tokenizer = vllm_model.llm.get_tokenizer()

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
                use_cache=False,  # HF Nemotron Parse crashes here without this
            )
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=[
                mask_bbox_tokens(output, tokenizer) for output in hf_outputs
            ],
            outputs_1_lst=[
                mask_bbox_tokens(output, tokenizer) for output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", ["nvidia/NVIDIA-Nemotron-Parse-v1.1"])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner, vllm_runner, model: str, dtype: str, num_logprobs: int
) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        inputs=[
            ([PROMPT] * 10, [IMAGE] * 10),
        ],
        model=model,
        dtype=dtype,
        max_tokens=100,
        num_logprobs=num_logprobs,
    )
