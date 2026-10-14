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
PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)


class DummyLogprobs(dict[int, Logprob]):
    def __init__(self, vocab_ids: Iterable[int]):
        super().__init__(dict.fromkeys(vocab_ids, Logprob(0.0)))

    def __repr__(self):
        return "DummyLogprobs()"


def mask_bbox_tokens(
    output: tuple[list[int], str, SampleLogprobs],
    tokenizer: TokenizerLike,
) -> tuple[list[int], str, SampleLogprobs]:
    """Always pass check_logprobs_close check for bounding box tokens
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


def generate_hf_logprobs_from_logits(
    hf_model: HfRunner,
    prompts: list[str],
    images: PromptImageInput,
    max_tokens: int,
    num_logprobs: int,
) -> list[tuple[list[int], str, SampleLogprobs]]:
    """Parse 3.5 exposes generation logits but not decoder hidden states."""
    inputs = hf_model.get_inputs(
        prompts,
        images=images,
        tokenization_kwargs={"add_special_tokens": False},
    )
    outputs = []
    for model_inputs in inputs:
        generated = hf_model.model.generate(
            **hf_model.wrap_device(model_inputs),
            do_sample=False,
            max_new_tokens=max_tokens,
            output_logits=True,
            return_dict_in_generate=True,
        )
        logprobs: SampleLogprobs = []
        for logits in generated.logits:
            values, indices = logits[0].float().log_softmax(dim=-1).topk(num_logprobs)
            logprobs.append(
                {
                    int(token_id): Logprob(float(value))
                    for token_id, value in zip(indices.tolist(), values.tolist())
                }
            )
        token_ids = generated.sequences[0, -len(logprobs) :].tolist()
        outputs.append((token_ids, hf_model.tokenizer.decode(token_ids), logprobs))
    return outputs


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
    is_parse_35 = model == "nvidia/Nemotron-3.5-Parse"
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=16384 if is_parse_35 else 1024,
        enable_chunked_prefill=is_parse_35,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                # The native causal-prefix processor expands this image placeholder.
                [f"<pad>{prompt}" for prompt in prompts] if is_parse_35 else prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs
        ]

        tokenizer = vllm_model.llm.get_tokenizer()

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs_per_case = [
            generate_hf_logprobs_from_logits(
                hf_model, prompts, images, max_tokens, num_logprobs
            )
            if is_parse_35
            else hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
                tokenization_kwargs={"add_special_tokens": False},
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


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        # v2.0's checkpoint ties lm_head to the decoder's input embeddings
        # instead of shipping a separate lm_head.weight tensor; this
        # exercises that path (see tie_weights() in nemotron_parse.py).
        "nvidia/NVIDIA-Nemotron-Parse-2.0",
        "nvidia/Nemotron-3.5-Parse",
    ],
)
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
