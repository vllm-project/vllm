# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF generation parity for Nemotron Parse 3.5."""

import pytest
from transformers import AutoModel

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close
from vllm.assets.image import ImageAsset
from vllm.logprobs import Logprob, SampleLogprobs

from ....conftest import HfRunner, PromptImageInput, VllmRunner
from .test_nemotron_parse import mask_bbox_tokens

MODEL = "nvidia/Nemotron-3.5-Parse"
PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
)


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


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("num_logprobs", [5])
def test_model(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    dtype: str,
    num_logprobs: int,
) -> None:
    model_info = HF_EXAMPLE_MODELS.get_hf_info("RadioMBartPrefixForCausalLM")
    model_info.check_available_online(on_fail="skip")

    image = ImageAsset("paper-11").pil_image_ext(ext="png").convert("RGB")
    prompts = [PROMPT] * 10
    images = [image] * 10
    max_tokens = 100

    with vllm_runner(
        MODEL,
        dtype=dtype,
        max_model_len=16384,
        enable_chunked_prefill=True,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    ) as vllm_model:
        # The native causal-prefix processor expands this image placeholder.
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [f"<pad>{prompt}" for prompt in prompts],
            max_tokens,
            num_logprobs=num_logprobs,
            images=images,
        )
        tokenizer = vllm_model.llm.get_tokenizer()

    with hf_runner(MODEL, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs = generate_hf_logprobs_from_logits(
            hf_model, prompts, images, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=[mask_bbox_tokens(output, tokenizer) for output in hf_outputs],
        outputs_1_lst=[mask_bbox_tokens(output, tokenizer) for output in vllm_outputs],
        name_0="hf",
        name_1="vllm",
    )
