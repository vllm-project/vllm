# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-OCR-2 multimodal generation.

Includes:
  1. Unit test verifying the architecture is in the model registry.
  2. Integration test that loads the model and runs image+text generation.
  3. Comparison test against HuggingFace reference outputs for numerical
     equivalence.
"""

import pytest
from transformers import AutoModelForCausalLM, BatchFeature

from tests.models.utils import check_logprobs_close
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.registry import ModelRegistry

from ....conftest import HfRunner, PromptImageInput, VllmRunner

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"

STOP_STR = ["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"]

IMAGE = ImageAsset("cherry_blossom").pil_image.convert("RGB")

PROMPT = "<|User|>: <image>\nDescribe this image in one sentence.\n\n<|Assistant|>: "


# --------------------------------------------------------------------------- #
# 1. Unit test: architecture is registered
# --------------------------------------------------------------------------- #
def test_deepseek_ocr2_registered():
    """DeepseekOCR2ForCausalLM must be discoverable in the model registry."""
    supported = ModelRegistry.get_supported_archs()
    assert "DeepseekOCR2ForCausalLM" in supported


def test_deepseek_ocr2_importable():
    """The model class must be importable from the registry."""
    model_cls = ModelRegistry._try_load_model_cls("DeepseekOCR2ForCausalLM")
    assert model_cls is not None


# --------------------------------------------------------------------------- #
# 2. Integration test: load model and generate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
def test_deepseek_ocr2_generation(
    vllm_runner: type[VllmRunner],
    dtype: str,
    max_tokens: int,
) -> None:
    """Load DeepSeek-OCR-2 in vLLM and verify it can generate text from an
    image prompt."""
    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            [PROMPT],
            max_tokens,
            images=[IMAGE],
        )

    assert len(outputs) == 1
    output_ids, output_str = outputs[0]
    # The model should produce a non-trivial response
    assert len(output_ids) > 0
    assert len(output_str.strip()) > 0


# --------------------------------------------------------------------------- #
# 3. Comparison test: vLLM vs HuggingFace reference outputs
# --------------------------------------------------------------------------- #
def _patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patch HfRunner to use the custom DeepseekOCRProcessor for OCR-2."""
    hf_processor = hf_model.processor

    def processor(*args, text="", images=None, **kwargs):
        from PIL.Image import Image

        if isinstance(images, Image):
            images = [images]
        inputs = hf_processor(
            *args,
            prompt=text,
            images=images,
            **kwargs,
        )
        inputs = {
            k: inputs[k]
            for k in inputs.keys()  # noqa: SIM118
            if k not in ("seq_lens", "sft_format")
        }
        return BatchFeature(data=inputs, tensor_type="pt")

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.language.model.embed_tokens
    )
    return hf_model


def _trunc_hf_output(
    hf_output: tuple[list[int], str, object],
    model: str,
) -> tuple[list[int], str, object]:
    output_ids, output_str, out_logprobs = hf_output
    for stop in STOP_STR:
        if output_str.endswith(stop):
            output_str = output_str.split(stop)[0]
    return output_ids, output_str, out_logprobs


def _run_comparison(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: list[tuple[list[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
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

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForCausalLM,
    ) as hf_model:
        hf_model = _patch_hf_runner(hf_model)
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(
        hf_outputs_per_case, vllm_outputs_per_case
    ):
        check_logprobs_close(
            outputs_0_lst=[
                _trunc_hf_output(o, model) for o in hf_outputs
            ],
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", [MODEL_NAME])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("num_logprobs", [5])
def test_deepseek_ocr2_vs_hf(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    num_logprobs: int,
) -> None:
    """Compare vLLM generation with HuggingFace reference for numerical
    equivalence on a single-image prompt."""
    _run_comparison(
        hf_runner,
        vllm_runner,
        inputs=[
            ([PROMPT], [IMAGE]),
        ],
        model=model,
        dtype=dtype,
        max_tokens=32,
        num_logprobs=num_logprobs,
    )
