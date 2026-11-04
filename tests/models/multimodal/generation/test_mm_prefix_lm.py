# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch
from transformers import AutoModelForImageTextToText

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from ....conftest import HfRunner, ImageTestAssets, VllmRunner
from .vlm_utils import model_utils

MODEL = "google/gemma-3-4b-it"
PROMPT = (
    "<bos><start_of_turn>user\n"
    "<start_of_image>What is the content in the center of the image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)
TINY_MODEL = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"


def _install_prefill_hidden_capture(model):
    model = getattr(model, "module", model)
    model._prefill_hidden = None

    language_model = model.language_model.model
    original_forward = language_model.forward

    def forward(*args, **kwargs):
        hidden_states = original_forward(*args, **kwargs)
        if model._prefill_hidden is None and torch.is_tensor(hidden_states):
            # Capturing hidden states for comparison is a deliberate D2H.
            with gpu_sync_allowed():
                model._prefill_hidden = hidden_states.detach().float().cpu()
        return hidden_states

    language_model.forward = forward


def _record_hidden_states(model):
    model = getattr(model, "module", model)
    model._hidden_by_position = {}

    language_model = model.language_model.model
    original_forward = language_model.forward

    def forward(input_ids, positions, *args, **kwargs):
        hidden_states = original_forward(input_ids, positions, *args, **kwargs)
        # A deliberate D2H, copied: the runner reuses its buffers next step.
        with gpu_sync_allowed():
            rows = hidden_states.to("cpu", torch.float32, copy=True)
            model._hidden_by_position.update(zip(positions.tolist(), rows))
        return hidden_states

    language_model.forward = forward


def _pop_recorded_hidden_states(model):
    model = getattr(model, "module", model)
    hidden, model._hidden_by_position = model._hidden_by_position, {}
    return hidden


def _get_prefill_hidden(model):
    model = getattr(model, "module", model)
    hidden = getattr(model, "_prefill_hidden", None)
    assert hidden is not None
    return hidden


def _get_hf_prefill_hidden(hf_model: HfRunner, image: Any):
    inputs = hf_model.get_inputs([PROMPT], images=[image])[0]
    with torch.no_grad():
        outputs = hf_model.model.model(
            **hf_model.wrap_device(inputs),
            use_cache=False,
        )
    return outputs.last_hidden_state[0].detach().float().cpu()


def _get_vllm_prefill_hidden(
    vllm_runner: type[VllmRunner],
    image: Any,
    vllm_runner_kwargs: dict[str, Any],
):
    with vllm_runner(
        MODEL,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        **vllm_runner_kwargs,
    ) as vllm_model:
        vllm_model.apply_model(_install_prefill_hidden_capture)
        vllm_model.generate_greedy([PROMPT], max_tokens=1, images=[image])
        return vllm_model.apply_model(_get_prefill_hidden)[0]


@pytest.mark.core_model
@pytest.mark.skipif(
    current_platform.is_rocm(), reason="ROCm attention has accuracy issue for this test"
)
def test_mm_prefix_lm_e2e(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: Gemma3 native prefill must apply image prefix-LM mask."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    image = image_assets[0].pil_image

    vllm_runner_kwargs: dict[str, Any] = {
        "mm_processor_cache_gb": 0,
        "mm_processor_kwargs": {"do_pan_and_scan": True},
    }
    vllm_hidden = _get_vllm_prefill_hidden(vllm_runner, image, vllm_runner_kwargs)

    hf_model = hf_runner(
        MODEL,
        auto_cls=AutoModelForImageTextToText,
    )
    hf_model = model_utils.gemma3_patch_hf_runner(hf_model)

    with hf_model:
        hf_hidden = _get_hf_prefill_hidden(hf_model, image)

    assert vllm_hidden.shape == hf_hidden.shape

    full_cos = torch.nn.functional.cosine_similarity(
        vllm_hidden.flatten(), hf_hidden.flatten(), dim=0
    )
    image_cos = torch.nn.functional.cosine_similarity(
        vllm_hidden[1:769].flatten(), hf_hidden[1:769].flatten(), dim=0
    )

    assert full_cos > 0.9, (
        "Gemma3 mm-prefix-LM full prefill hidden states should be close to HF; "
        f"got {full_cos=}"
    )
    assert image_cos > 0.9, (
        "Gemma3 mm-prefix-LM image prefill hidden states should be close to HF; "
        f"got {image_cos=}"
    )


@pytest.mark.core_model
def test_mm_prefix_lm_cache_hit_inside_image(
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """A prefix cache hit can end inside an image. The rest of the image must
    still be prefilled in one step, or its first part misses the rest."""
    # One process, so that all requests of a generate() call share a step.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Triton attention uses TF32 for fp32 by default.
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    # The image tokens span [100, 356), so a two-block hit ends inside it.
    prompt = {
        "prompt": "<bos><start_of_turn>user\n"
        + "word " * 92
        + "<start_of_image>What color is it?<end_of_turn>\n<start_of_turn>model\n",
        "multi_modal_data": {"image": image_assets[0].pil_image},
    }

    def image_after_hit(vllm_model: VllmRunner) -> torch.Tensor:
        hidden = vllm_model.apply_model(_pop_recorded_hidden_states)[0]
        return torch.stack([hidden[position] for position in range(256, 356)])

    with vllm_runner(
        TINY_MODEL,
        dtype="float32",
        block_size=128,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        disable_chunked_mm_input=True,
        limit_mm_per_prompt={"image": 1},
    ) as vllm_model:
        llm = vllm_model.get_llm()
        vllm_model.apply_model(_record_hidden_states)

        # Prefilled in one step; caches the first two blocks for the hit below.
        alone = SamplingParams(max_tokens=1, skip_reading_prefix_cache=True)
        llm.generate([prompt], alone)
        expected = image_after_hit(vllm_model)

        # Text-only requests below position 256 leave 40 tokens of budget for
        # the 112 after the hit.
        fillers = [{"prompt_token_ids": [10 + i] * 118} for i in range(4)]
        output = llm.generate([*fillers, prompt], SamplingParams(max_tokens=1))[-1]
        assert output.num_cached_tokens == 256

        torch.testing.assert_close(image_after_hit(vllm_model), expected)
