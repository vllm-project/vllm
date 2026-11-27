# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from PIL import Image

from vllm.renderers.base import BaseRenderer

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"


@pytest.mark.skip_global_cleanup
def test_encode_prompt_image_mode_none_overrides_engine_default(
    vllm_runner, monkeypatch
):
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    observed_media_io_kwargs = []
    original_process_multimodal = BaseRenderer._process_multimodal

    def process_multimodal_and_record_kwargs(renderer, *args, **kwargs):
        observed_media_io_kwargs.append(kwargs["media_io_kwargs"])
        return original_process_multimodal(renderer, *args, **kwargs)

    monkeypatch.setattr(
        BaseRenderer, "_process_multimodal", process_multimodal_and_record_kwargs
    )

    prompt = {
        "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"image": rgba},
        "media_io_kwargs": {"image": {"image_mode": None}},
    }

    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        media_io_kwargs={"image": {"image_mode": "RGB"}},
        enforce_eager=True,
        seed=0,
    ) as runner:
        outputs = runner.llm.encode(prompt, pooling_task="embed", use_tqdm=False)

    assert len(outputs) == 1
    assert outputs[0].finished
    assert observed_media_io_kwargs == [{"image": {"image_mode": None}}]
