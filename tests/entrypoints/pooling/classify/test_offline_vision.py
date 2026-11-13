# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from vllm import LLM, ClassificationRequestOutput
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.multimodal.utils import encode_image_url, fetch_image

MODEL_NAME = "vllm-project/Qwen2.5-VL-7B-for-VideoCls"
NUM_LABELS = 2
MAXIMUM_VIDEOS = 1

HF_OVERRIDES = {"architectures": ["Qwen2_5_VLForSequenceClassification"]}

TEXT = "This product was excellent and exceeded my expectations"
IMAGE_URL = f"{VLLM_S3_BUCKET_URL}/multimodal_asset/cat_snow.jpg"
VIDEO_URL = f"{VLLM_S3_BUCKET_URL}/multimodal_asset/slow_traffic_small.mp4"


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        max_model_len=16384,
        limit_mm_per_prompt={"video": MAXIMUM_VIDEOS},
        enforce_eager=True,
        seed=0,
        hf_overrides=HF_OVERRIDES,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def image_base64() -> str:
    return encode_image_url(fetch_image(IMAGE_URL))


def _assert_valid(outputs: list[ClassificationRequestOutput], n: int) -> None:
    """Validate `LLM.classify()` output shape.

    `LLM.classify` returns `ClassificationRequestOutput` whose `.outputs` is
    a `ClassificationOutput` with `.probs` as a `list[float]` of length
    `num_labels`. The exact prompt-token count depends on the tokenizer
    version, so it is not asserted here; only the structure is pinned.
    """
    assert len(outputs) == n
    for out in outputs:
        assert out.finished
        assert out.outputs.probs is not None
        assert len(out.outputs.probs) == NUM_LABELS
        assert out.prompt_token_ids, "rendered prompt ids must be non-empty"


# ---------------------------------------------------------------------------
# Text-only chat
# ---------------------------------------------------------------------------


class TestChatTextClassify:
    @pytest.mark.skip_global_cleanup
    def test_single_conversation(self, llm: LLM):
        convs = [
            {
                "role": "assistant",
                "content": "Please classify this text request.",
            },
            {"role": "user", "content": TEXT},
        ]

        outputs = llm.classify(convs, use_tqdm=False)

        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Image chat
# ---------------------------------------------------------------------------


class TestChatImageClassify:
    @pytest.mark.skip_global_cleanup
    def test_image_url(self, llm: LLM):
        convs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please classify this image."},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            }
        ]

        outputs = llm.classify(convs, use_tqdm=False)

        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_image_base64(self, llm: LLM, image_base64: str):
        convs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please classify this image."},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
        ]

        outputs = llm.classify(convs, use_tqdm=False)

        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Video chat
# ---------------------------------------------------------------------------


class TestChatVideoClassify:
    @pytest.mark.skip_global_cleanup
    def test_video_url(self, llm: LLM):
        convs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please classify this video."},
                    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                ],
            }
        ]

        outputs = llm.classify(convs, use_tqdm=False)

        _assert_valid(outputs, 1)
