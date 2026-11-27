# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from vllm import LLM, PoolingRequestOutput
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.utils import fetch_image

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/cat_snow.jpg"
)
TEXT = "A cat standing in the snow."
EMBED_DIM = 2048

DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        media_io_kwargs={"image": {"image_mode": "RGB"}},
        enforce_eager=True,
        seed=0,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def image():
    return fetch_image(IMAGE_URL)


def _format_chatml_prompt(body: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _prompt(
    text: str | None = TEXT,
    img: object | None = None,
) -> dict | str:
    """Build a Qwen3-VL prompt with explicit vision placeholders when needed."""
    parts: list[str] = []
    if text is not None:
        parts.append(text)
    body = IMAGE_PLACEHOLDER + "".join(parts) if img is not None else "".join(parts)
    formatted = _format_chatml_prompt(body)

    if img is None:
        return formatted
    return {
        "prompt": formatted,
        "multi_modal_data": {"image": img},
    }


def _assert_valid(outputs: list[PoolingRequestOutput], n: int) -> None:
    """Validate `LLM.encode(..., pooling_task="embed")` output shape.

    `LLM.encode` returns `PoolingRequestOutput` whose `.outputs` is a
    `PoolingOutput` with `.data` as a `torch.Tensor`. The `EmbeddingOutput`
    returned by `LLM.embed()` instead exposes `.embedding` as a
    `list[float]`, which is why this helper uses `.data.shape[-1]`.
    """
    assert len(outputs) == n
    for out in outputs:
        assert out.finished
        assert out.outputs.data is not None
        assert out.outputs.data.shape[-1] == EMBED_DIM
        assert out.prompt_token_ids, "rendered prompt ids must be non-empty"


# ---------------------------------------------------------------------------
# Baseline: prompt + image renders and embeds end-to-end
# ---------------------------------------------------------------------------


class TestBasicVisionEncode:
    """Smoke-level coverage of the offline prompt + multimodal path."""

    @pytest.mark.skip_global_cleanup
    def test_single_prompt(self, llm: LLM, image):
        outputs = llm.encode(_prompt(img=image), pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_batch_of_prompts(self, llm: LLM, image):
        prompts = [
            _prompt(TEXT, img=image),
            _prompt("A different caption.", img=image),
        ]
        outputs = llm.encode(prompts, pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 2)

    @pytest.mark.skip_global_cleanup
    def test_mixed_text_and_multimodal_prompts(self, llm: LLM, image):
        batch = [_prompt("A plain text prompt.", img=None), _prompt(img=image)]
        outputs = llm.encode(batch, pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 2)


# ---------------------------------------------------------------------------
# Multimodal combinations
# ---------------------------------------------------------------------------


class TestMultimodalCombinations:
    """`media_io_kwargs`, `mm_processor_kwargs` must compose cleanly on the
    offline `llm.encode` path.

    `mm_processor_kwargs` is deliberately passed as `{}`: the accepted key
    set is processor-specific, and validating it is out of scope here.
    """

    @pytest.mark.skip_global_cleanup
    def test_media_io_and_mm_processor_kwargs(self, llm: LLM, image):
        outputs = llm.encode(
            _prompt(img=image),
            media_io_kwargs={"image": {"timeout": 10}},
            mm_processor_kwargs={},
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Full combination
# ---------------------------------------------------------------------------


class TestFullCombination:
    """Every supported offline kwarg supplied at once."""

    @pytest.mark.skip_global_cleanup
    def test_everything_together(self, llm: LLM, image):
        outputs = llm.encode(
            _prompt(img=image),
            media_io_kwargs={"image": {"timeout": 10}},
            mm_processor_kwargs={},
            cache_salt="test-salt-value",
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestValidationCombinations:
    """Invalid inputs must be rejected early."""

    @pytest.mark.skip_global_cleanup
    def test_invalid_cache_salt_rejected(self, llm: LLM, image):
        with pytest.raises(VLLMValidationError):
            llm.encode(
                _prompt(img=image),
                cache_salt="",
                pooling_task="embed",
                use_tqdm=False,
            )
