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


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        seed=0,
        trust_request_chat_template=True,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def image():
    return fetch_image(IMAGE_URL)


@pytest.fixture(scope="module")
def official_chat_template():
    """The chat template shipped with the model's processor.

    Loaded once per module so the tests that override `chat_template` stay
    in sync with the model version instead of hard-coding a Jinja string.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_NAME).chat_template


def _conversation(text: str = TEXT) -> list[dict]:
    """A single user turn mixing an image and a text instruction."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "text", "text": text},
            ],
        }
    ]


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
# Baseline: chat + image renders and embeds end-to-end
# ---------------------------------------------------------------------------


class TestBasicVisionEncode:
    """Smoke-level coverage of the chat + multimodal path."""

    @pytest.mark.skip_global_cleanup
    def test_single_conversation(self, llm: LLM, image):
        outputs = llm.encode(_conversation(), pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_batch_of_conversations(self, llm: LLM, image):
        convs = [_conversation(TEXT), _conversation("A different caption.")]
        outputs = llm.encode(convs, pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 2)

    @pytest.mark.skip_global_cleanup
    def test_mixed_chat_and_completion(self, llm: LLM, image):
        # `is_conversations` is applied per element by the offline factory,
        # so a batch that mixes a plain-text completion and a chat
        # conversation must dispatch each element independently.
        mixed = ["A plain text prompt.", _conversation()]
        outputs = llm.encode(mixed, pooling_task="embed", use_tqdm=False)
        _assert_valid(outputs, 2)


# ---------------------------------------------------------------------------
# Chat template combinations
# ---------------------------------------------------------------------------


class TestChatTemplateCombinations:
    """`chat_template`, `chat_template_kwargs`, and prompt-shaping flags
    must compose without conflicting."""

    @pytest.mark.skip_global_cleanup
    def test_template_override_and_kwargs(
        self, llm: LLM, image, official_chat_template
    ):
        # Both `chat_template` and `chat_template_kwargs` supplied together.
        outputs = llm.encode(
            _conversation(),
            chat_template=official_chat_template,
            chat_template_kwargs={"add_generation_prompt": True},
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_template_with_explicit_generation_prompt(
        self, llm: LLM, image, official_chat_template
    ):
        # The dedicated `add_generation_prompt` kwarg must merge with the
        # request-level template correctly (not collide with kwargs above).
        outputs = llm.encode(
            _conversation(),
            chat_template=official_chat_template,
            add_generation_prompt=True,
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_continue_final_message(self, llm: LLM, image):
        convs = [
            *_conversation(),
            {"role": "assistant", "content": "The image shows"},
        ]
        outputs = llm.encode(
            convs,
            continue_final_message=True,
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Multimodal combinations
# ---------------------------------------------------------------------------


class TestMultimodalCombinations:
    """`media_io_kwargs` and `mm_processor_kwargs` must compose.

    `mm_processor_kwargs` is deliberately passed as `{}`: the accepted key
    set is processor-specific, and validating it is out of scope here. The
    plumbing (that the kwarg reaches the renderer) is covered by
    `test_io_processor.py::TestOfflineFactoryDefaults` with a mock.
    """

    @pytest.mark.skip_global_cleanup
    def test_media_io_and_mm_processor_kwargs(self, llm: LLM, image):
        outputs = llm.encode(
            _conversation(),
            media_io_kwargs={"image": {"timeout": 10}},
            mm_processor_kwargs={},
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Tokenization combinations
# ---------------------------------------------------------------------------


class TestTokenizationCombinations:
    """`add_special_tokens` (tri-state) and `cache_salt` must compose."""

    @pytest.mark.skip_global_cleanup
    def test_add_special_tokens_false_with_cache_salt(self, llm: LLM, image):
        outputs = llm.encode(
            _conversation(),
            add_special_tokens=False,
            cache_salt="test-salt-value",
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)

    @pytest.mark.skip_global_cleanup
    def test_add_special_tokens_true_with_cache_salt(self, llm: LLM, image):
        outputs = llm.encode(
            _conversation(),
            add_special_tokens=True,
            cache_salt="test-salt-value",
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Full combination
# ---------------------------------------------------------------------------


class TestFullCombination:
    """Every new offline kwarg supplied at once."""

    @pytest.mark.skip_global_cleanup
    def test_everything_together(self, llm: LLM, image, official_chat_template):
        outputs = llm.encode(
            _conversation(),
            chat_template=official_chat_template,
            chat_template_kwargs={"add_generation_prompt": True},
            media_io_kwargs={"image": {"timeout": 10}},
            mm_processor_kwargs={},
            cache_salt="test-salt-value",
            add_special_tokens=False,
            pooling_task="embed",
            use_tqdm=False,
        )
        _assert_valid(outputs, 1)


# ---------------------------------------------------------------------------
# Invalid combinations
# ---------------------------------------------------------------------------


class TestValidationCombinations:
    """Mutually exclusive / invalid combinations must be rejected early."""

    @pytest.mark.skip_global_cleanup
    def test_generation_prompt_and_continue_final_message_exclusive(
        self, llm: LLM, image
    ):
        with pytest.raises(
            VLLMValidationError,
            match="Cannot set both `continue_final_message`",
        ):
            llm.encode(
                _conversation(),
                add_generation_prompt=True,
                continue_final_message=True,
                pooling_task="embed",
                use_tqdm=False,
            )

    @pytest.mark.skip_global_cleanup
    def test_invalid_cache_salt_rejected(self, llm: LLM, image):
        with pytest.raises(VLLMValidationError):
            llm.encode(
                _conversation(),
                cache_salt="",
                pooling_task="embed",
                use_tqdm=False,
            )
