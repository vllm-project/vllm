# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, PoolingParams
from vllm.renderers.chat_utils import ChatTemplateContentFormatOption

MODEL_NAME = "vllm-project/Qwen3-Reranker-0.6B-seq-cls"
PROMPT = "The chef prepared a delicious meal."
TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]
TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

# Concatenates the content parts of every message, without any per-message
# decoration. Used to compare scoring against the raw `classify` path.
PLAIN_CONCAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% for part in message.content %}{{ part.text }}{% endfor %}"
    "{% endfor %}"
)

# Same as PLAIN_CONCAT_TEMPLATE but consumes an optional `prefix` kwarg.
PREFIXED_CONCAT_TEMPLATE = (
    "{{ prefix | default('') }}"
    "{% for message in messages %}"
    "{% for part in message.content %}{{ part.text }}{% endfor %}"
    "{% endfor %}"
)

# Same as PLAIN_CONCAT_TEMPLATE but exposes the two mutually exclusive
# prompt-shaping flags so tests can observe their effect.
TAGGED_CONCAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% for part in message.content %}{{ part.text }}{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|gen|>{% endif %}"
    "{% if continue_final_message %}<|cont|>{% endif %}"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy
        # to avoid retaining the LLM while VllmRunner.__exit__ releases ROCm
        # memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def hf_model(hf_runner):
    return hf_runner(MODEL_NAME, is_cross_encoder=True)


# =============================================================================
# Helpers
# =============================================================================


def _call_score(llm, data_1, data_2, **kwargs):
    """Call `llm.score` with `use_tqdm` disabled by default."""
    kwargs.setdefault("use_tqdm", False)
    return llm.score(data_1, data_2, **kwargs)


def _call_score_fresh(llm, data_1, data_2, **kwargs):
    """Reset the prefix cache, then call `llm.score`."""
    llm.reset_prefix_cache()
    return _call_score(llm, data_1, data_2, **kwargs)


def _call_classify_fresh(llm, prompt, **kwargs):
    """Reset the prefix cache, then call `llm.classify`."""
    llm.reset_prefix_cache()
    kwargs.setdefault("use_tqdm", False)
    return llm.classify(prompt, **kwargs)


def _score_single(llm, data_1, data_2, **kwargs):
    """Call `llm.score` and assert exactly one output is returned."""
    (out,) = _call_score(llm, data_1, data_2, **kwargs)
    return out


def _scores(outputs):
    return [out.outputs.score for out in outputs]


def _prompt_token_ids(output):
    return list(output.prompt_token_ids)


# =============================================================================
# TestPairingModes
# =============================================================================


class TestPairingModes:
    """Input pairing contract: `1 -> 1`, `1 -> N`, `N -> N`."""

    @pytest.mark.skip_global_cleanup
    def test_1_to_1_matches_hf(self, llm, hf_model):
        text_pair = [TEXTS_1[0], TEXTS_2[0]]

        hf_out = hf_model.predict([text_pair]).tolist()
        vllm_out = _scores(_call_score(llm, *text_pair))

        assert len(vllm_out) == len(hf_out) == 1
        assert hf_out[0] == pytest.approx(vllm_out[0], rel=0.01)

    @pytest.mark.skip_global_cleanup
    def test_1_to_n_matches_hf(self, llm, hf_model):
        text_pairs = [
            [TEXTS_1[0], TEXTS_2[0]],
            [TEXTS_1[0], TEXTS_2[1]],
        ]

        vllm_out = _scores(_call_score(llm, TEXTS_1[0], TEXTS_2))
        hf_out = hf_model.predict(text_pairs).tolist()

        assert len(vllm_out) == len(hf_out) == 2
        for hf, vl in zip(hf_out, vllm_out):
            assert hf == pytest.approx(vl, rel=0.01)

    @pytest.mark.skip_global_cleanup
    def test_n_to_n_matches_hf(self, llm, hf_model):
        text_pairs = [
            [TEXTS_1[0], TEXTS_2[0]],
            [TEXTS_1[1], TEXTS_2[1]],
        ]

        vllm_out = _scores(_call_score(llm, TEXTS_1, TEXTS_2))
        hf_out = hf_model.predict(text_pairs).tolist()

        assert len(vllm_out) == len(hf_out) == 2
        for hf, vl in zip(hf_out, vllm_out):
            assert hf == pytest.approx(vl, rel=0.01)


# =============================================================================
# TestContentFormat
# =============================================================================


class TestContentFormat:
    """`chat_template_content_format` controls message-content rendering."""

    @pytest.mark.parametrize(
        "content_format",
        [
            pytest.param("auto", id="detected-default"),
            pytest.param("openai", id="structured"),
            pytest.param("string", id="explicit-string"),
        ],
    )
    @pytest.mark.skip_global_cleanup
    def test_content_format_override_matches_classify(
        self, llm: LLM, content_format: ChatTemplateContentFormatOption
    ):
        query, document = TEXTS_1[0], TEXTS_2[0]
        expected_prompt = json.dumps(
            [
                {"role": "query", "content": [{"type": "text", "text": query}]},
                {
                    "role": "document",
                    "content": [{"type": "text", "text": document}],
                },
            ]
        )
        chat_template = "{{ messages | tojson }}"
        if content_format == "auto":
            chat_template = PLAIN_CONCAT_TEMPLATE
            expected_prompt = query + document
        elif content_format == "string":
            chat_template = (
                "{% for message in messages %}"
                "{% for char in message.content %}{{ char }}{% endfor %}"
                "{% endfor %}"
            )
            expected_prompt = query + document

        (scored,) = _call_score_fresh(
            llm,
            query,
            document,
            chat_template=chat_template,
            chat_template_content_format=content_format,
            add_special_tokens=False,
        )
        (classified,) = _call_classify_fresh(
            llm,
            expected_prompt,
            tokenization_kwargs={"add_special_tokens": False},
        )

        assert _prompt_token_ids(scored) == _prompt_token_ids(classified)
        assert scored.outputs.score == pytest.approx(classified.outputs.probs[0])

    @pytest.mark.skip_global_cleanup
    def test_invalid_content_format_raises(self, llm: LLM):
        with pytest.raises(ValueError, match="chat_template_content_format"):
            _call_score(
                llm,
                TEXTS_1[0],
                TEXTS_2[0],
                chat_template_content_format="invalid",  # type: ignore[arg-type]
            )


# =============================================================================
# TestAddSpecialTokens
# =============================================================================


class TestAddSpecialTokens:
    """`add_special_tokens=None` follows the per-path default."""

    @pytest.mark.skip_global_cleanup
    def test_completion_none_matches_true(self, llm: LLM):
        default = _score_single(llm, TEXTS_1[0], TEXTS_2[0], add_special_tokens=None)
        explicit = _score_single(llm, TEXTS_1[0], TEXTS_2[0], add_special_tokens=True)
        assert _prompt_token_ids(default) == _prompt_token_ids(explicit)
        assert default.outputs.score == pytest.approx(explicit.outputs.score)

    @pytest.mark.skip_global_cleanup
    def test_completion_false_matches_classify(self, llm: LLM):
        """`add_special_tokens=False` on completion inputs equals classify."""
        query, document = TEXTS_1[0], TEXTS_2[0]
        expected_prompt = query + document

        scored = _score_single(llm, query, document, add_special_tokens=False)
        (classified,) = _call_classify_fresh(
            llm,
            expected_prompt,
            tokenization_kwargs={"add_special_tokens": False},
        )

        assert _prompt_token_ids(scored) == _prompt_token_ids(classified)
        assert scored.outputs.score == pytest.approx(classified.outputs.probs[0])

    @pytest.mark.skip_global_cleanup
    def test_chat_none_matches_false(self, llm: LLM):
        kwargs = dict(
            chat_template=PLAIN_CONCAT_TEMPLATE,
            chat_template_content_format="auto",
        )
        default = _score_single(
            llm, TEXTS_1[0], TEXTS_2[0], add_special_tokens=None, **kwargs
        )
        explicit_false = _score_single(
            llm, TEXTS_1[0], TEXTS_2[0], add_special_tokens=False, **kwargs
        )
        assert _prompt_token_ids(default) == _prompt_token_ids(explicit_false)
        assert default.outputs.score == pytest.approx(explicit_false.outputs.score)

    @pytest.mark.skip_global_cleanup
    def test_chat_true_matches_classify(self, llm: LLM):
        """`add_special_tokens=True` on chat inputs equals classify."""
        query, document = TEXTS_1[0], TEXTS_2[0]
        kwargs = dict(
            chat_template=PLAIN_CONCAT_TEMPLATE,
            chat_template_content_format="auto",
        )
        expected_prompt = query + document

        scored = _score_single(llm, query, document, add_special_tokens=True, **kwargs)
        (classified,) = _call_classify_fresh(
            llm,
            expected_prompt,
            tokenization_kwargs={"add_special_tokens": True},
        )

        assert _prompt_token_ids(scored) == _prompt_token_ids(classified)
        assert scored.outputs.score == pytest.approx(classified.outputs.probs[0])


# =============================================================================
# TestChatTemplateKwargs
# =============================================================================


class TestChatTemplateKwargs:
    """`chat_template_kwargs` is forwarded into the chat template renderer."""

    @pytest.mark.skip_global_cleanup
    def test_empty_kwargs_matches_default(self, llm: LLM):
        kwargs = dict(
            chat_template=PREFIXED_CONCAT_TEMPLATE,
            chat_template_content_format="auto",
            add_special_tokens=False,
        )
        base = _score_single(llm, TEXTS_1[0], TEXTS_2[0], **kwargs)
        with_empty = _score_single(
            llm, TEXTS_1[0], TEXTS_2[0], chat_template_kwargs={"prefix": ""}, **kwargs
        )
        assert _prompt_token_ids(base) == _prompt_token_ids(with_empty)
        assert base.outputs.score == pytest.approx(with_empty.outputs.score)

    @pytest.mark.skip_global_cleanup
    def test_non_empty_kwargs_changes_prompt(self, llm: LLM):
        kwargs = dict(
            chat_template=PREFIXED_CONCAT_TEMPLATE,
            chat_template_content_format="auto",
            add_special_tokens=False,
        )
        base = _score_single(llm, TEXTS_1[0], TEXTS_2[0], **kwargs)
        with_prefix = _score_single(
            llm,
            TEXTS_1[0],
            TEXTS_2[0],
            chat_template_kwargs={"prefix": "PREFIX: "},
            **kwargs,
        )
        assert _prompt_token_ids(base) != _prompt_token_ids(with_prefix)
        assert len(_prompt_token_ids(with_prefix)) > len(_prompt_token_ids(base))


# =============================================================================
# TestGenerationPromptOptions
# =============================================================================


class TestGenerationPromptOptions:
    _KW = dict(
        chat_template=TAGGED_CONCAT_TEMPLATE,
        chat_template_content_format="auto",
        add_special_tokens=False,
    )

    @pytest.mark.skip_global_cleanup
    def test_add_generation_prompt_changes_prompt(self, llm: LLM):
        base = _score_single(llm, TEXTS_1[0], TEXTS_2[0], **self._KW)
        with_gp = _score_single(
            llm, TEXTS_1[0], TEXTS_2[0], add_generation_prompt=True, **self._KW
        )
        assert _prompt_token_ids(with_gp) != _prompt_token_ids(base)
        assert len(_prompt_token_ids(with_gp)) > len(_prompt_token_ids(base))

    @pytest.mark.skip_global_cleanup
    def test_continue_final_message_is_accepted(self, llm: LLM):
        """`continue_final_message` must be accepted on the scoring path."""
        base = _score_single(llm, TEXTS_1[0], TEXTS_2[0], **self._KW)
        cont = _score_single(
            llm, TEXTS_1[0], TEXTS_2[0], continue_final_message=True, **self._KW
        )
        assert isinstance(cont.outputs.score, float)
        assert cont.outputs.score == pytest.approx(base.outputs.score)

    @pytest.mark.skip_global_cleanup
    def test_both_false_matches_default(self, llm: LLM):
        base = _score_single(llm, TEXTS_1[0], TEXTS_2[0], **self._KW)
        both_false = _score_single(
            llm,
            TEXTS_1[0],
            TEXTS_2[0],
            add_generation_prompt=False,
            continue_final_message=False,
            **self._KW,
        )
        assert _prompt_token_ids(both_false) == _prompt_token_ids(base)
        assert both_false.outputs.score == pytest.approx(base.outputs.score)

    @pytest.mark.skip_global_cleanup
    def test_both_true_raises(self, llm: LLM):
        """`add_generation_prompt` and `continue_final_message`
        are mutually exclusive."""
        with pytest.raises(Exception, match="Cannot set both"):
            _call_score(
                llm,
                TEXTS_1[0],
                TEXTS_2[0],
                add_generation_prompt=True,
                continue_final_message=True,
            )


# =============================================================================
# TestPoolingParamsAndExtras
# =============================================================================


class TestPoolingParamsAndExtras:
    """`pooling_params` override the pooled activation; extras apply."""

    @pytest.mark.skip_global_cleanup
    def test_default_matches_explicit_activation(self, llm: LLM):
        def get(use_activation):
            return torch.tensor(
                _scores(
                    _call_score(
                        llm,
                        TEXTS_1[0],
                        TEXTS_2[0],
                        pooling_params=PoolingParams(use_activation=use_activation),
                    )
                )
            )

        default = get(use_activation=None)
        with_activation = get(use_activation=True)
        assert torch.allclose(default, with_activation, atol=1e-2), (
            "Default should use activation."
        )

    @pytest.mark.skip_global_cleanup
    def test_without_activation_differs(self, llm: LLM):
        def get(use_activation):
            return torch.tensor(
                _scores(
                    _call_score(
                        llm,
                        TEXTS_1[0],
                        TEXTS_2[0],
                        pooling_params=PoolingParams(use_activation=use_activation),
                    )
                )
            )

        with_activation = get(use_activation=True)
        without_activation = get(use_activation=False)
        assert not torch.allclose(with_activation, without_activation, atol=1e-2), (
            "use_activation=False should not use activation."
        )
        assert torch.allclose(
            softmax(without_activation), with_activation, atol=1e-2
        ), "use_activation=True should be softmax of the un-activated logits."

    @pytest.mark.skip_global_cleanup
    def test_max_tokens_per_doc_truncates(self, llm: LLM):
        long_doc = "The capital of France is Paris. " * 20

        no_limit = _score_single(llm, TEXTS_1[0], long_doc)
        with_limit = _score_single(
            llm,
            TEXTS_1[0],
            long_doc,
            pooling_params=PoolingParams(extra_kwargs={"max_tokens_per_doc": 10}),
        )
        assert len(_prompt_token_ids(with_limit)) < len(_prompt_token_ids(no_limit))

    @pytest.mark.skip_global_cleanup
    def test_conflicting_pooling_task_raises(self, llm: LLM):
        # The IO processor for this model exposes a fixed `pooling_task`
        # (typically `"score"`); overriding it with a different task must
        # raise rather than silently change semantics.
        with pytest.raises(ValueError, match="overwrite"):
            _call_score(
                llm,
                TEXTS_1[0],
                TEXTS_2[0],
                pooling_params=PoolingParams(task="embed"),
            )


# =============================================================================
# TestClassifyInterface
# =============================================================================


class TestClassifyInterface:
    """Cross encoder models expose a classification interface alongside `score`."""

    @pytest.mark.skip_global_cleanup
    def test_encode_with_classify_task_returns_single_label(self, llm: LLM):
        outputs = llm.encode(PROMPT, pooling_task="classify", use_tqdm=False)
        assert len(outputs) == 1
        assert len(outputs[0].outputs.data) == 1

    @pytest.mark.skip_global_cleanup
    def test_classify_entry_point_returns_single_prob(self, llm: LLM):
        outputs = llm.classify(PROMPT, use_tqdm=False)
        assert len(outputs) == 1
        assert len(outputs[0].outputs.probs) == 1
