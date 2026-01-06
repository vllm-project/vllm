# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateResolutionError
from vllm.entrypoints.score_utils import get_score_prompt
from vllm.inputs import TokensPrompt
from vllm.tokenizers import get_tokenizer

# A cross-encoder model for testing
CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def assert_prompt_tokenization_consistent(
    tokenizer, full_prompt, engine_prompt, add_special_tokens=True
):
    """Verify that engine_prompt token_ids match tokenizing full_prompt."""
    expected_ids = tokenizer(full_prompt, add_special_tokens=add_special_tokens)[
        "input_ids"
    ]
    actual_ids = engine_prompt["prompt_token_ids"]
    assert actual_ids == expected_ids, (
        f"Token IDs don't match.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )


@pytest.fixture(scope="module")
def cross_encoder_model_config():
    return ModelConfig(
        CROSS_ENCODER_MODEL_ID,
        runner="pooling",
    )


@pytest.fixture(scope="module")
def cross_encoder_tokenizer(cross_encoder_model_config):
    return get_tokenizer(
        CROSS_ENCODER_MODEL_ID,
        trust_remote_code=cross_encoder_model_config.trust_remote_code,
    )


@pytest.fixture(scope="module")
def llm_reranker_model_config():
    """Model config for LLM-as-reranker style (no pad token)."""
    config = ModelConfig(
        CROSS_ENCODER_MODEL_ID,
        runner="pooling",
    )
    # use_sep_token is a property that reads from hf_config,
    # so we set it there to override the default (True)
    config.hf_config.use_sep_token = False
    return config


@pytest.fixture
def tokenization_kwargs():
    """Common tokenization kwargs used across tests."""
    return {"add_special_tokens": True, "return_tensors": None}


@pytest.fixture
def mock_model_with_score_template():
    """Mock model class that supports score template and tracks post_process calls."""

    class MockModelWithScoreTemplate:
        supports_score_template = True
        post_process_called: list[TokensPrompt] = []

        @staticmethod
        def get_score_template(p1: str, p2: str) -> str:
            return f"[QUERY]{p1}[SEP][DOC]{p2}"

        @staticmethod
        def post_process_tokens(prompt: TokensPrompt) -> None:
            MockModelWithScoreTemplate.post_process_called.append(prompt)

    return MockModelWithScoreTemplate


@pytest.fixture
def mock_model_no_score_template():
    """Mock model class that does not support score template."""

    class MockModelNoScoreTemplate:
        supports_score_template = False

    return MockModelNoScoreTemplate


class TestGetScorePrompt:
    """Tests for the get_score_prompt function."""

    def test_tokenization_kwargs_passed_through(
        self,
        llm_reranker_model_config,
        cross_encoder_tokenizer,
    ):
        """Test that tokenization kwargs are properly passed through."""
        data_1 = "Query text"
        data_2 = "Document text"

        # Test with truncation - custom kwargs for this test
        custom_tokenization_kwargs = {
            "add_special_tokens": True,
            "return_tensors": None,
            "truncation": True,
            "max_length": 20,
        }

        full_prompt, engine_prompt = get_score_prompt(
            llm_reranker_model_config,
            cross_encoder_tokenizer,
            custom_tokenization_kwargs,
            data_1,
            data_2,
        )

        assert isinstance(full_prompt, str)
        assert "prompt_token_ids" in engine_prompt
        # With max_length=20 and truncation, should not exceed this
        assert len(engine_prompt["prompt_token_ids"]) <= 20
        # Since truncation was applied, token_ids should be a prefix of full encoding
        full_ids = cross_encoder_tokenizer(full_prompt, add_special_tokens=True)[
            "input_ids"
        ]
        actual_ids = engine_prompt["prompt_token_ids"]
        assert full_ids[: len(actual_ids)] == actual_ids, (
            f"Token IDs are not a prefix of full encoding.\n"
            f"Full IDs:   {full_ids}\n"
            f"Actual IDs: {actual_ids}"
        )

    def test_model_supports_score_template(
        self,
        cross_encoder_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_with_score_template,
    ):
        """Test when model supports score template (no score_template arg)."""
        with patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=mock_model_with_score_template,
        ):
            full_prompt, engine_prompt = get_score_prompt(
                cross_encoder_model_config,
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "query text",
                "document text",
            )

        assert full_prompt == "[QUERY]query text[SEP][DOC]document text"
        assert "prompt_token_ids" in engine_prompt
        assert len(engine_prompt["prompt_token_ids"]) > 0
        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer, full_prompt, engine_prompt
        )

    def test_model_supports_score_template_but_custom_template_provided(
        self,
        cross_encoder_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_with_score_template,
    ):
        """Test when model supports score template but custom template is provided."""
        template = (
            'TEMPLATE_USED {{ messages[0]["content"] }} {{ messages[1]["content"] }}'
        )
        with (
            patch(
                "vllm.model_executor.model_loader.get_model_cls",
                return_value=mock_model_with_score_template,
            ),
        ):
            full_prompt, engine_prompt = get_score_prompt(
                cross_encoder_model_config,
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "query",
                "doc",
                score_template=template,  # Providing a template
            )

        assert "prompt_token_ids" in engine_prompt
        assert full_prompt == "TEMPLATE_USED query doc"

        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer, full_prompt, engine_prompt
        )

    def test_not_using_default_template(
        self,
        llm_reranker_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_no_score_template,
    ):
        # FIXME: For now, we only apply a template when one is explicitly provided.
        # We cannot rely on the tokenizer's chat template because many models
        # inherit junk templates from their base LLM, which breaks both the models
        # and the tests that use them.
        with (
            patch(
                "vllm.model_executor.model_loader.get_model_cls",
                return_value=mock_model_no_score_template,
            ),
            patch(
                "vllm.entrypoints.score_utils.apply_hf_chat_template",
                return_value="test querytest doc",
            ),
        ):
            full_prompt, engine_prompt = get_score_prompt(
                llm_reranker_model_config,
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "test query",
                "test doc",
            )

        assert full_prompt == "test querytest doc"
        assert "prompt_token_ids" in engine_prompt
        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer, full_prompt, engine_prompt
        )

    def test_fallback_with_sep_token(
        self,
        cross_encoder_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_no_score_template,
    ):
        """Test fallback path when ChatTemplateResolutionError
        and use_sep_token=True."""
        with (
            patch(
                "vllm.model_executor.model_loader.get_model_cls",
                return_value=mock_model_no_score_template,
            ),
            patch(
                "vllm.entrypoints.score_utils.apply_hf_chat_template",
                side_effect=ChatTemplateResolutionError("No template"),
            ),
        ):
            full_prompt, engine_prompt = get_score_prompt(
                cross_encoder_model_config,  # use_sep_token=True
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "query",
                "document",
            )

        assert "prompt_token_ids" in engine_prompt
        # Should have token_type_ids from text_pair encoding
        assert "token_type_ids" in engine_prompt
        assert "query" in full_prompt
        assert "document" in full_prompt
        assert full_prompt != "querydocument"
        assert (
            engine_prompt["prompt_token_ids"]
            == cross_encoder_tokenizer(
                "query", text_pair="document", add_special_tokens=True
            )["input_ids"]
        )

        # FIXME(?): add_special_tokens=False is needed because in this case
        # full_prompt is obtained by decoding the tokenized prompt, which includes
        # special tokens and we would get duplicated special tokens otherwise.
        # This is inconsistent with other cases.
        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer,
            full_prompt,
            engine_prompt,
            add_special_tokens=False,
        )

    def test_fallback_without_sep_token(
        self,
        llm_reranker_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_no_score_template,
    ):
        """Test fallback path when ChatTemplateResolutionError
        and use_sep_token=False."""
        with (
            patch(
                "vllm.model_executor.model_loader.get_model_cls",
                return_value=mock_model_no_score_template,
            ),
            patch(
                "vllm.entrypoints.score_utils.apply_hf_chat_template",
                side_effect=ChatTemplateResolutionError("No template"),
            ),
        ):
            full_prompt, engine_prompt = get_score_prompt(
                llm_reranker_model_config,  # use_sep_token=False
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "query",
                "document",
            )

        assert full_prompt == "querydocument"
        assert "prompt_token_ids" in engine_prompt
        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer, full_prompt, engine_prompt
        )

    def test_post_process_tokens_called(
        self,
        cross_encoder_model_config,
        cross_encoder_tokenizer,
        tokenization_kwargs,
        mock_model_with_score_template,
    ):
        """Test that post_process_tokens is called on the engine prompt."""
        # Reset the call tracker
        mock_model_with_score_template.post_process_called.clear()

        with (
            patch(
                "vllm.model_executor.model_loader.get_model_cls",
                return_value=mock_model_with_score_template,
            ),
            patch(
                "vllm.entrypoints.score_utils.apply_hf_chat_template",
                side_effect=ChatTemplateResolutionError("No template"),
            ),
        ):
            full_prompt, engine_prompt = get_score_prompt(
                cross_encoder_model_config,
                cross_encoder_tokenizer,
                tokenization_kwargs,
                "query",
                "doc",
            )

        # post_process_tokens should have been called once
        assert len(mock_model_with_score_template.post_process_called) == 1
        assert mock_model_with_score_template.post_process_called[0] is engine_prompt
        assert_prompt_tokenization_consistent(
            cross_encoder_tokenizer, full_prompt, engine_prompt
        )
