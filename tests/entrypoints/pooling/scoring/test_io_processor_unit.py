# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor helpers."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import TokenizersBackend

from vllm import TokensPrompt
from vllm.entrypoints.pooling.scoring.io_processor import (
    BiEncoderIOProcessor,
    CrossEncoderIOProcessor,
    JinaRankingIOProcessor,
    LateInteractionIOProcessor,
    _apply_post_tokenization_to_token_type_ids,
)
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.renderers import TokenizeParams

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _DummyTokenizer:
    truncation_side: str = "left"
    # Outside the range of the prompt ids below, so a test can tell a pad
    # token apart from a real one.
    pad_token_id: int = 99999


@pytest.fixture
def llm_reranker_processor() -> CrossEncoderIOProcessor:
    vocab = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "a": 3,
        "bc": 4,
        "##b": 5,
        "##c": 6,
    }
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    processor = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace(enable_prompt_embeds=False)
    processor.tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    processor.supports_score_template = False
    processor.use_sep_token = False
    processor.model = None
    return processor


def test_llm_reranker_tokenization_is_independent_of_nonbinding_doc_limit(
    llm_reranker_processor: CrossEncoderIOProcessor,
):
    encode_kwargs = {"add_special_tokens": True}

    _, uncapped = llm_reranker_processor.get_score_prompt("a", "bc", encode_kwargs)
    _, nonbinding = llm_reranker_processor.get_score_prompt(
        "a", "bc", encode_kwargs, max_tokens_per_doc=10
    )

    expected = [1, 3, 5, 6, 2]
    assert uncapped["prompt_token_ids"] == expected
    assert nonbinding["prompt_token_ids"] == expected


def test_token_type_ids_stay_aligned_with_a_truncated_padded_prompt():
    """The cross-encoder segment boundary must survive truncate + pad.

    `token_type_ids` are parallel to `prompt_token_ids` and are reduced to a
    single boundary index by `compress_token_type_ids`. If the two arrays are
    truncated and padded in different orders they no longer describe the same
    positions, and the model is told the query segment is empty.
    """
    tokenizer = _DummyTokenizer()
    num_query, num_doc = 20, 30
    prompt = TokensPrompt(prompt_token_ids=list(range(num_query + num_doc)))
    token_type_ids = [0] * num_query + [1] * num_doc

    tok_params = TokenizeParams(
        max_total_tokens=100,
        pad_prompt_tokens=-1,
        truncate_prompt_tokens=40,
        truncation_side="left",
    )

    processed_prompt = tok_params.apply_post_tokenization(
        tokenizer,
        prompt,
    )
    assert "prompt_token_ids" in processed_prompt
    token_type_ids = _apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, token_type_ids
    )

    prompt_token_ids = processed_prompt["prompt_token_ids"]
    assert len(token_type_ids) == len(prompt_token_ids)

    # Keeping the last 40 tokens drops the first 10 query tokens, so 10 query
    # tokens survive and the document starts at index 10.
    first_doc = compress_token_type_ids(token_type_ids)
    assert first_doc == 10
    assert prompt_token_ids[:first_doc] == list(range(10, num_query))
    assert prompt_token_ids[first_doc:40] == list(range(num_query, 50))


def _pooling_output(
    request_id: str,
    data: torch.Tensor,
    error: RequestError | None = None,
) -> PoolingRequestOutput:
    return PoolingRequestOutput(
        request_id=request_id,
        outputs=PoolingOutput(data),
        prompt_token_ids=[1],
        num_cached_tokens=0,
        finished=True,
        error=error,
    )


@pytest.mark.parametrize(
    "processor_type", [BiEncoderIOProcessor, LateInteractionIOProcessor]
)
@pytest.mark.parametrize("failed_index", [0, 1])
def test_paired_scoring_preserves_request_error(processor_type, failed_index):
    processor = processor_type.__new__(processor_type)
    processor.pad_token_id = None
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    outputs = [
        _pooling_output("query", torch.tensor([1.0, 2.0])),
        _pooling_output("document", torch.tensor([3.0, 4.0])),
    ]
    outputs[failed_index] = _pooling_output(
        outputs[failed_index].request_id,
        torch.empty(0),
        error,
    )

    (result,) = processor._post_process(outputs, n_queries=1)

    assert result.request_id == "query_document"
    assert result.error is error
    assert result.outputs is outputs[failed_index].outputs


def test_jina_scoring_preserves_request_error_before_indexing_output():
    processor = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    failed = _pooling_output("request", torch.empty(0), error)

    assert processor._post_process([failed], n_queries=1) == [failed]
