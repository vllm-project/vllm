# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator, Sequence
from typing import cast

from vllm.config import VllmConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.utils.collection_utils import is_list_of

from .types import (
    QUERY_MAXLEN,
    ColBERTEmbeddingCompletionRequestMixin,
    ColBERTEmbeddingResponse,
    ColBERTEmbeddingResponseData,
)

QUERY_MARKER_TOKEN = "[QueryMarker]"
DOCUMENT_MARKER_TOKEN = "[DocumentMarker]"


class ColBERTQueryEmbeddingProcessor(
    IOProcessor[ColBERTEmbeddingCompletionRequestMixin, ColBERTEmbeddingResponse]
):
    """This IO processor only supports the ColBERT-style model jinaai/jina-colbert-v2.
    It does not support all ColBERT-style variants (e.g. colbert-ir/colbertv2.0).
    """

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        self.requests_cache: dict[str, ColBERTEmbeddingCompletionRequestMixin] = {}
        self.renderer: BaseRenderer = renderer
        # Context window (8192 for jinaai/jina-colbert-v2); caps document
        # content length minus the 3 special-token slots.
        self.max_model_len = vllm_config.model_config.max_model_len
        self._query_marker_id: int | None = None
        self._document_marker_id: int | None = None

    def __repr__(self) -> str:
        return (
            f"ColBERTQueryEmbeddingProcessor("
            f"query_maxlen={QUERY_MAXLEN}, "
            f"doc_maxlen={self.max_model_len}, "
            f"query_marker_token={QUERY_MARKER_TOKEN!r}, "
            f"document_marker_token={DOCUMENT_MARKER_TOKEN!r})"
        )

    def _resolve_marker_ids(self, tokenizer) -> tuple[int, int]:
        if self._query_marker_id is not None and self._document_marker_id is not None:
            return self._query_marker_id, self._document_marker_id

        unk_id = getattr(tokenizer, "unk_token_id", None)
        marker_ids: list[int] = []
        for marker in (QUERY_MARKER_TOKEN, DOCUMENT_MARKER_TOKEN):
            marker_id = tokenizer.convert_tokens_to_ids(marker)
            if marker_id is None or marker_id == unk_id:
                raise ValueError(
                    f"Marker token {marker!r} not found in the tokenizer "
                    "vocabulary. This plugin requires a ColBERT model whose "
                    "tokenizer defines both "
                    f"{QUERY_MARKER_TOKEN!r} and {DOCUMENT_MARKER_TOKEN!r} "
                    "(e.g. jinaai/jina-colbert-v2)."
                )
            marker_ids.append(marker_id)

        self._query_marker_id, self._document_marker_id = marker_ids
        return self._query_marker_id, self._document_marker_id

    def _iter_content_token_ids(
        self,
        tokenizer,
        request_input: list[int] | list[list[int]] | str | list[str],
    ) -> Iterator[list[int]]:
        if isinstance(request_input, str):
            yield tokenizer.encode(request_input, add_special_tokens=False)
            return

        if not isinstance(request_input, list) or not request_input:
            raise ValueError("input must be a non-empty string or list")

        if is_list_of(request_input, int):
            yield list(cast(list[int], request_input))
            return

        for item in request_input:
            if isinstance(item, str):
                yield tokenizer.encode(item, add_special_tokens=False)
            else:
                yield list(cast(list[int], item))

    def _build_query_prompt(
        self,
        tokenizer,
        content_ids: list[int],
    ) -> TokensPrompt:
        """[CLS] [QueryMarker] <tokens> [SEP] [MASK]... up to QUERY_MAXLEN."""
        query_marker_id, _ = self._resolve_marker_ids(tokenizer)
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token; cannot perform query expansion."
            )

        # [CLS], marker and [SEP] take 3 slots.
        content_ids = content_ids[: QUERY_MAXLEN - 3]
        token_ids = [
            tokenizer.cls_token_id,
            query_marker_id,
            *content_ids,
            tokenizer.sep_token_id,
        ]
        token_ids += [mask_token_id] * (QUERY_MAXLEN - len(token_ids))
        return TokensPrompt(prompt_token_ids=token_ids)

    def _build_document_prompt(
        self,
        tokenizer,
        content_ids: list[int],
    ) -> TokensPrompt:
        """[CLS] [DocumentMarker] <tokens> [SEP]"""
        _, document_marker_id = self._resolve_marker_ids(tokenizer)

        content_ids = content_ids[: self.max_model_len - 3]
        token_ids = [
            tokenizer.cls_token_id,
            document_marker_id,
            *content_ids,
            tokenizer.sep_token_id,
        ]
        return TokensPrompt(prompt_token_ids=token_ids)

    def parse_data(self, data: object) -> ColBERTEmbeddingCompletionRequestMixin:
        if isinstance(data, dict):
            return ColBERTEmbeddingCompletionRequestMixin(**data)
        raise TypeError("request data should be a dictionary")

    def pre_process(
        self,
        prompt: ColBERTEmbeddingCompletionRequestMixin,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        cache_key = request_id or "offline"
        assert cache_key not in self.requests_cache, "request_id duplicated"
        self.requests_cache[cache_key] = prompt

        tokenizer = self.renderer.get_tokenizer()
        prompts: list[TokensPrompt] = []
        for content_ids in self._iter_content_token_ids(tokenizer, prompt.input):
            if prompt.input_type == "query":
                prompts.append(self._build_query_prompt(tokenizer, content_ids))
            else:
                prompts.append(self._build_document_prompt(tokenizer, content_ids))
        return prompts

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is None:
            params = PoolingParams()
        params.task = "token_embed"
        params.skip_reading_prefix_cache = True
        return params

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> ColBERTEmbeddingResponse:
        raw_request = self.requests_cache.pop(request_id or "offline")

        num_prompt_tokens = 0
        response_data: list[ColBERTEmbeddingResponseData] = []
        for idx, output in enumerate(model_output):
            num_prompt_tokens += len(output.prompt_token_ids)
            response_data.append(
                ColBERTEmbeddingResponseData(
                    index=idx,
                    input_type=raw_request.input_type,
                    embedding=output.outputs.data.tolist(),
                )
            )

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )
        return ColBERTEmbeddingResponse(data=response_data, usage=usage)
