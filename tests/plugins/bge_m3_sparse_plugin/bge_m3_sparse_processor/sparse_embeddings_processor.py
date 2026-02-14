# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorResponse,
)
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import (
    IOProcessor,
    IOProcessorInput,
    IOProcessorOutput,
)
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer

from .types import (
    SparseEmbeddingCompletionRequestMixin,
    SparseEmbeddingResponse,
    SparseEmbeddingResponseData,
    SparseEmbeddingTokenWeight,
)

logger = init_logger(__name__)


class BgeM3SparseEmbeddingsProcessor(IOProcessor):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.offline_requests: list[SparseEmbeddingCompletionRequestMixin] = []
        self.online_requests: dict[str, SparseEmbeddingCompletionRequestMixin] = {}
        self.renderer: BaseRenderer = None

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
        request: Any = None,
    ) -> PoolingParams:
        if params is None:
            params = PoolingParams()
        # refer to PoolingCompletionRequest.to_pooling_params
        if request is not None:
            params.task = request.task
            params.truncate_prompt_tokens = request.truncate_prompt_tokens

    def parse_request(self, request_data: Any) -> IOProcessorInput:
        # for vllm.entrypoints.llm.LLM, offline mode, calls `encode` directly.
        if isinstance(request_data, dict):
            return SparseEmbeddingCompletionRequestMixin(**request_data)
        raise ValueError("Unable to parse request_data")

    def pre_process(
        self,
        prompt: SparseEmbeddingCompletionRequestMixin,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        if request_id is not None:
            assert request_id not in self.online_requests, "request_id duplicated"
            self.online_requests[request_id] = prompt
        else:
            self.offline_requests.append(prompt)
        if self.renderer is None and "renderer" in kwargs:
            self.renderer = kwargs["renderer"]
        return prompt.input

    def _get_sparse_embedding_request(self, request_id: str | None = None):
        if request_id:
            return self.online_requests.pop(request_id)
        return self.offline_requests.pop()

    def _build_sparse_embedding_token_weights(
        self, request_id: str | None, sparse_embedding: dict[int, float]
    ) -> list[SparseEmbeddingTokenWeight]:
        request = self._get_sparse_embedding_request(request_id)
        assert request, "illegal request"
        token_ids = sparse_embedding.keys()
        token_weights = sparse_embedding.values()
        tokens = [None] * len(token_ids)

        if request.return_token_id_texts_map:
            tokens = self.renderer.get_tokenizer().convert_ids_to_tokens(token_ids)
        sparse_embedding_output: list[SparseEmbeddingTokenWeight] = []
        for token_id, weight, token in zip(token_ids, token_weights, tokens):
            sparse_embedding_output.append(
                SparseEmbeddingTokenWeight(
                    token_id=token_id, weight=weight, token=token
                )
            )
        return sparse_embedding_output

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        num_prompt_tokens = 0
        response_data = []
        for idx in range(len(model_output)):
            mo = model_output[idx]
            sparse_embedding: dict[int, float] = {}
            num_prompt_tokens += len(mo.prompt_token_ids)
            if len(mo.prompt_token_ids) != len(mo.outputs.data):
                # this is the case that add_special_tokens is True,
                # which means first token and last token are special tokens
                mo.prompt_token_ids = mo.prompt_token_ids[1:]
            for token_id, weight in zip(mo.prompt_token_ids, mo.outputs.data):
                sparse_embedding[token_id] = max(
                    weight, sparse_embedding.get(token_id, 0.0)
                )
            response_data.append(
                SparseEmbeddingResponseData(
                    index=idx,
                    sparse_embedding=self._build_sparse_embedding_token_weights(
                        request_id, sparse_embedding
                    ),
                )
            )

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )
        resp = SparseEmbeddingResponse(
            request_id=request_id,
            data=response_data,
            usage=usage,
        )

        return resp

    def output_to_response(
        self, plugin_output: IOProcessorOutput
    ) -> IOProcessorResponse:
        return IOProcessorResponse(
            request_id=plugin_output.request_id,
            data=plugin_output,
        )
