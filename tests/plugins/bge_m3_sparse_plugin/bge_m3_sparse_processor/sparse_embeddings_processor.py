# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.config import ModelConfig, PoolerConfig, VllmConfig, get_current_vllm_config
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import (
    IOProcessor,
)
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens

from .types import (
    SparseEmbeddingCompletionRequestMixin,
    SparseEmbeddingResponse,
    SparseEmbeddingResponseData,
    SparseEmbeddingTokenWeight,
)

logger = init_logger(__name__)

_BGE_M3_EMBED_DIM = 1024


class BgeM3SparseEmbeddingsProcessor(
    IOProcessor[SparseEmbeddingCompletionRequestMixin, SparseEmbeddingResponse]
):
    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        self.offline_requests: list[SparseEmbeddingCompletionRequestMixin] = []
        self.online_requests: dict[str, SparseEmbeddingCompletionRequestMixin] = {}
        self.renderer: BaseRenderer = renderer

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is None:
            params = PoolingParams()
        # refer to PoolingCompletionRequest.to_pooling_params
        params.task = "embed&token_classify"
        # set and verify pooling params
        params.skip_reading_prefix_cache = True

        model_config: ModelConfig = get_current_vllm_config().model_config
        pooler_config: PoolerConfig = model_config.pooler_config
        if pooler_config is not None:
            for param in ["use_activation", "dimensions"]:
                if getattr(pooler_config, param, None) is None:
                    continue

                if getattr(params, param, None) is None:
                    setattr(params, param, getattr(pooler_config, param))

        if params.use_activation is None:
            params.use_activation = True
        if params.dimensions is not None:
            if not model_config.is_matryoshka:
                raise ValueError(
                    f'Model "{model_config.served_model_name}" does not '
                    f"support matryoshka representation, "
                    f"changing output dimensions will lead to poor results."
                )

            mds = model_config.matryoshka_dimensions
            if mds is not None:
                if params.dimensions not in mds:
                    raise ValueError(
                        f"Model {model_config.served_model_name!r} "
                        f"only supports {str(mds)} matryoshka dimensions, "
                        f"use other output dimensions will "
                        f"lead to poor results."
                    )
            elif params.dimensions < 1:
                raise ValueError("Dimensions must be greater than 0")
        return params

    def parse_request(
        self, request_data: object
    ) -> SparseEmbeddingCompletionRequestMixin:
        # for vllm.entrypoints.llm.LLM, offline mode, calls `encode` directly.
        if isinstance(request_data, dict):
            return SparseEmbeddingCompletionRequestMixin(**request_data)
        raise TypeError("request_data should be a dictionary")

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
        return prompt.input

    def _get_sparse_embedding_request(self, request_id: str | None = None):
        if request_id:
            return self.online_requests.pop(request_id, None)
        return self.offline_requests.pop()

    def _build_sparse_embedding_token_weights(
        self,
        sparse_embedding: dict[int, float],
        return_tokens: bool = False,
    ) -> list[SparseEmbeddingTokenWeight]:
        token_ids = sparse_embedding.keys()
        token_weights = sparse_embedding.values()
        tokens = [None] * len(token_ids)

        if return_tokens and self.renderer is not None:
            tokens = convert_ids_list_to_tokens(
                self.renderer.get_tokenizer(), token_ids
            )
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
    ) -> SparseEmbeddingResponse:
        num_prompt_tokens = 0
        response_data = []
        return_tokens = self._get_sparse_embedding_request(request_id).return_tokens
        for idx in range(len(model_output)):
            mo = model_output[idx]
            sparse_embedding: dict[int, float] = {}
            num_prompt_tokens += len(mo.prompt_token_ids)
            dense_embedding = mo.outputs.data[:_BGE_M3_EMBED_DIM].tolist()
            sparse_weights = mo.outputs.data[_BGE_M3_EMBED_DIM:].tolist()
            if len(mo.prompt_token_ids) != len(sparse_weights):
                # this is the case that add_special_tokens is True,
                # which means first token and last token are special tokens
                mo.prompt_token_ids = mo.prompt_token_ids[1:]
            for token_id, weight in zip(mo.prompt_token_ids, sparse_weights):
                sparse_embedding[token_id] = max(
                    weight, sparse_embedding.get(token_id, 0.0)
                )
            response_data.append(
                SparseEmbeddingResponseData(
                    index=idx,
                    sparse_embedding=self._build_sparse_embedding_token_weights(
                        sparse_embedding,
                        return_tokens,
                    ),
                    dense_embedding=dense_embedding,
                )
            )

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )
        resp = SparseEmbeddingResponse(
            data=response_data,
            usage=usage,
        )

        return resp
