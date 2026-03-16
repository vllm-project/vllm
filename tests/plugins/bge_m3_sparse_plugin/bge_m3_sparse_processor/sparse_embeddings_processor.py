# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.config import ModelConfig, PoolerConfig, VllmConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import EmbedRequestMixin
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
    EMBED_TASKS,
    SparseEmbeddingCompletionRequestMixin,
    SparseEmbeddingResponse,
    SparseEmbeddingResponseData,
    SparseEmbeddingTokenWeight,
)

logger = init_logger(__name__)


class BgeM3SparseEmbeddingsProcessor(
    IOProcessor[SparseEmbeddingCompletionRequestMixin, SparseEmbeddingResponse]
):
    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        self.offline_requests: list[SparseEmbeddingCompletionRequestMixin] = []
        self.online_requests: dict[str, SparseEmbeddingCompletionRequestMixin] = {}
        self.renderer: BaseRenderer = renderer
        self.default_pooling_params = {}
        pooler_config: PoolerConfig = vllm_config.model_config.pooler_config
        if pooler_config is not None:
            for param in ["use_activation", "dimensions"]:
                if getattr(pooler_config, param, None) is None:
                    continue
                self.default_pooling_params[param] = getattr(pooler_config, param)
        self.embed_dimensions = vllm_config.model_config.embedding_size
        self.embed_request_queue: list[EmbedRequestMixin] = []
        logger.info(self)

    def __repr__(self) -> str:
        return (
            f"BgeM3SparseEmbeddingsProcessor("
            f"embed_dimensions={self.embed_dimensions}, "
            f"default_pooling_params={self.default_pooling_params})"
        )

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is None:
            params = PoolingParams()
        # refer to PoolingCompletionRequest.to_pooling_params
        # set and verify pooling params
        params.skip_reading_prefix_cache = True

        raw_embed_request = self.embed_request_queue.pop(0)
        if raw_embed_request.task not in EMBED_TASKS:
            raise ValueError(
                f"Unsupported task {raw_embed_request}, "
                f"Supported tasks are {EMBED_TASKS}"
            )
        if raw_embed_request.task == "dense":
            params.task = "embed"
            params.skip_reading_prefix_cache = False
        elif raw_embed_request.task == "sparse":
            params.task = "token_classify"
        else:
            params.task = "embed&token_classify"
        params.use_activation = raw_embed_request.use_activation
        params.dimensions = raw_embed_request.dimensions

        model_config: ModelConfig = self.vllm_config.model_config
        for param in self.default_pooling_params:
            if getattr(params, param, None) is None:
                setattr(params, param, self.default_pooling_params[param])

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
            self.embed_request_queue.extend(prompt.to_embed_requests_online())
        else:
            self.offline_requests.append(prompt)
            self.embed_request_queue.extend(prompt.to_embed_requests_offline())
        return prompt.input

    def _get_sparse_embedding_request(self, request_id: str | None = None):
        if request_id:
            return self.online_requests.pop(request_id, None)
        return self.offline_requests.pop(0)

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
        raw_request = self._get_sparse_embedding_request(request_id)
        has_dense_embed = raw_request.task in ["dense", "dense&sparse"]
        has_sparse_embed = raw_request.task in ["sparse", "dense&sparse"]
        embed_dimensions = 0
        if has_dense_embed:
            embed_dimensions = (
                self.embed_dimensions
                if raw_request.dimensions is None
                else raw_request.dimensions
            )
        for idx in range(len(model_output)):
            mo = model_output[idx]
            sparse_embedding_dict: dict[int, float] = {}
            num_prompt_tokens += len(mo.prompt_token_ids)
            dense_embedding: list[float] | None = None
            sparse_embedding: list[SparseEmbeddingTokenWeight] | None = None
            if has_dense_embed:
                dense_embedding = mo.outputs.data[:embed_dimensions].tolist()
            if has_sparse_embed:
                sparse_weights = mo.outputs.data[embed_dimensions:].tolist()
                if len(mo.prompt_token_ids) != len(sparse_weights):
                    # this is the case that add_special_tokens is True,
                    # which means first token and last token are special tokens
                    mo.prompt_token_ids = mo.prompt_token_ids[1:]
                for token_id, weight in zip(mo.prompt_token_ids, sparse_weights):
                    sparse_embedding_dict[token_id] = max(
                        weight, sparse_embedding_dict.get(token_id, 0.0)
                    )
                sparse_embedding = self._build_sparse_embedding_token_weights(
                    sparse_embedding_dict,
                    raw_request.return_tokens,
                )

            response_data.append(
                SparseEmbeddingResponseData(
                    index=idx,
                    sparse_embedding=sparse_embedding,
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
