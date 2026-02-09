# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.protocol import CompletionRequestMixin
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
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
from vllm.renderers import TokenizeParams
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.sampling_params import SamplingParams

from .types import (
    SparseEmbeddingCompletionRequestMixin,
    SparseEmbeddingResponse,
    SparseEmbeddingResponseData,
)

logger = init_logger(__name__)


class BgeM3SparseEmbeddingsProcessor(IOProcessor):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len
        assert self.max_model_len, f"max_model_len is not configured, {vllm_config=}"

    def validate_or_generate_params(
        self, params: SamplingParams | PoolingParams | None = None, request: Any = None
    ) -> SamplingParams | PoolingParams:
        if request:
            params = PoolingParams(
                task=request.task,
                truncate_prompt_tokens=request.truncate_prompt_tokens
                if isinstance(request, IOProcessorRequest)
                else None,
            )
        return params

    def parse_request(self, request: Any) -> IOProcessorInput:
        # for vllm.entrypoints.llm.LLM, offline mode, calls `encode` directly.
        if type(request) is dict and "data" in request:
            return SparseEmbeddingCompletionRequestMixin(input=[request["data"]])

        # for online serving `pooling` endpoint
        if isinstance(request, IOProcessorRequest):
            if not hasattr(request, "data"):
                raise ValueError("missing 'data' field in OpenAIBaseModel Request")
            request_data = request.data
            kwargs = {"truncate_prompt_tokens": request.truncate_prompt_tokens}
            if type(request_data) is list:
                kwargs["input"] = request_data
                return SparseEmbeddingCompletionRequestMixin(**kwargs)
            if type(request_data) is str:
                kwargs["input"] = [request_data]
                return SparseEmbeddingCompletionRequestMixin(**kwargs)
            if type(request_data) is dict:
                kwargs.update(request_data)
                return SparseEmbeddingCompletionRequestMixin(**kwargs)
        raise ValueError("Unable to parse request")

    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        prompts = prompt_to_seq(prompt.input)
        if "renderer" not in kwargs:
            return prompts
        renderer = kwargs["renderer"]
        parsed_prompts = [
            parse_model_prompt(self.vllm_config.model_config, prompt)
            for prompt in prompts
        ]
        engine_prompts = renderer.render_cmpl(
            parsed_prompts,
            self._build_render_params(prompt),
        )
        return engine_prompts

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
            sparse_embedding = {}
            num_prompt_tokens += len(mo.prompt_token_ids)
            if len(mo.prompt_token_ids) != len(mo.outputs.data):
                # this is the case that add_special_tokens is True,
                # which means first token and last token are special tokens
                mo.prompt_token_ids = mo.prompt_token_ids[1:]
            for token_id, weight in zip(mo.prompt_token_ids, mo.outputs.data):
                if token_id not in sparse_embedding:
                    sparse_embedding[token_id] = weight
                    continue
                if weight > sparse_embedding[token_id]:
                    sparse_embedding[token_id] = weight
            response_data.append(
                SparseEmbeddingResponseData(
                    index=idx, sparse_embedding=sparse_embedding
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

    def _build_render_params(self, request: CompletionRequestMixin):
        encoder_config = self.vllm_config.model_config.encoder_config or {}
        return TokenizeParams(
            max_total_tokens=self.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=request.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )
