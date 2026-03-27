# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TypeAlias, cast

import torch.nn.functional as F

from vllm import PoolingRequestOutput
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
    PoolingServeContext,
)
from vllm.inputs import EngineInput
from vllm.renderers import TokenizeParams
from vllm.tasks import PoolingTask
from vllm.utils.mistral import is_mistral_tokenizer

from .protocol import RerankRequest, ScoreRequest, ScoreResponse
from .typing import ScoreInputs, ScoringData
from .utils import validate_score_input

ScoringServeContext: TypeAlias = PoolingServeContext[ScoreResponse]


class BiEncoderIOProcessor(PoolingIOProcessor):
    name = "bi-encoder"
    pooling_task: PoolingTask = "embed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tokenizer = self.renderer.get_tokenizer()
        if is_mistral_tokenizer(tokenizer):
            raise ValueError("MistralTokenizer not supported for cross-encoding")

        self.architecture = self.model_config.architecture
        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.pad_token_id = tokenizer.pad_token_id

    def create_pooling_params(self, request):
        return request.to_pooling_params(self.pooling_task)

    #######################################
    # online APIs

    def pre_process_online(self, ctx: ScoringServeContext):
        request = ctx.request

        if isinstance(request, ScoreRequest):
            data_1 = request.data_1
            data_2 = request.data_2
        elif isinstance(request, RerankRequest):
            data_1 = request.query
            data_2 = request.documents
        else:
            raise ValueError(f"Invalid {self.name} request type")

        scoring_data = self.validate_score_inputs(data_1, data_2)
        tok_params = request.build_tok_params(self.model_config)
        engine_inputs = self._pre_process(scoring_data, tok_params)

        ctx.engine_inputs = engine_inputs
        ctx.intermediates = len(scoring_data.data_1)

    def post_process_online(
        self,
        ctx: ScoringServeContext,
    ):
        if ctx.final_res_batch is None:
            raise ValueError("Final response batch not available")

        if ctx.intermediates is None:
            raise ValueError("data_1 len not available")

        ctx.final_res_batch = self._post_process(
            outputs=ctx.final_res_batch, offset=cast(int, ctx.intermediates)
        )

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}
        truncate_prompt_tokens = (
            None
            if ctx.tokenization_kwargs is None
            else ctx.tokenization_kwargs.pop("truncate_prompt_tokens", None)
        )

        tok_params = TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            max_total_tokens_param="max_model_len",
        )
        return self._pre_process(ctx.prompts, tok_params)

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        return self._post_process(outputs=ctx.outputs, offset=ctx.intermediates)

    #######################################
    # helpers

    def validate_score_inputs(
        self, data_1: ScoreInputs, data_2: ScoreInputs
    ) -> ScoringData:
        scoring_data = validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=self.is_multimodal_model,
            architecture=self.architecture,
        )
        return scoring_data

    def _pre_process(
        self,
        scoring_data: ScoringData,
        tok_params: TokenizeParams,
    ):
        prompts: list[str] = []
        for maybe_text in scoring_data.data_1 + scoring_data.data_2:
            if not isinstance(maybe_text, str):
                raise NotImplementedError(
                    "Embedding scores currently do not support multimodal input."
                )
            prompts.append(maybe_text)

        return self._preprocess_completion_offline(
            prompts=prompts, tok_params=tok_params
        )

    def _post_process(self, outputs: list[PoolingRequestOutput], offset: int):
        emb_data_1 = outputs[:offset]
        emb_data_2 = outputs[offset:]

        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        final_res_batch: list[PoolingRequestOutput] = []
        for emb_1, emb_2 in zip(emb_data_1, emb_data_2):
            pair_score = F.cosine_similarity(
                emb_1.outputs.data.float(), emb_2.outputs.data.float(), dim=0
            )

            padding: list[int] = []
            if self.pad_token_id is not None:
                padding = [self.pad_token_id]

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            final_res_batch.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=pair_score,
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )
        return final_res_batch
