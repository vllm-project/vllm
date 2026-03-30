# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import torch.nn.functional as F

from vllm import PoolingParams, PoolingRequestOutput, TokensPrompt
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
    PoolingServeContext,
)
from vllm.inputs import EngineInput, TokensInput, tokens_input
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import safe_apply_chat_template
from vllm.tasks import PoolingTask, ScoreType
from vllm.utils.mistral import is_mistral_tokenizer

from ...chat_utils import ChatTemplateResolutionError
from .protocol import RerankRequest, ScoreRequest, ScoringRequest
from .typing import ScoreData, ScoreInputs, ScoringData
from .utils import (
    _apply_model_score_template,
    compress_token_type_ids,
    compute_maxsim_score,
    parse_score_data,
    score_data_to_prompts,
    validate_score_input,
)

ScoringServeContext: TypeAlias = PoolingServeContext[ScoringRequest]


class ScoringIOProcessor(PoolingIOProcessor):
    name: ScoreType
    pooling_task: PoolingTask

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

    def valid_inputs(self, data_1: ScoreInputs, data_2: ScoreInputs) -> ScoringData:
        scoring_data = validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=self.is_multimodal_model,
            architecture=self.architecture,
        )
        return scoring_data


class BiEncoderIOProcessor(ScoringIOProcessor):
    name: ScoreType = "bi-encoder"
    pooling_task: PoolingTask = "embed"

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

        scoring_data = self.valid_inputs(data_1, data_2)
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
        assert isinstance(ctx.prompts, ScoringData)

        encoder_config = self.model_config.encoder_config or {}
        truncate_prompt_tokens = (
            None
            if ctx.tokenization_kwargs is None
            else ctx.tokenization_kwargs.pop("truncate_prompt_tokens", None)
        )

        tok_params = TokenizeParams(
            max_total_tokens=self.model_config.max_model_len,
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
        assert ctx.offset is not None
        return self._post_process(outputs=ctx.outputs, offset=ctx.offset)

    #######################################
    # helpers

    def _pre_process(
        self,
        scoring_data: ScoringData,
        tok_params: TokenizeParams,
    ) -> Sequence[EngineInput]:
        data_1 = score_data_to_prompts(scoring_data.data_1, "query", self.model_config)
        data_2 = score_data_to_prompts(
            scoring_data.data_2, "document", self.model_config
        )

        return self._preprocess_completion_offline(
            prompts=data_1 + data_2, tok_params=tok_params
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


class LateInteractionIOProcessor(BiEncoderIOProcessor):
    name: ScoreType = "late-interaction"
    pooling_task: PoolingTask = "token_embed"

    def _post_process(self, outputs: list[PoolingRequestOutput], offset: int):
        # Split into query and document embeddings
        emb_data_1 = outputs[:offset]
        emb_data_2 = outputs[offset:]

        # Expand queries if 1:N scoring
        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        tokenizer = self.renderer.get_tokenizer()

        final_res_batch: list[PoolingRequestOutput] = []
        padding: list[int] = []
        if (pad_token_id := tokenizer.pad_token_id) is not None:
            padding = [pad_token_id]

        # Compute MaxSim scores
        for emb_1, emb_2 in zip(emb_data_1, emb_data_2):
            # emb_1.outputs.data: [query_len, dim]
            # emb_2.outputs.data: [doc_len, dim]
            q_emb = emb_1.outputs.data
            d_emb = emb_2.outputs.data

            maxsim_score = compute_maxsim_score(q_emb, d_emb)

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            final_res_batch.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=maxsim_score,
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )
        return final_res_batch


class CrossEncoderIOProcessor(ScoringIOProcessor):
    name: ScoreType = "cross-encoder"
    pooling_task: PoolingTask = "classify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from vllm.model_executor.model_loader import get_model_cls
        from vllm.model_executor.models.interfaces import supports_score_template

        self.model = get_model_cls(self.model_config)
        self.supports_score_template = supports_score_template(self.model)
        self.use_sep_token = self.model_config.use_sep_token

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        assert isinstance(ctx.prompts, ScoringData)

        data_1 = ctx.prompts.data_1
        data_2 = ctx.prompts.data_2
        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        pooling_params = ctx.pooling_params

        if pooling_params is None:
            pooling_params = PoolingParams(task="classify")

        assert isinstance(pooling_params, PoolingParams)

        if pooling_params.task is None:
            pooling_params.task = "classify"

        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]

        pooling_params_list = list[PoolingParams]()
        engine_prompts = list[TokensInput]()
        for q, d in input_pairs:
            _, token_prompt = self.get_score_prompt(
                data_1=q,
                data_2=d,
                tokenization_kwargs=ctx.tokenization_kwargs or {},
                chat_template=ctx.chat_template,
            )

            if token_type_ids := token_prompt.pop("token_type_ids", None):
                params = pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                params.extra_kwargs = {"compressed_token_type_ids": compressed}
                pooling_params_list.append(params)
            else:
                pooling_params_list.append(pooling_params)

            engine_prompts.append(tokens_input(**token_prompt))

        ctx.pooling_params = pooling_params_list
        return engine_prompts

    def get_score_prompt(
        self,
        tokenization_kwargs: dict[str, Any],
        data_1: ScoreData,
        data_2: ScoreData,
        chat_template: str | None = None,
    ):
        model_config = self.model_config
        tokenizer = self.renderer.tokenizer

        prompt_1, prompt_2, mm_data, mm_uuids = parse_score_data(
            data_1,
            data_2,
            model_config,
        )

        def default_tokenizer_encode():
            if self.supports_score_template:
                full_prompt = _apply_model_score_template(
                    model_config, prompt_1, prompt_2
                )
                prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)
            else:
                if self.use_sep_token:
                    # cross_encoder models defaults to using separating token.
                    prompt_inputs = tokenizer(
                        text=prompt_1, text_pair=prompt_2, **tokenization_kwargs
                    )
                    full_prompt = tokenizer.decode(prompt_inputs["input_ids"])
                else:
                    # `llm as reranker` defaults to not using separating token.
                    full_prompt = prompt_1 + prompt_2
                    prompt_inputs = tokenizer(text=full_prompt, **tokenization_kwargs)
            return full_prompt, prompt_inputs

        # FIXME: For now, we only apply a template when one is explicitly provided.
        # We cannot rely on the tokenizer's chat template because many models
        # inherit junk templates from their base LLM, which breaks both the models
        # and the tests that use them.
        if chat_template is None:
            full_prompt, prompt_inputs = default_tokenizer_encode()
        else:
            # FIXME: Try applying a score template from the CLI arg or tokenizer_config.json
            # If that fails because there is no such template,
            # fall back to the default implementation.
            try:
                full_prompt = safe_apply_chat_template(
                    model_config,
                    tokenizer,
                    [
                        {"role": "query", "content": prompt_1},
                        {"role": "document", "content": prompt_2},
                    ],
                    chat_template=chat_template,
                    tools=None,
                    tokenize=False,
                )
                prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)
            except ChatTemplateResolutionError:
                full_prompt, prompt_inputs = default_tokenizer_encode()

        engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["input_ids"])

        if (token_type_ids := prompt_inputs.get("token_type_ids")) is not None:
            engine_prompt["token_type_ids"] = token_type_ids

        if self.supports_score_template:
            self.model.post_process_tokens(engine_prompt)

        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            engine_prompt["multi_modal_uuids"] = mm_uuids

        return full_prompt, engine_prompt


ScoringIOProcessors: dict[ScoreType, type[ScoringIOProcessor]] = {
    "bi-encoder": BiEncoderIOProcessor,
    "late-interaction": LateInteractionIOProcessor,
    "cross-encoder": CrossEncoderIOProcessor,
}
