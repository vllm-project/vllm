# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Sequence
from typing import Any, TypeAlias

import torch.nn.functional as F

from vllm import PoolingParams, PoolingRequestOutput, TokensPrompt
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
    PoolingServeContext,
)
from vllm.inputs import EngineInput
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import safe_apply_chat_template
from vllm.tasks import PoolingTask
from vllm.utils.mistral import is_mistral_tokenizer

from ...chat_utils import ChatTemplateResolutionError
from .protocol import RerankRequest, ScoreRequest, ScoringRequest
from .typing import ScoreData, ScoreInput, ScoringData
from .utils import (
    compress_token_type_ids,
    compute_maxsim_score,
    parse_score_data,
    score_data_to_prompts,
    validate_score_input,
)

ScoringServeContext: TypeAlias = PoolingServeContext[ScoringRequest]


class ScoringIOProcessor(PoolingIOProcessor):
    name: str
    pooling_task: PoolingTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = self.renderer.get_tokenizer()
        self.architecture = self.model_config.architecture
        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.pad_token_id = self.tokenizer.pad_token_id

    def create_pooling_params(self, request):
        return request.to_pooling_params(self.pooling_task)

    def valid_inputs(
        self,
        data_1: ScoreInput | list[ScoreInput],
        data_2: ScoreInput | list[ScoreInput],
    ) -> ScoringData:
        scoring_data = validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=self.is_multimodal_model,
            architecture=self.architecture,
        )
        return scoring_data


class BiEncoderIOProcessor(ScoringIOProcessor):
    name = "bi-encoder"
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
        engine_inputs = self._pre_process(
            scoring_data,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        ctx.engine_inputs = engine_inputs
        ctx.n_queries = len(scoring_data.data_1)

    def post_process_online(
        self,
        ctx: ScoringServeContext,
    ):
        assert ctx.final_res_batch is not None
        assert isinstance(ctx.n_queries, int)

        ctx.final_res_batch = self._post_process(
            outputs=ctx.final_res_batch, n_queries=ctx.n_queries
        )

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        assert isinstance(ctx.prompts, ScoringData)
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(
            **(ctx.tokenization_kwargs or {})
        )
        return self._pre_process(ctx.prompts, tok_params)

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        assert ctx.n_queries is not None
        return self._post_process(outputs=ctx.outputs, n_queries=ctx.n_queries)

    #######################################
    # helpers

    def _pre_process(
        self,
        scoring_data: ScoringData,
        tok_params: TokenizeParams,
        prompt_extras: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        data_1 = score_data_to_prompts(scoring_data.data_1, "query", self.model_config)
        data_2 = score_data_to_prompts(
            scoring_data.data_2, "document", self.model_config
        )

        return self._preprocess_completion_offline(
            prompts=data_1 + data_2, tok_params=tok_params, prompt_extras=prompt_extras
        )

    def _post_process(self, outputs: list[PoolingRequestOutput], n_queries: int):
        emb_data_1 = outputs[:n_queries]
        emb_data_2 = outputs[n_queries:]

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
    name = "late-interaction"
    pooling_task: PoolingTask = "token_embed"

    def _post_process(self, outputs: list[PoolingRequestOutput], n_queries: int):
        # Split into query and document embeddings
        emb_data_1 = outputs[:n_queries]
        emb_data_2 = outputs[n_queries:]

        # Expand queries if 1:N scoring
        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        final_res_batch: list[PoolingRequestOutput] = []
        padding: list[int] = []
        if (pad_token_id := self.pad_token_id) is not None:
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


class FlashLateInteractionIOProcessor(LateInteractionIOProcessor):
    name = "flash-late-interaction"

    def _post_process(self, outputs: list[PoolingRequestOutput], n_queries: int):
        return outputs


class CrossEncoderIOProcessor(ScoringIOProcessor):
    name = "cross-encoder"
    pooling_task: PoolingTask = "classify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if is_mistral_tokenizer(self.tokenizer):
            raise ValueError("MistralTokenizer not supported for cross-encoding")

        from vllm.model_executor.model_loader import get_model_cls
        from vllm.model_executor.models.interfaces import supports_score_template

        model = get_model_cls(self.model_config)
        self.supports_score_template = supports_score_template(model)
        self.model = model if self.supports_score_template else None
        self.use_sep_token = self.model_config.use_sep_token

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
        pooling_params = self.create_pooling_params(request)

        engine_inputs, pooling_params_list = self._pre_process(
            scoring_data,
            tok_params,
            pooling_params,
            chat_template=self.chat_template,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        ctx.engine_inputs = engine_inputs
        ctx.pooling_params = pooling_params_list

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        assert isinstance(ctx.prompts, ScoringData)
        assert not isinstance(ctx.pooling_params, Sequence)

        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(
            **(ctx.tokenization_kwargs or {})
        )
        engine_inputs, pooling_params_list = self._pre_process(
            ctx.prompts, tok_params, ctx.pooling_params, ctx.chat_template
        )
        ctx.pooling_params = pooling_params_list
        return engine_inputs

    #######################################
    # helpers

    def _pre_process(
        self,
        scoring_data: ScoringData,
        tok_params: TokenizeParams,
        pooling_params: PoolingParams | None,
        chat_template: str | None = None,
        prompt_extras: dict[str, Any] | None = None,
    ) -> tuple[Sequence[EngineInput], list[PoolingParams]]:
        # todo: support prompt_extras
        arrival_time = time.time()

        data_1 = scoring_data.data_1
        data_2 = scoring_data.data_2

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        if pooling_params is None:
            pooling_params = PoolingParams(task="classify")

        pooling_params_list = list[PoolingParams]()
        engine_inputs = list[EngineInput]()
        for q, d in zip(data_1, data_2):
            _, engine_prompt = self.get_score_prompt(
                data_1=q,
                data_2=d,
                encode_kwargs=tok_params.get_encode_kwargs(),
                chat_template=chat_template,
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):
                params = pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                params.extra_kwargs = {"compressed_token_type_ids": compressed}
                pooling_params_list.append(params)
            else:
                pooling_params_list.append(pooling_params)

            tok_params.apply_post_tokenization(self.tokenizer, engine_prompt)
            engine_inputs.append(
                self.renderer.process_for_engine(engine_prompt, arrival_time)
            )
        return engine_inputs, pooling_params_list

    def get_score_prompt(
        self,
        data_1: ScoreData,
        data_2: ScoreData,
        encode_kwargs: dict[str, Any],
        chat_template: str | None = None,
    ):
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_1, prompt_2, mm_data, mm_uuids = parse_score_data(
            data_1,
            data_2,
            model_config,
        )

        def default_tokenizer_encode():
            if self.supports_score_template:
                assert self.model is not None
                full_prompt = self.model.get_score_template(prompt_1, prompt_2)
                if full_prompt is None:
                    raise ValueError("Get empty score template from model")

                prompt_inputs = tokenizer(full_prompt, **encode_kwargs)
            else:
                if self.use_sep_token:
                    # cross_encoder models defaults to using separating token.
                    prompt_inputs = tokenizer(
                        text=prompt_1, text_pair=prompt_2, **encode_kwargs
                    )
                    full_prompt = tokenizer.decode(prompt_inputs["input_ids"])
                else:
                    # `llm as reranker` defaults to not using separating token.
                    full_prompt = prompt_1 + prompt_2
                    prompt_inputs = tokenizer(text=full_prompt, **encode_kwargs)
            return full_prompt, prompt_inputs

        # FIXME: For now, we only apply a template when one is explicitly provided.
        # We cannot rely on the tokenizer's chat template because many models
        # inherit junk templates from their base LLM, which breaks both the models
        # and the tests that use them.
        if chat_template is None:
            full_prompt, prompt_inputs = default_tokenizer_encode()
        else:
            # FIXME:
            # Try applying a score template from the CLI arg or tokenizer_config.json
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
                prompt_inputs = tokenizer(full_prompt, **encode_kwargs)
            except ChatTemplateResolutionError:
                full_prompt, prompt_inputs = default_tokenizer_encode()

        engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["input_ids"])

        if (token_type_ids := prompt_inputs.get("token_type_ids")) is not None:
            engine_prompt["token_type_ids"] = token_type_ids

        if self.model is not None:
            self.model.post_process_tokens(engine_prompt)

        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            engine_prompt["multi_modal_uuids"] = mm_uuids

        return full_prompt, engine_prompt


class JinaRankingIOProcessorMixin:
    @staticmethod
    def sanitize_input(text: str, special_tokens: dict[str, str]) -> str:
        for token in special_tokens.values():
            text = text.replace(token, "")
        return text

    @staticmethod
    def format_docs_prompts_func(
        query: str,
        docs: list[str],
        special_tokens: dict[str, str] | None = None,
        instruction: str | None = None,
        no_thinking: bool = True,
    ) -> str:
        # TODO: Try converting the code below into a chat template.

        default_special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        if special_tokens is None:
            special_tokens = default_special_tokens

        query = JinaRankingIOProcessorMixin.sanitize_input(query, special_tokens)
        docs = [
            JinaRankingIOProcessorMixin.sanitize_input(doc, special_tokens)
            for doc in docs
        ]

        prefix = (
            "<|im_start|>system\n"
            "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "  # noqa: E501
            "If the query is a question, how relevant a passage is depends on how well it answers the question. "  # noqa: E501
            "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "  # noqa: E501
            "If an instruction is provided, you should follow the instruction when determining the ranking."  # noqa: E501
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        if no_thinking:
            suffix += "<think>\n\n</think>\n\n"

        doc_emb_token = special_tokens["doc_embed_token"]
        query_emb_token = special_tokens["query_embed_token"]

        prompt = (
            f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "  # noqa: E501
            f"Rank the passages based on their relevance to query: {query}\n"
        )

        if instruction:
            prompt += f"<instruct>\n{instruction}\n</instruct>\n"

        doc_prompts = [
            f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
            for i, doc in enumerate(docs)
        ]
        prompt += "\n".join(doc_prompts) + "\n"
        prompt += f"<query>\n{query}{query_emb_token}\n</query>"

        return prefix + prompt + suffix

    @staticmethod
    def ensure_str(data: Sequence[Any]) -> list[str]:
        text: list[str] = []
        for prompt in data:
            if not isinstance(prompt, str):
                raise ValueError(
                    "The JinaForRanking model only supports text as input."
                )
            text.append(prompt)
        return text


class JinaRankingIOProcessor(LateInteractionIOProcessor, JinaRankingIOProcessorMixin):
    name = "jina-reranking-scoring"
    pooling_task: PoolingTask = "token_embed"

    def _pre_process(
        self,
        scoring_data: ScoringData,
        tok_params: TokenizeParams,
        prompt_extras: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        queries = self.ensure_str(scoring_data.data_1)
        docs = self.ensure_str(scoring_data.data_2)

        if len(queries) == 1:
            prompts = [self.format_docs_prompts_func(query=queries[0], docs=docs)]
        else:
            prompts = [
                self.format_docs_prompts_func(query=q, docs=[d])
                for q, d in zip(queries, docs)
            ]

        return self._preprocess_completion_offline(
            prompts=prompts, tok_params=tok_params, prompt_extras=prompt_extras
        )

    def _post_process(self, outputs: list[PoolingRequestOutput], n_queries: int):
        final_res_batch: list[PoolingRequestOutput] = []

        for i in range(len(outputs)):
            embeds = outputs[i].outputs.data.float()

            # The JinaForRanking model concatenates docs first, then query.
            # Let's stay consistent with this novel design.
            query_embeds = embeds[-1]
            doc_embeds = embeds[:-1]

            scores = F.cosine_similarity(query_embeds, doc_embeds)

            for score in scores:
                final_res_batch.append(
                    PoolingRequestOutput(
                        request_id=outputs[i].request_id,
                        outputs=score,
                        prompt_token_ids=outputs[i].prompt_token_ids,
                        num_cached_tokens=outputs[i].num_cached_tokens,
                        finished=True,
                    )
                )
        return final_res_batch


ScoringIOProcessors: dict[str, type[ScoringIOProcessor]] = {
    p.name: p
    for p in [
        BiEncoderIOProcessor,
        LateInteractionIOProcessor,
        JinaRankingIOProcessor,
        FlashLateInteractionIOProcessor,
        CrossEncoderIOProcessor,
    ]
}
