# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import torch.nn.functional as F

from vllm import PoolingParams, PoolingRequestOutput, TokensPrompt
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import safe_apply_chat_template
from vllm.renderers.inputs.preprocess import (
    extract_target_prompt,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.tasks import PoolingTask
from vllm.utils.mistral import is_mistral_tokenizer

from ...chat_utils import ChatTemplateResolutionError
from ..base.io_processor import PoolingIOProcessor
from ..pooling.protocol import PoolingCompletionRequest
from ..typing import (
    AnyOfflineInputsContext,
    AnyPoolingRequest,
    AnyRenderParam,
    EncodeChatRenderParams,
    EncodeCMPLRenderParams,
    OfflineEncodeInputsContext,
    OfflineOutputsContext,
    OfflineScoringInputsContext,
    PoolingEngineInput,
    PoolingServeContext,
    RequestFactory,
    RequestGenerator,
    ScoringRenderParams,
)
from .protocol import RerankRequest, ScoreRequest, ScoringRequest
from .typing import ScoreData, ScoreInput, ScoringData
from .utils import (
    compress_token_type_ids,
    compute_maxsim_score,
    get_num_special_tokens_for_pair,
    parse_score_data,
    score_data_to_prompts,
    truncate_text_to_tokens,
    validate_score_input,
)

ScoringServeContext: TypeAlias = PoolingServeContext[ScoringRequest]


def _apply_post_tokenization_to_token_type_ids(
    tokenizer: Any,
    tok_params: TokenizeParams,
    token_type_ids: list[int],
) -> list[int]:
    pad_length = tok_params.pad_prompt_tokens
    if pad_length is not None and pad_length < 0:
        pad_length = tok_params.max_input_tokens

    if pad_length is not None and pad_length > len(token_type_ids):
        pad_token_type_id = token_type_ids[-1] if token_type_ids else 0
        token_type_ids = token_type_ids + [pad_token_type_id] * (
            pad_length - len(token_type_ids)
        )

    max_length = tok_params.truncate_prompt_tokens
    if max_length is not None and max_length < 0:
        max_length = tok_params.max_input_tokens

    if max_length is None or max_length >= len(token_type_ids):
        return token_type_ids
    if max_length == 0:
        return token_type_ids[:0]

    side = tok_params.truncation_side or (
        tokenizer.truncation_side if tokenizer is not None else None
    )
    if side == "left":
        return token_type_ids[-max_length:]

    return token_type_ids[:max_length]


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

    def _validate_token_limit(self, value: int, name: str) -> None:
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if value >= self.model_config.max_model_len:
            raise ValueError(
                f"{name} ({value}) must be less "
                f"than max_model_len ({self.model_config.max_model_len})."
            )

    def _get_token_limits(
        self,
        request: ScoringRequest | None = None,
        pooling_params: PoolingParams | None = None,
    ) -> tuple[int, int]:
        """Extract and validate token limits from request or pooling_params."""
        if request is not None:
            max_tokens_per_query = getattr(request, "max_tokens_per_query", 0)
            max_tokens_per_doc = getattr(request, "max_tokens_per_doc", 0)
        else:
            extra = (
                (pooling_params.extra_kwargs or {})
                if pooling_params is not None
                else {}
            )
            max_tokens_per_query = extra.get("max_tokens_per_query", 0)
            max_tokens_per_doc = extra.get("max_tokens_per_doc", 0)

        if max_tokens_per_query != 0:
            self._validate_token_limit(max_tokens_per_query, "max_tokens_per_query")
        if max_tokens_per_doc != 0:
            self._validate_token_limit(max_tokens_per_doc, "max_tokens_per_doc")
        return max_tokens_per_query, max_tokens_per_doc

    def _truncate_scoring_data(
        self,
        scoring_data: ScoringData,
        max_tokens_per_query: int = 0,
        max_tokens_per_doc: int = 0,
    ) -> ScoringData:
        """Truncate query/document texts to token limits."""
        data_1 = scoring_data.data_1
        data_2 = scoring_data.data_2
        if max_tokens_per_query > 0:
            data_1 = [
                truncate_text_to_tokens(d, self.tokenizer, max_tokens_per_query)
                if isinstance(d, str)
                else d
                for d in data_1
            ]
        if max_tokens_per_doc > 0:
            data_2 = [
                truncate_text_to_tokens(d, self.tokenizer, max_tokens_per_doc)
                if isinstance(d, str)
                else d
                for d in data_2
            ]
        return ScoringData(data_1=data_1, data_2=data_2)

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

    def valid_inputs_online(self, request: AnyPoolingRequest):
        if isinstance(request, ScoreRequest):
            data_1 = request.data_1
            data_2 = request.data_2
        elif isinstance(request, RerankRequest):
            data_1 = request.query
            data_2 = request.documents
        else:
            raise ValueError(f"Invalid {request.__class__.__name__} request type")

        scoring_data = self.valid_inputs(data_1, data_2)
        return scoring_data


class BiEncoderIOProcessor(ScoringIOProcessor):
    name = "bi-encoder"
    pooling_task: PoolingTask = "embed"

    #######################################
    # online APIs

    def get_request_factory_online(
        self, ctx: PoolingServeContext
    ) -> Sequence[AnyRenderParam]:
        request = ctx.request
        scoring_data = self.valid_inputs_online(request)

        max_tokens_per_query, max_tokens_per_doc = self._get_token_limits(
            request=request
        )
        if max_tokens_per_query > 0 or max_tokens_per_doc > 0:
            scoring_data = self._truncate_scoring_data(
                scoring_data, max_tokens_per_query, max_tokens_per_doc
            )

        data_1 = score_data_to_prompts(scoring_data.data_1, "query", self.model_config)
        data_2 = score_data_to_prompts(
            scoring_data.data_2, "document", self.model_config
        )
        prompts = data_1 + data_2
        ctx.n_queries = len(data_1)

        prompts_seq = prompt_to_seq(prompts)
        parsed_prompts = [
            parse_model_prompt(self.model_config, prompt) for prompt in prompts_seq
        ]
        num_requests = len(parsed_prompts)

        tok_params = request.build_tok_params(self.model_config)
        params_seq = self._params_to_seq(ctx.pooling_params, num_requests)
        seq_lora_requests = self._lora_request_to_seq(ctx.lora_request, num_requests)
        seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

        requests = [
            EncodeCMPLRenderParams(
                prompts=parsed_prompts[i],
                tok_params=tok_params,
                prompt_extras=ctx.prompt_extras,
                skip_mm_cache=False,
                params=params_seq[i],
                lora_requests=seq_lora_requests[i],
                priorities=seq_priority[i],
            )
            for i in range(num_requests)
        ]
        return requests

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

    def get_request_factory_offline(
        self, ctx: AnyOfflineInputsContext
    ) -> tuple[RequestFactory, int]:
        assert isinstance(ctx, OfflineScoringInputsContext)

        max_tokens_per_query, max_tokens_per_doc = self._get_token_limits(
            pooling_params=ctx.pooling_params
        )

        scoring_data = ctx.scoring_data
        if max_tokens_per_query > 0 or max_tokens_per_doc > 0:
            scoring_data = self._truncate_scoring_data(
                scoring_data, max_tokens_per_query, max_tokens_per_doc
            )

        data_1 = score_data_to_prompts(scoring_data.data_1, "query", self.model_config)
        data_2 = score_data_to_prompts(
            scoring_data.data_2, "document", self.model_config
        )
        prompts = data_1 + data_2

        return super().get_request_factory_offline(
            OfflineEncodeInputsContext(
                pooling_task=self.pooling_task,
                prompts=prompts,
                tokenization_kwargs=ctx.tokenization_kwargs,
                pooling_params=ctx.pooling_params,
                lora_request=ctx.lora_request,
                priorities=ctx.priorities,
            )
        )

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        assert ctx.n_queries is not None
        return self._post_process(outputs=ctx.outputs, n_queries=ctx.n_queries)

    #######################################
    # helpers

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

    def post_process_online(
        self,
        ctx: ScoringServeContext,
    ):
        assert ctx.query_final_res_batch is not None
        assert ctx.final_res_batch is not None
        assert isinstance(ctx.n_queries, int)

        # Expand queries if 1:N scoring
        if len(ctx.query_final_res_batch) == 1:
            ctx.query_final_res_batch = ctx.query_final_res_batch * len(
                ctx.final_res_batch
            )

        final_res_batch: list[PoolingRequestOutput] = []
        for d1, d2 in zip(ctx.query_final_res_batch, ctx.final_res_batch):
            padding: list[int] = []
            if (pad_token_id := self.pad_token_id) is not None:
                padding = [pad_token_id]

            tokens = d1.prompt_token_ids + padding + d2.prompt_token_ids

            final_res_batch.append(
                PoolingRequestOutput(
                    request_id=f"{d1.request_id}_{d2.request_id}",
                    outputs=d2.outputs,
                    prompt_token_ids=tokens,
                    num_cached_tokens=d1.num_cached_tokens + d2.num_cached_tokens,
                    finished=True,
                )
            )
        ctx.final_res_batch = final_res_batch


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

    def get_request_factory_online(
        self, ctx: PoolingServeContext
    ) -> Sequence[AnyRenderParam]:
        request = ctx.request
        scoring_data = self.valid_inputs_online(request)
        data_1 = scoring_data.data_1
        data_2 = scoring_data.data_2
        num_requests = len(data_2)

        if len(data_1) == 1:
            data_1 = data_1 * num_requests

        max_tokens_per_query, max_tokens_per_doc = self._get_token_limits(
            request=request
        )

        tok_params = request.build_tok_params(self.model_config)
        seq_lora_requests = self._lora_request_to_seq(ctx.lora_request, num_requests)
        seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

        requests = [
            ScoringRenderParams(
                data_1=data_1[i],
                data_2=data_2[i],
                chat_template=self.chat_template,
                max_tokens_per_query=max_tokens_per_query,
                max_tokens_per_doc=max_tokens_per_doc,
                tok_params=tok_params,
                prompt_extras=ctx.prompt_extras,
                skip_mm_cache=False,
                params=ctx.pooling_params,
                lora_requests=seq_lora_requests[i],
                priorities=seq_priority[i],
            )
            for i in range(num_requests)
        ]

        return requests

    #######################################
    # offline APIs

    def get_request_factory_offline(
        self, ctx: AnyOfflineInputsContext
    ) -> tuple[RequestFactory, int]:
        assert isinstance(ctx, OfflineScoringInputsContext)

        data_1 = ctx.scoring_data.data_1
        data_2 = ctx.scoring_data.data_2
        num_requests = len(data_2)

        if len(data_1) == 1:
            data_1 = data_1 * num_requests

        max_tokens_per_query, max_tokens_per_doc = self._get_token_limits(
            pooling_params=ctx.pooling_params
        )
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(
            **(ctx.tokenization_kwargs or {})
        )
        prompt_extras = ctx.pooling_params.extra_kwargs

        seq_lora_requests = self._lora_request_to_seq(ctx.lora_request, num_requests)
        seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

        def request_factory() -> RequestGenerator:
            for i in range(num_requests):
                yield ScoringRenderParams(
                    data_1=data_1[i],
                    data_2=data_2[i],
                    chat_template=ctx.chat_template,
                    max_tokens_per_query=max_tokens_per_query,
                    max_tokens_per_doc=max_tokens_per_doc,
                    tok_params=tok_params,
                    prompt_extras=prompt_extras,
                    skip_mm_cache=False,
                    params=ctx.pooling_params,
                    lora_requests=seq_lora_requests[i],
                    priorities=seq_priority[i],
                )

        return request_factory, num_requests

    #######################################
    # helpers

    def render(
        self,
        render_params: EncodeCMPLRenderParams
        | EncodeChatRenderParams
        | ScoringRenderParams,
    ) -> PoolingEngineInput:
        if "data_1" not in render_params:
            raise ValueError(
                f"Unsupported render_params type {render_params.__class__.__name__}"
            )
        render_params = cast(ScoringRenderParams, render_params)

        arrival_time = time.time()

        tok_params = render_params["tok_params"]
        params = render_params["params"]
        prompt_extras = render_params["prompt_extras"]

        _, engine_prompt = self.get_score_prompt(
            data_1=render_params["data_1"],
            data_2=render_params["data_2"],
            encode_kwargs=tok_params.get_encode_kwargs(),
            chat_template=render_params["chat_template"],
            max_tokens_per_query=render_params["max_tokens_per_query"],
            max_tokens_per_doc=render_params["max_tokens_per_doc"],
            chat_template_kwargs=prompt_extras.get("chat_template_kwargs")
            if prompt_extras
            else None,
        )

        tok_params.apply_post_tokenization(self.tokenizer, engine_prompt)

        if token_type_ids := engine_prompt.pop("token_type_ids", None):
            params = params.clone()
            compressed = compress_token_type_ids(
                _apply_post_tokenization_to_token_type_ids(
                    self.tokenizer, tok_params, token_type_ids
                )
            )
            params.extra_kwargs = {
                **(params.extra_kwargs or {}),
                "compressed_token_type_ids": compressed,
            }

        engine_prompt_extras = (
            {
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := prompt_extras.get(k)) is not None
            }
            if prompt_extras
            else None
        )

        if engine_prompt_extras:
            target_prompt = extract_target_prompt(self.model_config, engine_prompt)
            target_prompt.update(engine_prompt_extras)

        engine_input = self.renderer.process_for_engine(engine_prompt, arrival_time)

        return PoolingEngineInput(
            prompts=engine_input,
            params=params,
            lora_requests=render_params["lora_requests"],
            priorities=render_params["priorities"],
        )

    def get_score_prompt(
        self,
        data_1: ScoreData,
        data_2: ScoreData,
        encode_kwargs: dict[str, Any],
        chat_template: str | None = None,
        max_tokens_per_query: int = 0,
        max_tokens_per_doc: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_1, prompt_2, mm_data, mm_uuids = parse_score_data(
            data_1,
            data_2,
            model_config,
        )

        # Apply truncation before defining closures
        if max_tokens_per_query > 0 and isinstance(prompt_1, str):
            prompt_1 = truncate_text_to_tokens(
                prompt_1, tokenizer, max_tokens_per_query
            )
        if max_tokens_per_doc > 0 and isinstance(prompt_2, str):
            prompt_2 = truncate_text_to_tokens(prompt_2, tokenizer, max_tokens_per_doc)

        def default_tokenizer_encode():
            local_kwargs = encode_kwargs.copy()

            if self.supports_score_template:
                assert self.model is not None
                full_prompt = self.model.get_score_template(prompt_1, prompt_2)
                if full_prompt is None:
                    raise ValueError("Get empty score template from model")

                prompt_inputs = tokenizer(full_prompt, **local_kwargs)
            else:
                if self.use_sep_token:
                    # cross_encoder models defaults to using separating token.
                    if max_tokens_per_doc > 0 and isinstance(prompt_2, str):
                        query_tokens = tokenizer.encode(
                            prompt_1, add_special_tokens=False
                        )
                        num_special = get_num_special_tokens_for_pair(tokenizer)
                        doc_limit_max_length = (
                            len(query_tokens) + max_tokens_per_doc + num_special
                        )
                        existing_max_length = local_kwargs.get("max_length")
                        if existing_max_length is not None:
                            effective_max_length = min(
                                doc_limit_max_length, existing_max_length
                            )
                        else:
                            effective_max_length = doc_limit_max_length
                        local_kwargs["truncation"] = "only_second"
                        local_kwargs["max_length"] = effective_max_length

                    prompt_inputs = tokenizer(
                        text=prompt_1, text_pair=prompt_2, **local_kwargs
                    )
                    full_prompt = tokenizer.decode(prompt_inputs["input_ids"])
                else:
                    # `llm as reranker` defaults to not using separating token.
                    if max_tokens_per_doc > 0 and isinstance(prompt_2, str):
                        query_ids = tokenizer.encode(prompt_1, add_special_tokens=False)
                        doc_ids = tokenizer.encode(prompt_2, add_special_tokens=False)
                        doc_ids = doc_ids[:max_tokens_per_doc]
                        input_ids = query_ids + doc_ids
                        full_prompt = tokenizer.decode(input_ids)
                        prompt_inputs = {"input_ids": input_ids}
                    else:
                        full_prompt = prompt_1 + prompt_2
                        prompt_inputs = tokenizer(text=full_prompt, **local_kwargs)
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
                _safe_kwargs = chat_template_kwargs or {}
                _reserved = {"chat_template", "tools", "tokenize"}
                _unexpected = _reserved & _safe_kwargs.keys()
                if _unexpected:
                    raise ValueError(
                        "chat_template_kwargs contains reserved keys that "
                        f"conflict with fixed scorer arguments: {_unexpected}"
                    )
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
                    **_safe_kwargs,
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

        def sanitize_input(text: str) -> str:
            for token in special_tokens.values():
                text = text.replace(token, "")
            return text

        query = sanitize_input(query)
        docs = [sanitize_input(doc) for doc in docs]

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
            instruction = sanitize_input(instruction)
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

    def get_request_factory_online(
        self, ctx: PoolingServeContext
    ) -> Sequence[AnyRenderParam]:
        request = ctx.request
        ctx.n_queries = 1

        prompt_extras = ctx.prompt_extras
        scoring_data = self.valid_inputs_online(request)

        max_tokens_per_query, max_tokens_per_doc = self._get_token_limits(
            request=request
        )

        if max_tokens_per_query > 0 or max_tokens_per_doc > 0:
            scoring_data = self._truncate_scoring_data(
                scoring_data, max_tokens_per_query, max_tokens_per_doc
            )

        queries = self.ensure_str(scoring_data.data_1)
        docs = self.ensure_str(scoring_data.data_2)

        chat_template_kwargs = (
            prompt_extras.get("chat_template_kwargs") if prompt_extras else None
        )
        instruction = (
            chat_template_kwargs.get("instruction") if chat_template_kwargs else None
        )

        if len(queries) == 1:
            prompts = [
                self.format_docs_prompts_func(
                    query=queries[0], docs=docs, instruction=instruction
                )
            ]
        else:
            prompts = [
                self.format_docs_prompts_func(
                    query=q, docs=[d], instruction=instruction
                )
                for q, d in zip(queries, docs)
            ]

        ctx.request = PoolingCompletionRequest(task="token_embed", input=prompts)
        requests = PoolingIOProcessor.get_request_factory_online(self, ctx)
        ctx.request = request
        return requests

    def get_request_factory_offline(
        self, ctx: AnyOfflineInputsContext
    ) -> tuple[RequestFactory, int]:
        assert isinstance(ctx, OfflineScoringInputsContext)

        scoring_data = ctx.scoring_data
        prompt_extras = ctx.pooling_params.extra_kwargs

        queries = self.ensure_str(scoring_data.data_1)
        docs = self.ensure_str(scoring_data.data_2)
        chat_template_kwargs = (
            prompt_extras.get("chat_template_kwargs") if prompt_extras else None
        )
        instruction = (
            chat_template_kwargs.get("instruction") if chat_template_kwargs else None
        )

        if len(queries) == 1:
            prompts = [
                self.format_docs_prompts_func(
                    query=queries[0], docs=docs, instruction=instruction
                )
            ]
        else:
            prompts = [
                self.format_docs_prompts_func(
                    query=q, docs=[d], instruction=instruction
                )
                for q, d in zip(queries, docs)
            ]

        return PoolingIOProcessor.get_request_factory_offline(
            self,
            OfflineEncodeInputsContext(
                pooling_task=self.pooling_task,
                prompts=prompts,
                tokenization_kwargs=ctx.tokenization_kwargs,
                pooling_params=ctx.pooling_params,
                lora_request=ctx.lora_request,
                priorities=ctx.priorities,
            ),
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
