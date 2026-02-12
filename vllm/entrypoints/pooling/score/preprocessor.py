import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vllm import TokensPrompt
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import make_async

from ...openai.chat_completion.protocol import ChatCompletionRequest
from ...openai.engine.serving import AnyRequest
from ...serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from .protocol import (
    RerankRequest,
    ScoreDataRequest,
    ScoreQueriesDocumentsRequest,
    ScoreRequest,
    ScoreTextRequest,
)
from .utils import ScoreData, get_score_prompt


class ScorePreProcessorBase:
    def __init__(
        self,
        model_config,
        renderer,
        io_processor,
        input_processor,
        score_template: str | None = None,
    ):
        self.score_template = score_template
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self.model_config = model_config
        self.renderer = renderer
        self.io_processor = io_processor
        self.input_processor = input_processor

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)
        max_model_len = self.model_config.max_model_len

        # Note: EmbeddingRequest, ClassificationRequest,
        # and ScoreRequest doesn't have max_tokens
        if isinstance(
            request,
            (
                ScoreDataRequest,
                ScoreTextRequest,
                ScoreQueriesDocumentsRequest,
                RerankRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreDataRequest: "score",
                    ScoreTextRequest: "score",
                    ScoreQueriesDocumentsRequest: "score",
                }
                operation = operations.get(type(request), "embedding generation")
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input.",
                    parameter="input_tokens",
                    value=token_num,
                )
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages.",
                parameter="input_tokens",
                value=token_num,
            )

        if max_tokens is not None and token_num + max_tokens > max_model_len:
            raise VLLMValidationError(
                "'max_tokens' or 'max_completion_tokens' is too large: "
                f"{max_tokens}. This model's maximum context length is "
                f"{max_model_len} tokens and your request has "
                f"{token_num} input tokens ({max_tokens} > {max_model_len}"
                f" - {token_num}).",
                parameter="max_tokens",
                value=max_tokens,
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)


class CrossEncoderPreProcessor(ScorePreProcessorBase):
    async def __call__(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ):
        tokenizer = self.renderer.get_tokenizer()
        model_config = self.model_config

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        tok_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]
        preprocess_async = make_async(
            self._preprocess_score,
            executor=self._tokenizer_executor,
        )
        preprocessed_prompts = await asyncio.gather(
            *(
                preprocess_async(
                    request=request,
                    tokenizer=tokenizer,
                    tokenization_kwargs=tok_kwargs,
                    data_1=t1,
                    data_2=t2,
                )
                for t1, t2 in input_pairs
            )
        )

        request_prompts: list[str] = []
        engine_prompts: list[TokensPrompt] = []
        for full_prompt, engine_prompt in preprocessed_prompts:
            request_prompts.append(full_prompt)
            engine_prompts.append(engine_prompt)

    def _preprocess_score(
        self,
        request: RerankRequest | ScoreRequest,
        tokenizer: TokenizerLike,
        tokenization_kwargs: dict[str, Any],
        data_1: ScoreData,
        data_2: ScoreData,
    ) -> tuple[str, TokensPrompt]:
        model_config = self.model_config
        full_prompt, engine_prompt = get_score_prompt(
            model_config=model_config,
            data_1=data_1,
            data_2=data_2,
            tokenizer=tokenizer,
            tokenization_kwargs=tokenization_kwargs,
            score_template=self.score_template,
        )
        self._validate_input(request, engine_prompt["prompt_token_ids"], full_prompt)
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return full_prompt, engine_prompt


class EmbeddingScorePreProcessor(ScorePreProcessorBase):
    async def __call__(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ):
        input_texts: list[str] = []
        for text in data_1 + data_2:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Embedding scores currently do not support multimodal input."
                )
            input_texts.append(text)

        model_config = self.model_config
        tokenizer = self.renderer.get_tokenizer()

        encode_async = make_async(
            tokenizer.encode,
            executor=self._tokenizer_executor,
        )

        tokenization_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
        tokenized_prompts = await asyncio.gather(
            *(encode_async(t, **tokenization_kwargs) for t in input_texts)
        )

        engine_prompts: list[TokensPrompt] = []
        for tok_result, input_text in zip(tokenized_prompts, input_texts):
            text_token_prompt = self._validate_input(request, tok_result, input_text)

            engine_prompts.append(
                TokensPrompt(prompt_token_ids=text_token_prompt["prompt_token_ids"])
            )

        return input_texts, engine_prompts


class LateInteractionPreProcessor(EmbeddingScorePreProcessor):
    pass
