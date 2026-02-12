# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Mapping
from typing import Any

from vllm import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import make_async

from ..base.io_processor import PoolingIOProcessor
from .protocol import (
    RerankRequest,
    ScoreRequest,
)
from .utils import ScoreData, get_score_prompt


class CrossEncoderIOProcessor(PoolingIOProcessor):
    async def pre_process(
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

        return engine_prompts, request_prompts

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
            score_template=self.chat_template,
        )
        self._validate_input(request, engine_prompt["prompt_token_ids"], full_prompt)
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return full_prompt, engine_prompt

    async def post_process(self, *args, **kwargs):
        pass


class EmbeddingScoreIOProcessor(PoolingIOProcessor):
    async def pre_process(
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


class LateInteractionIOProcessor(EmbeddingScoreIOProcessor):
    pass
