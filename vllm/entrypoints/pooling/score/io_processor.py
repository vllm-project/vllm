# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Mapping
from typing import Any

from vllm import PoolingOutput, PoolingRequestOutput, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import make_async

from ..base.io_processor import PoolingIOProcessor
from .protocol import (
    RerankRequest,
    ScoreRequest,
)
from .utils import (
    ScoreData,
    _cosine_similarity,
    compute_maxsim_score,
    get_score_prompt,
)


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
            self._pre_process,
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

    def _pre_process(
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

    async def post_process(self, engine_results):
        return [out for out in engine_results if out is not None]

    def create_pooling_params(self, request):
        return request.to_pooling_params("score")


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

    def create_pooling_params(self, request):
        return request.to_pooling_params("embed")

    async def post_process(
        self, data_1: list[ScoreData], data_2: list[ScoreData], engine_results
    ):
        embeddings = engine_results
        emb_data_1: list[PoolingRequestOutput] = []
        emb_data_2: list[PoolingRequestOutput] = []

        for i in range(0, len(data_1)):
            assert (emb := embeddings[i]) is not None
            emb_data_1.append(emb)

        for i in range(len(data_1), len(embeddings)):
            assert (emb := embeddings[i]) is not None
            emb_data_2.append(emb)

        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        tokenizer = self.renderer.get_tokenizer()
        final_res_batch = _cosine_similarity(
            tokenizer=tokenizer, embed_1=emb_data_1, embed_2=emb_data_2
        )
        return final_res_batch


class LateInteractionIOProcessor(EmbeddingScoreIOProcessor):
    def create_pooling_params(self, request):
        return request.to_pooling_params("token_embed")

    async def post_process(
        self, data_1: list[ScoreData], data_2: list[ScoreData], engine_results
    ):
        embeddings = engine_results

        # Split into query and document embeddings
        emb_data_1: list[PoolingRequestOutput] = []
        emb_data_2: list[PoolingRequestOutput] = []

        for i in range(0, len(data_1)):
            assert (emb := embeddings[i]) is not None
            emb_data_1.append(emb)

        for i in range(len(data_1), len(embeddings)):
            assert (emb := embeddings[i]) is not None
            emb_data_2.append(emb)

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
                    outputs=PoolingOutput(data=maxsim_score),
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )
        return final_res_batch
