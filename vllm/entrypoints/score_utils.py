# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any, Union

from torch.nn import CosineSimilarity

from vllm.inputs import SingletonPrompt, TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               PreTrainedTokenizer,
                                               PreTrainedTokenizerFast)
from typing_extensions import Required, TypeAlias, TypedDict
from vllm.entrypoints.chat_utils import ChatCompletionContentPartImageParam, ChatCompletionContentPartImageEmbedsParam

ScoreContentPartParam: TypeAlias = Union[ChatCompletionContentPartImageParam, ChatCompletionContentPartImageEmbedsParam]
class ScoreMultiModalParam(TypedDict, total=False):

    content: Required[list[ScoreContentPartParam]]
    """The contents of the message."""


def _cosine_similarity(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    embed_1: list[PoolingRequestOutput],
    embed_2: list[PoolingRequestOutput],
) -> list[PoolingRequestOutput]:

    scorer = CosineSimilarity(0)
    scores: Union[list[PoolingRequestOutput]] = []

    for emb_1, emb_2 in zip(embed_1, embed_2):
        pair_score = scorer(emb_1.outputs.data, emb_2.outputs.data)

        padding = []
        if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                    None)) is not None:
            padding = [pad_token_id]

        tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

        scores.append(
            PoolingRequestOutput(
                request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                outputs=pair_score,
                prompt_token_ids=tokens,
                finished=True))

    return scores


def _validate_score_input_lens(
    data_1: Union[Sequence[SingletonPrompt], ScoreMultiModalParam],
    data_2: Union[Sequence[SingletonPrompt], ScoreMultiModalParam],
):
    len_1 = len(data_1)
    len_2 = len(data_2)
    if isinstance(data_1, dict):
        len_1 = len(data_1["content"])
    if isinstance(data_2, dict):
        len_2 = len(data_2["content"])

    if len_1 > 1 and len_1 != len_2:
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len_1 == 0:
        raise ValueError("At least one text element must be given")
    if len_2 == 0:
        raise ValueError("At least one text_pair element must be given")

# TODO: it will be better to implement this as parse_chat_messages
def formatting_prompts(
    model_arch: str,
    tokenizer: AnyTokenizer,
    tokenization_kwargs: dict[str, Any],
    query: SingletonPrompt,
    doc: SingletonPrompt,
    query_type: str = 'text',
    doc_type: str = 'text',
    prefix_str: str = '',
) -> TokensPrompt:

    engine_prompt = TokensPrompt()
    if 'JinaVLForRanking' in model_arch:

        # Format content part
        if doc_type == 'image':
            doc_part = "**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
            if engine_prompt.get('multi_modal_data') is None:
                engine_prompt['multi_modal_data'] = {}

            if engine_prompt['multi_modal_data'].get('image') is None:
                engine_prompt['multi_modal_data']['image'] = []

            engine_prompt['multi_modal_data']['image'] += [
                doc['multi_modal_data']['image']
            ]
        else:
            doc_part = f"**Document**:\n{doc}"

        # Format query part
        if query_type == 'image':
            query_part = "**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
            if engine_prompt.get('multi_modal_data') is None:
                engine_prompt['multi_modal_data'] = {}

            if engine_prompt['multi_modal_data'].get('image') is None:
                engine_prompt['multi_modal_data']['image'] = []

            engine_prompt['multi_modal_data']['image'] += [
                query['multi_modal_data']['image']
            ]
        else:
            query_part = f"**Query**:\n{query}"

        # Combine parts
        prompt = doc_part + '\n' + query_part

        # Add prefix if provided
        if prefix_str:
            prompt = prefix_str + '\n' + prompt

        engine_prompt['prompt_token_ids'] = tokenizer(
            prompt, **tokenization_kwargs)["input_ids"] + [100]
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    return engine_prompt

# TODO: implement this as parse_chat_messages_futures
def parse_score_content_futures(
    content: list[str, ScoreContentPartParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> tuple[list[SingletonPrompt], Awaitable[Optional[MultiModalDataDict]]]:
    pass