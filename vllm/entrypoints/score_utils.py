# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Optional, Union

from torch.nn import CosineSimilarity
from typing_extensions import Required, TypeAlias, TypedDict

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker, ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam, ChatCompletionContentPartTextParam,
    MultiModalItemTracker, _parse_chat_message_content_part)
from vllm.inputs import SingletonPrompt, TokensPrompt
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.outputs import PoolingRequestOutput
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               PreTrainedTokenizer,
                                               PreTrainedTokenizerFast)

ScoreContentPartParam: TypeAlias = Union[
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartImageEmbedsParam]


class ScoreMultiModalParam(TypedDict, total=False):
    """
    A specialized parameter type for scoring multimodal content
    
    The reasons why don't reuse `CustomChatCompletionMessageParam` directly:
    1. Score tasks don't need the 'role' field (user/assistant/system) that's required in chat completions
    2. Including chat-specific fields would confuse users about their purpose in scoring
    3. This is a more focused interface that only exposes what's needed for scoring
    """
    content: Required[list[ScoreContentPartParam]]
    """The multimodal contents"""


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
    data_1: Union[Sequence[SingletonPrompt], list[ScoreContentPartParam]],
    data_2: Union[Sequence[SingletonPrompt], list[ScoreContentPartParam]],
):
    len_1 = len(data_1)
    len_2 = len(data_2)

    if len_1 > 1 and len_1 != len_2:
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len_1 == 0:
        raise ValueError("At least one text element must be given")
    if len_2 == 0:
        raise ValueError("At least one text_pair element must be given")


def parse_score_data(
    data_1: Union[str, ScoreContentPartParam],
    data_2: Union[str, ScoreContentPartParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
) -> tuple[SingletonPrompt, SingletonPrompt, Optional[MultiModalDataDict]]:
    mm_tracker = MultiModalItemTracker(model_config, tokenizer)

    content_1 = _parse_score_content(data_1, mm_tracker)

    content_2 = _parse_score_content(data_2, mm_tracker)

    return content_1, content_2, mm_tracker.all_mm_data()


def _parse_score_content(
    data: Union[str, ScoreContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
) -> SingletonPrompt:

    if isinstance(data, str):
        data = ChatCompletionContentPartTextParam(type="text", text=data)

    mm_parser = mm_tracker.create_parser()

    parse_res = _parse_chat_message_content_part(
        data,
        mm_parser,
        wrap_dicts=False,
    )

    if parse_res:
        return parse_res

    mm_placeholder_counts = mm_parser.mm_placeholder_counts()

    if len(mm_placeholder_counts) != 1 and mm_placeholder_counts[0] != 1:
        raise ValueError("Only one multi-modal item is supported")

    return next(iter(mm_placeholder_counts))


def apply_score_template(
    model_arch: str,
    prompt_1: SingletonPrompt,
    prompt_2: SingletonPrompt,
) -> SingletonPrompt:

    if 'JinaVLForRanking' in model_arch:
        return f"**Document**:\n{prompt_2}\n**Query**:\n{prompt_1}"

    raise ValueError(f"Unsupported model architecture: {model_arch}")


def post_process_tokens_mm_data(
    model_arch: str,
    prompt: TokensPrompt,
    mm_data: Optional[MultiModalDataDict],
):
    """
    Manipulation tokens and mm_data for some model architecture such as `JinaVLForRanking`
    """
    if 'JinaVLForRanking' in model_arch:
        prompt['prompt_token_ids'].append(100)  # add score target token

        if mm_data is not None:
            for key, value in mm_data.items():
                print(value)
                if isinstance(value, list):
                    mm_data[key].reverse()
