# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional, Union, cast

from torch.nn import CosineSimilarity
from typing_extensions import Required, TypeAlias, TypedDict

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker, ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam, ChatCompletionContentPartTextParam,
    MultiModalItemTracker, _ContentPart, _parse_chat_message_content_part)
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.interfaces import supports_score_template
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
    """ # noqa: E501
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
    data_1: Union[list[str], list[ScoreContentPartParam]],
    data_2: Union[list[str], list[ScoreContentPartParam]],
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
) -> tuple[str, str, Optional[MultiModalDataDict]]:
    mm_tracker = MultiModalItemTracker(model_config, tokenizer)

    content_1 = _parse_score_content(data_1, mm_tracker)

    content_2 = _parse_score_content(data_2, mm_tracker)

    def ensure_str(content: Optional[_ContentPart]) -> str:
        if content is not None and isinstance(content, str):
            return cast(str, content)
        else:
            raise ValueError(
                f"Only string content is supported, but got {content}.")

    prompt_1 = ensure_str(content_1)
    prompt_2 = ensure_str(content_2)

    return prompt_1, prompt_2, mm_tracker.all_mm_data()


def _parse_score_content(
    data: Union[str, ScoreContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
) -> Optional[_ContentPart]:

    if isinstance(data, str):
        data = ChatCompletionContentPartTextParam(type="text", text=data)

    mm_parser = mm_tracker.create_parser()

    parse_res = _parse_chat_message_content_part(
        data,
        mm_parser,
        wrap_dicts=False,
        interleave_strings=False,
    )

    if parse_res:
        return parse_res

    mm_placeholder_storage = mm_parser.mm_placeholder_storage()

    if len(mm_placeholder_storage) != 1 or len(
            next(iter(mm_placeholder_storage.values()))) != 1:
        raise ValueError("Only one multi-modal item is supported")

    return next(iter(mm_placeholder_storage.values()))[0]


def apply_score_template(
    model_config: ModelConfig,
    prompt_1: str,
    prompt_2: str,
) -> str:
    # NOTE(Simon): lazy import to avoid bring in all dependencies (e.g. gguf)
    from vllm.model_executor.model_loader import get_model_cls

    model = get_model_cls(model_config)
    if supports_score_template(model):
        full_prompt = model.get_score_template(prompt_1, prompt_2)
        if full_prompt is None:
            raise ValueError("Get empty score template from model")
        return full_prompt

    raise ValueError(
        f"Unsupported model architecture: {model_config.architecture}")


def post_process_tokens(
    model_config: ModelConfig,
    prompt: TokensPrompt,
) -> None:
    """
    Perform architecture-specific manipulations on the input tokens.
    
    Note:
        This is an in-place operation.
    """
    # NOTE(Simon): lazy import to avoid bring in all dependencies (e.g. gguf)
    from vllm.model_executor.model_loader import get_model_cls

    model = get_model_cls(model_config)
    if supports_score_template(model):
        model.post_process_tokens(prompt)


def get_score_prompt(
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    tokenization_kwargs: dict[str, Any],
    data_1: Union[str, ScoreContentPartParam],
    data_2: Union[str, ScoreContentPartParam],
) -> tuple[str, TokensPrompt]:
    prompt_1, prompt_2, mm_data = parse_score_data(
        data_1,
        data_2,
        model_config,
        tokenizer,
    )

    full_prompt = apply_score_template(model_config, prompt_1, prompt_2)

    prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)

    engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["input_ids"])

    post_process_tokens(model_config, engine_prompt)

    if mm_data is not None:
        engine_prompt["multi_modal_data"] = mm_data
    return full_prompt, engine_prompt
