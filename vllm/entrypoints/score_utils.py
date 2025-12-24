# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, TypeAlias, cast

from torch.nn import CosineSimilarity
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatTemplateResolutionError,
    MultiModalItemTracker,
    _ContentPart,
    _parse_chat_message_content_part,
    apply_hf_chat_template,
)
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.interfaces import supports_score_template
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.outputs import PoolingRequestOutput
from vllm.tokenizers import TokenizerLike

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam | ChatCompletionContentPartImageEmbedsParam
)


class ScoreMultiModalParam(TypedDict, total=False):
    """
    A specialized parameter type for scoring multimodal content

    The reasons why don't reuse `CustomChatCompletionMessageParam` directly:
    1. Score tasks don't need the 'role' field (user/assistant/system) that's required in chat completions
    2. Including chat-specific fields would confuse users about their purpose in scoring
    3. This is a more focused interface that only exposes what's needed for scoring
    """  # noqa: E501

    content: Required[list[ScoreContentPartParam]]
    """The multimodal contents"""


def _cosine_similarity(
    tokenizer: TokenizerLike,
    embed_1: list[PoolingRequestOutput],
    embed_2: list[PoolingRequestOutput],
) -> list[PoolingRequestOutput]:
    scorer = CosineSimilarity(0)
    scores: list[PoolingRequestOutput] = []

    for emb_1, emb_2 in zip(embed_1, embed_2):
        pair_score = scorer(emb_1.outputs.data, emb_2.outputs.data)

        padding: list[int] = []
        if (pad_token_id := tokenizer.pad_token_id) is not None:
            padding = [pad_token_id]

        tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

        scores.append(
            PoolingRequestOutput(
                request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                outputs=pair_score,
                prompt_token_ids=tokens,
                num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                finished=True,
            )
        )

    return scores


def _validate_score_input_lens(
    data_1: list[str] | list[ScoreContentPartParam],
    data_2: list[str] | list[ScoreContentPartParam],
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
    data_1: str | ScoreContentPartParam,
    data_2: str | ScoreContentPartParam,
    model_config: ModelConfig,
) -> tuple[str, str, MultiModalDataDict | None]:
    mm_tracker = MultiModalItemTracker(model_config)

    content_1 = _parse_score_content(data_1, mm_tracker)
    content_2 = _parse_score_content(data_2, mm_tracker)

    def ensure_str(content: _ContentPart | None) -> str:
        if content is not None and isinstance(content, str):
            return cast(str, content)
        else:
            raise ValueError(f"Only string content is supported, but got {content}.")

    prompt_1 = ensure_str(content_1)
    prompt_2 = ensure_str(content_2)

    return prompt_1, prompt_2, mm_tracker.all_mm_data()


def _parse_score_content(
    data: str | ScoreContentPartParam,
    mm_tracker: BaseMultiModalItemTracker,
) -> _ContentPart | None:
    if isinstance(data, str):
        part = ChatCompletionContentPartTextParam(type="text", text=data)
    else:
        part = data

    mm_parser = mm_tracker.create_parser()

    parse_res = _parse_chat_message_content_part(
        part,
        mm_parser,
        wrap_dicts=False,
        interleave_strings=False,
    )

    if parse_res:
        return parse_res

    mm_placeholder_storage = mm_parser.mm_placeholder_storage()

    if (
        len(mm_placeholder_storage) != 1
        or len(next(iter(mm_placeholder_storage.values()))) != 1
    ):
        raise ValueError("Only one multi-modal item is supported")

    return next(iter(mm_placeholder_storage.values()))[0]


def _apply_model_score_template(
    model_config: ModelConfig, prompt_1: str, prompt_2: str
) -> str:
    # NOTE(Simon): lazy import to avoid bring in all dependencies (e.g. gguf)
    from vllm.model_executor.model_loader import get_model_cls

    model = get_model_cls(model_config)
    if supports_score_template(model):
        full_prompt = model.get_score_template(prompt_1, prompt_2)
        if full_prompt is None:
            raise ValueError("Get empty score template from model")
        return full_prompt

    raise ValueError(f"Unsupported model architecture: {model_config.architecture}")


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
    tokenizer: TokenizerLike,
    tokenization_kwargs: dict[str, Any],
    data_1: str | ScoreContentPartParam,
    data_2: str | ScoreContentPartParam,
    score_template: str | None = None,
) -> tuple[str, TokensPrompt]:
    prompt_1, prompt_2, mm_data = parse_score_data(
        data_1,
        data_2,
        model_config,
    )
    from vllm.model_executor.model_loader import get_model_cls

    model = get_model_cls(model_config)

    def default_tokenizer_encode():
        if supports_score_template(model):
            full_prompt = _apply_model_score_template(model_config, prompt_1, prompt_2)
            prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)
        else:
            if model_config.use_pad_token:
                # cross_encoder models defaults to using pad_token.
                prompt_inputs = tokenizer(
                    text=prompt_1, text_pair=prompt_2, **tokenization_kwargs
                )
                full_prompt = tokenizer.decode(prompt_inputs["input_ids"])
            else:
                # `llm as reranker` models defaults to not using pad_token.
                full_prompt = prompt_1 + prompt_2
                prompt_inputs = tokenizer(text=full_prompt, **tokenization_kwargs)
        return full_prompt, prompt_inputs

    # FIXME: For now, we only apply a template when one is explicitly provided.
    # We cannot rely on the tokenizer's chat template because many models
    # inherit junk templates from their base LLM, which breaks both the models
    # and the tests that use them.
    if score_template is None:
        full_prompt, prompt_inputs = default_tokenizer_encode()
    else:
        # FIXME: Try applying a score template from the CLI arg or tokenizer_config.json
        # If that fails because there is no such template,
        # fall back to the default implementation.
        try:
            full_prompt = apply_hf_chat_template(
                tokenizer,
                [
                    {"role": "query", "content": prompt_1},
                    {"role": "document", "content": prompt_2},
                ],
                score_template,
                tools=None,
                model_config=model_config,
            )
            prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)
        except ChatTemplateResolutionError:
            full_prompt, prompt_inputs = default_tokenizer_encode()

    engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["input_ids"])

    if (token_type_ids := prompt_inputs.get("token_type_ids")) is not None:
        engine_prompt["token_type_ids"] = token_type_ids

    post_process_tokens(model_config, engine_prompt)

    if mm_data is not None:
        engine_prompt["multi_modal_data"] = mm_data
    return full_prompt, engine_prompt


def compress_token_type_ids(token_type_ids: list[int]) -> int:
    """
    Return position of the first 1 or the length of the list
    if not found.
    """
    first_one = len(token_type_ids)
    err_msg = (
        "Token type ids are expected to be a sequence"
        " of zeros followed by a sequence of ones"
    )
    for i, type_id in enumerate(token_type_ids):
        if type_id == 0 and first_one < i:
            raise ValueError(err_msg)
        elif type_id == 1 and first_one > i:
            first_one = i
        elif type_id > 1:
            raise ValueError(err_msg)

    return first_one
