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
    ChatCompletionContentPartVideoParam,
    ChatTemplateResolutionError,
    MultiModalItemTracker,
    _ContentPart,
    _parse_chat_message_content_part,
)
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.interfaces import supports_score_template
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.outputs import PoolingRequestOutput
from vllm.renderers.hf import safe_apply_chat_template
from vllm.tokenizers import TokenizerLike

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartTextParam
    | ChatCompletionContentPartVideoParam
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


def _get_num_special_tokens_for_pair(tokenizer: TokenizerLike) -> int:
    """Get number of special tokens added for a text pair encoding.

    This handles different tokenizer types by trying the HuggingFace method
    first and falling back to computing dynamically if needed.
    """
    # Try HuggingFace method with pair=True
    # Use getattr to bypass type checker since TokenizerLike protocol
    # doesn't define the 'pair' parameter that HuggingFace tokenizers have
    method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if method is not None:
        try:
            return method(pair=True)
        except TypeError:
            pass  # pair parameter not supported

    # Fallback: compute by tokenizing empty strings
    empty_encoding = tokenizer("", text_pair="", add_special_tokens=True)
    return len(empty_encoding["input_ids"])


def _truncate_text_to_tokens(
    text: str,
    tokenizer: TokenizerLike,
    max_tokens: int,
) -> str:
    """Truncate text to a maximum number of tokens.

    This is used as a fallback for cases where we can't use the tokenizer's
    built-in truncation (e.g., template-based models).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens])


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
) -> tuple[str, str, MultiModalDataDict | None, MultiModalUUIDDict | None]:
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
    mm_items, mm_uuids = mm_tracker.resolve_items()

    return prompt_1, prompt_2, mm_items, mm_uuids


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
    max_tokens_per_doc: int | None = None,
) -> tuple[str, TokensPrompt]:
    prompt_1, prompt_2, mm_data, mm_uuids = parse_score_data(
        data_1,
        data_2,
        model_config,
    )
    from vllm.model_executor.model_loader import get_model_cls

    model = get_model_cls(model_config)

    def default_tokenizer_encode():
        nonlocal prompt_2
        # Make a copy to avoid mutating the original
        local_kwargs = tokenization_kwargs.copy()

        if supports_score_template(model):
            # Template case - need to truncate text first (can't use only_second)
            if max_tokens_per_doc is not None and isinstance(prompt_2, str):
                prompt_2 = _truncate_text_to_tokens(
                    prompt_2, tokenizer, max_tokens_per_doc
                )
            full_prompt = _apply_model_score_template(model_config, prompt_1, prompt_2)
            prompt_inputs = tokenizer(full_prompt, **local_kwargs)
        else:
            if model_config.use_sep_token:
                # Cross-encoder case - use tokenizer's built-in truncation
                if max_tokens_per_doc is not None and isinstance(prompt_2, str):
                    # Calculate max_length to limit doc tokens
                    query_tokens = tokenizer.encode(prompt_1, add_special_tokens=False)
                    num_special = _get_num_special_tokens_for_pair(tokenizer)
                    doc_limit_max_length = (
                        len(query_tokens) + max_tokens_per_doc + num_special
                    )

                    # If truncate_prompt_tokens is also set, use the smaller
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
                # `llm as reranker` - build tokens directly for efficiency
                if max_tokens_per_doc is not None and isinstance(prompt_2, str):
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
    if score_template is None:
        full_prompt, prompt_inputs = default_tokenizer_encode()
    else:
        # FIXME: Try applying a score template from the CLI arg or tokenizer_config.json
        # If that fails because there is no such template,
        # fall back to the default implementation.
        try:
            # Template case - truncate text first if needed
            if max_tokens_per_doc is not None and isinstance(prompt_2, str):
                prompt_2 = _truncate_text_to_tokens(
                    prompt_2, tokenizer, max_tokens_per_doc
                )
            full_prompt = safe_apply_chat_template(
                model_config,
                tokenizer,
                [
                    {"role": "query", "content": prompt_1},
                    {"role": "document", "content": prompt_2},
                ],
                chat_template=score_template,
                tools=None,
                tokenize=False,
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
    if mm_uuids is not None:
        engine_prompt["multi_modal_uuids"] = mm_uuids

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
