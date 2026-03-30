# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any, cast

import torch

from vllm import PromptType, TextPrompt, TokensPrompt
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatTemplateResolutionError,
    ConversationMessage,
    MultiModalItemTracker,
    _parse_chat_message_content_parts,
)
from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.tokenizers import TokenizerLike

from .typing import (
    ScoreContentPartParam,
    ScoreData,
    ScoreInput,
    ScoreInputs,
    ScoringData,
)


def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    """
    Compute ColBERT MaxSim score.

    Args:
        q_emb: Query token embeddings [query_len, dim]
        d_emb: Document token embeddings [doc_len, dim]

    Returns:
        MaxSim score (sum over query tokens of max similarity to any doc token)
    """
    # [query_len, doc_len]
    token_scores = torch.matmul(q_emb, d_emb.T)
    # Max over document tokens, sum over query tokens
    return token_scores.amax(dim=-1).sum()


def _validate_mm_score_input(
    data: list[ScoreInput],
    is_multimodal_model: bool,
    architecture: str,
) -> list[ScoreData]:
    out: list[ScoreData] = []
    for d in data:
        if isinstance(d, str):
            out.append(d)
        else:
            if not is_multimodal_model:
                raise ValueError(f"MultiModalParam is not supported for {architecture}")
            content = cast(list[ScoreContentPartParam], d.get("content", []))
            out.append(content)
    return out


def _validate_score_input_lens(
    data_1: list[ScoreData],
    data_2: list[ScoreData],
):
    len_1 = len(data_1)
    len_2 = len(data_2)

    if len_1 > 1 and len_1 != len_2:
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len_1 == 0:
        raise ValueError("At least one text element must be given")
    if len_2 == 0:
        raise ValueError("At least one text_pair element must be given")


def validate_score_input(
    data_1: ScoreInputs,
    data_2: ScoreInputs,
    is_multimodal_model: bool,
    architecture: str,
) -> ScoringData:
    if not isinstance(data_1, list):
        data_1 = [data_1]

    if not isinstance(data_2, list):
        data_2 = [data_2]

    score_input_1 = _validate_mm_score_input(data_1, is_multimodal_model, architecture)
    score_input_2 = _validate_mm_score_input(data_2, is_multimodal_model, architecture)
    _validate_score_input_lens(score_input_1, score_input_2)
    return ScoringData(data_1=score_input_1, data_2=score_input_2)


def score_data_to_prompts(
    data_list: list[ScoreData],
    role: str,
    model_config: ModelConfig,
) -> list[PromptType]:
    """Convert a list of ScoreData into PromptType objects.

    For plain text inputs, returns the string directly.
    For multimodal inputs (list of content parts), parses them into
    a :class:`TextPrompt` with attached ``multi_modal_data`` /
    ``multi_modal_uuids``.

    This is used by late-interaction scoring where each query/document
    is encoded independently.
    """
    prompts: list[PromptType] = []
    for data in data_list:
        if isinstance(data, str):
            prompts.append(data)
        else:
            text, mm_data, mm_uuids = parse_score_data_single(data, role, model_config)
            prompt: TextPrompt = TextPrompt(prompt=text)
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data
            if mm_uuids is not None:
                prompt["multi_modal_uuids"] = mm_uuids
            prompts.append(prompt)
    return prompts


def _ensure_str(content: list[ConversationMessage]) -> str:
    """Extract a single string prompt from parsed conversation content."""
    assert len(content) == 1
    prompt = content[0]["content"]
    if prompt is not None and isinstance(prompt, str):
        return cast(str, prompt)
    raise ValueError(f"Only string content is supported, but got {content}.")


def _parse_score_content(
    role: str,
    data: ScoreData,
    mm_tracker: BaseMultiModalItemTracker,
) -> list[ConversationMessage]:
    parts: Iterable[ChatCompletionContentPartParam]
    if isinstance(data, str):
        parts = [ChatCompletionContentPartTextParam(type="text", text=data)]
    else:
        parts = cast(Iterable[ChatCompletionContentPartParam], data)

    mm_parser = mm_tracker.create_parser()

    parse_res = _parse_chat_message_content_parts(
        role=role,
        parts=parts,
        mm_tracker=mm_tracker,
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


def parse_score_data_single(
    data: ScoreData,
    role: str,
    model_config: ModelConfig,
) -> tuple[str, MultiModalDataDict | None, MultiModalUUIDDict | None]:
    """Parse **one** ScoreData into a text prompt and its own multi-modal
    data.

    Unlike :func:`parse_score_data`, each call creates an **independent**
    :class:`MultiModalItemTracker` so multi-modal items are kept separate.
    This is the correct behaviour for late-interaction scoring, where
    query and document are encoded independently.
    """
    mm_tracker = MultiModalItemTracker(model_config)
    content = _parse_score_content(role, data, mm_tracker)

    prompt = _ensure_str(content)
    mm_items, mm_uuids = mm_tracker.resolve_items()
    return prompt, mm_items, mm_uuids


def parse_score_data(
    data_1: ScoreData,
    data_2: ScoreData,
    model_config: ModelConfig,
) -> tuple[str, str, MultiModalDataDict | None, MultiModalUUIDDict | None]:
    """Parse a query-document pair into text prompts and shared multi-modal
    data.

    Uses a **single** :class:`MultiModalItemTracker` so that multi-modal
    items from both inputs are merged into one ``mm_data`` dict.  This is
    the correct behaviour for cross-encoder scoring, where query and
    document are concatenated into a single model prompt.
    """
    mm_tracker = MultiModalItemTracker(model_config)

    content_1 = _parse_score_content("query", data_1, mm_tracker)
    content_2 = _parse_score_content("document", data_2, mm_tracker)

    prompt_1 = _ensure_str(content_1)
    prompt_2 = _ensure_str(content_2)
    mm_items, mm_uuids = mm_tracker.resolve_items()

    return prompt_1, prompt_2, mm_items, mm_uuids


from vllm.model_executor.models.interfaces import supports_score_template
from vllm.renderers.hf import safe_apply_chat_template


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
    data_1: ScoreData,
    data_2: ScoreData,
    score_template: str | None = None,
) -> tuple[str, TokensPrompt]:
    prompt_1, prompt_2, mm_data, mm_uuids = parse_score_data(
        data_1,
        data_2,
        model_config,
    )
    from vllm.model_executor.model_loader import get_model_cls
    from vllm.model_executor.models.interfaces import supports_score_template

    model = get_model_cls(model_config)

    def default_tokenizer_encode():
        if supports_score_template(model):
            full_prompt = _apply_model_score_template(model_config, prompt_1, prompt_2)
            prompt_inputs = tokenizer(full_prompt, **tokenization_kwargs)
        else:
            if model_config.use_sep_token:
                # cross_encoder models defaults to using separating token.
                prompt_inputs = tokenizer(
                    text=prompt_1, text_pair=prompt_2, **tokenization_kwargs
                )
                full_prompt = tokenizer.decode(prompt_inputs["input_ids"])
            else:
                # `llm as reranker` defaults to not using separating token.
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
