# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias, cast

import torch
from torch.nn import CosineSimilarity
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoParam,
    ChatTemplateResolutionError,
    ConversationMessage,
    MultiModalItemTracker,
    _parse_chat_message_content_parts,
)
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType, TextPrompt
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


def compute_maxsim_scores(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 16,
    max_score_matrix_elements: int = 16_000_000,
) -> list[torch.Tensor]:
    """Compute ColBERT MaxSim scores in padded mini-batches."""
    if len(q_embs) != len(d_embs):
        raise ValueError("q_embs and d_embs must have the same length")

    num_pairs = len(q_embs)
    if num_pairs == 0:
        return []

    for q_emb, d_emb in zip(q_embs, d_embs):
        if q_emb.ndim != 2 or d_emb.ndim != 2:
            raise ValueError("Each embedding tensor must be 2-D")
        if q_emb.shape[1] != d_emb.shape[1]:
            raise ValueError("Query and document embeddings must have same dim")

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: list[torch.Tensor] = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(int(x.shape[0]) for x in q_embs[start:end])
        max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        # keep score matrix bounded to avoid oversized allocations.
        while (
            end - start > 1
            and (end - start) * max_q * max_d > max_score_matrix_elements
        ):
            end -= 1
            max_q = max(int(x.shape[0]) for x in q_embs[start:end])
            max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        batch_q = q_embs[start:end]
        batch_d = d_embs[start:end]
        batch_size = end - start
        dim = int(batch_q[0].shape[1])
        dtype = batch_q[0].dtype

        q_batch = torch.zeros(
            (batch_size, max_q, dim), dtype=dtype, device=compute_device
        )
        d_batch = torch.zeros(
            (batch_size, max_d, dim), dtype=dtype, device=compute_device
        )
        q_mask = torch.zeros(
            (batch_size, max_q), dtype=torch.bool, device=compute_device
        )
        d_mask = torch.zeros(
            (batch_size, max_d), dtype=torch.bool, device=compute_device
        )

        # copy to padded tensors
        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):
            q_len = int(q_emb.shape[0])
            d_len = int(d_emb.shape[0])
            q_batch[i, :q_len] = q_emb.to(device=compute_device, dtype=dtype)
            d_batch[i, :d_len] = d_emb.to(device=compute_device, dtype=dtype)
            q_mask[i, :q_len] = True
            d_mask[i, :d_len] = True

        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        max_per_query = token_scores.amax(dim=-1)
        max_per_query.masked_fill_(~q_mask, 0)
        batch_scores = max_per_query.sum(dim=-1).to("cpu")
        scores.extend(batch_scores.unbind(0))
        start = end

    return [cast(torch.Tensor, score) for score in scores]


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


# Raw input data with content key in ScoreMultiModalParam.
ScoreInput = str | ScoreMultiModalParam
ScoreInputs = ScoreInput | list[ScoreInput]
# Score data without content key.
ScoreData = str | list[ScoreContentPartParam]


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


def validate_score_input(
    data_1: ScoreInputs,
    data_2: ScoreInputs,
    is_multimodal_model: bool,
    architecture: str,
) -> tuple[list[ScoreData], list[ScoreData]]:
    if not isinstance(data_1, list):
        data_1 = [data_1]

    if not isinstance(data_2, list):
        data_2 = [data_2]

    score_input_1 = _validate_mm_score_input(data_1, is_multimodal_model, architecture)
    score_input_2 = _validate_mm_score_input(data_2, is_multimodal_model, architecture)
    _validate_score_input_lens(score_input_1, score_input_2)
    return score_input_1, score_input_2


def _ensure_str(content: list[ConversationMessage]) -> str:
    """Extract a single string prompt from parsed conversation content."""
    assert len(content) == 1
    prompt = content[0]["content"]
    if prompt is not None and isinstance(prompt, str):
        return cast(str, prompt)
    raise ValueError(f"Only string content is supported, but got {content}.")


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
