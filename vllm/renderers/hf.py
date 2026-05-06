# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import inspect
import itertools
import weakref
from collections import defaultdict, deque
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox
import torch
from typing_extensions import override

from vllm.entrypoints.chat_utils import (
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    ChatTemplateResolutionError,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import EmbedsPrompt
from vllm.inputs.engine import MultiModalInput
from vllm.logger import init_logger
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    apply_token_matches,
    find_mm_placeholders,
)
from vllm.tokenizers.hf import HfTokenizer, maybe_make_thread_pool
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.async_utils import make_async
from vllm.utils.func_utils import supports_kw

from .base import BaseRenderer
from .inputs.preprocess import parse_dec_only_prompt

if TYPE_CHECKING:
    from collections.abc import Set

    from vllm.config import ModelConfig, VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ChatTemplateContentFormat,
        ChatTemplateContentFormatOption,
        ConversationMessage,
    )
    from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict, TokensPrompt
    from vllm.inputs.engine import TokensInput
    from vllm.multimodal.processing.processor import (
        MultiModalPromptUpdates,
        ResolvedPromptUpdate,
    )

    from .inputs import DictPrompt
    from .params import ChatParams

logger = init_logger(__name__)


# Cache of `tokenizer -> prompt_embeds placeholder token ID`. Keyed by the
# tokenizer object (not `id(tokenizer)`) so a fresh tokenizer landing at a
# recycled memory address can't pick up a stale tid. Entries evict atomically
# with the tokenizer's garbage-collection.
_PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE: Final[
    weakref.WeakKeyDictionary[HfTokenizer, int]
] = weakref.WeakKeyDictionary()
_PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_ERROR: Final[str] = (
    "Expected {token!r} to tokenize to exactly 1 token, got {num_ids} ({ids!r})."
)
_PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR: Final[str] = (
    "Expected {expected} prompt_embeds placeholder spans in the "
    "tokenized prompt, found {actual}."
)
_MISSING_PROMPT_TOKEN_IDS_ERROR: Final[str] = (
    "Expected prompt_token_ids in rendered prompt when prompt_embeds "
    "are present. This indicates the chat template was invoked with "
    "tokenize=False."
)
_TOKENIZE_OVERRIDE_WARNING: Final[str] = (
    "Overriding `tokenize=False` to `True` because `prompt_embeds` "
    "post-processing requires tokenized IDs."
)


def _ensure_prompt_embeds_placeholder_token(tokenizer: HfTokenizer) -> int:
    """Register `PROMPT_EMBEDS_PLACEHOLDER_TOKEN` as a special token and return
    its token ID."""
    cached = _PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE.get(tokenizer)
    if cached is not None:
        return cached

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [PROMPT_EMBEDS_PLACEHOLDER_TOKEN]}
    )

    ids = tokenizer.encode(PROMPT_EMBEDS_PLACEHOLDER_TOKEN, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            _PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_ERROR.format(
                token=PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
                num_ids=len(ids),
                ids=ids,
            )
        )

    token_id = ids[0]
    _PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE[tokenizer] = token_id
    return token_id


def _build_prompt_embeds_updates(
    prompt_embeds_tensors: Sequence[torch.Tensor],
    placeholder_token_id: int,
) -> MultiModalPromptUpdates:
    """Build `MultiModalPromptUpdates` for `prompt_embeds` expansion.

    Each tensor produces a `PromptReplacement` that maps
    `[placeholder_token_id]` -> `[placeholder_token_id] x N`
    (where `N = tensor.shape[0]`).
    """
    updates: list[Sequence[ResolvedPromptUpdate]] = []
    for i, tensor in enumerate(prompt_embeds_tensors):
        update = PromptReplacement(
            modality="prompt_embeds",
            target=[placeholder_token_id],
            replacement=[placeholder_token_id] * tensor.shape[0],
        )
        updates.append([update.resolve(item_idx=i)])
    return {"prompt_embeds": updates}


def _expand_prompt_embeds_placeholders(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
) -> list[int]:
    """Expand each 1-token `prompt_embeds` sentinel into an N-token span.

    Uses `apply_token_matches`.  Each single placeholder token in
    `token_ids` is replaced with a consecutive span of
    `tensor.shape[0]` copies, following tensors in order.
    """
    expanded, _ = apply_token_matches(token_ids, mm_prompt_updates, tokenizer=None)
    return expanded


def _build_prompt_embeds_positions(
    token_ids: list[int],
    num_tensors: int,
    mm_prompt_updates: MultiModalPromptUpdates,
) -> list[tuple[int, int]]:
    """Locate each prompt_embeds placeholder span in `token_ids`.

    Expects `token_ids` to already contain expanded N-token spans.
    Returns `[(start_idx, length), ...]` aligned with the tensors.
    """
    placeholders = find_mm_placeholders(
        prompt=token_ids,
        mm_prompt_updates=mm_prompt_updates,
        tokenizer=None,
    )
    features = placeholders.get("prompt_embeds", [])

    if len(features) != num_tensors:
        raise ValueError(
            _PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR.format(
                expected=num_tensors,
                actual=len(features),
            )
        )

    return [(f.start_idx, f.length) for f in features]


def _build_mixed_prompt_embeds(
    token_ids: list[int],
    prompt_embeds_tensors: Sequence[torch.Tensor],
    positions: list[tuple[int, int]],
) -> tuple[torch.Tensor, list[bool]]:
    """Build the full-length `prompt_embeds` tensor and the `is_token_ids`
    mask aligned to `token_ids`."""
    total_len = len(token_ids)
    hidden_size = prompt_embeds_tensors[0].shape[1]
    dtype = prompt_embeds_tensors[0].dtype

    full_embeds = torch.zeros(total_len, hidden_size, dtype=dtype)
    is_token_ids = torch.ones(total_len, dtype=torch.bool)

    for (start, length), tensor in zip(positions, prompt_embeds_tensors, strict=True):
        full_embeds[start : start + length] = tensor
        is_token_ids[start : start + length] = False

    return full_embeds, is_token_ids.tolist()


_PROCESSOR_CHAT_TEMPLATES = dict[tuple[str, bool], str | None]()
"""
Used in `_try_get_processor_chat_template` to avoid calling
`cached_get_processor` again if the processor fails to be loaded.

This is needed because `lru_cache` does not cache when an exception happens.
"""


def _try_get_processor_chat_template(
    tokenizer: HfTokenizer,
    *,
    trust_remote_code: bool,
) -> str | None:
    cache_key = (tokenizer.name_or_path, trust_remote_code)
    if cache_key in _PROCESSOR_CHAT_TEMPLATES:
        return _PROCESSOR_CHAT_TEMPLATES[cache_key]

    from transformers import (
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )

    try:
        processor = cached_get_processor(
            tokenizer.name_or_path,
            processor_cls=(
                PreTrainedTokenizer,
                PreTrainedTokenizerFast,
                ProcessorMixin,
            ),
            trust_remote_code=trust_remote_code,
        )
        if (
            isinstance(processor, ProcessorMixin)
            and hasattr(processor, "chat_template")
            and (chat_template := processor.chat_template) is not None
        ):
            _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
            return chat_template
    except Exception:
        logger.debug(
            "Failed to load AutoProcessor chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    _PROCESSOR_CHAT_TEMPLATES[cache_key] = None
    return None


def resolve_chat_template(
    tokenizer: HfTokenizer,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
) -> str | None:
    # 1st priority: The given chat template
    if chat_template is not None:
        # Resolve template names (e.g. "tool_use") to actual Jinja content
        # so that downstream kwargs detection can parse template variables.
        return tokenizer.get_chat_template(chat_template, tools=tools)

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        chat_template = _try_get_processor_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        if chat_template is not None:
            return chat_template

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug(
            "Failed to load AutoTokenizer chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    if path is not None:
        logger.info_once(
            "Loading chat template fallback for %s as there isn't one "
            "defined on HF Hub.",
            tokenizer.name_or_path,
        )
        chat_template = load_chat_template(path)
    else:
        logger.debug_once(
            "There is no chat template fallback for %s", tokenizer.name_or_path
        )

    return chat_template


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str | None = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


# NOTE: The proper way to handle this is to build a CFG so that we can handle
# the scope in which each variable is defined, but that is too complicated
def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node):
    messages_varnames = [
        varname for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
    ]

    # Search for {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Search for {%- for content in message['content'] -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _try_extract_ast(chat_template: str) -> jinja2.nodes.Template | None:
    import transformers.utils.chat_template_utils as hf_chat_utils

    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception:
        logger.exception("Error when compiling Jinja template")
        return None


@lru_cache(maxsize=32)
def _detect_content_format(
    chat_template: str,
    *,
    default: ChatTemplateContentFormat,
) -> ChatTemplateContentFormat:
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return default

    try:
        next(_iter_nodes_assign_content_item(jinja_ast))
    except StopIteration:
        return "string"
    except Exception:
        logger.exception("Error when parsing AST of Jinja template")
        return default
    else:
        return "openai"


def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    resolved_chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    jinja_text = (
        resolved_chat_template
        if isinstance(resolved_chat_template, str)
        else load_chat_template(chat_template, is_literal=True)
    )

    detected_format = (
        "string"
        if jinja_text is None
        else _detect_content_format(jinja_text, default="string")
    )

    return detected_format


@lru_cache
def _log_chat_template_content_format(
    chat_template: str | None,  # For caching purposes
    given_format: ChatTemplateContentFormatOption,
    detected_format: ChatTemplateContentFormatOption,
):
    logger.info(
        "Detected the chat template content format to be '%s'. "
        "You can set `--chat-template-content-format` to override this.",
        detected_format,
    )

    if given_format != "auto" and given_format != detected_format:
        logger.warning(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )


def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    if given_format != "auto":
        return given_format

    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    _log_chat_template_content_format(
        chat_template,
        given_format=given_format,
        detected_format=detected_format,
    )

    return detected_format


# adapted from https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py#L398-L412
# only preserve the parse function used to resolve chat template kwargs
class AssistantTracker(jinja2.ext.Extension):
    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_generation_support")
        call_block = jinja2.nodes.CallBlock(call, [], [], body)
        return call_block.set_lineno(lineno)


def _resolve_chat_template_kwargs(chat_template: str) -> Set[str]:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    parsed_content = env.parse(chat_template)
    template_vars = jinja2.meta.find_undeclared_variables(parsed_content)
    return template_vars


_cached_resolve_chat_template_kwargs = lru_cache(_resolve_chat_template_kwargs)


@lru_cache
def _get_hf_base_chat_template_params() -> frozenset[str]:
    from transformers import PreTrainedTokenizer

    # Get standard parameters from HuggingFace's base tokenizer class.
    # This dynamically extracts parameters from PreTrainedTokenizer's
    # apply_chat_template method, ensuring compatibility with tokenizers
    # that use **kwargs to receive standard parameters.

    # Read signature from HF's base class - the single source of truth
    base_sig = inspect.signature(PreTrainedTokenizer.apply_chat_template)

    # Exclude VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) placeholders
    return frozenset(
        p.name
        for p in base_sig.parameters.values()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


def resolve_chat_template_kwargs(
    tokenizer: HfTokenizer,
    chat_template: str,
    chat_template_kwargs: dict[str, Any],
    raise_on_unexpected: bool = True,
) -> dict[str, Any]:
    # We exclude chat_template from kwargs here, because
    # chat template has been already resolved at this stage
    unexpected_vars = {"chat_template", "tokenize"}
    if raise_on_unexpected and (
        unexpected_in_kwargs := unexpected_vars & chat_template_kwargs.keys()
    ):
        raise ValueError(
            "Found unexpected chat template kwargs from request: "
            f"{unexpected_in_kwargs}"
        )

    fn_kw = {
        k
        for k in chat_template_kwargs
        if supports_kw(tokenizer.apply_chat_template, k, allow_var_kwargs=False)
    }
    template_vars = _cached_resolve_chat_template_kwargs(chat_template)

    # Allow standard HF parameters even if tokenizer uses **kwargs to receive them
    hf_base_params = _get_hf_base_chat_template_params()

    accept_vars = (fn_kw | template_vars | hf_base_params) - unexpected_vars
    return {k: v for k, v in chat_template_kwargs.items() if k in accept_vars}


@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[True] = ...,
    **kwargs,
) -> list[int]: ...
@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[False] = ...,
    **kwargs,
) -> str: ...
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    tokenize: bool = True,
    **kwargs,
) -> str | list[int]:
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )
    if chat_template is None:
        raise ChatTemplateResolutionError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        )

    resolved_kwargs = resolve_chat_template_kwargs(
        tokenizer=tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=kwargs,
    )

    # transformers v5 changed the default of `return_dict` to True, which
    # makes `apply_chat_template(tokenize=True)` return a `BatchEncoding`
    # instead of `list[int]`. Force `return_dict=False` so downstream code
    # that expects a flat token list (e.g. `parse_dec_only_prompt`) works
    # consistently across v4 and v5.
    if tokenize and "return_dict" not in resolved_kwargs:
        resolved_kwargs["return_dict"] = False

    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=chat_template,
            tokenize=tokenize,
            **resolved_kwargs,
        )
    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template"
        )
        raise ValueError(str(e)) from e


def rebuild_mm_uuids_from_mm_data(
    mm_uuids: MultiModalUUIDDict,
    mm_data: MultiModalDataDict,
) -> MultiModalUUIDDict:
    """Rebuild mm_uuids after vision_chunk processing.

    When videos are split into chunks, the original UUIDs need to be updated
    to reflect the new UUIDs generated for each chunk.

    Args:
        mm_uuids: Original UUIDs dictionary
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        Updated UUIDs dictionary with chunk UUIDs
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return mm_uuids

    assert all(isinstance(item, dict) for item in vision_chunks), (
        "Expected all vision_chunk items to be dicts"
    )
    vision_chunks = cast(list[dict[str, Any]], vision_chunks)
    vision_chunk_uuids = [
        uuid_val for item in vision_chunks if (uuid_val := item.get("uuid")) is not None
    ]

    if vision_chunk_uuids:
        mm_uuids = dict(mm_uuids)
        mm_uuids["vision_chunk"] = vision_chunk_uuids

    return mm_uuids


def build_video_prompts_from_mm_data(
    mm_data: MultiModalDataDict,
) -> list[str]:
    """Build video prompts from vision_chunk data.

    Collects prompts from video chunks and groups them by video_idx.

    Args:
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        List of video prompts, one per video.
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return []

    # Group chunks by video_idx
    video_prompts_dict: dict[int, list[str]] = defaultdict(list)

    for item in vision_chunks:
        # vision_chunk items are always dicts (VisionChunkImage/VisionChunkVideo)
        assert isinstance(item, dict)
        if item.get("type") == "video_chunk":
            video_idx = item.get("video_idx", 0)
            prompt = item.get("prompt", "")
            video_prompts_dict[video_idx].append(prompt)

    # Build prompts in video order
    video_prompts = [
        "".join(video_prompts_dict[video_idx])
        for video_idx in sorted(video_prompts_dict.keys())
    ]

    return video_prompts


def replace_vision_chunk_video_placeholder(
    prompt_raw: str | list[int],
    mm_data: MultiModalDataDict,
    video_placeholder: str | None,
) -> str | list[int]:
    # get video placeholder, replace it with runtime video-chunk prompts
    if video_placeholder and isinstance(prompt_raw, str):
        video_prompts = build_video_prompts_from_mm_data(mm_data)

        # replace in order
        prompt_raw_parts = prompt_raw.split(video_placeholder)
        if len(prompt_raw_parts) == len(video_prompts) + 1:
            prompt_raw = "".join(
                itertools.chain.from_iterable(zip(prompt_raw_parts, video_prompts))
            )
            prompt_raw += prompt_raw_parts[-1]
        else:
            logger.warning(
                "Number of video placeholders (%d) does not match "
                "number of videos (%d) in the request.",
                len(prompt_raw_parts) - 1,
                len(video_prompts),
            )
    return prompt_raw


class HfRenderer(BaseRenderer[HfTokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        # Ensure the og tokenizer is never modified by maybe_make_thread_pool
        tokenizer = copy.copy(tokenizer)
        if (
            # Skip for mock configs and tokenizers
            getattr(config.model_config, "enable_prompt_embeds", False)
            and isinstance(tokenizer, HfTokenizer)
        ):
            _ensure_prompt_embeds_placeholder_token(tokenizer)
        super().__init__(config, tokenizer)

        self.use_unified_vision_chunk = getattr(
            config.model_config.hf_config, "use_unified_vision_chunk", False
        )

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        prompt_embeds_placeholder_token_id: int | None = None
        if model_config.enable_prompt_embeds:
            prompt_embeds_placeholder_token_id = (
                _ensure_prompt_embeds_placeholder_token(tokenizer)
            )

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        # prompt_embeds tensors are carried by the tracker through mm_data,
        # but they must NOT be fed to the MM processor (which would reject
        # the unknown key). Extract them here.
        prompt_embeds_tensors: list[torch.Tensor] | None = None
        if mm_data is not None and "prompt_embeds" in mm_data:
            prompt_embeds_tensors = list(
                cast(Sequence[torch.Tensor], mm_data["prompt_embeds"])
            )
            mm_data = {k: v for k, v in mm_data.items() if k != "prompt_embeds"}
            if not mm_data:
                mm_data = None

        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if prompt_embeds_tensors:
            # prompt_embeds post-processing requires prompt_token_ids.
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **chat_template_kwargs,
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            mm_uuids = rebuild_mm_uuids_from_mm_data(mm_uuids, mm_data)

            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = cast(
                list[int],
                replace_vision_chunk_video_placeholder(
                    prompt_raw,
                    mm_data,
                    video_placeholder,
                ),
            )

        prompt = parse_dec_only_prompt(prompt_raw)

        # When `prompt_embeds` is mixed with other modality data,
        # `_process_tokens` runs `_process_multimodal` first (expanding
        # `<|AUDIO|>` / `<|IMAGE|>` placeholders) and then
        # `_apply_prompt_embeds_to_engine_input` augments the result.
        # Stash the tensors and placeholder ID for that override to consume.
        if prompt_embeds_tensors and mm_data:
            assert prompt_embeds_placeholder_token_id is not None
            cast(dict, prompt)["_prompt_embeds"] = (
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )
            if params.mm_processor_kwargs:
                cast(dict, prompt)["mm_processor_kwargs"] = params.mm_processor_kwargs
        elif prompt_embeds_tensors:
            # Pure mode: no other MM data, mutate prompt to EmbedsPrompt shape.
            assert prompt_embeds_placeholder_token_id is not None
            self._apply_prompt_embeds_to_prompt(
                prompt,
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        prompt_embeds_placeholder_token_id: int | None = None
        if model_config.enable_prompt_embeds:
            prompt_embeds_placeholder_token_id = (
                _ensure_prompt_embeds_placeholder_token(tokenizer)
            )

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_embeds_tensors: list[torch.Tensor] | None = None
        if mm_data is not None and "prompt_embeds" in mm_data:
            prompt_embeds_tensors = list(
                cast(Sequence[torch.Tensor], mm_data["prompt_embeds"])
            )
            mm_data = {k: v for k, v in mm_data.items() if k != "prompt_embeds"}
            if not mm_data:
                mm_data = None

        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if prompt_embeds_tensors:
            # prompt_embeds post-processing requires prompt_token_ids.
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        prompt_raw = await self._apply_chat_template_async(
            model_config,
            tokenizer,
            conversation,
            **chat_template_kwargs,
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = cast(
                list[int],
                replace_vision_chunk_video_placeholder(
                    prompt_raw,
                    mm_data,
                    video_placeholder,
                ),
            )

        prompt = parse_dec_only_prompt(prompt_raw)

        # See `render_messages` for the rationale.
        if prompt_embeds_tensors and mm_data:
            assert prompt_embeds_placeholder_token_id is not None
            cast(dict, prompt)["_prompt_embeds"] = (
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )
            if params.mm_processor_kwargs:
                cast(dict, prompt)["mm_processor_kwargs"] = params.mm_processor_kwargs
        elif prompt_embeds_tensors:
            assert prompt_embeds_placeholder_token_id is not None
            self._apply_prompt_embeds_to_prompt(
                prompt,
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    @override
    def _process_tokens(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput | MultiModalInput:
        """Pre-expand `prompt_embeds` sentinels before delegating to the MM
        processor, then attach `prompt_embeds` modality data to the result.

        Mixed mode only: the `_prompt_embeds` stash is set by
        `render_messages` when `prompt_embeds` co-exist with other MM data
        (images, audio, …).  We expand each 1-token sentinel to an N-token
        span *before* calling `super()._process_tokens()` so the MM
        processor records all placeholder offsets in the final (post-expansion)
        coordinate space, no offset shifting needed afterwards.
        """
        prompt_embeds_info = cast(dict, prompt).pop("_prompt_embeds", None)
        if prompt_embeds_info is not None:
            tensors, placeholder_token_id = prompt_embeds_info
            mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
            cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
                list(prompt["prompt_token_ids"]), mm_updates
            )
        engine_input = super()._process_tokens(prompt, skip_mm_cache=skip_mm_cache)
        if prompt_embeds_info is not None:
            tensors, _ = prompt_embeds_info
            self._apply_prompt_embeds_to_engine_input(
                cast(MultiModalInput, engine_input),
                tensors,
                mm_updates,
            )
        return engine_input

    @override
    async def _process_tokens_async(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput | MultiModalInput:
        """Async equivalent of `_process_tokens`."""
        prompt_embeds_info = cast(dict, prompt).pop("_prompt_embeds", None)
        if prompt_embeds_info is not None:
            tensors, placeholder_token_id = prompt_embeds_info
            mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
            cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
                list(prompt["prompt_token_ids"]), mm_updates
            )
        engine_input = await super()._process_tokens_async(
            prompt, skip_mm_cache=skip_mm_cache
        )
        if prompt_embeds_info is not None:
            tensors, _ = prompt_embeds_info
            self._apply_prompt_embeds_to_engine_input(
                cast(MultiModalInput, engine_input),
                tensors,
                mm_updates,
            )
        return engine_input

    @staticmethod
    def _apply_prompt_embeds_to_prompt(
        prompt: DictPrompt,
        prompt_embeds_tensors: list[torch.Tensor],
        placeholder_token_id: int,
    ) -> None:
        """Mutate `prompt` from `TokensPrompt` to `EmbedsPrompt` shape.

        Pure `prompt_embeds` path only (no other MM modalities).  Expands
        each `<prompt_embeds>` sentinel token into an N-token span and builds
        the full-length `prompt_embeds` tensor + `prompt_is_token_ids` mask
        that the engine's `enable_prompt_embeds` worker branch consumes.
        """
        token_ids = cast(list[int] | None, prompt.get("prompt_token_ids"))
        if token_ids is None:
            raise RuntimeError(_MISSING_PROMPT_TOKEN_IDS_ERROR)

        embeds_orig_positions: list[int] = [
            i for i, tok in enumerate(token_ids) if tok == placeholder_token_id
        ]
        if len(embeds_orig_positions) != len(prompt_embeds_tensors):
            raise ValueError(
                f"Expected {len(prompt_embeds_tensors)} prompt_embeds "
                f"placeholder tokens in the rendered prompt, found "
                f"{len(embeds_orig_positions)}."
            )

        mm_updates = _build_prompt_embeds_updates(
            prompt_embeds_tensors, placeholder_token_id
        )
        expanded = _expand_prompt_embeds_placeholders(token_ids, mm_updates)
        positions = _build_prompt_embeds_positions(
            expanded, len(prompt_embeds_tensors), mm_updates
        )

        embeds_prompt = cast(EmbedsPrompt, prompt)
        embeds_prompt["prompt_token_ids"] = expanded
        full_embeds, is_token_ids_mask = _build_mixed_prompt_embeds(
            expanded, prompt_embeds_tensors, positions
        )
        embeds_prompt["prompt_embeds"] = full_embeds
        embeds_prompt["prompt_is_token_ids"] = is_token_ids_mask

    @staticmethod
    def _apply_prompt_embeds_to_engine_input(
        engine_input: MultiModalInput,
        prompt_embeds_tensors: list[torch.Tensor],
        mm_updates: MultiModalPromptUpdates,
    ) -> None:
        """Augment `engine_input` in-place with a `prompt_embeds` modality.

        Mixed mode: called after `_process_multimodal` has already run on the
        pre-expanded token IDs (expansion was done in `_process_tokens` before
        calling `super()`).  Locates the already-expanded `prompt_embeds` spans
        and adds `prompt_embeds` entries to `mm_kwargs`, `mm_hashes`, and
        `mm_placeholders`.
        """
        # token_ids already contain the pre-expanded N-token spans.
        token_ids = list(engine_input["prompt_token_ids"])

        positions = _build_prompt_embeds_positions(
            token_ids, len(prompt_embeds_tensors), mm_updates
        )

        pe_kwargs_items: list[MultiModalKwargsItem] = []
        pe_hashes: list[str] = []
        pe_placeholders: list[PlaceholderRange] = []
        for tensor, (start, length) in zip(
            prompt_embeds_tensors, positions, strict=True
        ):
            pe_kwargs_items.append(
                MultiModalKwargsItem(
                    {
                        "embedding": MultiModalFieldElem(
                            data=tensor,
                            field=MultiModalSharedField(batch_size=1),
                        )
                    }
                )
            )
            pe_hashes.append(MultiModalHasher.hash_kwargs(prompt_embeds=tensor))
            # `is_embed=None` matches the existing image_embeds-style
            # "no encoder, just splice the tensor directly" semantics.
            pe_placeholders.append(
                PlaceholderRange(offset=start, length=length, is_embed=None)
            )

        cast(
            MultiModalKwargsItems[MultiModalKwargsItem | None],
            engine_input["mm_kwargs"],
        )["prompt_embeds"] = pe_kwargs_items
        engine_input["mm_hashes"] = {
            **engine_input["mm_hashes"],
            "prompt_embeds": pe_hashes,
        }
        cast(dict, engine_input["mm_placeholders"])["prompt_embeds"] = pe_placeholders
