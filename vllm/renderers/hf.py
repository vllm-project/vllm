# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio
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

from vllm import envs
from vllm.exceptions import ChatTemplateResolutionError, VLLMValidationError
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
from vllm.renderers.chat_utils import (
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.chat_utils.content_parser import (
    PROMPT_EMBEDS_PLACEHOLDER_ID_PREFIX,
)
from vllm.tokenizers.hf import HfTokenizer
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path,
    get_chat_template_override_path,
)
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.async_utils import make_async
from vllm.utils.func_utils import supports_kw

from .base import BaseRenderer
from .inputs.preprocess import parse_dec_only_prompt

if TYPE_CHECKING:
    from collections.abc import Set

    from vllm.config import ModelConfig, VllmConfig
    from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict, TokensPrompt
    from vllm.inputs.engine import TokensInput
    from vllm.multimodal.processing.processor import (
        MultiModalPromptUpdates,
        ResolvedPromptUpdate,
    )
    from vllm.renderers.chat_utils import (
        ChatCompletionMessageParam,
        ChatTemplateContentFormat,
        ChatTemplateContentFormatOption,
        ConversationMessage,
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


def _filter_prompt_embeds_by_template(
    conversation: list[ConversationMessage],
    prompt_embeds_tensors: list[torch.Tensor],
    rendered: str,
) -> tuple[list[torch.Tensor], list[ConversationMessage]]:
    """Drop embeddings whose placeholders are omitted by the chat template."""
    markers = [
        f"{PROMPT_EMBEDS_PLACEHOLDER_ID_PREFIX}{index}__"
        for index in range(len(prompt_embeds_tensors))
    ]

    def normalize_placeholders(value: Any) -> Any:
        if isinstance(value, str):
            for marker in markers:
                value = value.replace(marker, PROMPT_EMBEDS_PLACEHOLDER_TOKEN)
            return value
        if isinstance(value, list):
            return [normalize_placeholders(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize_placeholders(item) for key, item in value.items()}
        return value

    visible_markers: list[tuple[int, torch.Tensor]] = []
    for marker, tensor in zip(markers, prompt_embeds_tensors, strict=True):
        occurrences = rendered.count(marker)
        if occurrences > 1:
            raise ValueError(
                f"Chat template rendered prompt_embeds placeholder {marker!r} "
                f"{occurrences} times; each placeholder must appear at most once."
            )
        if occurrences == 1:
            visible_markers.append((rendered.index(marker), tensor))
    visible_tensors = [
        tensor for _, tensor in sorted(visible_markers, key=lambda item: item[0])
    ]
    normalized_conversation = normalize_placeholders(copy.deepcopy(conversation))
    return visible_tensors, normalized_conversation


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
    expanded, _ = apply_token_matches(token_ids, mm_prompt_updates)
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
    placeholders = find_mm_placeholders(token_ids, mm_prompt_updates)
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


_PROCESSOR_CHAT_TEMPLATES = dict[tuple[str, str | None, str | None, bool], str | None]()
"""
Used in `_try_get_processor_chat_template` to avoid calling
`cached_get_processor` again if the processor fails to be loaded.

This is needed because `lru_cache` does not cache when an exception happens.
"""


def _try_get_processor_chat_template(
    tokenizer: HfTokenizer,
    *,
    revision: str | None,
    code_revision: str | None,
    trust_remote_code: bool,
) -> str | None:
    cache_key = (
        tokenizer.name_or_path,
        revision,
        code_revision,
        trust_remote_code,
    )
    if cache_key in _PROCESSOR_CHAT_TEMPLATES:
        return _PROCESSOR_CHAT_TEMPLATES[cache_key]

    from transformers import ProcessorMixin, PythonBackend, TokenizersBackend

    try:
        processor = cached_get_processor(
            tokenizer.name_or_path,
            processor_cls=(PythonBackend, TokenizersBackend, ProcessorMixin),
            revision=revision,
            code_revision=code_revision,
            trust_remote_code=trust_remote_code,
        )
        if (
            isinstance(processor, ProcessorMixin)
            and hasattr(processor, "chat_template")
            and (chat_template := processor.chat_template) is not None
        ):
            if isinstance(chat_template, dict) and "default" in chat_template:
                chat_template = chat_template["default"]
            if not isinstance(chat_template, str):
                raise TypeError(
                    "Expected a str or a dict with a 'default' entry, "
                    f"got {type(chat_template).__name__}"
                )
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

    # Predefined templates that replace the ones shipped on HF Hub
    path = get_chat_template_override_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    if path is not None:
        return load_chat_template(path)

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        chat_template = _try_get_processor_chat_template(
            tokenizer,
            revision=model_config.tokenizer_revision,
            code_revision=(
                model_config.code_revision
                if model_config.tokenizer == model_config.model
                else None
            ),
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

    # `ns.rest` matches the dotted alias "ns.rest".
    if isinstance(node, jinja2.nodes.Getattr) and "." in varname:
        base, _, attr = varname.rpartition(".")
        return node.attr == attr and _is_var_access(node.node, base)

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


def _attr_dotted_name(node: jinja2.nodes.Node) -> str | None:
    """Render an attribute-assignment target like `ns.rest` as a dotted name."""
    if isinstance(node, jinja2.nodes.NSRef):
        return f"{node.name}.{node.attr}"
    parts = []
    while isinstance(node, jinja2.nodes.Getattr):
        parts.append(node.attr)
        node = node.node
    if isinstance(node, jinja2.nodes.Name):
        parts.append(node.name)
        return ".".join(reversed(parts))
    return None


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    seen = {varname}
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if isinstance(lhs, jinja2.nodes.Name) and _is_var_or_elems_access(
                rhs, related_varname
            ):
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname and lhs.name not in seen:
                    seen.add(lhs.name)
                    related_varnames.append(lhs.name)

            # Namespace aliases: `namespace(rest=messages)` or `ns.rest = messages`.
            aliases: list[str] = []
            if (
                isinstance(lhs, jinja2.nodes.Name)
                and isinstance(rhs, jinja2.nodes.Call)
                and isinstance(rhs.node, jinja2.nodes.Name)
                and rhs.node.name == "namespace"
            ):
                aliases = [
                    f"{lhs.name}.{kwarg.key}"
                    for kwarg in rhs.kwargs
                    if _is_var_or_elems_access(kwarg.value, related_varname)
                ]
            elif isinstance(
                lhs, (jinja2.nodes.Getattr, jinja2.nodes.NSRef)
            ) and _is_var_or_elems_access(rhs, related_varname):
                dotted = _attr_dotted_name(lhs)
                aliases = [dotted] if dotted is not None else []

            for alias in aliases:
                yield assign_ast, alias
                if alias != related_varname and alias not in seen:
                    seen.add(alias)
                    related_varnames.append(alias)


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
    """Yield loops that iterate over message content or macro-bound content."""
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Track macro parameters that receive message.content as an argument.
    # Some templates pass message.content through a macro parameter whose
    # name is not literally "content".
    macro_content_params_by_loop: dict[int, set[str]] = {}
    loops_in_macros: set[int] = set()
    for macro_node in root.find_all(jinja2.nodes.Macro):
        macro_param_names = {arg.name for arg in macro_node.args}
        macro_content_params: set[str] = set()
        for call_node in root.find_all(jinja2.nodes.Call):
            if (
                isinstance(call_node.node, jinja2.nodes.Name)
                and call_node.node.name == macro_node.name
            ):
                for i, arg in enumerate(call_node.args):
                    if i < len(macro_node.args) and any(
                        _is_var_or_elems_access(arg, varname, "content")
                        for varname in message_varnames
                    ):
                        macro_content_params.add(macro_node.args[i].name)
                for kwarg in call_node.kwargs:
                    if (
                        isinstance(kwarg, jinja2.nodes.Keyword)
                        and kwarg.key in macro_param_names
                        and any(
                            _is_var_or_elems_access(kwarg.value, varname, "content")
                            for varname in message_varnames
                        )
                    ):
                        macro_content_params.add(kwarg.key)

        for loop_ast in macro_node.find_all(jinja2.nodes.For):
            loops_in_macros.add(id(loop_ast))
            if macro_content_params:
                macro_content_params_by_loop[id(loop_ast)] = macro_content_params

    # Search for {%- for content in message['content'] -%} loops
    # or {%- for item in content -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break
        else:
            macro_content_params_for_loop = macro_content_params_by_loop.get(
                id(loop_ast)
            )
            if (
                isinstance(loop_iter, jinja2.nodes.Name)
                and macro_content_params_for_loop is not None
                and loop_iter.name in macro_content_params_for_loop
            ):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                continue

            if (
                id(loop_ast) not in loops_in_macros
                and isinstance(loop_iter, jinja2.nodes.Name)
                and loop_iter.name == "content"
            ):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name


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


@lru_cache(maxsize=32)
def _detect_developer_role_support(chat_template: str) -> bool:
    return '"developer"' in chat_template or "'developer'" in chat_template


_DeveloperRoleMode = Literal["native", "system", "leading_system"]
"""
How developer messages reach a chat template, as in the Rust frontend's
`DeveloperRole`. Unless the template names `developer` (`native`), developer
messages in the leading run of system messages are merged with it into one
`system` message. A later developer message becomes a `system` message in
place if the template renders one mid-conversation (`system`), else it joins
the adjacent user turn, the following one if any (`leading_system`).
"""

_DEVELOPER_ROLE_MODES_MAXSIZE = 32
_DEVELOPER_ROLE_MODES = dict[
    tuple[str, str, tuple[tuple[str, str], ...]], _DeveloperRoleMode
]()
"""
Used in `_resolve_developer_role_mode` to probe each chat template once. The
probe renders with the tokenizer's special tokens and the request's template
variables (e.g. `enable_thinking`), so both are in the key.
"""

# A system message after an assistant turn, as clients add context
# mid-conversation. Some templates accept one only after a user turn.
_LATER_SYSTEM_PROBE: Final = (
    ("system", "PROBE_SYSTEM"),
    ("user", "PROBE_USER"),
    ("assistant", "PROBE_ASSISTANT"),
    ("system", "PROBE_LATER_SYSTEM"),
    ("user", "PROBE_USER_2"),
)

_PROBE_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _renders_later_system(
    tokenizer: HfTokenizer, chat_template: str, template_kwargs: dict[str, Any]
) -> bool:
    conversation = [
        {"role": role, "content": text} for role, text in _LATER_SYSTEM_PROBE
    ]
    # A `tool_use` template may render nothing without tools.
    for tools in (None, [_PROBE_TOOL]):
        try:
            prompt = tokenizer.apply_chat_template(
                conversation=conversation,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:
            continue
        if isinstance(prompt, str) and all(
            text in prompt for _, text in _LATER_SYSTEM_PROBE
        ):
            return True
    return False


def _resolve_developer_role_mode(
    tokenizer: HfTokenizer,
    chat_template: str,
    template_kwargs: dict[str, Any] | None = None,
) -> _DeveloperRoleMode:
    # A render probe cannot tell whether the template knows `developer`,
    # since templates with a generic role branch render any role.
    if _detect_developer_role_support(chat_template):
        return "native"

    template_kwargs = template_kwargs or {}
    cache_key = (
        chat_template,
        tokenizer.name_or_path,
        tuple(sorted((k, repr(v)) for k, v in template_kwargs.items())),
    )
    if (mode := _DEVELOPER_ROLE_MODES.get(cache_key)) is None:
        mode = (
            "system"
            if _renders_later_system(tokenizer, chat_template, template_kwargs)
            else "leading_system"
        )
        # Requests may carry their own chat template, so bound the cache.
        if len(_DEVELOPER_ROLE_MODES) >= _DEVELOPER_ROLE_MODES_MAXSIZE:
            _DEVELOPER_ROLE_MODES.clear()
        _DEVELOPER_ROLE_MODES[cache_key] = mode
    return mode


_MessageContent = str | list[dict[str, str]] | None


def _is_text_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") == "text"


def _is_empty_content(content: _MessageContent) -> bool:
    if isinstance(content, list):
        return all(_is_text_part(part) and not part.get("text") for part in content)
    return not content


def _append_content(
    content: _MessageContent, other: _MessageContent
) -> _MessageContent:
    """Append `other` to `content`, keeping parts in order. Text on both sides
    of the boundary is joined with a blank line."""
    if _is_empty_content(other):
        return content
    if _is_empty_content(content):
        return other
    if isinstance(content, str) and isinstance(other, str):
        return f"{content}\n\n{other}"

    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    others = [{"type": "text", "text": other}] if isinstance(other, str) else other
    assert parts and others
    if _is_text_part(parts[-1]) and _is_text_part(others[0]):
        joined = {**parts[-1], "text": f"{parts[-1]['text']}\n\n{others[0]['text']}"}
        return [*parts[:-1], joined, *others[1:]]
    return [*parts, *others]


def _join_previous_user(
    lowered: list[ConversationMessage], content: _MessageContent
) -> None:
    # Developer content that no user turn follows joins the previous user
    # turn, or becomes one if the previous turn is not one.
    if lowered and lowered[-1]["role"] == "user":
        previous = lowered[-1]
        lowered[-1] = {
            **previous,
            "content": _append_content(previous.get("content"), content),
        }
    else:
        lowered.append({"role": "user", "content": content})


def _lower_developer_messages(
    conversation: list[ConversationMessage],
    mode: _DeveloperRoleMode,
) -> list[ConversationMessage]:
    """Lower developer messages for a chat template, keeping them in place.

    Developer messages are merged at most with an adjacent turn, so media
    placeholders stay in order and the prompt prefix stays stable as later
    turns are appended. The caller's own system messages are left to the
    template.
    """
    if mode == "native":
        return conversation

    leading_run = next(
        (
            i
            for i, msg in enumerate(conversation)
            if msg["role"] not in ("system", "developer")
        ),
        len(conversation),
    )
    lowered: list[ConversationMessage] = []
    if any(msg["role"] == "developer" for msg in conversation[:leading_run]):
        merged: _MessageContent = ""
        for msg in conversation[:leading_run]:
            merged = _append_content(merged, msg.get("content"))
        lowered.append({"role": "system", "content": merged})
    else:
        lowered.extend(conversation[:leading_run])

    # Developer content waiting for the next user turn.
    pending: _MessageContent = None
    for msg in conversation[leading_run:]:
        if msg["role"] == "developer":
            if mode == "system":
                system_msg = {k: v for k, v in msg.items() if k != "tools"}
                system_msg["role"] = "system"
                lowered.append(cast("ConversationMessage", system_msg))
            else:
                pending = _append_content(pending or "", msg.get("content"))
        elif msg["role"] == "user" and pending is not None:
            content = _append_content(pending, msg.get("content"))
            lowered.append({**msg, "content": content})
            pending = None
        else:
            if pending is not None:
                _join_previous_user(lowered, pending)
                pending = None
            lowered.append(msg)
    if pending is not None:
        _join_previous_user(lowered, pending)
    return lowered


_CONTENT_FORMATS_MAXSIZE = 32
_CONTENT_FORMATS = dict[
    tuple[str | None, bool, str, str | None, str | None, bool],
    "ChatTemplateContentFormat",
]()
"""
Used in `_resolve_chat_template_content_format` to avoid resolving and parsing
the chat template on every request.

`lru_cache` cannot be used because `tools` is a list and `ModelConfig` defines
`__eq__` without `__hash__`, so the key holds the fields `resolve_chat_template`
selects the template by. Only the presence of `tools` is part of it: it decides
whether the processor template is tried, and its contents never reach the AST.
"""


def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    cache_key = (
        chat_template,
        tools is None,
        tokenizer.name_or_path,
        model_config.revision,
        model_config.code_revision,
        model_config.trust_remote_code,
    )
    if (cached_format := _CONTENT_FORMATS.get(cache_key)) is not None:
        return cached_format

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

    # Requests may carry their own chat template, so bound the cache.
    if len(_CONTENT_FORMATS) >= _CONTENT_FORMATS_MAXSIZE:
        _CONTENT_FORMATS.clear()
    _CONTENT_FORMATS[cache_key] = detected_format

    return detected_format


def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    if given_format == "auto":
        logger.info_once(
            "Detected the chat template content format to be '%s'. "
            "You can set `--chat-template-content-format` to override this.",
            detected_format,
        )
    elif given_format != detected_format:
        logger.warning_once(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )

    return detected_format if given_format == "auto" else given_format


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
    from transformers import PythonBackend

    # Get standard parameters from HuggingFace's base tokenizer class.
    # This dynamically extracts parameters from PythonBackend's
    # apply_chat_template method, ensuring compatibility with tokenizers
    # that use **kwargs to receive standard parameters.

    # Read signature from HF's base class - the single source of truth
    base_sig = inspect.signature(PythonBackend.apply_chat_template)

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


def _template_error_reason(exc: BaseException) -> str:
    # Extract the most specific reason from a chat template error chain.
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, jinja2.TemplateError):
            return str(current)
        current = current.__cause__ or current.__context__
    return str(exc)


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
    if any(msg["role"] == "developer" for msg in conversation):
        hf_params = _get_hf_base_chat_template_params()
        template_kwargs = {
            k: v for k, v in resolved_kwargs.items() if k not in hf_params
        }
        mode = _resolve_developer_role_mode(tokenizer, chat_template, template_kwargs)
        if mode != "native":
            conversation = _lower_developer_messages(conversation, mode)
            logger.info_once(
                "Chat template does not support the 'developer' message role. "
                "Developer messages after the leading system message become %s.",
                "system messages in place"
                if mode == "system"
                else "part of the adjacent user turn",
            )

    # transformers v5 changed the default of `return_dict` to True, which
    # makes `apply_chat_template(tokenize=True)` return a `BatchEncoding`
    # instead of `list[int]`. Force `return_dict=False` so downstream code
    # that expects a flat token list (e.g. `parse_dec_only_prompt`) works
    # consistently across v4 and v5.
    if tokenize and "return_dict" not in resolved_kwargs:
        resolved_kwargs["return_dict"] = False

    try:
        plain = tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=chat_template,
            tokenize=tokenize,
            **resolved_kwargs,
        )
    except Exception as e:
        # Chat templates reject invalid user input (e.g. an unsupported
        # `reasoning_effort` value) by raising from within the template.
        # Surface those as a 400 Bad Request carrying the template's own
        # reason (which typically lists the supported values) instead of a
        # 500 or any generic upstream wrapper message.
        logger.warning("Chat template rejected the request: %s", e)
        raise VLLMValidationError(_template_error_reason(e)) from e

    return plain


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
        super().__init__(config, tokenizer)

        self.use_unified_vision_chunk = getattr(
            config.model_config.hf_config, "use_unified_vision_chunk", False
        )

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

    def _prepare_tokenizer(self, tokenizer: HfTokenizer) -> None:
        # Skip for mock configs
        if getattr(self.model_config, "enable_prompt_embeds", False):
            _ensure_prompt_embeds_placeholder_token(tokenizer)

    def _replace_executor(self) -> None:
        """Replace the inner render pool after a stuck worker.

        Called after a render timeout so that the stuck thread (which CPython
        cannot interrupt) does not permanently block all subsequent renders.
        The executor *object* is kept so `make_async` wrappers bound at
        init (tokenize, decode, pooling, derender) keep submitting work.
        The old inner pool is abandoned with wait=False — its thread will
        eventually finish or be reaped on process exit.
        """
        self._executor.replace_inner()
        logger.warning(
            "Chat template render timed out — executor pool replaced "
            "(%d worker(s)). The stuck thread may continue consuming "
            "CPU until the process exits.",
            self.model_config.renderer_num_workers,
        )

    async def _render_with_timeout(self, coro):
        """Run a template-rendering coroutine with a timeout.

        On timeout, replaces the executor so subsequent renders are not
        permanently blocked by the stuck thread.
        """
        timeout = envs.VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT
        if timeout <= 0:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError:
            self._replace_executor()
            raise TimeoutError(
                f"Chat template rendering timed out after "
                f"{timeout}s. This may indicate a malicious or "
                f"excessively complex template. Adjust "
                f"VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT to override."
            ) from None

    def _can_produce_offsets(self) -> bool:
        # Only fast tokenizers expose offset_mapping.
        return self.tokenizer is not None and self.tokenizer.is_fast

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
            visibility_check_kwargs = chat_template_kwargs.copy()
            visibility_check_kwargs["tokenize"] = False
            rendered = safe_apply_chat_template(
                model_config,
                tokenizer,
                conversation,
                **visibility_check_kwargs,
            )
            assert isinstance(rendered, str)
            prompt_embeds_tensors, conversation = _filter_prompt_embeds_by_template(
                conversation,
                prompt_embeds_tensors,
                rendered,
            )

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
            visibility_check_kwargs = chat_template_kwargs.copy()
            visibility_check_kwargs["tokenize"] = False
            rendered = await self._render_with_timeout(
                self._apply_chat_template_async(
                    model_config,
                    tokenizer,
                    conversation,
                    **visibility_check_kwargs,
                )
            )
            assert isinstance(rendered, str)
            prompt_embeds_tensors, conversation = _filter_prompt_embeds_by_template(
                conversation,
                prompt_embeds_tensors,
                rendered,
            )

        if prompt_embeds_tensors:
            # prompt_embeds post-processing requires prompt_token_ids.
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        prompt_raw = await self._render_with_timeout(
            self._apply_chat_template_async(
                model_config,
                tokenizer,
                conversation,
                **chat_template_kwargs,
            )
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
        if prompt_embeds_info is None:
            return super()._process_tokens(prompt, skip_mm_cache=skip_mm_cache)

        tensors, placeholder_token_id = prompt_embeds_info
        mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
        cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
            list(prompt["prompt_token_ids"]), mm_updates
        )
        engine_input = super()._process_tokens(prompt, skip_mm_cache=skip_mm_cache)
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
        if prompt_embeds_info is None:
            return await super()._process_tokens_async(
                prompt, skip_mm_cache=skip_mm_cache
            )

        tensors, placeholder_token_id = prompt_embeds_info
        mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
        cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
            list(prompt["prompt_token_ids"]), mm_updates
        )
        engine_input = await super()._process_tokens_async(
            prompt, skip_mm_cache=skip_mm_cache
        )
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

    def _apply_prompt_embeds_to_engine_input(
        self,
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
        mm_config = self.model_config.get_multimodal_config()
        for tensor, (start, length) in zip(
            prompt_embeds_tensors, positions, strict=True
        ):
            token_ids[start : start + length] = [0] * length
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
            pe_hashes.append(
                MultiModalHasher.hash_kwargs(
                    mm_config.mm_hasher_algorithm, prompt_embeds=tensor
                )
            )
            # `is_embed=None` matches the existing image_embeds-style
            # "no encoder, just splice the tensor directly" semantics.
            pe_placeholders.append(
                PlaceholderRange(offset=start, length=length, is_embed=None)
            )

        engine_input["prompt_token_ids"] = token_ids
        cast(
            MultiModalKwargsItems[MultiModalKwargsItem | None],
            engine_input["mm_kwargs"],
        )["prompt_embeds"] = pe_kwargs_items
        engine_input["mm_hashes"] = {
            **engine_input["mm_hashes"],
            "prompt_embeds": pe_hashes,
        }
        cast(dict, engine_input["mm_placeholders"])["prompt_embeds"] = pe_placeholders
