# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere prompt renderer.

Templates the Cohere Command-family prompt formats (cmd3 / cmd4) using
the ``cohere_melody`` Rust bindings instead of Jinja. Enabled by
passing ``--tokenizer-mode cohere`` on the engine; tokenization itself
still flows through the cached HuggingFace tokenizer.

The renderer's client surface is ``chat_template_kwargs`` (vLLM's
generic passthrough for template-time inputs). It is fed by two
different code paths, which speak different vocabularies:

1. :class:`vllm.entrypoints.cohere.serving.CohereServingChatV2` --
   translates a native `Cohere Chat v2 request
   <https://docs.cohere.com/reference/chat>`__ into
   ``chat_template_kwargs``.
2. :class:`OpenAIServingChat` -- forwards whatever the client puts in
   ``request.chat_template_kwargs`` directly, without translation.

Accordingly the renderer accepts keys from both vocabularies.

**Cohere Chat v2 request fields** (populated automatically by
``CohereServingChatV2``; a direct ``chat_template_kwargs`` caller may
also pass them):

* ``documents``: list of documents (v2 ``documents``)
* ``tools``: list of tools (v2 ``tools``)
* ``safety_mode``: v2 ``safety_mode`` (``CONTEXTUAL`` / ``STRICT`` /
  ``NONE``) -- forwarded to melody's cmd3 ``safety_mode`` slot;
  lowercased.
* ``citation_options``: v2 ``citation_options``. Its ``.mode``
  (``ENABLED`` / ``DISABLED`` / ``FAST`` / ``ACCURATE`` / ``OFF``) is
  normalized into melody's cmd3 ``citation_quality`` (``on`` / ``off``)
  and cmd4 ``grounding`` (``enabled`` / ``disabled`` / ``unknown``)
  slots. cmd4 has no fast/accurate distinction at the prompt-template
  layer -- both mean grounding-on.
* ``response_format``: v2 ``response_format`` (``json_object`` /
  ``json_schema``) -- mapped onto melody's ``json_mode`` /
  ``json_schema`` config slots.
* ``thinking``: v2 ``thinking``; its ``.type`` (``enabled`` /
  ``disabled``) becomes melody's ``reasoning_type``.

**Melody template-config knobs** (accepted for direct-passthrough
callers; not part of the Cohere Chat v2 surface):

* ``cohere_format``: renderer selector, ``cmd3`` or ``cmd4`` (default
  ``cmd4``).
* ``template_id``: pick one of melody's built-in template variants.
* ``available_tools``: melody's own name for tools; takes precedence
  over ``tools`` when both are set.
* ``reasoning_type``: direct ``enabled`` / ``disabled`` toggle;
  overrides derivation from ``thinking.type``.
* ``dev_instruction``: developer-instruction override.
* ``json_mode`` / ``json_schema``: direct structured-output toggles;
  override derivation from ``response_format``.
* cmd3-only: ``citation_quality`` (``on`` / ``off``), ``skip_preamble``.
* cmd4-only: ``grounding`` (``enabled`` / ``disabled`` / ``unknown``;
  overrides derivation from ``citation_options.mode``),
  ``platform_instruction``.

**Rejected inputs**: ``template_jinja`` and ``template`` are Cohere's
own inlets for raw Jinja template source. They are valid at Cohere's
API surface but not at vLLM's -- raw template source must flow through
the standard ``chat_template`` request field (guarded by
``--trust-request-chat-template``), which this renderer forwards to
melody as ``template_jinja``. Setting these keys explicitly in
``chat_template_kwargs`` raises ``ValueError`` so client
misconfiguration surfaces loudly.

**Everything else** in ``chat_template_kwargs`` is forwarded verbatim
to melody as ``additional_template_fields`` and becomes accessible as
Jinja variables inside the template. This matches vLLM's documented
contract for ``chat_template_kwargs`` ("kwargs accessible by the
template"), so e.g. ``chat_template_kwargs={"reasoning_effort":
"low"}`` resolves ``{{ reasoning_effort }}`` inside cmd3 / cmd4
templates.

Citations produced by Cohere models are surfaced through the
Cohere-scoped ``CohereChatMessage.citations`` /
``CohereDeltaMessage.citations`` fields (see
:mod:`vllm.entrypoints.cohere.cohere_chat_message`), populated by the
``cohere2`` reasoning parser. The base OpenAI ``ChatMessage`` /
``DeltaMessage`` keep their declared schemas unchanged; the response
envelope declares them as ``SerializeAsAny[...]`` so the subclass
fields survive JSON serialization.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.logger import init_logger
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)


_DEFAULT_FORMAT = "cmd4"
_VALID_FORMATS = ("cmd3", "cmd4")


class MelodyContentType(str, Enum):
    """Wire-format discriminator for melody content blocks.

    These strings are what the cmd3 / cmd4 Jinja templates check against
    in ``message.content[0].type`` (e.g. the ``thinking`` branch in
    ``cmd4-v1.jinja``). Keep new values in sync with melody's template
    schema.
    """

    TEXT = "text"
    THINKING = "thinking"
    IMAGE = "image"
    DOCUMENT = "document"


# Reserved ``chat_template_kwargs`` key that carries
# ``AssistantChatMessageV2.citations`` from a Cohere v2 request through
# to the renderer, keyed by request-message index. Populated by
# ``CohereServingChatV2._apply_cohere_template_kwargs`` and consumed by
# ``_conversation_to_melody_messages``. The value is a
# ``dict[int, list[dict]]`` of melody-shape ``FilterCitation`` dicts.
# The leading underscore signals "internal, not part of the public
# ``chat_template_kwargs`` surface" -- clients should not set this key
# directly.
MESSAGES_CITATIONS_KEY = "_messages_citations"


# Reserved ``chat_template_kwargs`` key carrying a request-side
# ``dict[tuple[int, int], CitationSource]`` map from melody's numeric
# ``(tool_call_index, tool_result_index)`` addressing to fully-resolved
# wire-shape sources (type, id, document/tool_output payload).
# Populated by :meth:`CohereServingChatV2._apply_cohere_template_kwargs`
# and consumed by :class:`BaseCohereCommandReasoningParser` at request
# start so the parser can attach real document ids on each melody source
# it emits -- eliminating the numeric-coord round-trip through the
# internal streaming pipeline. Leading underscore signals "internal,
# not part of the public ``chat_template_kwargs`` surface" -- clients
# should not set this key directly.
POSITION_TO_SOURCE_KEY = "_position_to_source"


# Keys this renderer interprets directly from ``chat_template_kwargs`` and
# maps onto typed melody render-config fields. Everything *not* in this
# set is forwarded verbatim to melody as ``additional_template_fields``
# (i.e. as Jinja template variables), so callers can write
# ``chat_template_kwargs = {"my_var": "..."}`` and have ``{{ my_var }}``
# resolve inside the template -- matching vLLM's documented contract
# for ``chat_template_kwargs``.
_RENDERER_CONSUMED_KEYS = frozenset(
    {
        "cohere_format",
        "template_id",
        # ``template_jinja`` / ``template`` are Cohere-only inputs for
        # raw Jinja source -- valid at Cohere's own API surface but not
        # at vLLM's. In vLLM, raw template source must flow through the
        # standard ``chat_template`` request field, so a non-``None``
        # value here is rejected in ``_build_render_config``. The keys
        # stay in the consumed set so an explicit ``None`` (e.g. from
        # ``.model_dump(exclude_none=False)``) is dropped rather than
        # surfacing as a stray ``{{ template_jinja }}`` Jinja variable.
        "template_jinja",
        "template",
        "use_jinja",
        "documents",
        "available_tools",
        "tools",
        "reasoning_type",
        "thinking",
        "dev_instruction",
        "response_format",
        "json_schema",
        "json_mode",
        "safety_mode",
        "citation_quality",
        "citation_options",
        "skip_preamble",
        "grounding",
        "platform_instruction",
        MESSAGES_CITATIONS_KEY,
        POSITION_TO_SOURCE_KEY,
    }
)


def _try_import_melody():
    try:
        import cohere_melody  # type: ignore

        return cohere_melody
    except ImportError as e:  # pragma: no cover - exercised at runtime
        raise ImportError(
            "The `cohere` tokenizer/renderer mode requires the "
            "`cohere_melody` package. Install it via "
            "`pip install cohere-melody` or build from "
            "https://github.com/cohere-ai/melody."
        ) from e


_MELODY_ROLES = frozenset({"system", "user", "chatbot", "tool"})


# Cohere v2 ``citation_options.mode`` -> melody cmd4 ``grounding``.
# v2 surfaces five modes (``ENABLED`` / ``DISABLED`` / ``FAST`` /
# ``ACCURATE`` / ``OFF``) but cmd4's template doesn't differentiate
# fast vs accurate at the prompt layer -- both just turn grounding on,
# and ``ENABLED`` / ``DISABLED`` are direct aliases. Melody itself
# accepts ``unknown`` / ``enabled`` / ``disabled``.
_CMD4_GROUNDING_FROM_MODE = {
    "fast": "enabled",
    "accurate": "enabled",
    "on": "enabled",
    "enabled": "enabled",
    "off": "disabled",
    "disabled": "disabled",
    "unknown": "unknown",
}


def _normalize_cmd4_grounding(value: Any) -> str:
    """Coerce a user-facing grounding/citation mode to melody's vocab.

    Raises ``ValueError`` rather than letting an unrecognized value
    slip through to ``render_cmd4`` and surface as a generic
    ``Invalid config: grounding`` from melody.
    """
    out = _CMD4_GROUNDING_FROM_MODE.get(value.lower())
    if out is None:
        raise ValueError(
            f"Unrecognized cmd4 grounding value: {value!r}. Expected one "
            f"of ENABLED / DISABLED / FAST / ACCURATE / OFF (from "
            f"citation_options.mode), or the direct melody values "
            f"enabled / disabled / unknown."
        )
    return out


def _role_to_melody(role: str) -> str:
    """Map an OpenAI role to the role string melody expects.

    The cmd3 / cmd4 jinja templates only recognize ``system``, ``user``,
    ``assistant``/``chatbot``, and ``tool``: any other role is silently
    dropped by the template's role-dispatch chain (no fallback branch),
    which produces a malformed prompt without any error. We therefore
    refuse unknown roles up front rather than letting them disappear.

    Aliases:

    * ``assistant`` -> ``chatbot`` (Cohere's historical assistant role
      name used by the templates).
    * ``developer`` -> ``system`` (OpenAI's ``developer`` role is
      documented as high-priority instructions, which maps onto the
      ``system`` slot in Cohere's prompt format).
    """
    role = role.lower()
    if role == "assistant":
        return "chatbot"
    if role == "developer":
        return "system"
    if role in _MELODY_ROLES:
        return role
    raise ValueError(
        f"Unsupported message role for the cohere renderer: {role!r}. "
        "Expected one of: system, developer, user, assistant, chatbot, tool."
    )


def _normalize_tool_call(tc: dict[str, Any] | Any) -> dict[str, Any]:
    """Normalize an OpenAI tool call dict into melody's tool call shape.

    melody expects ``{id, name, parameters: <json str>}`` whereas OpenAI
    delivers ``{id, type, function: {name, arguments: <json|str>}}``.
    """
    # Pydantic objects
    if hasattr(tc, "model_dump"):
        tc = tc.model_dump()
    if not isinstance(tc, dict):
        raise TypeError(f"Unexpected tool_call value: {tc!r}")

    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or ""
    args = fn.get("arguments")
    if args is None:
        args = tc.get("arguments", {})
    # melody expects a JSON-encoded string
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": tc.get("id") or "",
        "name": name,
        "parameters": args,
    }


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI ``ConversationMessage.content`` into melody content blocks.

    The chat_utils ``content_format="openai"`` produces a list of
    ``{type: text, text: ...}`` dicts already; we pass those through with
    minimal coercion. Plain string content is wrapped in a single text block.
    Image / multimodal placeholder dicts (``{"type": "image"}``) are
    forwarded as-is so that templates can reference image placeholders that
    the upstream tokenizer expands separately.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": MelodyContentType.TEXT, "text": content}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": MelodyContentType.TEXT, "text": part})
            continue
        if not isinstance(part, dict):
            raise TypeError(f"Unexpected content part: {part!r}")
        part_type = part.get("type", MelodyContentType.TEXT)
        if part_type in ("text", "input_text", "output_text", "refusal"):
            blocks.append(
                {"type": MelodyContentType.TEXT, "text": part.get("text", "")}
            )
        elif part_type == MelodyContentType.THINKING:
            blocks.append(
                {
                    "type": MelodyContentType.THINKING,
                    "thinking": part.get("thinking", ""),
                }
            )
        elif part_type == MelodyContentType.IMAGE:
            blocks.append(
                {
                    "type": MelodyContentType.IMAGE,
                    "image": {
                        "template_placeholder": part.get(
                            "template_placeholder", "<image>"
                        ),
                    },
                }
            )
        elif part_type == MelodyContentType.DOCUMENT:
            doc = part.get("document")
            if isinstance(doc, dict):
                blocks.append({"type": MelodyContentType.DOCUMENT, "document": doc})
            else:
                # Fall back to wrapping arbitrary string as text.
                blocks.append({"type": MelodyContentType.TEXT, "text": json.dumps(doc)})
        elif part_type == "tool_reference":
            # Tool references are rendered by name; emit as text so the
            # renderer downstream is content-format agnostic.
            blocks.append(
                {
                    "type": MelodyContentType.TEXT,
                    "text": part.get("name") or part.get("text", ""),
                }
            )
        else:
            # Unknown block type: render as text fallback.
            text = part.get("text") or part.get(part_type) or ""
            text_str = text if isinstance(text, str) else json.dumps(text)
            blocks.append({"type": MelodyContentType.TEXT, "text": text_str})
    return blocks


def _document_to_melody(doc: Any) -> dict[str, Any]:
    """Coerce a Cohere v2 document into melody's ``Document`` (dict) shape."""
    match doc:
        case str():
            return {"text": doc}
        # Cohere v2 wraps documents in {id, data: {...}}; the ``**rest``
        # capture keeps the outer ``id`` (and any other outer keys) so we
        # can lift them onto the payload without losing them.
        case {"data": dict() as data, **rest}:
            payload = dict(data)
            if (outer_id := rest.get("id")) is not None and "id" not in payload:
                payload["id"] = outer_id
            return payload
        case dict():
            return dict(doc)
        case _:
            raise TypeError(f"Unsupported document type: {type(doc).__name__}")


def _tool_to_melody(tool: Any) -> dict[str, Any]:
    """Coerce a chat completions tool definition into melody's ``Tool`` shape.

    Accepts either the raw OpenAI tool wrapper ``{type:"function", function:
    {name, description, parameters}}`` or a flat ``{name, description,
    parameters}`` dict (which is what melody itself expects).
    """
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump()
    if not isinstance(tool, dict):
        raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    if "function" in tool and isinstance(tool["function"], dict):
        fn = tool["function"]
    else:
        fn = tool
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", "") or "",
        "parameters": fn.get("parameters") or {},
    }


def _conversation_to_melody_messages(
    conversation: list[ConversationMessage],
    messages_citations: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Convert vLLM ``ConversationMessage``s into melody's message dict shape.

    ``messages_citations`` is an optional index-keyed lookup of
    per-message citations from the request, populated by
    ``CohereServingChatV2`` under the
    :data:`MESSAGES_CITATIONS_KEY` chat_template_kwargs entry. Values
    must already be in melody's ``FilterCitation`` dict shape (see
    ``CohereServingChatV2._sdk_citation_to_melody``). Index is the
    position in the *input* ``conversation`` list, which currently maps
    1-to-1 with ``CohereChatV2Request.messages`` for text-only Cohere
    requests. This mapping breaks silently if ``parse_chat_messages``
    ever expands a single input message into multiple entries (e.g.
    multi-modal splitting), which is not the case today.
    """
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(conversation):
        role = _role_to_melody(msg.get("role", "user"))
        content_blocks = _content_blocks(msg.get("content"))

        # Treat reasoning content as a thinking block on assistant turns
        # so multi-turn reasoning is preserved in the rendered prompt.
        #
        # Cohere models output thought as either a ``thinking``
        # block (reasoning models) or a ``tool_plan`` field (older non-
        # reasoning Command models), but vLLM's ConversationMessage has a
        # single unified ``reasoning`` field that drops that difference.
        # The cmd3 / cmd4 jinja templates render ``thinking`` blocks in both
        # the case of tool calls and regular thinking blocks, so this is fine
        # from the renderer's perspective.
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if role == "chatbot" and reasoning:
            content_blocks.insert(
                0,
                {"type": MelodyContentType.THINKING, "thinking": reasoning},
            )

        tool_calls = [_normalize_tool_call(tc) for tc in (msg.get("tool_calls") or [])]

        out_msg: dict[str, Any] = {
            "role": role,
            "content": content_blocks,
            "tool_calls": tool_calls,
        }
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            out_msg["tool_call_id"] = tool_call_id
        if messages_citations and (cites := messages_citations.get(i)):
            out_msg["citations"] = cites
        out.append(out_msg)
    return out


def _build_render_config(
    conversation: list[ConversationMessage],
    chat_template_kwargs: dict[str, Any],
    chat_template: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the ``render_cmd3`` / ``render_cmd4`` config dict.

    ``chat_template`` is the standard vLLM per-request template override
    (from ``ChatCompletionRequest.chat_template`` / the server-side
    ``--chat-template`` flag). When non-``None`` it is forwarded to melody
    as ``template_jinja`` so a single vLLM-native path drives template
    source through the trust guard.

    Returns ``(format, config_dict)`` where ``format`` is either ``"cmd3"``
    or ``"cmd4"``.
    """
    fmt = chat_template_kwargs.get("cohere_format", _DEFAULT_FORMAT)
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"Invalid cohere_format={fmt!r}; expected one of {_VALID_FORMATS}"
        )

    # Reject Cohere-only inlets for raw Jinja source. These keys are
    # valid at Cohere's own API surface but not at vLLM's: callers must
    # use the standard vLLM ``chat_template`` request field so the
    # ``--trust-request-chat-template`` guard applies uniformly.
    for cohere_only_key in ("template_jinja", "template"):
        if chat_template_kwargs.get(cohere_only_key) is not None:
            raise ValueError(
                f"chat_template_kwargs.{cohere_only_key!r} is not accepted "
                "in vLLM (it is a Cohere-only field); pass the template "
                "body via the standard 'chat_template' request field "
                "instead (requires --trust-request-chat-template)."
            )

    config: dict[str, Any] = {
        "messages": _conversation_to_melody_messages(
            conversation,
            chat_template_kwargs.get(MESSAGES_CITATIONS_KEY),
        ),
    }

    # ``template_id`` is a selector for one of melody's built-in template
    # variants (not raw source), so it is safe to accept from the client.
    if template_id := chat_template_kwargs.get("template_id"):
        config["template_id"] = template_id
    # Raw template source flows exclusively through the standard vLLM
    # ``chat_template`` field so it is subject to the
    # ``--trust-request-chat-template`` guard in OnlineRenderer.
    if chat_template:
        config["template_jinja"] = chat_template
    # Only support Jinja with vllm
    config["use_jinja"] = True

    # Documents
    documents = chat_template_kwargs.get("documents") or []
    if documents:
        config["documents"] = [_document_to_melody(d) for d in documents]

    # Tools - prefer explicit ``available_tools``, fall back to OpenAI ``tools``
    tools = (
        chat_template_kwargs.get("available_tools")
        or chat_template_kwargs.get("tools")
        or []
    )
    if tools:
        config["available_tools"] = [_tool_to_melody(t) for t in tools]

    # Reasoning toggle (cmd3 + cmd4)
    if (rt := chat_template_kwargs.get("reasoning_type")) is not None:
        config["reasoning_type"] = str(rt)
    elif "thinking" in chat_template_kwargs:
        # Cohere v2 ``thinking: {type: enabled|disabled}`` shorthand
        thinking = chat_template_kwargs["thinking"]
        t = thinking.get("type") if isinstance(thinking, dict) else thinking
        if t in ("enabled", "disabled"):
            config["reasoning_type"] = t

    if (di := chat_template_kwargs.get("dev_instruction")) is not None:
        config["dev_instruction"] = str(di)

    # JSON / structured outputs
    if (rf := chat_template_kwargs.get("response_format")) is not None:
        rf = rf.model_dump() if hasattr(rf, "model_dump") else dict(rf)
        rf_type = rf.get("type")
        if rf_type == "json_object":
            config["json_mode"] = True
        elif rf_type in ("json_schema", "json"):
            schema = rf.get("schema") or rf.get("json_schema")
            if isinstance(schema, dict) and "schema" in schema:
                schema = schema["schema"]
            if schema is not None:
                config["json_schema"] = (
                    schema if isinstance(schema, str) else json.dumps(schema)
                )
    if (js := chat_template_kwargs.get("json_schema")) is not None:
        config["json_schema"] = js if isinstance(js, str) else json.dumps(js)
    if "json_mode" in chat_template_kwargs:
        config["json_mode"] = bool(chat_template_kwargs["json_mode"])

    if fmt == "cmd3":
        if (sm := chat_template_kwargs.get("safety_mode")) is not None:
            config["safety_mode"] = str(sm).lower()
        # citation_quality: ``on`` / ``off``
        cq = chat_template_kwargs.get("citation_quality")
        if cq is None and (co := chat_template_kwargs.get("citation_options")):
            mode = co.get("mode") if isinstance(co, dict) else None
            if mode is not None:
                cq = "on" if str(mode).lower() != "off" else "off"
        if cq is not None:
            config["citation_quality"] = str(cq).lower()
        if "skip_preamble" in chat_template_kwargs:
            config["skip_preamble"] = bool(chat_template_kwargs["skip_preamble"])
    else:  # cmd4
        # cmd4 uses ``grounding`` rather than safety_mode/citation_quality.
        # melody's cmd4 only accepts ``unknown`` / ``enabled`` / ``disabled``,
        # so the Cohere v2 ``citation_options.mode`` values
        # (``FAST`` / ``ACCURATE`` / ``OFF``) have to be normalized -- a
        # raw lowercased passthrough would raise from
        # ``render_cmd4`` (cmd4 has no fast/accurate distinction at the
        # prompt-template layer; both request grounding-on).
        if (gr := chat_template_kwargs.get("grounding")) is not None:
            config["grounding"] = _normalize_cmd4_grounding(gr)
        elif co := chat_template_kwargs.get("citation_options"):
            mode = co.get("mode") if isinstance(co, dict) else None
            if mode is not None:
                config["grounding"] = _normalize_cmd4_grounding(mode)
        if (pi := chat_template_kwargs.get("platform_instruction")) is not None:
            config["platform_instruction"] = str(pi)

    # Anything we didn't explicitly interpret above is forwarded to melody
    # as a Jinja template variable. This matches vLLM's documented contract
    # for ``chat_template_kwargs`` ("kwargs accessible by the template")
    # and lets callers write e.g. ``{"reasoning_effort": "low"}`` directly
    # without a nested ``additional_template_fields`` wrapper.
    extra = {
        k: v
        for k, v in chat_template_kwargs.items()
        if k not in _RENDERER_CONSUMED_KEYS
    }
    if extra:
        config["additional_template_fields"] = extra

    return fmt, config


class CohereRenderer(BaseRenderer[HfTokenizer]):
    """Renderer that templates Cohere prompts via the melody Rust bindings.

    Tokenization is delegated to the standard HF tokenizer; only the
    chat-template step is replaced with ``cohere_melody.render_cmd3`` /
    ``render_cmd4``. Enabled via ``--tokenizer-mode cohere``.
    """

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        # Match HfRenderer in not mutating the cached tokenizer instance
        tokenizer = copy.copy(tokenizer)
        super().__init__(config, tokenizer)

        # Lazy import to keep `cohere_melody` an optional dependency
        self._melody = _try_import_melody()
        # ``render_cmd3`` / ``render_cmd4`` are pure CPU work; cache the
        # thread-pool wrapper once so the async path doesn't allocate a new
        # adapter on every request.
        self._render_async = make_async(self._render, executor=self._executor)

    def _render(self, fmt: str, config_dict: dict[str, Any]) -> str:
        if fmt == "cmd3":
            return self._melody.render_cmd3(config_dict)
        return self._melody.render_cmd4(config_dict)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        chat_template_kwargs = dict(params.chat_template_kwargs)
        fmt, config_dict = _build_render_config(
            conversation, chat_template_kwargs, params.chat_template
        )
        prompt_text = self._render(fmt, config_dict)
        prompt = parse_dec_only_prompt(prompt_text)

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
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        chat_template_kwargs = dict(params.chat_template_kwargs)
        fmt, config_dict = _build_render_config(
            conversation, chat_template_kwargs, params.chat_template
        )
        prompt_text = await self._render_async(fmt, config_dict)
        prompt = parse_dec_only_prompt(prompt_text)

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
