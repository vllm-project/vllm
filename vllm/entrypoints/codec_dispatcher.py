"""
Codec v0.5 #87: bolt-on tool dispatcher.

Closes the loop on @codecai/tool-kit (shipped at v0.4.1). The SDK lets
operators publish a tool manifest at build time; this module lets the
engine dispatch to those tools without round-tripping through detokenize
→ JSON → re-tokenize.

Three pieces (contract in spec/versions/v0.5.md § "v0.5-5"):

1. Tool registry — reads CODEC_TOOL_MANIFEST_URLS at startup, fetches
   each manifest, validates `tokenizerHash` against the active model's
   tokenizer SHA-256. Mismatched tools load in `text-fallback` mode
   (the engine still surfaces them but doesn't dispatch).

2. MCP-style HTTP client — when ToolWatcher detects a `<tool_call>`
   region, POST `CodecToolCall` (msgpack-framed) to the tool's
   `/codec/tool/v1/call` endpoint.

3. Reinjection path — `CodecToolResult.response_ids` get inserted into
   the model's generation context where `<tool_call>` was detected.
   Model continues generation without detokenize/retokenize on the
   response.

Gated behind CODEC_BOLT_ON_DISPATCH=1, default off. When off, the
engine surfaces `<tool_call>` regions to the client unchanged
(existing v0.2-v0.4 behaviour).

This module ships the contract + helper primitives. Wiring it into
serving_completions._generate_binary_stream is a follow-up integration
pass: that change touches the model's KV cache reinjection path and
needs review against SGLang's batching semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import msgspec.msgpack

log = logging.getLogger(__name__)

CODEC_BOLT_ON_DISPATCH = os.environ.get("CODEC_BOLT_ON_DISPATCH", "0") == "1"
CODEC_TOOL_MANIFEST_URLS = os.environ.get("CODEC_TOOL_MANIFEST_URLS", "").strip()
CODEC_TOOL_MANIFEST_REQUIRED = os.environ.get("CODEC_TOOL_MANIFEST_REQUIRED", "0") == "1"


# ── Wire types (mirror @codecai/tool-kit's CodecToolCall / Result shapes) ──


@dataclass
class CodecToolCall:
    """Outgoing call to a Codec-aware tool endpoint."""

    tool_name: str
    arguments_json: str
    call_id: str
    tokenizer_hash: str


@dataclass
class CodecToolResult:
    """Response from a Codec-aware tool endpoint."""

    response_ids: list[int]
    call_id: str
    is_error: bool = False
    error_message: Optional[str] = None


# msgspec encoders for the wire shape.
_call_encoder = msgspec.msgpack.Encoder()
_result_decoder = msgspec.msgpack.Decoder()


def encode_tool_call(call: CodecToolCall) -> bytes:
    """Encode a tool call to msgpack bytes for POSTing to a tool endpoint."""
    return _call_encoder.encode({
        "tool_name": call.tool_name,
        "arguments_json": call.arguments_json,
        "call_id": call.call_id,
        "tokenizer_hash": call.tokenizer_hash,
    })


def decode_tool_result(data: bytes) -> CodecToolResult:
    """Decode a tool endpoint's response into a CodecToolResult."""
    parsed = _result_decoder.decode(data)
    return CodecToolResult(
        response_ids=list(parsed.get("response_ids", [])),
        call_id=str(parsed.get("call_id", "")),
        is_error=bool(parsed.get("is_error", False)),
        error_message=parsed.get("error_message"),
    )


# ── Manifest registry ──────────────────────────────────────────────────────


@dataclass
class RegisteredTool:
    """One entry in the engine's tool registry."""

    manifest_url: str
    name: str
    endpoint: str
    tokenizer_hash: str
    """The tokenizer the tool was built against (sha256 of the tokenizer
    map bytes). MUST match the active model's tokenizer for dispatch to
    fire; mismatched tools load in text-fallback mode."""
    mode: str  # "dispatch" | "text-fallback"


class ToolRegistry:
    """Engine-side tool registry. Loaded once at boot from
    CODEC_TOOL_MANIFEST_URLS; mutable thereafter only via
    register_tool() (operator-side admin path)."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    @classmethod
    def from_env(cls, active_tokenizer_hash: str) -> "ToolRegistry":
        """Load tools from CODEC_TOOL_MANIFEST_URLS env var.

        ``active_tokenizer_hash`` is the sha256 of the model's tokenizer
        map bytes — used to decide which tools load in dispatch mode vs
        text-fallback mode.
        """
        registry = cls()
        if not CODEC_TOOL_MANIFEST_URLS:
            return registry
        urls = [u.strip() for u in CODEC_TOOL_MANIFEST_URLS.split(",") if u.strip()]
        for url in urls:
            try:
                manifest = _fetch_manifest(url)
                tool_hash = manifest.get("tokenizerHash", "")
                mode = "dispatch" if tool_hash == active_tokenizer_hash else "text-fallback"
                registry.register_tool(RegisteredTool(
                    manifest_url=url,
                    name=manifest["name"],
                    endpoint=manifest["endpoint"],
                    tokenizer_hash=tool_hash,
                    mode=mode,
                ))
                log.info(
                    "codec_dispatcher: registered tool %s (mode=%s) from %s",
                    manifest["name"], mode, url,
                )
            except Exception as e:
                if CODEC_TOOL_MANIFEST_REQUIRED:
                    raise RuntimeError(
                        f"codec_dispatcher: required manifest {url} failed to load: {e}"
                    ) from e
                log.warning(
                    "codec_dispatcher: dropping tool from %s (manifest load failed): %s",
                    url, e,
                )
        return registry


def _fetch_manifest(url: str) -> dict:
    """Fetch a tool manifest JSON from the URL. Stdlib-only to keep the
    engine startup path free of heavy HTTP deps."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read()
    parsed = json.loads(body)
    # Minimum required fields per @codecai/tool-kit's manifest.json shape.
    for required in ("name", "endpoint", "tokenizerHash"):
        if required not in parsed:
            raise ValueError(f"manifest missing required field {required!r}")
    return parsed


# ── Dispatch (MCP-style HTTP client) ───────────────────────────────────────


def dispatch_call(
    tool: RegisteredTool,
    arguments_json: str,
    call_id: str,
) -> CodecToolResult:
    """POST a CodecToolCall to ``tool.endpoint`` and decode the response.

    Synchronous + stdlib-only. Engines that want async dispatch can
    wrap this in a thread or replace with their preferred HTTP client.
    """
    import urllib.request
    call = CodecToolCall(
        tool_name=tool.name,
        arguments_json=arguments_json,
        call_id=call_id,
        tokenizer_hash=tool.tokenizer_hash,
    )
    req = urllib.request.Request(
        tool.endpoint,
        data=encode_tool_call(call),
        headers={"Content-Type": "application/x-msgpack"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    return decode_tool_result(body)


# ── Reinjection hook ───────────────────────────────────────────────────────


def reinject_ids_into_context(context_ids: list[int], reinject_ids: list[int]) -> list[int]:
    """Insert ``reinject_ids`` into ``context_ids`` at the end (append).

    This is the simplest reinjection model — equivalent to the tool
    response being the next chunk the model "reads". Engines with KV-
    cache-aware models MAY do something smarter (insert at the position
    where the <tool_call> region was detected), but append is the
    correct fallback and what the SGLang ToolWatcher integration
    currently uses.
    """
    return context_ids + reinject_ids


__all__ = [
    "CODEC_BOLT_ON_DISPATCH",
    "CODEC_TOOL_MANIFEST_URLS",
    "CODEC_TOOL_MANIFEST_REQUIRED",
    "CodecToolCall",
    "CodecToolResult",
    "RegisteredTool",
    "ToolRegistry",
    "decode_tool_result",
    "dispatch_call",
    "encode_tool_call",
    "reinject_ids_into_context",
]
