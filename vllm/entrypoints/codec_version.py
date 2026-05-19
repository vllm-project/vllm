# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec version negotiation — opt-on, two-stage, graceful downgrade.

Implements the rules in `spec/versions/v0.4.md`:

  - § Capabilities are opt-on at the server (two-stage)
  - § Graceful downgrade (response shaping)
  - § Version Compatibility Signaling (Codec-Client-Version, 426 path)

Two-stage model per capability, both off by default:

  - ENABLE:  capability is *available*. Server emits the v0.X-specific
             headers to v0.X+ clients; v0.(X-1) clients see v0.(X-1) wire.
  - ENFORCE: capability is *mandatory*. v0.(X-1) clients get 426
             Upgrade Required with structured upgrade prompt.

Env-var configuration (matches the spec table):

  CODEC_SAFETY_POLICY=<id>            enable safety-policy v0.4
  CODEC_SAFETY_POLICY_REQUIRED=1      enforce  safety-policy v0.4
  CODEC_VERSION_POLICY=advisory|strict  enable + enforce version policy

All operator config flows through env vars so the supervisor can mount
them via docker-compose or systemd without code changes.
"""

from __future__ import annotations

import json
import os

from fastapi import Request
from fastapi.responses import Response

# ── Version comparison ───────────────────────────────────────────────────────


def _parse(v: str) -> tuple[int, int]:
    """Parse "0.X" or "0.X.Y" → (0, X). Patch part is ignored for wire
    decisions — only minor matters per the versioning policy."""
    s = v.strip().lstrip("v")
    parts = s.split(".")
    if len(parts) < 2:
        return (0, 0)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (0, 0)


def version_ge(a: str, b: str) -> bool:
    """True iff version a >= version b. Used to gate header emission."""
    return _parse(a) >= _parse(b)


def version_lt(a: str, b: str) -> bool:
    return not version_ge(a, b)


# ── Request-side ─────────────────────────────────────────────────────────────


# Per spec § Version Compatibility Signaling: a request without the header
# is treated as v0.2 (the oldest published version) — the most conservative
# choice. v0.4+ clients always send the header.
DEFAULT_CLIENT_VERSION = "0.2"


def parse_client_version(request: Request) -> str:
    """Return the client's advertised Codec version, or DEFAULT_CLIENT_VERSION
    if the header is absent.

    Lower-cased lookup because Starlette normalizes header keys.
    """
    raw = request.headers.get("codec-client-version", "").strip()
    if not raw:
        return DEFAULT_CLIENT_VERSION
    # Tolerate a leading "v": "v0.4" → "0.4".
    return raw.lstrip("v")


# ── Header version-introduced floor ──────────────────────────────────────────


# Mirrors `spec/versions/v0.4.md` § Graceful downgrade § reference floor table.
# Server MUST suppress a header when client_version < the floor.
HEADER_VERSION_INTRODUCED: dict[str, str] = {
    "Codec-Tokenizer-Map": "0.2",
    "Codec-Zstd-Dict": "0.2",
    "Codec-Latent-Map": "0.3",
    "Codec-Map": "0.3",  # modality-agnostic alias
    "Codec-Safety-Policy": "0.4",
    "Codec-Safety-Policy-Hash": "0.4",
    # Codec-Min-Version and Codec-Required-Features are v0.4 but emitted
    # ONLY on 426 responses — see make_426_response() — so they're not
    # gated through this table for normal 2xx flow.
}


def should_emit_header(header_name: str, client_version: str) -> bool:
    """True iff the server may emit this header to a client speaking
    `client_version`. Headers not in the registry default to True
    (assumed v0.2 baseline or non-Codec headers like Content-Encoding).
    """
    floor = HEADER_VERSION_INTRODUCED.get(header_name)
    if floor is None:
        return True
    return version_ge(client_version, floor)


def filter_codec_headers(headers: dict, client_version: str) -> dict:
    """Strip any Codec-* header whose version-introduced floor exceeds
    the client's advertised version. Pass-through for non-Codec headers
    (Content-Type, Vary, Content-Encoding, etc.).

    Used at the response-build seam so callers don't have to inline
    a per-header check.
    """
    out = {}
    for k, v in headers.items():
        if should_emit_header(k, client_version):
            out[k] = v
    return out


# ── Capability config (stage-1: enable; stage-2: enforce) ────────────────────


def safety_policy_enabled() -> bool:
    """Stage-1: capability is available. v0.4+ clients see the headers."""
    return bool(os.environ.get("CODEC_SAFETY_POLICY", "").strip())


def safety_policy_required() -> bool:
    """Stage-2: mandatory. v0.3- clients get 426."""
    if not safety_policy_enabled():
        return False
    return os.environ.get("CODEC_SAFETY_POLICY_REQUIRED", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def version_policy_mode() -> str:
    """One of: "off" (default), "advisory" (header set but no 426),
    "strict" (emit 426 when client < 0.4 and any capability is required).
    """
    mode = os.environ.get("CODEC_VERSION_POLICY", "off").strip().lower()
    return mode if mode in ("off", "advisory", "strict") else "off"


def any_v04_mandatory() -> bool:
    """True iff any v0.4 capability is set to ENFORCE."""
    return safety_policy_required() or version_policy_mode() == "strict"


# ── 426 builder ──────────────────────────────────────────────────────────────


def collect_required_features() -> list[str]:
    """Per spec § Required-features registry: which feature names should
    appear on a 426 right now, based on the server's enforce-stage
    config."""
    out: list[str] = []
    if safety_policy_required():
        out.append("safety-policy-enforcement")
    # Future: mandatory-classifier when that capability lands.
    return out


def needs_upgrade(client_version: str, min_version: str = "0.4") -> bool:
    """True iff the deployment mandates `min_version` and the client's
    advertised version is below it."""
    if not any_v04_mandatory():
        return False
    return version_lt(client_version, min_version)


def make_426_response(
    *,
    client_version: str,
    min_version: str = "0.4",
    deployment_id: str | None = None,
) -> Response:
    """Build the structured 426 Upgrade Required response.

    See `spec/versions/v0.4.md § Version Compatibility Signaling` —
    HTTP-transport shape. JSON body degrades gracefully for older
    clients that don't know the field shape (`error` is a string they
    can render).
    """
    required = collect_required_features()
    body = {
        "error": "codec_version_required",
        "minimum_version": min_version,
        "required_features": required,
        "client_version": client_version,
        "docs_url": "https://codecai.net/docs/version-negotiation/",
    }
    if deployment_id is None:
        deployment_id = os.environ.get("CODEC_DEPLOYMENT_ID", "").strip()
    if deployment_id:
        body["deployment_id"] = deployment_id

    headers = {
        "Codec-Min-Version": min_version,
        "Codec-Required-Features": ", ".join(required) if required else "",
    }
    if not headers["Codec-Required-Features"]:
        # Don't ship an empty header value.
        del headers["Codec-Required-Features"]

    return Response(
        content=json.dumps(body),
        status_code=426,
        media_type="application/json",
        headers=headers,
    )


# ── Well-known version-policy descriptor ─────────────────────────────────────


def version_policy_document() -> dict | None:
    """Return the `.well-known/codec/version-policy.json` content for the
    current server state, or None when no v0.4 capability is mandatory
    (the spec says deployments without mandatory features SHOULD NOT
    publish this).
    """
    if not any_v04_mandatory():
        return None
    return {
        "minimum_version": "0.4",
        "required_features": collect_required_features(),
        "deployment_id": os.environ.get("CODEC_DEPLOYMENT_ID", "").strip() or None,
        "docs_url": "https://codecai.net/docs/version-negotiation/",
    }
