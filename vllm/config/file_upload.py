# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration dataclass for the opt-in `/v1/files` upload endpoint."""

from pydantic import Field

from vllm.config.utils import config


@config
class FileUploadConfig:
    """Configuration for the `/v1/files` upload endpoint.

    This subsystem is off by default. When enabled, it exposes an
    OpenAI-compatible file upload API that lets clients upload multimodal
    files (video/image/audio) once and reference them in subsequent
    chat-completion requests via the `vllm-file://<id>` URL scheme.

    See `docs/serving/openai_compatible_server.md` for the security posture
    and deployment patterns.
    """

    enabled: bool = False
    """Whether the `/v1/files` endpoint is exposed. When False (the default),
    all `/v1/files*` paths return 404."""

    dir: str = ""
    """Directory under which uploaded file bytes are stored. Each file is
    stored with a random sha256 on-disk name (the client `filename` is kept
    only in metadata). If empty, defaults to `$TMPDIR/vllm-uploads-<pid>`,
    which is cleared on server startup."""

    ttl_seconds: int = 3600
    """Time-to-live for uploaded files, in seconds. Measured from last
    access (atime). Set to `-1` to disable time-based expiry (quota-based
    LRU eviction still applies). When disabled, `expires_at` is omitted
    from API responses."""

    max_size_mb: int = Field(default=512, gt=0)
    """Maximum size of a single uploaded file, in megabytes. Uploads
    exceeding this limit are rejected with 413 during streaming (no memory
    spike)."""

    max_total_gb: int = Field(default=5, gt=0)
    """Total disk quota across all uploaded files, in gigabytes. When a
    new upload would exceed this limit, oldest files are evicted (LRU).
    Uploads larger than the total quota are rejected outright."""

    max_concurrent: int = Field(default=4, gt=0)
    """Maximum number of simultaneous in-flight upload operations. Requests
    beyond this limit receive 503. Prevents disk-fill DoS via concurrent
    max-size uploads."""

    scope_header: str = ""
    """Name of a request header whose value scopes uploaded files. When
    set, files are tagged at upload time with the header value, and
    subsequent operations only succeed when the caller presents the same
    header value. If set but the header is missing from a request, the
    response is 400. When empty (the default), files are server-global —
    possession of the 128-bit file ID is the sole access control.

    Common values for gateway-fronted deployments:
    - `OpenAI-Project` (native OpenAI SDK; SDK auto-sends from
      `OPENAI_PROJECT_ID` env or the `project=...` client parameter)
    - `X-Consumer-ID` (Kong, Apigee custom policies)
    - `X-Auth-Request-User` (oauth2-proxy from JWT `sub` claim)"""

    disable_listing: bool = False
    """When True, the `GET /v1/files` (list) endpoint returns 404. Individual
    file operations via known ID still work. Removes the enumeration surface
    for capability-only deployments."""
