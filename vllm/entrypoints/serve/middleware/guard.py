# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional fastapi-guard security middleware for the API server.

fastapi-guard (https://github.com/Guard-Core/fastapi-guard) provides IP
block/allow lists, rate limiting with auto-ban, user-agent blocking,
penetration-attempt detection, security headers, optional Redis-backed
shared state, and optional IPInfo geo/cloud-provider lookups.

The middleware is disabled unless VLLM_GUARD_ENABLED is set. When enabled
without the package installed, registration raises an actionable error
instead of silently disabling security.
"""

from typing import TYPE_CHECKING

from fastapi import FastAPI

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from guard import SecurityConfig

logger = init_logger(__name__)


def is_guard_available() -> bool:
    """Return True when the fastapi-guard package is importable."""
    try:
        import guard  # noqa: F401
    except ImportError:
        return False
    return True


def _csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _build_guard_config() -> "SecurityConfig":
    """Build the SecurityConfig from the VLLM_GUARD_* environment variables."""
    from guard import SecurityConfig

    kwargs: dict[str, object] = {
        "enable_rate_limiting": True,
        "rate_limit": envs.VLLM_GUARD_RATE_LIMIT,
        "rate_limit_window": envs.VLLM_GUARD_RATE_LIMIT_WINDOW,
        "enable_ip_banning": True,
        "auto_ban_threshold": envs.VLLM_GUARD_AUTO_BAN_THRESHOLD,
        "auto_ban_duration": envs.VLLM_GUARD_AUTO_BAN_DURATION,
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable.
        "enable_redis": False,
        "passive_mode": envs.VLLM_GUARD_PASSIVE_MODE,
        "enable_rate_limit_auto_ban": envs.VLLM_GUARD_RATE_LIMIT_AUTO_BAN,
        "blacklist": _csv(envs.VLLM_GUARD_BLOCKED_IPS),
        "blocked_user_agents": list(_csv(envs.VLLM_GUARD_BLOCKED_USER_AGENTS)),
        "trusted_proxies": _csv(envs.VLLM_GUARD_TRUSTED_PROXIES),
        "trusted_proxy_depth": envs.VLLM_GUARD_TRUSTED_PROXY_DEPTH,
        "exclude_paths": list(_csv(envs.VLLM_GUARD_EXCLUDED_PATHS)),
        "custom_log_file": envs.VLLM_GUARD_LOG_FILE,
        "log_format": envs.VLLM_GUARD_LOG_FORMAT,
        "security_headers": (
            {
                "enabled": True,
                "hsts": {"max_age": 31536000, "include_subdomains": True},
                "frame_options": "SAMEORIGIN",
                "content_type_options": "nosniff",
                "referrer_policy": "strict-origin-when-cross-origin",
            }
            if envs.VLLM_GUARD_SECURITY_HEADERS
            else None
        ),
        "enforce_https": envs.VLLM_GUARD_ENFORCE_HTTPS,
        # Behind a TLS-terminating proxy, enforce_https must read the
        # forwarded scheme or every request looks like plain HTTP.
        "trust_x_forwarded_proto": (
            envs.VLLM_GUARD_TRUST_X_FORWARDED_PROTO or envs.VLLM_GUARD_ENFORCE_HTTPS
        ),
    }

    if allowed_ips := _csv(envs.VLLM_GUARD_ALLOWED_IPS):
        kwargs["whitelist"] = allowed_ips
    if blocked_countries := _csv(envs.VLLM_GUARD_BLOCKED_COUNTRIES):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _csv(envs.VLLM_GUARD_ALLOWED_COUNTRIES):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _csv(envs.VLLM_GUARD_BLOCK_CLOUD_PROVIDERS):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if envs.VLLM_GUARD_REDIS_URL:
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = envs.VLLM_GUARD_REDIS_URL
        kwargs["redis_prefix"] = "vllm_guard:"

    if envs.VLLM_GUARD_IPINFO_TOKEN:
        kwargs["ipinfo_token"] = envs.VLLM_GUARD_IPINFO_TOKEN

    return SecurityConfig(**kwargs)  # type: ignore[arg-type]


def init_guard_middleware(app: FastAPI) -> None:
    """Attach the fastapi-guard middleware when VLLM_GUARD_ENABLED is set.

    No-op unless the env flag is on. Fails loudly when the flag is on but
    the package is missing: a misconfiguration must never silently disable
    security.
    """
    if not envs.VLLM_GUARD_ENABLED:
        return
    if not is_guard_available():
        raise RuntimeError(
            "VLLM_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "vllm[guard]"'
        )

    from guard import SecurityMiddleware

    app.add_middleware(SecurityMiddleware, config=_build_guard_config())
    logger.info(
        "fastapi-guard security middleware enabled (passive_mode=%s, redis=%s)",
        envs.VLLM_GUARD_PASSIVE_MODE,
        bool(envs.VLLM_GUARD_REDIS_URL),
    )
