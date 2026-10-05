# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_default_cache_root() -> str:
    """Return default root for cache files, honoring XDG_CACHE_HOME."""
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    """Return default root for config files, honoring XDG_CONFIG_HOME."""
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggesting Kubernetes
            service discovery issue.

    """
    if "VLLM_PORT" not in os.environ:
        return None

    port = os.getenv("VLLM_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"VLLM_PORT '{port}' appears to be a URI. "
                "This may be caused by a Kubernetes service discovery issue, "
                "check the warning in: https://docs.vllm.ai/en/latest/configuration/env_vars.html"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


@dataclass(frozen=True)
class EnvVarDef(Generic[T]):
    """Definition and specification for a vLLM environment variable."""

    name: str
    var_type: type
    default: Any
    doc: str
    getter: Callable[[], T] | None = None
    choices: list[Any] | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    def resolve(self) -> T:
        """Resolve and return current runtime value of the environment variable."""
        if self.getter is not None:
            return self.getter()

        raw = os.getenv(self.name)
        if raw is None:
            if callable(self.default):
                return self.default()
            return self.default

        if self.var_type is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")  # type: ignore[return-value]
        if self.var_type is int:
            return int(raw)  # type: ignore[return-value]
        if self.var_type is float:
            return float(raw)  # type: ignore[return-value]
        return raw  # type: ignore[return-value]

    def resolve_default(self) -> Any:
        """Resolve the default value without reading the environment."""
        if callable(self.default):
            return self.default()
        return self.default

    def is_set(self) -> bool:
        """Check if this environment variable is explicitly set in os.environ."""
        return self.name in os.environ


ENV_VAR_REGISTRY: dict[str, EnvVarDef[Any]] = {}


def register_env_var(env_def: EnvVarDef[Any]) -> EnvVarDef[Any]:
    """Register an environment variable definition into the registry."""
    ENV_VAR_REGISTRY[env_def.name] = env_def
    return env_def


def get_env_var_def(name: str) -> EnvVarDef[Any] | None:
    """Get the definition for an environment variable by name."""
    return ENV_VAR_REGISTRY.get(name)


def list_env_vars() -> list[EnvVarDef[Any]]:
    """Return a list of all registered environment variable definitions."""
    return list(ENV_VAR_REGISTRY.values())


def format_env_vars_table() -> str:
    """Format registered environment variables as a human-readable table."""
    lines = [
        f"{'Variable':<36} {'Type':<12} {'Default':<28} {'Description'}",
        "-" * 110,
    ]
    for var in sorted(ENV_VAR_REGISTRY.values(), key=lambda v: v.name):
        type_str = getattr(var.var_type, "__name__", str(var.var_type))
        try:
            default_val = str(var.resolve_default())
        except Exception:
            default_val = "<computed>"
        if len(default_val) > 25:
            default_val = default_val[:22] + "..."
        doc_summary = var.doc.strip().split("\n")[0] if var.doc else ""
        if len(doc_summary) > 42:
            doc_summary = doc_summary[:39] + "..."
        lines.append(f"{var.name:<36} {type_str:<12} {default_val:<28} {doc_summary}")
    return "\n".join(lines)


# ================== Core Environment Variable Definitions ==================

register_env_var(
    EnvVarDef(
        name="VLLM_TARGET_DEVICE",
        var_type=str,
        default="cuda",
        doc="Target device of vLLM, supporting [cuda (by default), rocm, cpu, etc.]",
        getter=lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_CACHE_ROOT",
        var_type=str,
        default=lambda: os.path.expanduser(
            os.path.join(get_default_cache_root(), "vllm")
        ),
        doc=(
            "Root directory for vLLM cache files. Defaults to "
            "~/.cache/vllm unless XDG_CACHE_HOME is set."
        ),
        getter=lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CACHE_ROOT",
                os.path.join(get_default_cache_root(), "vllm"),
            )
        ),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_CONFIG_ROOT",
        var_type=str,
        default=lambda: os.path.expanduser(
            os.path.join(get_default_config_root(), "vllm")
        ),
        doc=(
            "Root directory for vLLM config files. Defaults to "
            "~/.config/vllm unless XDG_CONFIG_HOME is set."
        ),
        getter=lambda: os.path.expanduser(
            os.getenv(
                "VLLM_CONFIG_ROOT",
                os.path.join(get_default_config_root(), "vllm"),
            )
        ),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_HOST_IP",
        var_type=str,
        default="",
        doc=(
            "Host IP address for vLLM internal communication when "
            "multiple interfaces exist."
        ),
        getter=lambda: os.getenv("VLLM_HOST_IP", ""),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_PORT",
        var_type=int,
        default=None,
        doc=(
            "Port for vLLM internal RPC and distributed communication. "
            "Raises if URI is passed."
        ),
        getter=get_vllm_port,
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_RPC_BASE_PATH",
        var_type=str,
        default=tempfile.gettempdir,
        doc=(
            "Base path used for IPC when frontend API server "
            "communicates with engine in MP mode."
        ),
        getter=lambda: os.getenv("VLLM_RPC_BASE_PATH", tempfile.gettempdir()),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_USE_MODELSCOPE",
        var_type=bool,
        default=False,
        doc="If true, loads models from ModelScope instead of Hugging Face Hub.",
        getter=lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").strip().lower()
        in ("1", "true"),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_USE_FASTOKENS",
        var_type=bool,
        default=False,
        doc="If true, uses fastokens library for tokenization when available.",
        getter=lambda: os.getenv("VLLM_USE_FASTOKENS", "False").lower()
        in ("true", "1"),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_RINGBUFFER_WARNING_INTERVAL",
        var_type=int,
        default=60,
        doc="Interval in seconds for ringbuffer warnings.",
        getter=lambda: int(os.getenv("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_ENGINE_READY_TIMEOUT_S",
        var_type=int,
        default=600,
        doc="Timeout in seconds to wait for engine to be ready.",
        getter=lambda: int(os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "600")),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT",
        var_type=float,
        default=30.0,
        doc="Timeout in seconds for chat template rendering to prevent CPU hang.",
        getter=lambda: float(os.getenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", "30.0")),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_API_KEY",
        var_type=str,
        default=None,
        doc="API key for API server authorization.",
        getter=lambda: os.getenv("VLLM_API_KEY", None),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_CONFIGURE_LOGGING",
        var_type=bool,
        default=True,
        doc="Whether to configure vLLM logging. Set to 0 to disable.",
        getter=lambda: bool(int(os.getenv("VLLM_CONFIGURE_LOGGING", "1"))),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOGGING_LEVEL",
        var_type=str,
        default="INFO",
        doc="Default logging level for vLLM (e.g. DEBUG, INFO, WARNING, ERROR).",
        getter=lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOGGING_PREFIX",
        var_type=str,
        default="",
        doc="Prefix string prepended to all vLLM log messages.",
        getter=lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOGGING_STREAM",
        var_type=str,
        default="ext://sys.stdout",
        doc="Default logging stream (e.g. ext://sys.stdout, ext://sys.stderr).",
        getter=lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout"),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOGGING_CONFIG_PATH",
        var_type=str,
        default=None,
        doc="Path to a custom logging configuration file.",
        getter=lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOGGING_COLOR",
        var_type=str,
        default="auto",
        doc="Controls colored logging output ('auto', '1', '0').",
        getter=lambda: os.getenv("VLLM_LOGGING_COLOR", "auto"),
    )
)

register_env_var(
    EnvVarDef(
        name="NO_COLOR",
        var_type=bool,
        default=False,
        doc="Standard unix flag for disabling ANSI color codes.",
        getter=lambda: os.getenv("NO_COLOR", "0") != "0",
    )
)

register_env_var(
    EnvVarDef(
        name="FORCE_COLOR",
        var_type=bool,
        default=False,
        doc="De-facto standard flag for forcing ANSI color codes.",
        getter=lambda: os.getenv("FORCE_COLOR", "0") != "0",
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_LOG_STATS_INTERVAL",
        var_type=float,
        default=10.0,
        doc="Interval in seconds between logging engine statistics.",
        getter=lambda: (
            val
            if (val := float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))) > 0.0
            else 10.0
        ),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_USAGE_STATS_SERVER",
        var_type=str,
        default="https://stats.vllm.ai",
        doc="Server URL for vLLM anonymous usage statistics.",
        getter=lambda: os.getenv("VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_NO_USAGE_STATS",
        var_type=bool,
        default=False,
        doc="Disable sending anonymous usage statistics.",
        getter=lambda: (
            os.getenv("VLLM_NO_USAGE_STATS", "0").strip().lower() in ("1", "true")
        ),
    )
)

register_env_var(
    EnvVarDef(
        name="VLLM_DO_NOT_TRACK",
        var_type=bool,
        default=False,
        doc=(
            "Disable sending anonymous usage statistics "
            "(alias for VLLM_NO_USAGE_STATS; checks DO_NOT_TRACK too)."
        ),
        getter=lambda: (
            (
                os.environ.get("VLLM_DO_NOT_TRACK", None)
                or os.environ.get("DO_NOT_TRACK", None)
                or "0"
            )
            == "1"
        ),
    )
)
