# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Declarative definitions for vLLM environment variables.

Each :class:`EnvVarDef` is the single source of truth for a variable's name,
type, default and documentation. Plain ``str``/``int``/``float``/``bool``
variables are parsed by :meth:`EnvVarDef.resolve` from ``var_type`` and
``default``. A custom ``getter`` is only needed for non-standard parsing.
"""

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def get_vllm_port() -> int | None:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.

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
                "This may be caused by a Kubernetes service discovery issue,"
                "check the warning in: https://docs.vllm.ai/en/latest/configuration/env_vars.html"
            ) from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err


@dataclass(frozen=True)
class EnvVarDef:
    """Definition of a vLLM environment variable.

    Attributes:
        name: Environment variable name.
        var_type: One of ``str``, ``int``, ``float`` or ``bool``. Booleans are
            parsed as ``bool(int(value))``.
        default: Value (or zero-arg callable producing it) used when unset.
        doc: One-line description.
        getter: Optional override for variables that need custom parsing.

    """

    name: str
    var_type: type
    default: Any
    doc: str
    getter: Callable[[], Any] | None = None

    def resolve_default(self) -> Any:
        return self.default() if callable(self.default) else self.default

    def resolve(self) -> Any:
        if self.getter is not None:
            return self.getter()
        raw = os.environ.get(self.name)
        if raw is None:
            return self.resolve_default()
        if self.var_type is bool:
            return bool(int(raw))
        return self.var_type(raw)

    def is_set(self) -> bool:
        return self.name in os.environ


ENV_VAR_REGISTRY: dict[str, EnvVarDef] = {}


def _register(*defs: EnvVarDef) -> None:
    for env_def in defs:
        if env_def.name in ENV_VAR_REGISTRY:
            raise ValueError(f"Duplicate environment variable: {env_def.name}")
        ENV_VAR_REGISTRY[env_def.name] = env_def


def get_env_var_def(name: str) -> EnvVarDef | None:
    """Return the definition for an environment variable, if registered."""
    return ENV_VAR_REGISTRY.get(name)


def list_env_vars() -> list[EnvVarDef]:
    """Return all registered environment variable definitions."""
    return list(ENV_VAR_REGISTRY.values())


def format_env_vars_table() -> str:
    """Format registered environment variables as a plain-text table."""
    lines = [f"{'Variable':<36} {'Type':<8} {'Default':<28} Description"]
    lines.append("-" * 110)
    for var in sorted(ENV_VAR_REGISTRY.values(), key=lambda v: v.name):
        default = str(var.resolve_default())
        if len(default) > 25:
            default = default[:22] + "..."
        summary = var.doc.split("\n")[0]
        lines.append(
            f"{var.name:<36} {var.var_type.__name__:<8} {default:<28} {summary}"
        )
    return "\n".join(lines)


def _default_cache_root() -> str:
    return os.path.join(get_default_cache_root(), "vllm")


def _default_config_root() -> str:
    return os.path.join(get_default_config_root(), "vllm")


def _get_log_stats_interval() -> float:
    val = float(os.getenv("VLLM_LOG_STATS_INTERVAL", "10."))
    return val if val > 0.0 else 10.0


def _get_do_not_track() -> bool:
    return (
        os.environ.get("VLLM_DO_NOT_TRACK", None)
        or os.environ.get("DO_NOT_TRACK", None)
        or "0"
    ) == "1"


_register(
    EnvVarDef(
        "VLLM_TARGET_DEVICE",
        str,
        "cuda",
        "Target device of vLLM: cuda (default), rocm or cpu.",
        getter=lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda").lower(),
    ),
    EnvVarDef(
        "VLLM_CACHE_ROOT",
        str,
        lambda: os.path.expanduser(_default_cache_root()),
        "Root directory for vLLM cache files (honors XDG_CACHE_HOME).",
        getter=lambda: os.path.expanduser(
            os.getenv("VLLM_CACHE_ROOT", _default_cache_root())
        ),
    ),
    EnvVarDef(
        "VLLM_CONFIG_ROOT",
        str,
        lambda: os.path.expanduser(_default_config_root()),
        "Root directory for vLLM config files (honors XDG_CONFIG_HOME).",
        getter=lambda: os.path.expanduser(
            os.getenv("VLLM_CONFIG_ROOT", _default_config_root())
        ),
    ),
    EnvVarDef(
        "VLLM_HOST_IP",
        str,
        "",
        "IP address of this node for distributed communication.",
    ),
    EnvVarDef(
        "VLLM_PORT",
        int,
        None,
        "Base port for distributed communication; raises if given a URI.",
        getter=get_vllm_port,
    ),
    EnvVarDef(
        "VLLM_RPC_BASE_PATH",
        str,
        tempfile.gettempdir,
        "Base path for IPC between the API server and engine processes.",
    ),
    EnvVarDef(
        "VLLM_USE_MODELSCOPE",
        bool,
        False,
        "Load models from ModelScope instead of Hugging Face Hub.",
        getter=lambda: (
            os.environ.get("VLLM_USE_MODELSCOPE", "False").strip().lower()
            in ("1", "true")
        ),
    ),
    EnvVarDef(
        "VLLM_USE_FASTOKENS",
        bool,
        False,
        "Replace the HF fast tokenizer Rust backend with `fastokens`.",
    ),
    EnvVarDef(
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        int,
        60,
        "Seconds between warnings when the ring buffer is full.",
    ),
    EnvVarDef(
        "VLLM_ENGINE_READY_TIMEOUT_S",
        int,
        600,
        "Seconds to wait for engine cores to become ready at startup.",
    ),
    EnvVarDef(
        "VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT",
        float,
        30.0,
        "Max seconds for one chat template render; 0 disables the timeout.",
    ),
    EnvVarDef("VLLM_API_KEY", str, None, "API key for the vLLM API server."),
    EnvVarDef(
        "VLLM_CONFIGURE_LOGGING",
        bool,
        True,
        "Whether vLLM configures logging; 0 disables it.",
    ),
    EnvVarDef(
        "VLLM_LOGGING_CONFIG_PATH",
        str,
        None,
        "Path to a custom logging configuration file.",
    ),
    EnvVarDef(
        "VLLM_LOGGING_LEVEL",
        str,
        "INFO",
        "Default logging level.",
        getter=lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),
    ),
    EnvVarDef(
        "VLLM_LOGGING_STREAM",
        str,
        "ext://sys.stdout",
        "Default logging stream.",
    ),
    EnvVarDef(
        "VLLM_LOGGING_PREFIX",
        str,
        "",
        "Prefix prepended to all log messages.",
    ),
    EnvVarDef(
        "VLLM_LOGGING_COLOR",
        str,
        "auto",
        "Colored logging: 'auto' (terminal only), '1' (always), '0' (never).",
    ),
    EnvVarDef(
        "NO_COLOR",
        bool,
        False,
        "Standard unix flag for disabling ANSI color codes.",
        getter=lambda: os.getenv("NO_COLOR", "0") != "0",
    ),
    EnvVarDef(
        "FORCE_COLOR",
        bool,
        False,
        "De-facto standard flag for forcing ANSI color codes.",
        getter=lambda: os.getenv("FORCE_COLOR", "0") != "0",
    ),
    EnvVarDef(
        "VLLM_LOG_STATS_INTERVAL",
        float,
        10.0,
        "Seconds between stats log lines; non-positive values fall back to 10.",
        getter=_get_log_stats_interval,
    ),
    EnvVarDef(
        "VLLM_USAGE_STATS_SERVER",
        str,
        "https://stats.vllm.ai",
        "Server that receives anonymous usage stats.",
    ),
    EnvVarDef(
        "VLLM_NO_USAGE_STATS",
        bool,
        False,
        "Set to 1 to disable anonymous usage stats.",
        getter=lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    ),
    EnvVarDef(
        "VLLM_DO_NOT_TRACK",
        bool,
        False,
        "Set to 1 to disable usage stats; DO_NOT_TRACK is also honored.",
        getter=_get_do_not_track,
    ),
)
