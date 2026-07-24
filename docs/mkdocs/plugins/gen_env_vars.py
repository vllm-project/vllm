# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mkdocs-gen-files plugin: generates docs/configuration/env_vars.md.

Walks ``vllm.envs.Settings`` sub-models and emits a Markdown reference
with each environment variable as an H3 heading. Run automatically by
mkdocs at build time when this script is registered under the
``gen-files`` plugin in ``mkdocs.yaml``.

The pure rendering logic lives in :func:`render_env_vars_page` so it can
be unit-tested without needing a mkdocs build.
"""

from __future__ import annotations

import inspect
import io
import types
import typing
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

import vllm.envs as envs

# ----------------------------------------------------------------------------
# Page header (curated by humans; mirrors the original env_vars.md)
# ----------------------------------------------------------------------------

PAGE_HEADER = """\
# Environment Variables

vLLM uses the following environment variables to configure the system. Each
variable is documented under the configuration domain it belongs to.

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for
    vLLM's **internal usage**. It is not the port and ip for the API server.
    If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the
    API server, it will not work.

    All environment variables used by vLLM are prefixed with `VLLM_`.
    **Special care should be taken for Kubernetes users**: please do not name
    the service as `vllm`, otherwise environment variables set by Kubernetes
    might conflict with vLLM's environment variables, because [Kubernetes
    sets environment variables for each service with the capitalized service
    name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

"""


# ----------------------------------------------------------------------------
# Stable display strings for default_factory fields whose runtime default
# would bake in the build machine's home directory.
# ----------------------------------------------------------------------------

_PATH_FACTORY_DISPLAY: dict[str, str] = {
    # field_name -> display string
    "config_root": "~/.config/vllm",
    "cache_root": "~/.cache/vllm",
    "assets_cache": "~/.cache/vllm/assets",
    "xla_cache_path": "~/.cache/vllm/xla_cache",
    "rpc_base_path": "<system tempdir>",
}


# ----------------------------------------------------------------------------
# Type & default formatting
# ----------------------------------------------------------------------------


def format_type(annotation: Any) -> str:
    """Render a type annotation as a compact human-readable string."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if annotation is type(None):
        return "None"
    if origin is None:
        # Plain class
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return repr(annotation)
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(format_type(a) for a in args)
    # list[str], set[str], dict[str, X]
    inner = ", ".join(format_type(a) for a in args)
    name = getattr(origin, "__name__", repr(origin))
    return f"{name}[{inner}]"


def format_default(field_name: str, info: FieldInfo) -> str:
    """Render a field's default value as a compact display string."""
    if field_name in _PATH_FACTORY_DISPLAY:
        return _PATH_FACTORY_DISPLAY[field_name]

    if info.default_factory is not None:
        try:
            value = info.default_factory()  # type: ignore[call-arg]
        except Exception:
            return "<computed at runtime>"
        return repr(value)

    # info.default may be PydanticUndefined for required fields, but vllm.envs
    # makes everything optional, so this branch should always have a default.
    default = info.default
    if default is None:
        return "None"
    return repr(default)


# ----------------------------------------------------------------------------
# Page rendering
# ----------------------------------------------------------------------------


def render_env_vars_page() -> str:
    """Render the entire env_vars.md page as a single markdown string."""
    out = io.StringIO()
    out.write(PAGE_HEADER)

    for sub_attr, sub_field in envs.Settings.model_fields.items():
        annotation = sub_field.annotation
        if not (isinstance(annotation, type) and issubclass(annotation, BaseSettings)):
            continue
        sub_cls: type[BaseSettings] = annotation

        out.write(f"## {sub_cls.__name__}\n\n")

        # Only render a class's own docstring, not one inherited from
        # BaseSettings (whose docstring describes pydantic-settings itself).
        sub_doc = sub_cls.__dict__.get("__doc__")
        if sub_doc:
            out.write(f"{inspect.cleandoc(sub_doc)}\n\n")

        prefix = sub_cls.model_config.get("env_prefix", "") or ""
        for field_name, info in sub_cls.model_fields.items():
            env_name = envs.resolve_env_name(info, field_name, prefix)
            type_str = format_type(info.annotation)
            default_str = format_default(field_name, info)

            out.write(f"### `{env_name}`\n\n")
            out.write(f"- **Type:** `{type_str}`\n")
            out.write(f"- **Default:** `{default_str}`\n\n")

            description = info.description
            if description:
                out.write(f"{description}\n\n")

    return out.getvalue()


# ----------------------------------------------------------------------------
# mkdocs-gen-files entry point
# ----------------------------------------------------------------------------

# Only call ``mkdocs_gen_files.open()`` when we're running inside an active
# mkdocs build. Importing this module from pytest (or any plain Python
# context) must remain side-effect-free, otherwise mkdocs_gen_files triggers
# a full ``mkdocs.yaml`` load, which executes unrelated hooks such as
# ``generate_argparse.py``.

try:
    import mkdocs_gen_files
    from mkdocs_gen_files.editor import FilesEditor
except ImportError:
    pass
else:
    if FilesEditor._current is not None:
        with mkdocs_gen_files.open("configuration/env_vars.md", "w") as f:
            f.write(render_env_vars_page())
