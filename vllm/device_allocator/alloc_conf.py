# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for reading and editing PyTorch's allocator configuration.

Torch's allocator config is parsed as a whole: ``parseArgs`` resets every
option that is not explicitly present in the string it is handed. Writing a
single field therefore silently drops the rest of the user's configuration,
for the remainder of the process:

    >>> # PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
    >>> torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    >>> # max_split_size is now SIZE_MAX, not 512 MiB -- and writing
    >>> # "expandable_segments:True" back does not bring it home either.

So anything that wants to toggle one option has to read the current config,
flip just that field, and write the whole string back.
"""

import os

import torch

# Checked in torch's own precedence order: the accelerator-agnostic variable
# wins, then the CUDA one, then the HIP one (which is what ROCm users set).
ALLOC_CONF_ENV_VARS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
)

EXPANDABLE_SEGMENTS = "expandable_segments"


def alloc_conf_from_env(
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> str:
    """The allocator config string the process was started with, if any."""
    env = os.environ if environ is None else environ
    for name in ALLOC_CONF_ENV_VARS:
        conf = env.get(name)
        if conf:
            return conf
    return ""


def live_alloc_conf() -> str | None:
    """The allocator config string currently in effect, or ``None``.

    Read from the live allocator rather than the environment, so that a
    *runtime* write by another component is visible.
    """
    try:
        settings = torch.cuda.memory._snapshot().get("allocator_settings")
        if settings is not None:
            for name in ALLOC_CONF_ENV_VARS:
                conf = settings.get(name)
                if conf is not None:
                    return conf
    except Exception:  # noqa: BLE001
        pass
    return None


def current_alloc_conf() -> str:
    """The live allocator config, falling back to the environment."""
    live = live_alloc_conf()
    return live if live is not None else alloc_conf_from_env()


def conf_flag_enabled(conf: str, key: str) -> bool:
    """Parse ``<key>:<bool>`` out of an allocator config string."""
    for field in conf.split(","):
        name, _, value = field.partition(":")
        if name.strip() == key:
            return value.strip().lower() in ("true", "1")
    return False


def with_conf_flag(conf: str, key: str, enabled: bool) -> str:
    """Return ``conf`` with only ``key`` flipped, every other field verbatim."""
    want = f"{key}:{'True' if enabled else 'False'}"
    fields = [f for f in (f.strip() for f in conf.split(",")) if f]
    for i, field in enumerate(fields):
        if field.partition(":")[0].strip() == key:
            fields[i] = want
            return ",".join(fields)
    fields.append(want)
    return ",".join(fields)


def expandable_segments_enabled_from_env(
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> bool:
    return conf_flag_enabled(alloc_conf_from_env(environ), EXPANDABLE_SEGMENTS)


def expandable_segments_enabled() -> bool | None:
    """Live ``expandable_segments`` state, or ``None`` if it cannot be read.

    ``None`` is deliberately distinct from ``False``: the environment can never
    reflect a runtime write, so a caller verifying a write it just issued must
    not read "cannot tell" as "the write did not take".
    """
    try:
        settings = torch.cuda.memory._snapshot().get("allocator_settings")
        if settings is not None and EXPANDABLE_SEGMENTS in settings:
            return bool(settings[EXPANDABLE_SEGMENTS])
    except Exception:  # noqa: BLE001
        pass
    return None


def set_alloc_conf(conf: str) -> None:
    """Write a complete allocator config string."""
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is not None:
        setter(conf)
        return
    torch.cuda.memory._set_allocator_settings(conf)
