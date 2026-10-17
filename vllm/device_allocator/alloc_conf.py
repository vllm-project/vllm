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
from collections.abc import Generator
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

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


_non_expandable_depth = 0


@contextmanager
def non_expandable_allocations(enabled: bool = True) -> Generator[bool, None, None]:
    """Run the body with ``expandable_segments`` off, then restore the config.

    Yields whether the toggle was actually applied. Re-entrant: only the
    outermost context writes to the allocator. The restore is unconditional --
    a stale or unavailable reader costs a log line, never allocator state.
    """
    global _non_expandable_depth
    if not enabled or _non_expandable_depth > 0:
        yield False
        return

    prev_conf = current_alloc_conf()
    live = expandable_segments_enabled()
    was_enabled = (
        live if live is not None else conf_flag_enabled(prev_conf, EXPANDABLE_SEGMENTS)
    )
    if not was_enabled:
        yield False
        return

    _non_expandable_depth += 1
    prev_env = {n: os.environ[n] for n in ALLOC_CONF_ENV_VARS if n in os.environ}
    try:
        try:
            set_alloc_conf(with_conf_flag(prev_conf, EXPANDABLE_SEGMENTS, False))
            for name, value in prev_env.items():
                os.environ[name] = with_conf_flag(value, EXPANDABLE_SEGMENTS, False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not disable expandable_segments (%s); "
                "leaving the allocator alone.",
                exc,
            )
            yield False
            return

        if expandable_segments_enabled():
            logger.warning(
                "Requested non-expandable allocations, but the allocator still "
                "reports expandable_segments=True; allocations made here may "
                "stay VMM-backed and non-IPC-exportable."
            )
            yield False
            return

        logger.info(
            "Disabled expandable_segments so that allocations made here stay "
            "IPC-exportable; the previous allocator config is restored on exit."
        )
        yield True
    finally:
        _non_expandable_depth -= 1
        for name, value in prev_env.items():
            os.environ[name] = value
        try:
            set_alloc_conf(prev_conf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore the allocator config: %s", exc)
