# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Generator
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALLOC_CONF_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def expandable_segments_enabled_from_env(
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> bool:
    env = os.environ if environ is None else environ
    for name in _ALLOC_CONF_ENV_VARS:
        conf = env.get(name)
        if not conf:
            continue
        for field in conf.split(","):
            key, _, value = field.partition(":")
            if key.strip() == "expandable_segments":
                return value.strip().lower() in ("true", "1")
    return False


def expandable_segments_enabled() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = torch.cuda.memory._snapshot().get("allocator_settings")
        if settings is not None and "expandable_segments" in settings:
            return bool(settings["expandable_segments"])
    except Exception:  # noqa: BLE001
        pass
    return expandable_segments_enabled_from_env()


def _set_expandable_segments(enabled: bool) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.cuda.memory._set_allocator_settings(
            f"expandable_segments:{'True' if enabled else 'False'}"
        )


_depth = 0


@contextmanager
def non_expandable_allocations(enabled: bool = True) -> Generator[bool, None, None]:
    global _depth
    if not enabled or _depth > 0 or not expandable_segments_enabled():
        yield False
        return

    _set_expandable_segments(False)
    if expandable_segments_enabled():
        logger.warning(
            "Requested a non-expandable CUDA graph pool, but the allocator "
            "still reports expandable_segments=True; leaving it alone."
        )
        yield False
        return

    _depth += 1
    logger.info(
        "Disabled expandable_segments for CUDA graph capture so that "
        "graph-pool activations stay IPC-exportable (custom all-reduce can "
        "then skip its staging copy-in)."
    )
    try:
        yield True
    finally:
        _depth -= 1
        _set_expandable_segments(True)
        logger.info("Restored expandable_segments after CUDA graph capture.")
