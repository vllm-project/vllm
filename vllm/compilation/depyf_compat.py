# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep depyf usable against Inductor versions newer than its last release.

``depyf.prepare_debug`` replaces three torch functions with hooks of its own.
Two of them tolerate new arguments, but ``load_by_key_path`` has its signature
frozen at ``(key, path, linemap, attrs)``, and Inductor now also passes
``set_sys_modules`` (pytorch#184285). Every compilation that reaches an
autotune code block while depyf is active therefore dies with a ``TypeError``,
which surfaces as ``InductorError: Failed to run autotuning code block``.

depyf has hit this before -- ``linemap`` and ``attrs`` broke the same hook in
thuml/depyf#80 -- but 0.20.0 is both the pinned version and the last release
upstream published, so a version bump cannot resolve it. Patch the hook here
instead, following what depyf itself does for ``lazy_format_graph_code``:
accept ``**kwargs`` and forward them.

Route every depyf entry point through ``prepare_debug`` below so the patch is
never missed.
"""

import inspect
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_patched = False


def _hook_accepts_inductor_keywords(hook: Callable[..., Any]) -> bool:
    """Whether depyf's hook already takes everything Inductor passes it.

    Compares against the live ``PyCodeCache.load_by_key_path``, so this only
    means anything before depyf swaps that out for the hook.
    """
    import torch._inductor.codecache as codecache

    accepted = inspect.signature(hook).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return True
    expected = inspect.signature(codecache.PyCodeCache.load_by_key_path).parameters
    return expected.keys() <= accepted.keys()


def _patch_load_by_key_path() -> None:
    """Forward unrecognised keywords through depyf's ``load_by_key_path`` hook.

    Mirrors depyf's own implementation -- dump the generated source alongside
    the rest of the debug output, then defer to the function depyf displaced --
    and adds ``**kwargs`` so the call survives arguments depyf never knew about.
    """
    import depyf.explain.enable_debugging as enable_debugging
    from depyf.explain.global_variables import data
    from depyf.explain.utils import (
        get_current_compiled_fn_name,
        write_code_to_file_template,
    )

    def load_by_key_path(key, path, linemap=None, attrs=None, **kwargs):
        with open(path) as f:
            src = f.read()
        func_name = get_current_compiled_fn_name()
        dumped_path = write_code_to_file_template(
            src, os.path.join(data["dump_src_dir"], func_name + ".kernel_%s.py")
        )
        unpatched = data["unpatched_load_by_key_path"]
        return unpatched(key, dumped_path, linemap, attrs, **kwargs)

    enable_debugging.patched_load_by_key_path = load_by_key_path


def prepare_debug(dump_src_dir: str) -> AbstractContextManager[None]:
    """``depyf.prepare_debug`` with the compatibility patches applied."""
    global _patched
    import depyf
    import depyf.explain.enable_debugging as enable_debugging

    if not _patched and not _hook_accepts_inductor_keywords(
        enable_debugging.patched_load_by_key_path
    ):
        logger.debug("Patching depyf's load_by_key_path hook for this Inductor")
        _patch_load_by_key_path()
        _patched = True

    return depyf.prepare_debug(dump_src_dir)
