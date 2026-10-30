# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import patch


def _patched_load_by_key_path(
    key: str,
    path: str,
    linemap: list[tuple[int, str]] | None = None,
    attrs: dict[str, Any] | None = None,
    *,
    set_sys_modules: bool | None = None,
    **kwargs: Any,
):
    """Load an Inductor module while retaining its source for depyf.

    depyf 0.20.0 predates the ``set_sys_modules`` keyword added to
    ``PyCodeCache.load_by_key_path``. This is the depyf hook with the newer
    Torch arguments forwarded to the original loader.
    """
    from depyf.explain.global_variables import data
    from depyf.explain.utils import (
        get_current_compiled_fn_name,
        write_code_to_file_template,
    )

    dump_src_dir = data["dump_src_dir"]
    unpatched_load_by_key_path = data["unpatched_load_by_key_path"]

    with open(path) as source_file:
        src = source_file.read()

    func_name = get_current_compiled_fn_name()
    new_filepath = write_code_to_file_template(
        src, os.path.join(dump_src_dir, func_name + ".kernel_" + "%s" + ".py")
    )
    return unpatched_load_by_key_path(
        key,
        new_filepath,
        linemap,
        attrs,
        set_sys_modules=set_sys_modules,
        **kwargs,
    )


def _needs_loader_compat(torch_loader: Any, depyf_loader: Any) -> bool:
    torch_parameters = inspect.signature(torch_loader).parameters
    depyf_parameters = inspect.signature(depyf_loader).parameters
    if "set_sys_modules" not in torch_parameters:
        return False
    return "set_sys_modules" not in depyf_parameters and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in depyf_parameters.values()
    )


@contextlib.contextmanager
def prepare_debug(dump_src_dir: str) -> Generator[None, None, None]:
    """Prepare depyf debugging with compatibility for Torch loader changes."""
    import depyf
    from depyf.explain import enable_debugging
    from torch._inductor import codecache

    torch_loader = codecache.PyCodeCache.load_by_key_path
    depyf_loader = enable_debugging.patched_load_by_key_path
    if not _needs_loader_compat(torch_loader, depyf_loader):
        with depyf.prepare_debug(dump_src_dir):
            yield
        return

    with (
        patch.object(
            enable_debugging, "patched_load_by_key_path", _patched_load_by_key_path
        ),
        depyf.prepare_debug(dump_src_dir),
    ):
        yield
