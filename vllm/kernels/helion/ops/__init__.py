# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion kernel implementation.

Importing this package does NOT register any kernels. Runtime code imports the
specific op module it needs, e.g.::

    from vllm.kernels.helion.ops import scaled_mm  # noqa: F401

which triggers that op's ``@register_kernel`` as an import side effect.

Tools that need the full registry (e.g. scripts/autotune_helion_kernels.py)
call ``import_all_ops()`` to force every op module to register.
"""

import importlib
import pkgutil


def import_all_kernels() -> list[str]:
    """Import every kernel submodule so all ``@register_kernel`` decorators run.

    Returns:
        The fully-qualified module names that were imported.
    """
    imported: list[str] = []
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.ispkg:
            continue
        module_name = f"{__name__}.{module_info.name}"
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported
