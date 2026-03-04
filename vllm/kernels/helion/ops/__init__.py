# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-import all Helion op modules to trigger kernel registration."""

import importlib
import pkgutil

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if has_helion() and current_platform.is_cuda_alike():
    # Phase 1: Import all submodules so that @register_kernel
    # decorators execute and register ops with the global registry.
    for _module_info in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{_module_info.name}")

    # Phase 2: Resolve best version for each kernel, export as module-level names.
    from vllm.kernels.helion.register import resolve_all_kernels as _resolve

    for _name, _wrapper in _resolve().items():
        globals()[_name] = _wrapper
