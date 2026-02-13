# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-import all Helion op modules to trigger kernel registration."""

import importlib
import pkgutil

# Automatically import all submodules so that @register_kernel
# decorators execute and register ops with torch.ops.vllm_helion.
for _module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_info.name}")
