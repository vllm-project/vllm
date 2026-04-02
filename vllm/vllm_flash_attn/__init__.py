# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.abc
import os
import sys

# When developing with VLLM_FLASH_ATTN_SRC_DIR, the cute/ directory is
# symlinked rather than copied, so its source files contain unmodified
# `flash_attn.cute.*` imports. This finder redirects those imports to
# `vllm.vllm_flash_attn.cute.*`, ensuring a single set of module objects
# (important for JIT cache correctness).
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.islink(_cute_dir):

    class _CuteImportRedirector(importlib.abc.MetaPathFinder):
        _PREFIX = "flash_attn.cute"
        _TARGET = "vllm.vllm_flash_attn.cute"

        def find_module(self, fullname, path=None):
            if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            target = self._TARGET + fullname[len(self._PREFIX) :]
            mod = importlib.import_module(target)
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.insert(0, _CuteImportRedirector())

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
