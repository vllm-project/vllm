# SPDX-License-Identifier: Apache-2.0
"""Compat re-export shim — `vllm._genesis.compat.gpu_profile`.

Genesis is moving detectors into the unified `compat/` package as part
of the v8.0 reorganization. To avoid breaking any caller (including
patches that import the legacy path), this shim re-exports every
public symbol from the legacy `vllm._genesis.gpu_profile` module.

Both import paths work simultaneously during the migration window:

    # Legacy (still works)
    from vllm._genesis.gpu_profile import detect_gpu_class
    # New (preferred for new code)
    from vllm._genesis.compat.gpu_profile import detect_gpu_class

When the migration is complete, the legacy module will become a thin
shim pointing at this one (so the source of truth lives under compat/).

Author: Sandermage(Sander) Barzov Aleksandr.
"""
from __future__ import annotations

# Forward EVERYTHING the legacy module exposes — including private helpers
# and module-level constants — so wildcard imports / hasattr checks keep
# working. We use `from ... import *` then re-export via __all__ if the
# legacy module has it.
from vllm._genesis.gpu_profile import *  # noqa: F401, F403
from vllm._genesis import gpu_profile as _legacy

# If the legacy module declared __all__, mirror it. Otherwise, copy
# every public attribute.
__all__ = list(getattr(_legacy, "__all__", [
    name for name in dir(_legacy) if not name.startswith("_")
]))

# Also forward the module itself for any `import gpu_profile as gp` style
# imports already in the codebase.
for _name in __all__:
    globals()[_name] = getattr(_legacy, _name)
