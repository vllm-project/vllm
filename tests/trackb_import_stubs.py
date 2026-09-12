# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import stubs for running Track-B unit tests against an unbuilt source tree.

Installs lightweight stand-ins for compiled extensions so pure-Python producer
and Triton consumer tests can import shipped modules without a full CUDA build.
Safe no-op when real extensions are already importable.
"""

from __future__ import annotations

import sys
import types


def install_trackb_import_stubs() -> None:
    for name in ("vllm._C", "vllm._C_stable_libtorch", "vllm._version"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except Exception:
            pass
        mod = types.ModuleType(name)
        if name.endswith("_version"):
            mod.__version__ = "0.0.0+trackb"
            mod.__version_tuple__ = (0, 0, 0)
        sys.modules[name] = mod

    try:
        import torch

        class _Ops:
            def __getattr__(self, _name):
                def _f(*_a, **_k):
                    return False

                return _f

        # Only replace when custom ops namespace is missing attributes we need.
        try:
            _ = torch.ops._C.cutlass_scaled_mm_supports_block_fp8  # type: ignore[attr-defined]
        except Exception:
            torch.ops._C = _Ops()  # type: ignore[attr-defined]

        # Re-importing checkout vllm after a site-packages import can re-register
        # opaque types and abort collection. Make registration idempotent.
        _reg = getattr(torch._C, "_register_opaque_type", None)
        if callable(_reg) and not getattr(_reg, "_trackb_idempotent", False):

            def _idempotent_register_opaque_type(name, *args, **kwargs):
                try:
                    return _reg(name, *args, **kwargs)
                except RuntimeError as exc:
                    if "already registered" not in str(exc):
                        raise
                    return None

            _idempotent_register_opaque_type._trackb_idempotent = True  # type: ignore[attr-defined]
            torch._C._register_opaque_type = _idempotent_register_opaque_type  # type: ignore[attr-defined]
    except Exception:
        pass
