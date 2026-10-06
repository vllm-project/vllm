# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-enable vLLM's custom all-reduce on gfx11, capped to single-session sizes.

`custom_all_reduce.py` disables this collective on **all** of gfx11 because it is
a pull (every rank reads its peers' buffers) and a desktop root complex forwards
peer writes but returns zeros for peer reads.  That is true of the local box,
whose P2P only exists because 8086:a700 was added to the kernel P2PDMA
whitelist; it is NOT true of the ainodes, whose EPYC exposes legitimate P2P.

Measured on ainode2 (world 4, fp16, eager, `checks/allreduce_bench3.py`), with
each rank sending DIFFERENT data and the reference summed on the host:

    bytes    PYNCCL   CUSTOM   err(CUSTOM)
    30 KB     58.7     39.0     2.4e-04   <- 1 session, MTP k=2 (M=3)
    41 KB     52-74    42-46    4.6e-04   <- 1 session, MTP k=3 (M=4)
    61 KB     53.6     59.1     2.7e-04   <- 2 sessions: PYNCCL already wins
   123 KB     68.1     96.9     2.6e-04
   184 KB     67.1    132.5     2.4e-04

The error is fp16 rounding of a 4-way sum, i.e. it computes the real sum and not
its own buffer four times -- which is the failure mode the guard exists for, and
which this test can tell apart.

So: the win is real but it is **only** in the single-session regime, and above
~48 KB this collective LOSES to PYNCCL. Hence the cap: `should_custom_ar`
admits a message only while `size < max_size`, so 48 KB sends M<=4 through the
custom kernel and leaves everything else on PYNCCL.

Graph capture is already handled upstream in this fork:
`RocmPlatform.use_custom_allreduce_graph_registration()` returns False on gfx11,
so captured all-reduces go through the buffer registered at init instead of
registering the graph's own.

⛔ Do NOT enable this on the local box: there the collective returns garbage
(see rdna3_p2p/13_CAP_CUSTOM_ALLREDUCE.md). It is gated behind
``VLLM_RDNA3_CUSTOM_AR_CAP`` (bytes) on purpose: no flag, no change.
"""

import os
import sys

_TARGET = "vllm.distributed.device_communicators.cuda_communicator"


class _FakeArch:
    """Hide gfx11 from the arch guard, only while the constructor runs."""

    def __enter__(self):
        import torch

        self._torch = torch
        self._orig = torch.cuda.get_device_properties

        def shim(dev=0):
            props = self._orig(dev)

            class _P:
                def __getattr__(self, k):
                    return getattr(props, k)

            p = _P()
            object.__setattr__(p, "gcnArchName", "gfx942")
            return p

        torch.cuda.get_device_properties = shim
        return self

    def __exit__(self, *a):
        self._torch.cuda.get_device_properties = self._orig


def apply() -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    if getattr(CustomAllreduce, "_rdna3_forced", False):
        return
    cap = int(os.environ.get("VLLM_RDNA3_CUSTOM_AR_CAP", "0"))
    if cap <= 0:
        return

    original = CustomAllreduce.__init__

    def patched(self, group, device, *args, **kwargs):
        kwargs["max_size"] = cap
        with _FakeArch():
            original(self, group, device, *args, **kwargs)
        if not self.disabled:
            print(f"[custom-ar] activado en gfx11 con tope {cap} B "
                  f"(fully_connected={getattr(self, 'fully_connected', '?')})")
        else:
            print("[custom-ar] sigue deshabilitado por otra comprobacion",
                  file=sys.stderr)

    CustomAllreduce.__init__ = patched
    CustomAllreduce._rdna3_forced = True


def apply_when_imported() -> None:
    if _TARGET in sys.modules:
        apply()
        return

    import importlib.abc
    import importlib.machinery

    class _PatchAfterLoad(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(module):
                inner(module)
                try:
                    apply()
                except Exception as e:  # noqa: BLE001
                    print("[custom-ar] no aplicado:", e, file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterLoad())


if int(os.environ.get("VLLM_RDNA3_CUSTOM_AR_CAP", "0")) > 0:
    apply_when_imported()
