# SPDX-License-Identifier: Apache-2.0
"""Genesis v7.0 wiring framework — the layer that actually binds kernels to vLLM.

Phase 2 delivered kernels as pure-python modules (router_softmax, dequant_buffer,
etc.) with full TDD. Phase 3 (this package) connects them to vLLM at runtime via:

  1. **Text-patch injection** — for in-method code that cannot be monkey-patched
     cleanly (e.g. arg_utils.py control flow with `raise NotImplementedError`).
     The patcher reads the target file, matches anchors, writes the result.
     Idempotent via marker comments. Uses a verification-before-write approach
     borrowed from the v5.14.1 monolith but with:
       - Stronger failure-isolation (one patch's failure does not abort siblings)
       - Anchor drift detection + graceful skip (NOT a hard crash)
       - Immutability: writes once per container filesystem layer, then registry
         flag prevents re-application

  2. **Attribute rebind** — for standalone functions/methods we can swap out
     cleanly (e.g. `vllm.model_executor.layers.fused_moe.softmax`). Uses
     `setattr(module, name, genesis_fn)` with the original stashed in a
     per-patch registry for rollback testing.

  3. **Import-time hooks** — for cases where the target needs to be rewritten
     before first-import (e.g. compile-time Triton kernel parameters). Uses
     `importlib.abc.MetaPathFinder` insertion.

Design goals
------------
- **Idempotent per process**: every wiring step checks "already applied" before
  running. Safe to re-call in worker spawn scenarios.
- **Graceful**: any wiring failure becomes a logged skip, never a crash. vLLM
  continues with upstream behavior for that patch.
- **Verifiable**: each wiring step provides an `assert_applied()` helper that
  can be called post-register to confirm the rebind is live.
- **Reversible** (mostly): for attribute rebinds, we save the original ref and
  can restore it. Text-patches are one-way within a container filesystem.
- **Platform-guard first**: no wiring step runs if `should_apply()` says the
  platform doesn't need it.

Modules
-------
  text_patch.py    — Safe file-text anchor replacement with idempotency
  rebind.py        — Attribute rebinding helper with original-ref registry
  (future: import_hook.py for AST-level rewrite)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

from vllm._genesis.wiring.rebind import AttributeRebinder, WiringRegistry
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatchFailure,
)

__all__ = [
    "AttributeRebinder",
    "WiringRegistry",
    "TextPatcher",
    "TextPatchResult",
    "TextPatchFailure",
]
