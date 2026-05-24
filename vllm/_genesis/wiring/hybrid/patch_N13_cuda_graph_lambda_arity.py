# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N13 — CUDAGraphWrapper gc.collect/empty_cache lambda arity fix.

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/41235
"[Bugfix][Compile] Fix gc.collect/empty_cache patch arity in CUDAGraphWrapper"
by @roikoren755, OPEN as of 2026-04-29.

================================================================
WHAT IT DOES
================================================================

`CUDAGraphWrapper.__call__` (vllm/compilation/cuda_graph.py:290-296) patches
`gc.collect` and `torch.accelerator.empty_cache` with **0-arg lambdas** to
suppress them during piecewise cudagraph capture:

    stack.enter_context(patch("gc.collect", lambda: None))
    stack.enter_context(patch("torch.accelerator.empty_cache", lambda: None))

But at least `gc.collect(generation)` is sometimes called WITH a positional
argument — specifically from `torch._dynamo.convert_frame._compile` whenever
dynamo recompiles a nested `@torch.compile` callable INSIDE the capture
region. The 0-arg lambda then raises:

    TypeError: <lambda>() takes 0 positional arguments but 1 was given

…and the worker dies.

The fix is a 2-line change: accept any args/kwargs and discard them.

    stack.enter_context(patch("gc.collect", lambda *args, **kwargs: None))
    stack.enter_context(patch("torch.accelerator.empty_cache",
                              lambda *args, **kwargs: None))

================================================================
APPLICABILITY TO GENESIS
================================================================

This bug specifically fires when dynamo RE-compiles a nested @torch.compile
callable inside the cudagraph capture region. Our P67/P67b/P78/P85 family
does exactly this — we have nested compiled kernels (multi-query verify,
spec-decode safety nets) that occasionally trigger dynamo recompiles when
shape changes happen mid-capture.

PR author reports the bug "consistent on GB200 nightly". Sander's planned
RTX PRO 6000 Blackwell upgrade (Q3 2026) sits on the same Blackwell SM
class — this would hit us too.

For our current 2× A5000 stack the bug is intermittent rather than
consistent, but when it fires the worker dies and we lose the engine.
Defensive patch — costs nothing.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1`).
- Pure text-patch, idempotent via marker.
- Drift-aware: when upstream PR #41235 merges, the new lambdas with `*args,
  **kwargs` will already exist. Our marker detects and self-retires.
- Anchor missing → SKIPPED, source stays vanilla. Zero regression risk.
- Worst case: no bug to fix on a particular workload = no-op runtime cost.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Source PR: vllm-project/vllm#41235 by @roikoren755.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN13_cuda_graph_lambda_arity")

GENESIS_PN13_MARKER = (
    "Genesis PN13 CUDAGraphWrapper lambda arity (vllm#41235) v7.62.x"
)


# ─── Sub-patch: replace 2 zero-arg lambdas with var-arg lambdas ─────────

PN13_ANCHOR = (
    "                    stack.enter_context(patch(\"gc.collect\", lambda: None))\n"
    "                    stack.enter_context(\n"
    "                        patch(\"torch.accelerator.empty_cache\", lambda: None)\n"
    "                    )\n"
)

PN13_REPLACEMENT = (
    "                    # [Genesis PN13 vllm#41235 backport] gc.collect is\n"
    "                    # sometimes called with a positional arg (dynamo\n"
    "                    # convert_frame _compile invokes gc.collect(generation)\n"
    "                    # when re-compiling nested @torch.compile callables\n"
    "                    # inside cudagraph capture). 0-arg lambda raises\n"
    "                    # TypeError, worker dies. Accept any args/kwargs.\n"
    "                    stack.enter_context(\n"
    "                        patch(\"gc.collect\", lambda *args, **kwargs: None)\n"
    "                    )\n"
    "                    stack.enter_context(\n"
    "                        patch(\n"
    "                            \"torch.accelerator.empty_cache\",\n"
    "                            lambda *args, **kwargs: None,\n"
    "                        )\n"
    "                    )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("compilation/cuda_graph.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN13 compilation/cuda_graph.py — CUDAGraphWrapper gc.collect/"
            "empty_cache lambda arity fix (vllm#41235)"
        ),
        target_file=str(target),
        marker=GENESIS_PN13_MARKER,
        sub_patches=[
            TextPatch(
                name="pN13_lambda_arity",
                anchor=PN13_ANCHOR,
                replacement=PN13_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN13",
            # If upstream PR #41235 lands, the var-arg lambda string appears
            # in vanilla source and our anchor (zero-arg version) won't match.
            "patch(\"gc.collect\", lambda *args, **kwargs: None)",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN13 — CUDAGraphWrapper lambda arity fix (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN13")
    log_decision("PN13", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN13 applied: CUDAGraphWrapper.__call__ now accepts var-args on "
            "patched gc.collect / empty_cache lambdas (vllm#41235). Defensive "
            "fix — prevents TypeError worker-death class when dynamo recompiles "
            "nested @torch.compile callables inside cudagraph capture region."
        ),
        patch_name=patcher.patch_name,
    )
