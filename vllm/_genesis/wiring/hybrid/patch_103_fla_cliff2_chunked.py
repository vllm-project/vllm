# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 103 — FLA Cliff 2 chunked fwd_h+fwd_o orchestrator.

Backport-equivalent for the upstream `fla` package issue raised by
@noonghunna in https://github.com/noonghunna/qwen36-27b-single-3090/issues/1
("Cliff 2 — DeltaNet GDN forward at 50-60K single-prompt OOM"). Addressed
by Sander 2026-04-28 per direct ask.

================================================================
v7.69 INSTALL MODEL — text-patch + setattr (survives `exec vllm serve`)
================================================================

The v7.62.20 design installed only via `setattr(chunk_mod, ...)` from
`apply()`. That works when `apply_all()` runs in the same process as
`vllm serve` (e.g. via the genesis_vllm_plugin entry point). But it
DIES on the entrypoint pattern:

    python3 -m vllm._genesis.patches.apply_all   # apply runs in shell process
    exec vllm serve "$@"                          # exec REPLACES the process

`exec` discards the Python state. The setattr is gone. Workers spawn
fresh, import the original `chunk_gated_delta_rule_fwd`, hit Cliff 2.

Cross-rig finding: noonghunna club-3090#19 (2026-05-02) — `[Genesis]
applied: P103` log appeared in entrypoint, but Cliff 2 still fired
because the wrap was lost. Sub-finding: P103's "rebound at 0 caller
sites" log is misleading — internal callers in chunk.py resolve via
chunk_mod.__dict__ (which our setattr DID update), so 0 external
callers is normal. The real problem was the process replacement.

v7.69 fix: text-patch `chunk.py` to APPEND a self-install hook at end
of file. The hook checks `GENESIS_ENABLE_P103=1` at module-import time
and calls `_genesis_p103_install_at_import(globals())` which monkey-
patches `chunk_gated_delta_rule_fwd` in the just-loaded module dict.

Why this survives:
- Text-patch modifies the file on disk (under vllm install root).
- Workers spawn → import chunk.py → run our appended block → install
  wrapper → ChunkGatedDeltaRuleFunction.forward's name lookup hits
  the wrapper.
- Setattr-based apply() is preserved as defense-in-depth (it still
  helps the current process and any module that captured a stale
  reference via `from .chunk import chunk_gated_delta_rule_fwd`).

================================================================
PROBLEM
================================================================

`vllm/model_executor/layers/fla/ops/chunk_delta_h.py:301` defines
`chunk_gated_delta_rule_fwd_h` which allocates a recurrent hidden-state
tensor:

    h = k.new_empty(B, NT, H, V, K)

For Qwen3.6 GDN (H=48 per rank under TP=2 H=24, K=V=128, fp16) at T=64K,
that single allocation is **805 MiB**. On a 24 GB single-GPU (3090/4090/
A6000) running the model with `--gpu-memory-utilization 0.93`, this pushes
prefill OOM somewhere between 50-60K context length.

The `h` tensor is consumed by `chunk_fwd_o` (defined in `chunk_o.py`)
which indexes per chunk via Triton block-ptr — meaning `h` MUST be fully
materialized for the consumer kernel. So Option A (smaller chunk_size)
saves nothing (`B*NT*H*V*K` invariant under BT) and Option B (Triton
stream-accumulation) requires kernel rewrite (out of scope for text-patch).

================================================================
SOLUTION (Option C+ — chained fwd_h + fwd_o per sub-prompt)
================================================================

Wrap the high-level orchestrator
`vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule_fwd`
which is the function that calls BOTH `fwd_h` and `fwd_o` adjacent
(chunk.py:60 + chunk.py:64). For prompts with T > MAX_T, split along the
T dim, call fwd_h on each sub-T slice (small `h_sub`), feed straight into
fwd_o to produce `o_sub`, drop `h_sub` (no need to retain), chain
`final_state` between sub-calls. Concat all `o_sub` along T → final `o`.

This NEVER materializes the full `(B, NT, H, V, K)` tensor — only one
sub-segment's worth at a time. For sub_T=16K vs T=64K: peak `h` allocation
shrinks **4×** (805 → 200 MiB on Qwen3.6-27B per-rank).

The pre-`fwd_h` setup (cumsum, kkt, solve_tril, recompute_w_u) runs ONCE
on full inputs because those don't allocate the large h tensor.

================================================================
WHO BENEFITS
================================================================

- **single-GPU users (3090/4090/A6000)** running Qwen3.6-27B/35B in-prompt
  >50K — they were the original Cliff 2 reporters
- **our TP=2 setup** also benefits (~400 MiB headroom per rank at 128K
  context) but we don't currently OOM at any prompt length we run

================================================================
SAFETY MODEL
================================================================

- Default OFF; opt-in via `GENESIS_ENABLE_P103=1`
- Sub-T threshold via `GENESIS_FLA_FWD_H_MAX_T` (default 16384)
- Path falls back to original when:
  - cu_seqlens is not None (variable-length batches don't trigger Cliff 2)
  - T <= MAX_T (no benefit from splitting)
  - SUPPRESS_LEVEL >= 3 (caller wants raw h tensor — incompatible with chunking)
- Module-level monkey-patch on `chunk.py::chunk_gated_delta_rule_fwd`
  + rebind on caller in same module (chunk.py:110)
- KDA path (kda.py:1205 calls fwd_h directly) NOT covered — KDA is a
  separate model class, Qwen3.6 GDN uses the chunk.py orchestrator path
- Numerical correctness: chained final_state propagation preserves the
  recurrent state. Tested via tiny-tensor unit test (synthetic dims).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original Cliff 2 report: noonghunna (qwen36-27b-single-3090#1).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

log = logging.getLogger("genesis.wiring.p103_fla_cliff2_chunked")

_GENESIS_P103_MARKER_ATTR = "_genesis_p103_chunked_wrap"

_TARGET_MODULE = "vllm.model_executor.layers.fla.ops.chunk"
_FN_NAME = "chunk_gated_delta_rule_fwd"


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    if os.environ.get("GENESIS_ENABLE_P103", "").strip().lower() not in (
        "1", "true", "yes", "on"
    ):
        return False
    return True


def _make_chunked_wrapper(
    original_fwd: Any,
    chunk_local_cumsum: Any,
    chunk_scaled_dot_kkt_fwd: Any,
    solve_tril: Any,
    recompute_w_u_fwd: Any,
    chunk_gated_delta_rule_fwd_h: Any,
    chunk_fwd_o_callable: Any,
    fla_chunk_size: int,
    suppress_level: int,
):
    """Build the wrapped orchestrator.

    All dependencies passed by closure to avoid module-scope imports
    inside the hot path.
    """
    import torch

    _MAX_T = max(int(os.environ.get("GENESIS_FLA_FWD_H_MAX_T", "16384")), fla_chunk_size)
    # Round MAX_T down to multiple of FLA_CHUNK_SIZE so per-chunk slicing
    # aligns with the kernel's chunk_size = 64.
    _MAX_T = (_MAX_T // fla_chunk_size) * fla_chunk_size

    # v7.62.20b — minimum-overhead fall-through. Decode (T=1) and
    # short-prefill (T <= MAX_T) batches are 99%+ of calls; that path
    # MUST be near-free. We use *args/**kwargs passthrough so the
    # interpreter handles arg forwarding in C, not Python. The cu_seqlens
    # check is positional/kwarg-aware.
    _SUPPRESS_GE_3 = suppress_level >= 3

    if _SUPPRESS_GE_3:
        # SUPPRESS_LEVEL >= 3: caller wants the raw h tensor, which our
        # chunked path doesn't produce. Return identity wrapper —
        # marker is set so is_applied() reports True, but no behavior change.
        def chunked_fwd(*args, **kwargs):
            return original_fwd(*args, **kwargs)
        chunked_fwd.__name__ = "chunk_gated_delta_rule_fwd"
        chunked_fwd.__doc__ = (
            "[Genesis P103] identity wrapper "
            f"(SUPPRESS_LEVEL={suppress_level} >=3, chunked path bypassed)"
        )
        setattr(chunked_fwd, _GENESIS_P103_MARKER_ATTR, True)
        return chunked_fwd

    def chunked_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
    ):
        # Hot-path: T <= MAX_T (always true for decode T=1, and most prefills)
        # OR cu_seqlens indicates a true multi-seq batch.
        #
        # [Issue #18 fix v7.71] Single-seq prefill is the dominant path on
        # vLLM serving — cu_seqlens is virtually always non-None there
        # (varlen attention API marker). For B=1 the shape is `[2]` =
        # `[0, T]`, semantically equivalent to dense B=1. The original
        # gate over-rejected this case, preventing chunking from EVER
        # engaging on real serving (442/442 invocations bypassed per
        # noonghunna diagnostic on 60K Cliff 2 probe). Detect single-seq
        # cu_seqlens and let it fall through to the chunked path.
        _single_seq_cu = (
            cu_seqlens is not None
            and hasattr(cu_seqlens, "shape")
            and cu_seqlens.shape == (2,)
        )
        if (cu_seqlens is not None and not _single_seq_cu) or q.shape[1] <= _MAX_T:
            return original_fwd(
                q, k, v, g, beta, scale,
                initial_state, output_final_state,
                cu_seqlens, chunk_indices, chunk_offsets,
            )

        # Step 1: full-input setup. These allocations are small — they
        # don't materialize the (B, NT, H, V, K) tensor that triggers Cliff 2.
        g_full = chunk_local_cumsum(
            g, chunk_size=fla_chunk_size, cu_seqlens=None, chunk_indices=None,
        )
        A = chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g=g_full,
            cu_seqlens=None, chunk_indices=None,
            output_dtype=torch.float32,
        )
        A = solve_tril(
            A=A, cu_seqlens=None, chunk_indices=None, output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k, v=v, beta=beta, A=A, g_cumsum=g_full,
            cu_seqlens=None, chunk_indices=None,
        )

        # Step 2: chained per-sub-prompt fwd_h + fwd_o. Never materialize
        # the full h. Chain final_state between sub-calls.
        #
        # T = sequence-length dim (audit fix 2026-05-02 — was undefined,
        # would NameError on every Cliff 2 trigger; bug present since
        # P103 v7.62.20 ship). The hot-path guard at line 166 already
        # uses `q.shape[1]` to decide whether to enter the chunked
        # branch, so by the time we're here we know `q.shape[1] > _MAX_T`.
        T = q.shape[1]
        o_segments = []
        state = initial_state

        for start in range(0, T, _MAX_T):
            end = min(start + _MAX_T, T)

            # Slice all per-T tensors
            q_sub = q[:, start:end]
            k_sub = k[:, start:end]
            w_sub = w[:, start:end]
            u_sub = u[:, start:end]
            g_sub = g_full[:, start:end]

            # fwd_h on sub: small h_sub, plus state for chaining
            h_sub, v_new_sub, state_next = chunk_gated_delta_rule_fwd_h(
                k=k_sub, w=w_sub, u=u_sub, g=g_sub,
                initial_state=state,
                # Always materialize state for chaining; only the last
                # iteration's state is exposed back to caller.
                output_final_state=True,
                cu_seqlens=None,
                chunk_indices=None,
                chunk_offsets=None,
            )

            # fwd_o consumes h_sub → o_sub. After this, h_sub falls out
            # of scope (next loop iteration overwrites the local), so
            # the (B, NT_sub, H, V, K) allocation is freed before the
            # next sub-call's allocation. Peak transient is one slab,
            # not the full concatenation.
            o_sub = chunk_fwd_o_callable(
                q=q_sub, k=k_sub, v=v_new_sub, h=h_sub, g=g_sub,
                scale=scale,
                cu_seqlens=None, chunk_indices=None,
            )
            o_segments.append(o_sub)

            # Drop references — explicit hint to allocator
            del h_sub, v_new_sub
            state = state_next

        o = torch.cat(o_segments, dim=1)
        final_state = state if output_final_state else None

        # Match the original return signature. SUPPRESS_LEVEL was
        # checked above; we never reach here for >= 3.
        return g_full, o, A, final_state, None, None, None

    chunked_fwd.__name__ = "chunk_gated_delta_rule_fwd"
    chunked_fwd.__doc__ = (
        f"[Genesis P103] chunked fwd_h+fwd_o wrapper "
        f"(MAX_T={_MAX_T}, FLA_CHUNK_SIZE={fla_chunk_size})"
    )
    setattr(chunked_fwd, _GENESIS_P103_MARKER_ATTR, True)
    return chunked_fwd


def _genesis_p103_install_at_import(module_globals: Any) -> bool:
    """Install the chunked-fwd wrapper into a chunk.py module's globals.

    Called from the text-patched bottom of chunk.py at MODULE-IMPORT
    TIME. Survives `exec vllm serve` and worker spawn because every
    fresh import of chunk.py runs the appended block which calls this
    helper.

    Args:
      module_globals: chunk.py's `globals()` dict (passed by the text-
        patched call site). Must contain `chunk_gated_delta_rule_fwd`
        and the closure deps below.

    Returns:
      True if wrapper installed (or was already installed), False if
      installation skipped (env not set, deps missing, etc). Never
      raises — failure paths log debug and return False so the
      text-patched chunk.py keeps importing cleanly.
    """
    try:
        if os.environ.get("GENESIS_ENABLE_P103", "").strip().lower() not in (
            "1", "true", "yes", "on"
        ):
            return False

        original = module_globals.get(_FN_NAME)
        if original is None:
            return False
        if getattr(original, _GENESIS_P103_MARKER_ATTR, False):
            # Already wrapped — idempotent.
            return True

        # Resolve closure deps from the module dict directly
        try:
            wrapper = _make_chunked_wrapper(
                original_fwd=original,
                chunk_local_cumsum=module_globals["chunk_local_cumsum"],
                chunk_scaled_dot_kkt_fwd=module_globals[
                    "chunk_scaled_dot_kkt_fwd"
                ],
                solve_tril=module_globals["solve_tril"],
                recompute_w_u_fwd=module_globals["recompute_w_u_fwd"],
                chunk_gated_delta_rule_fwd_h=module_globals[
                    "chunk_gated_delta_rule_fwd_h"
                ],
                chunk_fwd_o_callable=module_globals["chunk_fwd_o"],
                fla_chunk_size=module_globals.get("FLA_CHUNK_SIZE", 64),
                suppress_level=module_globals.get("SUPPRESS_LEVEL", 0),
            )
        except KeyError as e:
            log.debug(
                "[Genesis P103 self-install] interface drift: "
                "missing symbol in chunk.py module globals — %s", e,
            )
            return False

        module_globals[_FN_NAME] = wrapper
        log.info(
            "[Genesis P103 self-install] wrapper installed in chunk.py "
            "at module-import time (survives `exec vllm serve` + worker "
            "spawn)"
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P103 self-install] non-fatal failure: %s", e)
        return False


# ─── v7.69 text-patch: append self-install hook to chunk.py ──────────
#
# Inserted at end of `vllm/model_executor/layers/fla/ops/chunk.py`.
# Anchor is the last line of the file's last function definition; we
# append after it. The hook runs every time chunk.py is imported (in
# main process, in spawn workers, in any child interpreter) — so the
# wrapper is installed regardless of how vllm was started.

_P103_SELF_INSTALL_BLOCK = (
    "\n\n"
    "# ============================================================\n"
    "# [Genesis P103 v7.69 self-install] — module-import-time hook\n"
    "# ============================================================\n"
    "# When GENESIS_ENABLE_P103=1, monkey-patch chunk_gated_delta_rule_fwd\n"
    "# at module-import time so the chunked-h wrapper survives any startup\n"
    "# mechanism: `exec vllm serve` from an entrypoint shell, worker\n"
    "# spawn, etc. The v7.62.20 setattr-only path died on the entrypoint\n"
    "# pattern (`python3 -m vllm._genesis.patches.apply_all && exec\n"
    "# vllm serve`). See club-3090#19 finding 2 (2026-05-02).\n"
    "#\n"
    "# Lazy import — if vllm._genesis isn't on sys.path (test env, partial\n"
    "# install), the try/except keeps chunk.py importable.\n"
    "try:\n"
    "    import os as _genesis_p103_os\n"
    "    if _genesis_p103_os.environ.get(\n"
    "        \"GENESIS_ENABLE_P103\", \"\"\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "        from vllm._genesis.wiring.hybrid.patch_103_fla_cliff2_chunked "
    "import (\n"
    "            _genesis_p103_install_at_import as "
    "_genesis_p103_install,\n"
    "        )\n"
    "        _genesis_p103_install(globals())\n"
    "except Exception:  # noqa: BLE001\n"
    "    # Never break chunk.py import — Genesis is opt-in defense-in-depth.\n"
    "    pass\n"
)

# Anchor: the last `elif SUPPRESS_LEVEL >= 3:` block of
# `chunk_gated_delta_rule` function (chunk.py:128-234). After this `return`,
# the rest of the file is just the high-level wrapper docstring + return —
# safe to append after.
_P103_SELF_INSTALL_ANCHOR = (
    "    o, final_state = ChunkGatedDeltaRuleFunction.apply(\n"
    "        q,\n"
    "        k,\n"
    "        v,\n"
    "        g,\n"
    "        beta,\n"
    "        scale,\n"
    "        initial_state,\n"
    "        output_final_state,\n"
    "        cu_seqlens,\n"
    "        chunk_indices,\n"
    "        chunk_offsets,\n"
    "        use_qk_l2norm_in_kernel,\n"
    "    )\n"
    "    return o, final_state\n"
)

_P103_SELF_INSTALL_REPLACEMENT = (
    _P103_SELF_INSTALL_ANCHOR + _P103_SELF_INSTALL_BLOCK
)

_GENESIS_P103_SELF_INSTALL_MARKER = (
    "Genesis P103 self-install hook (v7.69, club-3090#19 finding 2)"
)


def _make_self_install_text_patcher():
    """Build the v7.69 text-patch that appends the self-install hook to
    chunk.py. Returns None if vllm tree is not resolvable."""
    from vllm._genesis.guards import resolve_vllm_file
    from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher

    target = resolve_vllm_file(
        "model_executor/layers/fla/ops/chunk.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P103 model_executor/layers/fla/ops/chunk.py — self-install "
            "hook (v7.69, club-3090#19 finding 2)"
        ),
        target_file=str(target),
        marker=_GENESIS_P103_SELF_INSTALL_MARKER,
        sub_patches=[
            TextPatch(
                name="p103_self_install_at_chunk_py_end",
                anchor=_P103_SELF_INSTALL_ANCHOR,
                replacement=_P103_SELF_INSTALL_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Specific to our own insertion (re-runs hit Layer 2
            # IDEMPOTENT first via the wiring marker).
            "[Genesis P103 v7.69 self-install]",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P103 — chunked fwd_h+fwd_o orchestrator wrap.

    v7.69 strategy:
      1. Apply text-patch on chunk.py (appends self-install hook).
         This is the durable mechanism — survives `exec vllm serve`
         and worker spawn because chunk.py on disk now imports our
         install helper at every module load.
      2. ALSO do the legacy setattr-based wrap on the current process
         (defense-in-depth — helps modules that captured a stale
         reference via `from .chunk import chunk_gated_delta_rule_fwd`,
         and triggers the wrapper immediately if chunk.py was already
         imported before our text-patch landed on disk).

    Never raises. Returns (status, reason).
    """
    if not should_apply():
        return "skipped", "GENESIS_ENABLE_P103 not set or platform not NVIDIA SM 8.0+"

    # Hybrid-active dispatch gate — Cliff 2 only triggers on FLA-GDN models
    try:
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P103 FLA Cliff 2 chunked",
                "pure-attention model (no GDN — Cliff 2 only affects DeltaNet)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P103] model_detect probe failed (proceeding): %s", e)

    # ─── Step 1 (v7.69): durable text-patch on chunk.py ─────────────
    # This is what survives `exec vllm serve` + worker spawn.
    text_patch_status = "skipped"
    text_patch_reason = "vllm tree not resolvable"
    try:
        from vllm._genesis.guards import vllm_install_root
        from vllm._genesis.wiring.text_patch import TextPatchResult

        if vllm_install_root() is not None:
            patcher = _make_self_install_text_patcher()
            if patcher is not None:
                result, failure = patcher.apply()
                if result in (
                    TextPatchResult.APPLIED,
                    TextPatchResult.IDEMPOTENT,
                ):
                    text_patch_status = "applied" if (
                        result == TextPatchResult.APPLIED
                    ) else "idempotent"
                    text_patch_reason = (
                        "chunk.py self-install hook appended (survives "
                        "`exec vllm serve` + worker spawn)"
                    )
                else:
                    text_patch_status = "skipped"
                    text_patch_reason = (
                        f"text-patch did not land: "
                        f"{failure.reason if failure else 'unknown'} — "
                        f"{failure.detail if failure and failure.detail else 'unknown'}"
                    )
    except Exception as e:  # noqa: BLE001
        log.debug("[Genesis P103] text-patch step non-fatal failure: %s", e)
        text_patch_reason = f"text-patch raised: {e}"

    # ─── Step 2 (legacy setattr): current-process wrap ──────────────
    # Defense-in-depth — wraps the function in THIS process's chunk.py
    # module dict, helping any module that already imported it (e.g.
    # `from .chunk import chunk_gated_delta_rule_fwd` style aliases).
    import importlib
    try:
        chunk_mod = importlib.import_module(_TARGET_MODULE)
    except ImportError as e:
        return (
            "applied" if text_patch_status == "applied" else "skipped",
            f"text-patch={text_patch_status} ({text_patch_reason}); "
            f"setattr-step skipped (FLA module unavailable: {e})",
        )

    original = getattr(chunk_mod, _FN_NAME, None)
    if original is None:
        return (
            "applied" if text_patch_status == "applied" else "skipped",
            f"text-patch={text_patch_status} ({text_patch_reason}); "
            f"setattr-step skipped ({_FN_NAME!r} not in {_TARGET_MODULE!r})",
        )

    if getattr(original, _GENESIS_P103_MARKER_ATTR, False):
        return "applied", (
            f"text-patch={text_patch_status} ({text_patch_reason}); "
            f"setattr already idempotent (chunk.py self-install fired or "
            f"prior apply still active in this process)"
        )

    # Resolve closure dependencies — fail soft if any symbol moved
    try:
        chunk_local_cumsum = getattr(chunk_mod, "chunk_local_cumsum")
        chunk_scaled_dot_kkt_fwd = getattr(chunk_mod, "chunk_scaled_dot_kkt_fwd")
        solve_tril = getattr(chunk_mod, "solve_tril")
        recompute_w_u_fwd = getattr(chunk_mod, "recompute_w_u_fwd")
        fwd_h = getattr(chunk_mod, "chunk_gated_delta_rule_fwd_h")
        chunk_fwd_o_callable = getattr(chunk_mod, "chunk_fwd_o")
        FLA_CHUNK_SIZE = getattr(chunk_mod, "FLA_CHUNK_SIZE", 64)
        SUPPRESS_LEVEL = getattr(chunk_mod, "SUPPRESS_LEVEL", 0)
    except AttributeError as e:
        return (
            "applied" if text_patch_status == "applied" else "skipped",
            f"text-patch={text_patch_status} ({text_patch_reason}); "
            f"setattr-step skipped (P49 interface drift: missing symbol "
            f"in chunk.py — {e})",
        )

    wrapper = _make_chunked_wrapper(
        original_fwd=original,
        chunk_local_cumsum=chunk_local_cumsum,
        chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
        solve_tril=solve_tril,
        recompute_w_u_fwd=recompute_w_u_fwd,
        chunk_gated_delta_rule_fwd_h=fwd_h,
        chunk_fwd_o_callable=chunk_fwd_o_callable,
        fla_chunk_size=FLA_CHUNK_SIZE,
        suppress_level=SUPPRESS_LEVEL,
    )

    # Rebind in the defining module + walk sys.modules for any other
    # callers that did `from .chunk import chunk_gated_delta_rule_fwd`.
    # rebound_count is INFORMATIONAL — internal callers in chunk.py
    # resolve via chunk_mod.__dict__ which we just updated, so 0 is
    # normal. The count tracks only `from .chunk import X` style aliases.
    setattr(chunk_mod, _FN_NAME, wrapper)
    import sys
    rebound_count = 0
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not hasattr(mod, _FN_NAME):
            continue
        captured = getattr(mod, _FN_NAME, None)
        if captured is original:
            try:
                setattr(mod, _FN_NAME, wrapper)
                rebound_count += 1
            except Exception:
                # Frozen module / read-only namespace — text-patch is primary path
                pass

    max_t = max(int(os.environ.get("GENESIS_FLA_FWD_H_MAX_T", "16384")), 64)
    return (
        "applied",
        (
            f"P103 v7.69 applied: text-patch={text_patch_status} "
            f"({text_patch_reason}); setattr-step wrapped chunk.py::"
            f"{_FN_NAME} (MAX_T={max_t}, rebound at {rebound_count} "
            f"external alias sites — internal callers go through "
            f"chunk_mod.__dict__ and resolve to wrapper). For "
            f"T <= MAX_T or cu_seqlens != None, falls through to "
            f"original. NO-OP for non-hybrid models."
        ),
    )


def is_applied() -> bool:
    """Return True iff our wrapper marker is present."""
    try:
        import importlib
        chunk_mod = importlib.import_module(_TARGET_MODULE)
        fn = getattr(chunk_mod, _FN_NAME, None)
        return fn is not None and getattr(fn, _GENESIS_P103_MARKER_ATTR, False)
    except Exception:
        return False
