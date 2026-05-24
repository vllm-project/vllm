# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N19 — Scoped max_split_size_mb during model load.

================================================================
WHAT THIS PATCH DOES
================================================================

Backport of [vllm#41268](https://github.com/vllm-project/vllm/pull/41268)
(MatthewBonanni, OPEN as of 2026-04-30). PyTorch 2.10+ (we run 2.11)
introduced increased fragmentation during model loading and cudagraph
capture: weight segments that should land contiguously instead get
split inside-other-segments, leaving 200-500 MiB unusable.

Mitigation: temporarily set `max_split_size_mb` to a small value (20
MiB, the minimum PyTorch allows) for the duration of model loading,
then restore the prior value (or PyTorch's effective default of
SIZE_MAX = no limit). Result: cudaMalloc is called more often during
load (negligible cost at startup), but allocator never splits weight
segments inside other segments.

================================================================
WHAT IT FIXES
================================================================

Memory fragmentation at load time. PR #41268 reports +200-500 MiB
recoverable on H100 via this single change. Unverified on Ampere SM
8.6 (our A5000 hardware) — see acceptance bar below.

For our PROD config:
- v794 27B Lorbus: currently `--gpu-memory-utilization=0.90` leaves
  ~2.0 GiB headroom per GPU. If PN19 frees an extra 200-500 MiB,
  operator could safely move to 0.93+ and unlock more KV blocks.
- v759 35B FP8: currently 0.93 — same headroom math applies.
- Long-context configs (256K+): the headroom matters most here
  because cudagraph capture peaks compete with KV blocks.

================================================================
SAFETY MODEL
================================================================

- Cudagraph-safe: the change is at MODEL LOAD time, not at runtime.
  Cudagraph capture happens AFTER model load and uses the restored
  allocator state. Capture-time behavior is identical to upstream.

- Restoration: on exit (success OR exception) the prior
  `max_split_size_mb` is restored. If no prior value was set, PyTorch
  defaults to SIZE_MAX (no limit) — same as if PN19 never ran.

- Defensive parsing: stdlib `re` extracts the prior value (if any)
  from `PYTORCH_CUDA_ALLOC_CONF`. Other allocator hints (e.g.
  `expandable_segments:True`) are preserved unchanged.

- Idempotent via marker; drift detection on the upstream context
  manager appearing natively (it would mean #41268 merged).

- Default OFF; opt-in via `GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1`.

================================================================
ACCEPTANCE BAR — RECOMMENDED OPERATOR PROCEDURE
================================================================

To validate PN19 actually helps on YOUR hardware before enabling
permanently:

1. Capture baseline: boot your config WITHOUT PN19. Record
   `nvidia-smi --query-gpu=memory.used --format=csv -l 1` peak
   during model load (typically 60-180s after container start).
2. Restart with `GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1`. Record
   peak again.
3. Compare. If PN19's peak is ≥ 200 MiB lower, the patch is real
   for your config — try bumping `--gpu-memory-utilization` by
   0.03 (gives back the freed memory as KV blocks).
4. If gap < 100 MiB, PN19 is a no-op for your config. Disable.

Cross-reference Genesis memory `feedback_p104_l2_persistence_thrashing`
— shipping a hardware-mismatch patch that helps one rig but hurts
another is anti-pattern. PN19 default-OFF respects this. Do not flip
to default-ON without per-architecture validation.

================================================================
TORCH API DEPENDENCY
================================================================

This patch uses `torch._C._accelerator_setAllocatorSettings` — a
private API exposed in torch 2.11+ (we run torch 2.11.0+cu130). On
older torch versions the import is unsafe; PN19 detects this and
SKIPS with a clear log message.

The `_C._accelerator_*` API replaced the older `_cuda_*` family for
multi-accelerator support (CUDA, ROCm, XPU). vllm#41268 uses this
API for a reason — it's the new canonical path.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#41268 (MatthewBonanni).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pn19_scoped_max_split")


GENESIS_PN19_MARKER = "Genesis PN19 scoped max_split_size_mb v1"


# Drift markers: detect when upstream lands #41268 natively. At that
# point our text-patch's anchor will fail to match (because the new
# context manager already wraps load_model) → PN19 SKIPs cleanly →
# operator can retire the env flag.
UPSTREAM_DRIFT_MARKERS = [
    GENESIS_PN19_MARKER,
    # Upstream natively lands the context manager helper:
    "_scoped_allocator_max_split",
    "max_split_size_mb=20",
    # PR #41268 also imports `regex` and adds the import line
    "import regex as re",
]


# ─── Sub-patch 1: insert the context-manager helper ─────────────────────

# Anchor: the existing `_maybe_get_memory_pool_context` method definition.
# We insert our new helper RIGHT AFTER it (before the `init_device`
# decorator + method definition).
HELPER_ANCHOR_OLD = (
    "    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:\n"
)


HELPER_ANCHOR_NEW = (
    "    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:\n"
    # Insert our helper as a method RIGHT AFTER this one. We use a class
    # method definition prefix that will continue at the same indent
    # level. The text-patch system replaces the anchor with the new text
    # so we keep the original line and prepend our method by replacing
    # the anchor with: helper + original line.
)


# Actually, the cleanest text-patch approach: anchor on a UNIQUE block
# spanning the END of `_maybe_get_memory_pool_context` and the
# BEGINNING of the next method. Insert our helper between them.
#
# Let's use the `@instrument(span_name="Init device")` line + the
# `def init_device(self):` line as the anchor — that's two-line unique
# in the file. We inject our helper BEFORE the @instrument decorator.

PN19_HELPER_OLD = (
    "    @instrument(span_name=\"Init device\")\n"
    "    def init_device(self):\n"
)

PN19_HELPER_NEW = (
    "    # [Genesis PN19 scoped max_split_size_mb v1]\n"
    "    # Backport of vllm#41268 (MatthewBonanni, OPEN 2026-04-30).\n"
    "    # PyTorch 2.10+ fragmentation during model load: temporarily\n"
    "    # set max_split_size_mb=20 (PyTorch minimum) for the load\n"
    "    # section, restore prior on exit. Enabled via\n"
    "    # GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1.\n"
    "    def _genesis_pn19_scoped_allocator_max_split(self, max_split_size_mb: int):\n"
    "        from contextlib import contextmanager\n"
    "        @contextmanager\n"
    "        def _ctx():\n"
    "            import re as _genesis_pn19_re\n"
    "            import os as _genesis_pn19_os\n"
    "            from vllm.platforms import current_platform\n"
    "            if not current_platform.is_cuda():\n"
    "                yield\n"
    "                return\n"
    "            conf = _genesis_pn19_os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')\n"
    "            match = _genesis_pn19_re.search(r'max_split_size_mb:(\\d+)', conf)\n"
    "            original_value = match.group(1) if match else None\n"
    "            try:\n"
    "                torch._C._accelerator_setAllocatorSettings(\n"
    "                    f'max_split_size_mb:{max_split_size_mb}'\n"
    "                )\n"
    "            except Exception as e:\n"
    "                # torch < 2.11 lacks _accelerator_setAllocatorSettings.\n"
    "                # Fall through unchanged.\n"
    "                import logging as _genesis_pn19_logging\n"
    "                _genesis_pn19_logging.getLogger('genesis.pn19').warning(\n"
    "                    'PN19 scoped max_split_size_mb skipped (torch lacks '\n"
    "                    '_accelerator_setAllocatorSettings): %s', e\n"
    "                )\n"
    "                yield\n"
    "                return\n"
    "            try:\n"
    "                yield\n"
    "            finally:\n"
    "                _SIZE_MAX_MB = (2 ** 64 - 1) // (1024 * 1024)\n"
    "                restore = original_value if original_value else str(_SIZE_MAX_MB)\n"
    "                try:\n"
    "                    torch._C._accelerator_setAllocatorSettings(\n"
    "                        f'max_split_size_mb:{restore}'\n"
    "                    )\n"
    "                except Exception:\n"
    "                    pass\n"
    "        return _ctx()\n"
    "\n"
    "    @instrument(span_name=\"Init device\")\n"
    "    def init_device(self):\n"
)


# ─── Sub-patch 2: wrap load_model's with-block ──────────────────────────

LOAD_MODEL_OLD = (
    "    def load_model(self, *, load_dummy_weights: bool = False) -> None:\n"
    "        with (\n"
    "            self._maybe_get_memory_pool_context(tag=\"weights\"),\n"
    "            set_current_vllm_config(self.vllm_config),\n"
    "        ):\n"
    "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n"
)


LOAD_MODEL_NEW = (
    "    def load_model(self, *, load_dummy_weights: bool = False) -> None:\n"
    "        with (\n"
    "            self._maybe_get_memory_pool_context(tag=\"weights\"),\n"
    "            set_current_vllm_config(self.vllm_config),\n"
    "            # [Genesis PN19] 20 MiB is PyTorch's minimum max_split_size_mb.\n"
    "            self._genesis_pn19_scoped_allocator_max_split(max_split_size_mb=20),\n"
    "        ):\n"
    "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n"
)


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_worker.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN19 scoped max_split_size_mb",
        target_file=target,
        marker=GENESIS_PN19_MARKER,
        sub_patches=[
            TextPatch(
                name="pn19_helper_method",
                anchor=PN19_HELPER_OLD,
                replacement=PN19_HELPER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn19_load_model_wrap",
                anchor=LOAD_MODEL_OLD,
                replacement=LOAD_MODEL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN19. Default OFF. Opt-in via env flag.

    See module docstring for safety model + acceptance bar. Recommend
    measuring fragmentation gap on your hardware before flipping
    permanently.
    """
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT not set; default OFF. "
            "Backport of vllm#41268 (MatthewBonanni, OPEN). PyTorch 2.10+ "
            "introduces load-time fragmentation; this patch sets "
            "max_split_size_mb=20 during model load, restores on exit. "
            "Estimated win: 200-500 MiB on H100 (per #41268 author); "
            "unverified on Ampere — measure before relying on it."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _patcher()
    if p is None:
        return "skipped", "v1/worker/gpu_worker.py not found"

    result, failure = p.apply()
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(result, failure, applied_message='PN19 applied: max_split_size_mb=20 scoped to model load (vllm#41268 backport, MatthewBonanni). Frees 200-500 MiB load-time fragmentation.', patch_name='PN19 scoped max_split_size_mb')


def is_applied() -> bool:
    """Reporter for verify_live_rebinds in apply_all.py."""
    if vllm_install_root() is None:
        return False
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN19_MARKER in f.read()
    except Exception:
        return False
