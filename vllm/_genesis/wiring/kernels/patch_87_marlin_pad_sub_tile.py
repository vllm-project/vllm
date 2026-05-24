# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 87 — Marlin W4A16/W8A16 sub-tile output dim pad-on-load.

Backport of [vllm#40361](https://github.com/vllm-project/vllm/pull/40361)
("[Kernel][Bugfix] Marlin W4A16: pad sub-tile output dims on load").

================================================================
v7.62.10 REWRITE: text-patch instead of class-rebind
================================================================

The v7.62 implementation used class-rebind (capture original methods +
monkey-patch new ones). Empirically observed 2026-04-28: under torch.compile
+ FULL cudagraph capture (which Marlin path enters during decode),
dynamo refused to trace through the wrapper indirection and crashed
with `torch._dynamo.exc.Unsupported: Attempted to call function marked
as skipped`. The wrapper closure over `_ORIGINAL_APPLY_WEIGHTS` global
was the trigger.

This rewrite uses text-patch (like P91) — modifies the actual marlin.py
source directly. dynamo then treats the patched code as native vLLM
source and traces it cleanly. No wrapper, no closure, no indirection.

================================================================
WHAT THE PATCH DOES (faithful port of PR #40361 diff)
================================================================

5 sub-patches on `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`:

  1. **imports** — add `dataclasses`, `torch.nn.functional as F`,
     `init_logger`, `GPTQ_MARLIN_MIN_THREAD_N`, `round_up`
  2. **can_implement** — wrap `c.partition_weight_shape[1]` with
     `round_up(..., GPTQ_MARLIN_MIN_THREAD_N)` so sub-tile shards report
     supported (the actual padding happens in process_weights_after_loading)
  3. **_maybe_pad_n method** — insert new private method right before
     process_weights_after_loading; zero-pads qweight/scales/qzeros/bias
     along the output dim to the next tile multiple, stores `_marlin_orig_n`
     on the layer, swaps self.config to report padded shape
  4. **process_weights_after_loading prelude** — call `self._maybe_pad_n(layer)`
     as the very first statement, so all downstream repack/permute see
     the padded shape consistently
  5. **apply_weights output slice** — pad bias to padded_n if caller
     supplied at orig_n, pass padded_n to apply_gptq_marlin_linear,
     slice the extra columns off the output

================================================================
SAFETY MODEL
================================================================

- Idempotent via marker, drift detection on multiple anchors
- When `padded_n == orig_n` (already tile-aligned), `_maybe_pad_n`
  returns early — pure no-op, identical behavior to upstream
- All sub-patches required (textual integrity); failure at any one
  leaves marlin.py in a partially-patched state on disk → next boot
  detects via marker absence and re-applies cleanly
- Default OFF; opt-in via `GENESIS_ENABLE_P87=1`

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#40361.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p87_marlin_pad_sub_tile")

GENESIS_P87_MARKER = (
    "Genesis P87 Marlin sub-tile output dim pad-on-load (vllm#40361) v7.62.10_textpatch"
)


# ─── Sub-patch 1: imports block ──────────────────────────────────────────

P87_IMPORTS_OLD = (
    "import torch\n"
    "\n"
    "from vllm import _custom_ops as ops\n"
    "from vllm.model_executor.layers.quantization.utils.marlin_utils import (\n"
    "    MARLIN_SUPPORTED_GROUP_SIZES,\n"
)

P87_IMPORTS_NEW = (
    "# [Genesis P87 vllm#40361 backport] additional imports\n"
    "import dataclasses\n"
    "\n"
    "import torch\n"
    "import torch.nn.functional as F  # noqa: N812 — Genesis P87\n"
    "\n"
    "from vllm import _custom_ops as ops\n"
    "from vllm.logger import init_logger as _genesis_p87_init_logger\n"
    "from vllm.model_executor.layers.quantization.utils.marlin_utils import (\n"
    "    GPTQ_MARLIN_MIN_THREAD_N as _GENESIS_P87_MIN_THREAD_N,\n"
    "    MARLIN_SUPPORTED_GROUP_SIZES,\n"
)


# ─── Sub-patch 2: round_up import (separate site near math_utils) + logger ──

P87_LOGGER_OLD = (
    "from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig\n"
    "\n"
    "\n"
    "class MarlinLinearKernel(MPLinearKernel):\n"
)

P87_LOGGER_NEW = (
    "from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig\n"
    "\n"
    "# [Genesis P87 vllm#40361 backport] round_up + logger for sub-tile pad\n"
    "from vllm.utils.math_utils import round_up as _genesis_p87_round_up\n"
    "_genesis_p87_logger = _genesis_p87_init_logger(__name__)\n"
    "\n"
    "\n"
    "class MarlinLinearKernel(MPLinearKernel):\n"
)


# ─── Sub-patch 3: can_implement uses padded_n ────────────────────────────

P87_CAN_IMPLEMENT_OLD = (
    "        return check_marlin_supports_shape(\n"
    "            c.partition_weight_shape[1],  # out_features\n"
    "            c.partition_weight_shape[0],  # in_features\n"
    "            c.full_weight_shape[0],  # in_features\n"
    "            c.group_size,\n"
    "        )\n"
)

P87_CAN_IMPLEMENT_NEW = (
    "        # [Genesis P87 vllm#40361 backport] allow shapes where\n"
    "        # out_features is not divisible by Marlin tile (MIN_THREAD_N=64).\n"
    "        # Validation against round_up(n, 64); actual zero-padding happens\n"
    "        # in process_weights_after_loading via _maybe_pad_n. Runtime cost\n"
    "        # is zero (load-time padding), VRAM cost is a few KB per layer.\n"
    "        _genesis_p87_padded_n = _genesis_p87_round_up(\n"
    "            c.partition_weight_shape[1], _GENESIS_P87_MIN_THREAD_N\n"
    "        )\n"
    "        return check_marlin_supports_shape(\n"
    "            _genesis_p87_padded_n,  # out_features (possibly padded)\n"
    "            c.partition_weight_shape[0],  # in_features\n"
    "            c.full_weight_shape[0],  # in_features\n"
    "            c.group_size,\n"
    "        )\n"
)


# ─── Sub-patch 4: insert _maybe_pad_n method + call from PWA ─────────────

P87_PWA_OLD = (
    "    # note assumes that\n"
    "    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}\n"
    "    #  `weight_scale` is: {input_dim = 0, output_dim = 1}\n"
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        device = getattr(layer, self.w_q_name).device\n"
)

P87_PWA_NEW = (
    "    # ════════════════════════════════════════════════════════════════\n"
    "    # [Genesis P87 vllm#40361 backport] sub-tile output dim pad-on-load\n"
    "    # ════════════════════════════════════════════════════════════════\n"
    "    def _maybe_pad_n(self, layer: torch.nn.Module) -> None:\n"
    "        \"\"\"Pad qweight/scales/qzeros/bias along the output dim to the\n"
    "        next multiple of GPTQ_MARLIN_MIN_THREAD_N. Sets _marlin_orig_n on\n"
    "        the layer for later output slicing in apply_weights. No-op when\n"
    "        already tile-aligned (sets _marlin_orig_n and returns).\n"
    "        \"\"\"\n"
    "        c = self.config\n"
    "        orig_n = c.partition_weight_shape[1]\n"
    "        padded_n = _genesis_p87_round_up(orig_n, _GENESIS_P87_MIN_THREAD_N)\n"
    "        layer._marlin_orig_n = orig_n\n"
    "        if padded_n == orig_n:\n"
    "            return\n"
    "        pad = padded_n - orig_n\n"
    "        pack_factor = 32 // c.weight_type.size_bits\n"
    "        # qweight: [k/pack, n] int32, output_dim=1. Pad with zeros ->\n"
    "        # those output columns decode to weight 0.\n"
    "        q = getattr(layer, self.w_q_name)\n"
    "        q.data = F.pad(q.data, (0, pad), value=0)\n"
    "        # scales: [num_groups, n], output_dim=1. Pad with zeros (values\n"
    "        # don't matter; padded weight columns are already zero).\n"
    "        s = getattr(layer, self.w_s_name)\n"
    "        s.data = F.pad(s.data, (0, pad), value=0)\n"
    "        # qzeros: [num_groups, n/pack] int32, packed_dim=1. Pad by pad/pack\n"
    "        # extra packed columns. Pad value 0 is safe (used only for padded\n"
    "        # weight columns, which are themselves zero).\n"
    "        if c.zero_points and self.w_zp_name is not None:\n"
    "            zp = getattr(layer, self.w_zp_name, None)\n"
    "            if zp is not None:\n"
    "                zp_pad_cols = pad // pack_factor\n"
    "                if zp_pad_cols > 0:\n"
    "                    zp.data = F.pad(zp.data, (0, zp_pad_cols), value=0)\n"
    "        # bias: [n] -> [padded_n]\n"
    "        if hasattr(layer, \"bias\") and layer.bias is not None:\n"
    "            layer.bias.data = F.pad(layer.bias.data, (0, pad), value=0)\n"
    "        # Swap config so all downstream transforms use padded n.\n"
    "        self.config = dataclasses.replace(\n"
    "            c,\n"
    "            partition_weight_shape=(c.partition_weight_shape[0], padded_n),\n"
    "        )\n"
    "        _genesis_p87_logger.info_once(\n"
    "            \"[Genesis P87] padded output dim %d -> %d (tile=%d)\",\n"
    "            orig_n, padded_n, _GENESIS_P87_MIN_THREAD_N,\n"
    "        )\n"
    "\n"
    "    # note assumes that\n"
    "    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}\n"
    "    #  `weight_scale` is: {input_dim = 0, output_dim = 1}\n"
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        # [Genesis P87 vllm#40361 backport] pad before any repack/permute\n"
    "        self._maybe_pad_n(layer)\n"
    "        device = getattr(layer, self.w_q_name).device\n"
)


# ─── Sub-patch 5: apply_weights output slice ─────────────────────────────

P87_APPLY_OLD = (
    "        return apply_gptq_marlin_linear(\n"
    "            input=x,\n"
    "            weight=w_q,\n"
    "            weight_scale=w_s,\n"
    "            weight_zp=w_zp,  # type: ignore\n"
    "            g_idx=w_gidx,  # type: ignore\n"
    "            g_idx_sort_indices=layer.g_idx_sort_indices,\n"
    "            workspace=self.workspace,\n"
    "            wtype=c.weight_type,\n"
    "            input_size_per_partition=c.partition_weight_shape[0],\n"
    "            output_size_per_partition=c.partition_weight_shape[1],\n"
    "            is_k_full=self.is_k_full,\n"
    "            input_global_scale=getattr(layer, \"input_global_scale\", None),\n"
    "            bias=bias,\n"
    "            input_dtype=c.act_type,\n"
    "        )\n"
)

P87_APPLY_NEW = (
    "        # [Genesis P87 vllm#40361 backport] handle padded out-dim\n"
    "        _genesis_p87_padded_n = c.partition_weight_shape[1]\n"
    "        _genesis_p87_orig_n = getattr(\n"
    "            layer, \"_marlin_orig_n\", _genesis_p87_padded_n\n"
    "        )\n"
    "        if bias is not None and bias.shape[-1] != _genesis_p87_padded_n:\n"
    "            bias = F.pad(\n"
    "                bias, (0, _genesis_p87_padded_n - bias.shape[-1]), value=0\n"
    "            )\n"
    "        _genesis_p87_out = apply_gptq_marlin_linear(\n"
    "            input=x,\n"
    "            weight=w_q,\n"
    "            weight_scale=w_s,\n"
    "            weight_zp=w_zp,  # type: ignore\n"
    "            g_idx=w_gidx,  # type: ignore\n"
    "            g_idx_sort_indices=layer.g_idx_sort_indices,\n"
    "            workspace=self.workspace,\n"
    "            wtype=c.weight_type,\n"
    "            input_size_per_partition=c.partition_weight_shape[0],\n"
    "            output_size_per_partition=_genesis_p87_padded_n,\n"
    "            is_k_full=self.is_k_full,\n"
    "            input_global_scale=getattr(layer, \"input_global_scale\", None),\n"
    "            bias=bias,\n"
    "            input_dtype=c.act_type,\n"
    "        )\n"
    "        # [Genesis P87] discard the extra columns produced by the padded matmul\n"
    "        if _genesis_p87_orig_n != _genesis_p87_padded_n:\n"
    "            _genesis_p87_out = _genesis_p87_out[\n"
    "                ..., :_genesis_p87_orig_n\n"
    "            ].contiguous()\n"
    "        return _genesis_p87_out\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/kernels/linear/mixed_precision/marlin.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P87 marlin.py — sub-tile output dim pad-on-load (vllm#40361)"
        ),
        target_file=str(target),
        marker=GENESIS_P87_MARKER,
        sub_patches=[
            TextPatch(
                name="p87_imports",
                anchor=P87_IMPORTS_OLD,
                replacement=P87_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="p87_logger_round_up_imports",
                anchor=P87_LOGGER_OLD,
                replacement=P87_LOGGER_NEW,
                required=True,
            ),
            TextPatch(
                name="p87_can_implement_padded",
                anchor=P87_CAN_IMPLEMENT_OLD,
                replacement=P87_CAN_IMPLEMENT_NEW,
                required=True,
            ),
            TextPatch(
                name="p87_pwa_with_maybe_pad_n",
                anchor=P87_PWA_OLD,
                replacement=P87_PWA_NEW,
                required=True,
            ),
            TextPatch(
                name="p87_apply_weights_slice",
                anchor=P87_APPLY_OLD,
                replacement=P87_APPLY_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P87",
            # Upstream-side markers if PR #40361 (or equivalent) merges:
            "_marlin_orig_n",
            "GPTQ_MARLIN_MIN_THREAD_N as _GENESIS",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P87 — Marlin sub-tile output dim pad-on-load (text-patch v7.62.10)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P87")
    log_decision("P87", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "marlin.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P87] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #40361 (or equivalent) appears merged",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "P87 v7.62.10 applied (TEXT-PATCH, 5 sub-patches): marlin.py "
        "MarlinLinearKernel.{can_implement, _maybe_pad_n (NEW), "
        "process_weights_after_loading prelude, apply_weights output slice} "
        "rewritten in-place. Sub-tile output dims now zero-padded at load + "
        "sliced at apply. Replaces v7.62 class-rebind which crashed under "
        "torch.dynamo cudagraph capture."
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
