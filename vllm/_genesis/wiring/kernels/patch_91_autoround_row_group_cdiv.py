# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 91 — AutoRound row-parallel group ceil-div + start-idx fix.

Backport of the *non-MoE-specific* portion of [vllm#39460](
https://github.com/vllm-project/vllm/pull/39460) ("[Bugfix][Quantization] Fix
Gemma4 AutoRound serving gaps on top of GPTQMarlin row groups").

================================================================
ROOT CAUSE
================================================================

`vllm/model_executor/layers/quantization/gptq_marlin.py:395-402` computes:

    scales_and_zp_input_dim = 0
    scales_and_zp_size = input_size_per_partition // group_size

When `input_size_per_partition % group_size != 0` (e.g. AutoRound INT4/INT8
checkpoints where a row-parallel layer's per-rank shard does NOT divide
cleanly into quant groups), this floor-div drops the *trailing partial group*
silently:

  - The checkpoint stores `cdiv(input_size, group_size)` scales / qzeros per
    output column (the trailing group covers the remainder).
  - `create_weights` allocates only `floor` rows for the parameter tensor.
  - The loader then narrows the source tensor to `[tp_rank * shard_size,
    +shard_size)` (`vllm/model_executor/parameter.py:222-225`), which uses
    the WRONG start_idx for the second rank — it's measured in scales-rows
    but should be measured in input-element units divided by group_size.

Combined symptom: at TP>=2 with `group_size=128` and any non-divisible
input_size_per_partition, rank-1 scales are loaded from the wrong source
row, producing silently wrong dequant — either visibly broken output or a
fallback to a slow non-Marlin path (depending on which downstream check
trips). Sister bug #38064 (W4A8 silently runs as W4A16) had a 2.72x
latency improvement when its silent-fallback was removed; this one closes
the *correctness* hole on the W4A16 / W8A16 path.

================================================================
WHY OUR INT4/INT8 27B IS LIKELY HIT
================================================================

`Lorbus/Qwen3.6-27B-int4-AutoRound` (W4A16, group=128, sym, packing
auto_round:auto_gptq) and `Minachist/Qwen3.6-27B-INT8-AutoRound` (W8A16,
same shape) at TP=2 split GDN `in_proj_a` across two ranks. With low-rank
projections (e.g. lora-rank 512) and group_size=128 the partition divides
cleanly — but the GDN-specific `in_proj_ba` and several dense layers DO
hit non-divisible per-rank input dims when num_v_heads * head_dim_v
configurations land on awkward numbers (the same arithmetic that creates
the MIN_THREAD_N=64 problem P87 addresses on the output side).

Lorbus INT4 measured at 87/61/67 t/s vs Minachist INT8 at 93/77/86 t/s on
identical HW. Cross-engine analysis (2026-04-28) hypothesizes this is the
dominant cause of INT4 < INT8 perf gap on our hardware — verify by A/B
benchmark of Lorbus before/after this patch.

================================================================
FIX (text-patch, 3 anchored sub-patches in 2 files)
================================================================

1. **gptq_marlin.py L402** — replace `input_size // group_size` with
   `cdiv(input_size, group_size)` (the `repeat_scales_on_all_ranks` branch).
2. **gptq_marlin.py L407** — replace `input_size_per_partition // group_size`
   with `cdiv(input_size_per_partition, group_size)` (the row-parallel branch).
3. **gptq_marlin.py L470** — register `row_group_size` and
   `row_input_size_per_partition` on `scales` and `qzeros` so the loader
   can compute the correct global group offset.
4. **parameter.py L219-225** — replace the start_idx computation in
   `RowvLLMParameter.load_row_parallel_weight` with the group-aware variant
   from the PR.

We deliberately DO NOT port the MoE-side changes (`gate_linear.py`,
`moe_wna16.py`, `gemma4.py`) — those are Gemma4-specific and unrelated to
our Qwen3.6-MoE prod path. If Gemma4 enters our model rotation, those
patches can be added as P91b.

================================================================
SAFETY MODEL
================================================================

- `cdiv(x, group_size)` >= `floor(x // group_size)` always; equal when x
  is divisible. So this change can only INCREASE allocated rows, never
  shrink — no chance of breaking checkpoints that previously loaded fine.
- The loader change is gated by `getattr(self, 'row_group_size', None)`:
  parameters NOT carrying the new attrs (everything except scales/qzeros
  for GPTQMarlin row-parallel) take the original code path unchanged.
- Default OFF (env-gated). When OFF, behavior is upstream nightly.
- Idempotent via marker; drift-aware on three anchor sites.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#39460.
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

log = logging.getLogger("genesis.wiring.p91_autoround_row_group_cdiv")


GENESIS_P91_MARKER = "Genesis P91 AutoRound row-group cdiv (vllm#39460) v7.62.1"


# ─── gptq_marlin.py: 3 sub-patches ─────────────────────────────────────────

P91_GM_ANCHOR_FLOOR_INPUT_SIZE = (
    "            scales_and_zp_size = input_size // group_size\n"
)

P91_GM_REPLACE_FLOOR_INPUT_SIZE = (
    "            # [Genesis P91 vllm#39460 backport] cdiv not floor-div: when\n"
    "            # input_size % group_size != 0, AutoRound stores cdiv() many\n"
    "            # scales (trailing partial group covers the remainder); floor\n"
    "            # silently drops the partial group's scales.\n"
    "            from vllm.utils.math_utils import cdiv as _genesis_p91_cdiv\n"
    "            scales_and_zp_size = _genesis_p91_cdiv(input_size, group_size)\n"
)


P91_GM_ANCHOR_FLOOR_PARTITION = (
    "            scales_and_zp_size = input_size_per_partition // group_size\n"
)

P91_GM_REPLACE_FLOOR_PARTITION = (
    "            # [Genesis P91 vllm#39460 backport] cdiv for per-partition too.\n"
    "            from vllm.utils.math_utils import cdiv as _genesis_p91_cdiv\n"
    "            scales_and_zp_size = _genesis_p91_cdiv(\n"
    "                input_size_per_partition, group_size\n"
    "            )\n"
)


# Anchor: register_parameter("scales", scales) — insert row_group_attrs setattr
# just before the register_parameter calls so loader sees them.
P91_GM_ANCHOR_REGISTER_SCALES = (
    "        layer.register_parameter(\"qweight\", qweight)\n"
    "        layer.register_parameter(\"g_idx\", g_idx)\n"
    "        layer.register_parameter(\"scales\", scales)\n"
    "        layer.register_parameter(\"qzeros\", qzeros)\n"
)

P91_GM_REPLACE_REGISTER_SCALES = (
    "        # [Genesis P91 vllm#39460 backport] tag scales/qzeros with their\n"
    "        # row-group metadata so RowvLLMParameter.load_row_parallel_weight\n"
    "        # can compute the correct global group offset for TP>1 instead of\n"
    "        # the broken `tp_rank * shard_size` (which is in scales-rows units\n"
    "        # but applied to the input-element-indexed source tensor).\n"
    "        # We use direct setattr (no import dependency) — equivalent to\n"
    "        # the upstream `set_weight_attrs` helper but robust to module\n"
    "        # path drift across vLLM releases.\n"
    "        try:\n"
    "            for _genesis_p91_obj in (scales, qzeros):\n"
    "                setattr(_genesis_p91_obj, 'row_group_size', group_size)\n"
    "                setattr(\n"
    "                    _genesis_p91_obj,\n"
    "                    'row_input_size_per_partition',\n"
    "                    input_size_per_partition,\n"
    "                )\n"
    "        except Exception:\n"
    "            pass  # tagging best-effort; loader falls back to original\n"
    "                  # non-grouped start_idx via getattr default.\n"
    "        layer.register_parameter(\"qweight\", qweight)\n"
    "        layer.register_parameter(\"g_idx\", g_idx)\n"
    "        layer.register_parameter(\"scales\", scales)\n"
    "        layer.register_parameter(\"qzeros\", qzeros)\n"
)


def _make_gm_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/quantization/gptq_marlin.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91 gptq_marlin.py — cdiv groups + row-group attrs (vllm#39460)"
        ),
        target_file=str(target),
        marker=GENESIS_P91_MARKER + "_gptq_marlin",
        sub_patches=[
            TextPatch(
                name="p91_gm_floor_input_size_to_cdiv",
                anchor=P91_GM_ANCHOR_FLOOR_INPUT_SIZE,
                replacement=P91_GM_REPLACE_FLOOR_INPUT_SIZE,
                required=True,
            ),
            TextPatch(
                name="p91_gm_floor_partition_to_cdiv",
                anchor=P91_GM_ANCHOR_FLOOR_PARTITION,
                replacement=P91_GM_REPLACE_FLOOR_PARTITION,
                required=True,
            ),
            TextPatch(
                name="p91_gm_register_row_group_attrs",
                anchor=P91_GM_ANCHOR_REGISTER_SCALES,
                replacement=P91_GM_REPLACE_REGISTER_SCALES,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91",
            "_genesis_p91_cdiv",
            # Upstream-side markers if vLLM merges the original PR or equivalent
            "scales_and_zp_size = cdiv(input_size",
            "row_input_size_per_partition",
        ],
    )


# ─── parameter.py: 1 sub-patch ─────────────────────────────────────────────

P91_PARAM_ANCHOR = (
    "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
    "        shard_size = self.data.shape[self.input_dim]\n"
    "        loaded_weight = loaded_weight.narrow(\n"
    "            self.input_dim, self.tp_rank * shard_size, shard_size\n"
    "        )\n"
)

P91_PARAM_REPLACE = (
    "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
    "        shard_size = self.data.shape[self.input_dim]\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P91 vllm#39460 backport]\n"
    "        # When the parameter is tagged with row_group_size +\n"
    "        # row_input_size_per_partition (set by GPTQMarlin.create_weights\n"
    "        # for `scales` and `qzeros` row-parallel layers), compute\n"
    "        # start_idx in source-tensor row units (== scale-rows = input\n"
    "        # element units / group_size). The default tp_rank * shard_size\n"
    "        # is wrong for partial-group shards and silently corrupts dequant.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        _genesis_p91_group_size = getattr(self, 'row_group_size', None)\n"
    "        _genesis_p91_input_partition = getattr(\n"
    "            self, 'row_input_size_per_partition', None\n"
    "        )\n"
    "        if (_genesis_p91_group_size is not None\n"
    "                and _genesis_p91_input_partition is not None):\n"
    "            _genesis_p91_start = (\n"
    "                self.tp_rank * _genesis_p91_input_partition\n"
    "            ) // _genesis_p91_group_size\n"
    "        else:\n"
    "            _genesis_p91_start = self.tp_rank * shard_size\n"
    "        loaded_weight = loaded_weight.narrow(\n"
    "            self.input_dim, _genesis_p91_start, shard_size\n"
    "        )\n"
)


def _make_param_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/parameter.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P91 parameter.py — group-aware row_parallel start_idx (vllm#39460)"
        ),
        target_file=str(target),
        marker=GENESIS_P91_MARKER + "_parameter",
        sub_patches=[
            TextPatch(
                name="p91_param_group_aware_start_idx",
                anchor=P91_PARAM_ANCHOR,
                replacement=P91_PARAM_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P91",
            "_genesis_p91_start",
            "row_group_size",
            "row_input_size_per_partition",
        ],
    )


# ─── apply / is_applied / revert ──────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Apply P91 — AutoRound row-parallel group cdiv + start-idx fix."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P91")
    log_decision("P91", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    gm = _make_gm_patcher()
    if gm is None:
        return "skipped", "gptq_marlin.py not found"
    param = _make_param_patcher()
    if param is None:
        return "skipped", "parameter.py not found"

    for p in (gm, param):
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"

    # Idempotency check on both files
    with open(gm.target_file) as f:
        gm_content = f.read()
    with open(param.target_file) as f:
        param_content = f.read()

    gm_already = gm.marker in gm_content
    param_already = param.marker in param_content

    if gm_already and param_already:
        return "applied", "idempotent (both markers present)"

    # Drift detection
    for marker in gm.upstream_drift_markers:
        if marker.startswith("[Genesis"):
            continue
        if marker in gm_content and not gm_already:
            return (
                "skipped",
                f"upstream drift in gptq_marlin.py: {marker!r} present "
                "without our marker — upstream may have merged equivalent fix",
            )
    for marker in param.upstream_drift_markers:
        if marker.startswith("[Genesis"):
            continue
        if marker in param_content and not param_already:
            return (
                "skipped",
                f"upstream drift in parameter.py: {marker!r} present "
                "without our marker — upstream may have merged equivalent fix",
            )

    # Audit G-POST-03 fix 2026-05-05 (genesis_post_fix_rescan_audit):
    # SKIPPED was being masked as final "applied" — surface it honestly.
    if not gm_already:
        result, failure = gm.apply()
        if result == TextPatchResult.SKIPPED:
            _r = failure.reason if failure else "anchor drift / not eligible"
            _d = f" ({failure.detail})" if (failure and failure.detail) else ""
            return "skipped", f"{gm.patch_name}: {_r}{_d}"
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{gm.patch_name}: "
                f"{failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''})"
            )

    if not param_already:
        result, failure = param.apply()
        if result == TextPatchResult.SKIPPED:
            _r = failure.reason if failure else "anchor drift / not eligible"
            _d = f" ({failure.detail})" if (failure and failure.detail) else ""
            return "skipped", (
                f"{param.patch_name}: {_r}{_d} "
                "(P91 partial: gptq_marlin.py applied but parameter.py "
                "skipped — re-apply needed for matching pair)"
            )
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{param.patch_name}: "
                f"{failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''})"
            )

    return (
        "applied",
        "P91 applied (DUAL FILE): gptq_marlin.py uses cdiv() for scale rows "
        "and tags scales/qzeros with row_group_size + "
        "row_input_size_per_partition; parameter.py uses group-aware "
        "start_idx for row-parallel scale/zero loading. Fixes silent dequant "
        "corruption when input_size_per_partition % group_size != 0 at TP>1."
    )


def is_applied() -> bool:
    """Return True iff both files contain our marker."""
    if vllm_install_root() is None:
        return False
    gm = _make_gm_patcher()
    param = _make_param_patcher()
    if gm is None or param is None:
        return False
    try:
        with open(gm.target_file) as f:
            gm_content = f.read()
        with open(param.target_file) as f:
            param_content = f.read()
    except Exception:
        return False
    return gm.marker in gm_content and param.marker in param_content


def revert() -> tuple[str, str]:
    """Revert is text-patch driven — caller should re-image the container.
    For safety we don't attempt in-place restore (the original lines are
    not preserved verbatim once replaced)."""
    return (
        "skipped",
        "P91 text-patch revert not supported in-place; redeploy a fresh "
        "container or `git checkout` the affected vLLM files",
    )
