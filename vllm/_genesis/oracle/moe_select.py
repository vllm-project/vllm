# SPDX-License-Identifier: Apache-2.0
"""MoE expert implementation oracle — pattern from vllm#41436.

Centralizes the decision "which fused_moe expert impl to use" based on
(quant_format × hardware × workload regime). Replaces ad-hoc if/elif
scattered across PN8/P81/P83/P87/P91 patches.

Doesn't ENABLE patches — that's still done by `should_apply` in dispatcher.
This oracle just provides a single function operators / our own validators
can call to ANSWER the question "for this config, which expert path will
fire?".

Why this matters
----------------
Memory `feedback_v764_swap_findings` shows we silently shipped P87 enabled
on Minachist INT8 (group_size=-1 → AllSparkLinearKernel selected → P87
no-op for that checkpoint). An oracle would have flagged this earlier:
"on quant=int8_gs-1, P87 is no-op; consider keeping disabled".

Models on Genesis stack:
- 27B Lorbus AutoRound INT4 (group_size=128) → Marlin path → P87/P91 fire
- 27B Minachist INT8 (group_size=-1) → AllSpark → P87/P91 NO-OP
- 35B-A3B FP8 → fused_moe block-quantized → P81 fires for low-M
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoESelection:
    """Decision result for a given (quant, hw, regime) combination."""
    impl_name: str           # "marlin", "allspark", "fp8_block_scaled", etc.
    relevant_patches: tuple[str, ...]   # Genesis patches that affect this path
    notes: str               # human-readable explanation


def select_moe_expert_impl(
    quant_format: str,
    sm_major: int,
    num_tokens: int,
    *,
    has_calibrated_kv_scales: bool = False,
    group_size: int | None = None,
) -> MoESelection:
    """Pure decision function — what fused_moe impl will fire?

    Args:
      quant_format: "fp8", "compressed_tensors_int4_marlin", "int4_autoround",
                     "int8_gs128", "int8_gs-1" (no group)
      sm_major: GPU compute capability major (e.g. 8 for A5000 SM86)
      num_tokens: scheduled batch tokens (M dimension)
      has_calibrated_kv_scales: checkpoint provides k_scale/v_scale
      group_size: weight quant group size (None for FP8, 128 for AutoRound, -1 for no-group)

    Returns:
      MoESelection — which impl, which Genesis patches matter, why.
    """
    # Per-tensor MM low-M optimization (Genesis P81 vllm#37035 hot path)
    if quant_format == "fp8" and num_tokens <= 8:
        return MoESelection(
            impl_name="fp8_block_scaled_low_m",
            relevant_patches=("P81", "P82", "PN8"),
            notes="FP8 block-scaled MM, low-M decode path; P81 fires when M<=8 "
                  "(35B PROD MTP K=3 verify hits this every step)",
        )
    if quant_format == "fp8":
        return MoESelection(
            impl_name="fp8_block_scaled",
            relevant_patches=("PN8", "P82"),
            notes="Standard FP8 block-scaled fused_moe (35B prefill / large batch)",
        )
    if quant_format in ("compressed_tensors_int4_marlin", "int4_autoround"):
        if group_size in (None, 128):
            return MoESelection(
                impl_name="marlin",
                relevant_patches=("P87", "P91"),
                notes="GPTQ Marlin path (27B Lorbus AutoRound g=128). P87 fixes "
                      "sub-tile shard padding (vllm#40361); P91 fixes row-parallel "
                      "scales group cdiv (vllm#39660). Both REQUIRED for our model.",
            )
        if group_size == -1:
            return MoESelection(
                impl_name="allspark_no_group",
                relevant_patches=(),
                notes="AllSpark linear kernel for no-group quant. P87/P91 are NO-OP "
                      "here — Marlin not selected. (Memory feedback_v764_swap_findings)",
            )
    if quant_format == "int8_gs128":
        return MoESelection(
            impl_name="marlin_int8",
            relevant_patches=("P87",),
            notes="INT8 group=128 → Marlin int8 path. P87 sub-tile padding fires.",
        )
    if quant_format == "int8_gs-1":
        return MoESelection(
            impl_name="allspark_no_group",
            relevant_patches=(),
            notes="INT8 no-group → AllSpark; Marlin patches NO-OP. "
                  "(memory feedback_v764_swap_findings — Minachist 27B INT8 case)",
        )
    return MoESelection(
        impl_name="vllm_default",
        relevant_patches=(),
        notes=f"Unknown/unhandled quant_format={quant_format!r}; "
              "vllm default path; no Genesis MoE patches apply",
    )


def explain_for_config(quant_format: str, sm_major: int, **kwargs) -> str:
    """Human-readable explanation — used by `genesis doctor` and CLI."""
    sel = select_moe_expert_impl(
        quant_format, sm_major, num_tokens=kwargs.get("num_tokens", 256),
        **{k: v for k, v in kwargs.items() if k != "num_tokens"},
    )
    parts = [
        f"MoE oracle: quant={quant_format} sm{sm_major}.x → impl={sel.impl_name}",
        f"  Relevant Genesis patches: {', '.join(sel.relevant_patches) or '(none)'}",
        f"  Notes: {sel.notes}",
    ]
    return "\n".join(parts)
