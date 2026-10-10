# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the GLM-5.2 MonoKernel runs, and the per-step rule its spec cannot express.

Under MTP with k draft tokens a verify step is a pure-decode batch whose requests carry
up to 1 + k query rows. The kernel treats every row as its own request row (position,
slot, CSR range), and a row sees the KV of earlier rows of the same launch through the
kvnew / penew mailboxes, so verify steps need no kernel change."""

import torch

from vllm import envs
from vllm.models.common.mono import MonoSpec

# widths the GLM kernel builds (S <= 12 at TP8)
KERNEL_WIDTHS = (1, 2, 4, 5, 6, 8, 10, 12)

GLM5_MONO = MonoSpec(
    name="GLM-5.2 MonoKernel",
    architectures=("glm_moe_dsa",),
    dtypes=(torch.bfloat16,),
    kv_cache_dtypes=("auto", "bfloat16"),
    tp_sizes=(8,),
    cdna_versions=(4,),
    widths=KERNEL_WIDTHS,
    opt_in=lambda c: bool(envs.VLLM_ROCM_USE_GLM5_MONOKERNEL),
    on_refusal="raise",
)


def step_reason(md, T: int, has_residual: bool, max_query_len: int, topk: int) -> str:
    """'' if a step's attention metadata (identical on every TP rank) allows the mono
    path, else why not. The width check belongs to the runtime."""
    if not has_residual:
        return "no_residual"
    n_dec = int(getattr(md, "num_decode_tokens", 0))
    if md.num_prefills != 0 or n_dec != md.num_actual_tokens:
        return "prefill_or_mixed"
    if md.max_query_len < 1 or md.max_query_len > max_query_len:
        return "query_len"
    if md.num_actual_tokens != T:
        return "padded_T"
    if md.topk_tokens != topk:
        return "topk"
    return ""


def spec_widths(k: int, max_width: int = 12) -> tuple[int, ...]:
    """Kernel widths that are multiples of 1 + k: FULL-graph capture sizes under spec
    decode are multiples of the verify length and must all be kernel widths."""
    return tuple(w for w in KERNEL_WIDTHS if w % (1 + k) == 0 and w <= max_width)
