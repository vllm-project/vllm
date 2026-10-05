# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative decoding (vLLM native MTP) on the MonoKernel: the per-step go / no-go
rule and the width map. Plain Python (CPU-testable; no torch / vLLM imports).

Under MTP with k draft tokens, a verify step is a pure-decode batch whose requests carry
up to 1 + k query rows (contiguous per request, consecutive positions; sparse-MLA
metadata gives every row its own causal CSR range min(ctx_start + j + 1, 2048)). The
kernel treats every row as an independent request row -- its own position, slot, CSR
range -- and a row sees the KV of earlier rows written in the same launch through the
kvnew / penew mailboxes (patch_new_kv matches each key slot against all rows' new
slots), so no kernel change is needed for the vLLM-indexer path."""

# widths the GLM kernel builds (S <= 12 at TP8)
KERNEL_WIDTHS = (1, 2, 4, 5, 6, 8, 10, 12)


def step_reason(
    md, T: int, has_residual: bool, max_query_len: int, fits: bool, topk: int = 2048
) -> str:
    """'' if a step's attention metadata allows the mono path, else why not (identical
    on every TP rank). Taken when the step is pure decode, every request has 1 ..
    ``max_query_len`` query rows (1; 1 + k under MTP spec decode), all T rows are this
    step's rows and a built width fits T."""
    if not has_residual:
        return "no_residual"
    n_dec = int(getattr(md, "num_decode_tokens", 0))
    if md.num_prefills != 0 or n_dec != md.num_actual_tokens:
        return "prefill_or_mixed"
    if md.max_query_len < 1 or md.max_query_len > max_query_len:
        return "query_len"
    if md.num_actual_tokens != T:
        return "padded_T"
    if not fits:
        return "too_many_rows"
    if md.topk_tokens != topk:
        return "topk"
    return ""


def spec_widths(k: int, max_width: int = 12) -> tuple[int, ...]:
    """Kernel widths that are multiples of 1 + k (vLLM's FULL-graph capture sizes under
    spec decode are multiples of the uniform verify length, and every capture size must
    be a kernel width, guards.py)."""
    return tuple(w for w in KERNEL_WIDTHS if w % (1 + k) == 0 and w <= max_width)


def width_for(rows: int, sizes) -> int | None:
    """The kernel width a step with ``rows`` rows (B requests x (1 + k)) runs on: the
    smallest built width >= rows."""
    for s in sorted(sizes):
        if s >= rows:
            return s
    return None
