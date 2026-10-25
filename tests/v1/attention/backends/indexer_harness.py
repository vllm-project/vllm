# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-agnostic pieces of the DSA sparse-indexer suites.

The DeepSeek V3.2 indexer scores every visible key ``s`` of a query token
``t`` as ``sum_h w[t, h] * relu(q[t, h] . k[s])`` with fp8 ``q`` and per-row
fp8 ``k`` (one scale per 128-element row), and returns the top-k positions.

- ``quantize_indexer_k``: the per-row fp8 quantization the cache stores.
- ``ref_indexer_logits``: float32 scores from the quantized operands.
- ``check_topk_rows``: the output contract the sparse attention relies on.
"""

import torch

from vllm.platforms import current_platform


def quantize_indexer_k(
    k: torch.Tensor, *, ue8m0: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row fp8 quantization: returns (fp8 values, float32 scale per row)."""
    fp8 = current_platform.fp8_dtype()
    # The ROCm cache writers divide by 224, not 240, for e4m3fnuz.
    fp8_max = 224.0 if fp8 == torch.float8_e4m3fnuz else 448.0
    amax = k.float().abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / fp8_max
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    k_fp8 = (k.float() / scale[:, None]).to(fp8)
    return k_fp8, scale


def ref_indexer_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """[T, S] scores of every query token against every key row.

    Args:
        q_fp8: [T, H, D] fp8 queries.
        k_fp8: [S, D] fp8 keys; k_scale: [S] their float32 scales.
        weights: [T, H] float32 per-head weights.

    """
    k = k_fp8.float() * k_scale[:, None]
    dots = torch.einsum("thd,sd->ths", q_fp8.float(), k)
    return (dots.relu() * weights.float()[:, :, None]).sum(dim=1)


def check_topk_rows(
    topk: torch.Tensor,
    logits: list[torch.Tensor],
    k: int,
    *,
    rtol: float = 1e-3,
) -> None:
    """Assert each row is a valid top-k of its visible scores.

    Args:
        topk: [T, >= k] indices the indexer wrote.
        logits: per token, the float32 scores of its visible positions
            (positions 0..len-1 of its request).

    The first ``min(len, k)`` entries must be distinct visible positions and
    the rest -1 (sparse MLA walks exactly that prefix), and every selected
    score must reach the k-th largest one up to fp32 summation-order noise.

    """
    topk = topk.cpu()
    for t, scores in enumerate(logits):
        scores = scores.cpu()
        visible = scores.numel()
        expected = min(visible, k)
        row = topk[t, :k]
        assert (row[:expected] >= 0).all(), f"token {t}: invalid entry in prefix"
        assert (row[expected:] == -1).all(), f"token {t}: entries past the prefix"
        chosen = row[:expected].long()
        assert chosen.max().item() < visible, f"token {t}: selected a future key"
        assert chosen.unique().numel() == expected, f"token {t}: duplicates"
        if visible > k:
            kth = scores.topk(k).values[-1]
            tol = rtol * (scores.abs().max() + 1)
            worst = scores[chosen].min()
            assert worst >= kth - tol, (
                f"token {t}: selected score {worst:.5f} < k-th {kth:.5f}"
            )
