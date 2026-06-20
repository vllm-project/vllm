# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PixelPrune: pre-ViT visual token pruning via 2D predictive coding.

Implements the Pred-2D selector from the PixelPrune paper
(arXiv:2604.00886): a LOCO-I-style predictor flags redundant patches on
the merged-token grid so they can be dropped before the ViT. Patches
are expected in *packed merge order* (``spatial_merge_size ** 2``
consecutive patches per merged token, Qwen-VL convention).
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from numba import njit


def _patches_equal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """True where every element along the last dim matches."""
    return (x - y).abs().amax(dim=-1) == 0


# ------------------------------------------------------------------
# Pred-2D LOCO-I selector
# ------------------------------------------------------------------


def select_patches_pred2d(
    tokens: torch.Tensor,
    h: int,
    w: int,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Compute sorted keep-indices for one image's merged-token grid.

    LOCO-I causal neighbourhood at position ``(r, c)``::

        C  B
        A  X

    Predict B on a vertical edge (C≈A, C≠B); otherwise predict A.
    Drop X if it matches the prediction; otherwise keep it as an anchor.

    ``threshold == 0`` uses a vectorised byte-hash compare; ``threshold > 0``
    uses an anchored serial scan against reconstructed neighbours.
    """
    if threshold == 0.0:
        return _select_pred2d_vectorised(tokens, h, w)
    return _select_pred2d_anchored(tokens, h, w, threshold)


@functools.lru_cache(maxsize=4)
def _get_hash_weights(D_half: int) -> np.ndarray:
    """Fixed-seed int64 weights for the byte-view polynomial hash.

    Full int64 entropy (vs a 32-bit-embedded-in-int64 variant): universal
    hashing gives collision probability ~2^-64 per comparison instead of
    ~2^-32, at zero runtime cost — einsum is still int64 multiply-add.
    """
    rng = np.random.RandomState(0xC0FFEE)
    lo = rng.randint(0, 2**32, D_half, dtype=np.uint64)
    hi = rng.randint(0, 2**32, D_half, dtype=np.uint64)
    return ((hi << np.uint64(32)) | lo).view(np.int64)


def _select_pred2d_vectorised(
    tokens: torch.Tensor,
    h: int,
    w: int,
    fingerprint: bool = True,
) -> torch.Tensor:
    """Vectorised exact-match path (threshold == 0).

    Args:
        fingerprint: When ``True`` (default), hash each D-dim patch to one
            int64 via byte-view polynomial hashing so the predictor compares
            scalars instead of D-dim vectors. CPU-only. When ``False``,
            compares full D-dim vectors — pure-torch, device-agnostic.
    """
    D = tokens.shape[-1]
    device = tokens.device

    keep = torch.zeros(h, w, dtype=torch.bool, device=device)
    keep[0, 0] = True  # anchor

    if fingerprint:
        if D % 2 != 0:
            raise ValueError(
                f"PixelPrune fingerprint requires even D for byte-view "
                f"reinterpret, got D={D}."
            )
        # fp32 → int64 byte-view (zero-copy) → numpy int einsum.
        # numpy's int einsum outpaces torch's int sum on CPU here.
        t_np = tokens.to(torch.float32).contiguous().cpu().numpy()
        t_int64 = t_np.view(np.int64).reshape(t_np.shape[0], D // 2)
        hashed = np.einsum("nd,d->n", t_int64, _get_hash_weights(D // 2))
        g = torch.from_numpy(hashed).to(device).view(h, w)

        if w > 1:
            keep[0, 1:] = g[0, 1:] != g[0, :-1]
        if h > 1:
            keep[1:, 0] = g[1:, 0] != g[:-1, 0]
        if h > 1 and w > 1:
            X = g[1:, 1:]  # noqa: N806
            A = g[1:, :-1]  # noqa: N806
            B = g[:-1, 1:]  # noqa: N806
            C = g[:-1, :-1]  # noqa: N806
            # Predict B on a vertical edge (C==A but C!=B); else A.
            pred = torch.where((C == A) & (C != B), B, A)
            keep[1:, 1:] = pred != X
    else:
        g = tokens.view(h, w, D)

        if w > 1:
            keep[0, 1:] = ~_patches_equal(g[0:1, 1:], g[0:1, :-1])[0]
        if h > 1:
            keep[1:, 0] = ~_patches_equal(g[1:, 0:1], g[:-1, 0:1])[:, 0]
        if h > 1 and w > 1:
            X = g[1:, 1:]  # noqa: N806
            A = g[1:, :-1]  # noqa: N806
            B = g[:-1, 1:]  # noqa: N806
            C = g[:-1, :-1]  # noqa: N806
            use_b = _patches_equal(C, A) & ~_patches_equal(C, B)
            pred = torch.where(use_b.unsqueeze(-1), B, A)
            keep[1:, 1:] = ~_patches_equal(X, pred)

    return keep.flatten().nonzero(as_tuple=False)[:, 0]


@njit(cache=True, nogil=True, fastmath=True, boundscheck=False)
def _anchored_inner(g: np.ndarray, threshold: float) -> np.ndarray:
    """Numba-compiled inner loop for the anchored LOCO-I scan.

    Three L∞ reductions per cell, each with **early-exit** on first
    over-threshold dim (avoids scanning all D dims when the answer is
    already determined):

      * ``|C-A| > t``: pred is A regardless of cb, skip ``|C-B|`` entirely.
      * ``|x-pred| > t``: cell is kept, don't bother with max scalar.

    ``np.empty_like`` instead of ``g.copy()`` saves the upfront h·w·D
    memcpy; we write ``anchor[r, c]`` once per cell explicitly.
    ``nogil=True`` releases the GIL so multiple images can run in parallel
    through the renderer pool. ``fastmath`` lets LLVM reorder/vectorise
    the abs-and-compare loops freely (safe — L∞ is associative).
    """
    h, w, D = g.shape
    keep = np.zeros((h, w), dtype=np.bool_)
    keep[0, 0] = True
    anchor = np.empty_like(g)
    anchor[0, 0] = g[0, 0]

    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                continue

            # Top row: B falls back to A; left column: C falls back to B.
            A = anchor[r, c - 1] if c > 0 else anchor[r - 1, 0]
            B = anchor[r - 1, c] if r > 0 else A

            if r > 0 and c > 0:
                C = anchor[r - 1, c - 1]
                ca_violated = False
                for d in range(D):
                    if abs(C[d] - A[d]) > threshold:
                        ca_violated = True
                        break
                if ca_violated:
                    pred = A
                else:
                    # Pick B on a vertical edge (C≈A but C≠B); else A.
                    cb_violated = False
                    for d in range(D):
                        if abs(C[d] - B[d]) > threshold:
                            cb_violated = True
                            break
                    pred = B if cb_violated else A
            else:
                pred = A

            x = g[r, c]
            keep_this = False
            for d in range(D):
                if abs(x[d] - pred[d]) > threshold:
                    keep_this = True
                    break

            if keep_this:
                keep[r, c] = True
                # Store raw x as anchor (cell becomes its own anchor).
                for d in range(D):
                    anchor[r, c, d] = x[d]
            else:
                # Store pred (not x): downstream sees what the receiver sees.
                for d in range(D):
                    anchor[r, c, d] = pred[d]

    return keep


def _select_pred2d_anchored(
    tokens: torch.Tensor,
    h: int,
    w: int,
    threshold: float,
) -> torch.Tensor:
    """Anchored serial scan (threshold > 0); compares against reconstructed
    neighbours so per-cell error stays bounded by one threshold step."""
    device = tokens.device
    # numpy doesn't support bfloat16; also numba needs contiguous fp32.
    g = (
        tokens.to(torch.float32)
        .contiguous()
        .cpu()
        .numpy()
        .reshape(h, w, tokens.shape[-1])
    )
    keep = _anchored_inner(g, float(threshold))
    kept = np.flatnonzero(keep.ravel())
    return torch.from_numpy(kept).to(device=device, dtype=torch.long)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def compute_pixel_prune_keep_indices(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    threshold: float = 0.0,
) -> list[torch.Tensor]:
    """Per-image patch-level keep-indices for PixelPrune.

    Images only — each ``image_grid_thw`` row must have ``t == 1``. Caller
    decides the numerical space of ``pixel_values``: ``threshold == 0`` is
    invariant to any per-channel affine normalization (byte-hash compares
    bit-exact); ``threshold > 0`` is an L∞ distance in the same space as
    the tensor, so caller must keep them consistent.

    Returns one int64 patch-index tensor per image.
    """
    block_size = spatial_merge_size**2
    device = pixel_values.device

    merged_pv = pixel_values.reshape(-1, pixel_values.shape[-1] * block_size)

    keep_indices_list: list[torch.Tensor] = []
    offset = 0
    block_offsets = torch.arange(block_size, device=device, dtype=torch.long)
    for t, h, w in image_grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        if t != 1:
            raise ValueError(f"PixelPrune only supports images (t == 1), got t={t}.")
        length = h * w // block_size
        img_merged = merged_pv[offset : offset + length]
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size

        merged_keep = select_patches_pred2d(
            img_merged, merged_h, merged_w, threshold=threshold
        )
        patch_indices = (
            merged_keep.unsqueeze(1) * block_size + block_offsets
        ).flatten()
        keep_indices_list.append(patch_indices)
        offset += length

    return keep_indices_list
