# SPDX-License-Identifier: Apache-2.0
"""Fold / unfold logic for multi-object-group prefix-cache hit computation.

With hybrid models, an LMCache request is split across several *object groups*
(full attention, sliding window, mamba, ...), each stored as its own
``MemoryObj`` and each with a different token-dependency rule:

- **full attention** needs every chunk of the prefix present to serve it;
- a **sliding window of ``w`` chunks** only needs the last ``w`` chunks of the
  prefix present (mamba is the ``w == 1`` case).

A model-wide prefix-cache hit of length ``L`` requires *every* object group to
be able to serve a prefix of length ``L`` under its own rule. This module turns
the per-group presence bitmaps into that single answer in three steps:

1. **fold** — combine the per-group presence into a servable bitmap: bit ``j``
   is set iff every group can serve a length-``j + 1`` prefix;
2. **highest-set-bit** — the model-wide hit length is the highest set bit plus
   one (``-1`` -> 0, no hit);
3. **unfold** — expand the hit length into the concrete chunks each group needs
   (``unfold_range``), producing the retain mask used to load / lock / transfer.

When every group is full attention the servable set is a downward-closed prefix,
so the hit length equals the leading-ones count of the AND of the per-group
presences -- i.e. the plain ``TrimPolicy.PREFIX`` / require-all intersection.
Fold/unfold is a strict generalization of that behavior.

Bitmaps here are laid out **group-major**: bit ``g * num_chunks + j`` is set iff
chunk ``j`` is available for object group ``g``.
"""

# Standard
from collections.abc import Iterable, Sequence

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.native_storage_ops import fold as _native_fold
from lmcache.native_storage_ops import unfold as _native_unfold
from lmcache.v1.distributed.api import TrimPolicy

FULL_ATTENTION_WINDOW = -1
"""Sentinel ``group_windows`` value marking a full-attention object group
(needs the whole prefix). Any value ``<= 0`` is treated as full attention."""


def unfold_range(prefix_len: int, window: int) -> tuple[int, int]:
    """Chunk range ``[lo, hi)`` one object group needs to serve a prefix.

    Full attention needs ``[0, prefix_len)``; a window of ``w`` chunks needs
    ``[max(0, prefix_len - w), prefix_len)``.

    Args:
        prefix_len: model-wide hit length in chunks.
        window: cross-chunk sliding-window size in chunks; ``<= 0`` means full
            attention.

    Returns:
        The half-open chunk range this group must keep. ``(0, 0)`` when
        ``prefix_len <= 0``.
    """
    if prefix_len <= 0:
        return (0, 0)
    lo = 0 if window <= 0 else max(0, prefix_len - window)
    return (lo, prefix_len)


def fold(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Fold per-(group, chunk, rank) presence into servable prefix lengths.

    For each object group, computes which prefix lengths it can serve under its
    rule (a length-``L`` prefix needs the last ``min(window, L)`` chunks
    present), and intersects across groups. A chunk is present for a group only
    when **every** kv_rank shard is present.

    Args:
        found: presence bitmap of length
            ``len(group_windows) * num_chunks * num_ranks``; bit
            ``g * (num_chunks * num_ranks) + j * num_ranks + r`` set iff chunk
            ``j`` of object group ``g`` is present for kv_rank ``r``.
        num_chunks: number of LMCache chunks in the request.
        num_ranks: number of kv_rank shards per chunk (``world_size`` at lookup).
        group_windows: per-object-group cross-chunk sliding-window size in
            chunks, in object-group order; ``<= 0`` means full attention.

    Returns:
        A bitmap of size ``num_chunks``; bit ``j`` set iff every group can
        serve a length-``j + 1`` prefix. Feed it to :func:`highest_set_bit`;
        the model-wide hit length is that index plus one (``-1`` -> 0).

    Raises:
        ValueError: If ``group_windows`` is empty, ``num_chunks`` is negative,
            or ``num_ranks`` is not positive.
    """
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    return _native_fold(found, num_chunks, num_ranks, list(group_windows))


def highest_set_bit(servable: Bitmap) -> int:
    """Model-wide prefix hit length: the highest set bit of a fold output.

    Args:
        servable: A servable-lengths bitmap from :func:`fold`.

    Returns:
        The highest set bit index. On :func:`fold`'s output this is the
        model-wide hit length, always ``>= 0`` because :func:`fold` always sets
        bit 0 (the empty prefix is servable by every group).

    Raises:
        ValueError: If ``servable`` has no set bit. This cannot occur on
            :func:`fold`'s output, which always sets bit 0.
    """
    return servable.highest_set_bit()


def unfold(
    hit_length: int,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Expand a model-wide hit length into the per-group retain mask.

    Each group retains the chunks it needs to serve ``hit_length``:
    ``[0, hit_length)`` for full attention, ``[hit_length - window, hit_length)``
    for a sliding window, with **all** kv_ranks of each retained
    ``(group, chunk)`` set.

    Args:
        hit_length: model-wide prefix hit length in chunks (clamped to
            ``num_chunks``).
        num_chunks: number of LMCache chunks in the request.
        num_ranks: number of kv_rank shards per chunk.
        group_windows: per-object-group cross-chunk sliding-window size in
            chunks, in object-group order; ``<= 0`` means full attention.

    Returns:
        A retain mask over the same ranked layout as :func:`fold`'s input,
        length ``len(group_windows) * num_chunks * num_ranks``.

    Raises:
        ValueError: If ``group_windows`` is empty, ``num_chunks`` is negative,
            or ``num_ranks`` is not positive.
    """
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    return _native_unfold(hit_length, num_chunks, num_ranks, list(group_windows))


def fold_unfold_ranked(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> tuple[int, Bitmap]:
    """Compose :func:`fold` -> :func:`highest_set_bit` -> :func:`unfold`.

    Convenience for the full pipeline over the ``group x chunk x kv_rank``
    lookup key layout: the model-wide hit length and the keys each group must
    retain to serve it.

    Args:
        found: presence bitmap (see :func:`fold`).
        num_chunks: number of LMCache chunks in the request.
        num_ranks: number of kv_rank shards per chunk.
        group_windows: per-object-group cross-chunk window sizes.

    Returns:
        ``(hit_length, retain_mask)`` over the same ranked layout as ``found``.
    """
    servable = fold(found, num_chunks, num_ranks, group_windows)
    # fold's bits are chunk-indexed (bit j == prefix length j + 1), so the hit
    # length is the highest set bit plus one; -1 (no servable prefix) -> 0.
    hit_length = highest_set_bit(servable) + 1
    return hit_length, unfold(hit_length, num_chunks, num_ranks, group_windows)


def fold_unfold(
    found: Bitmap,
    num_chunks: int,
    group_windows: Sequence[int],
) -> tuple[int, Bitmap]:
    """:func:`fold_unfold_ranked` for the single-rank (group-major) layout.

    Args:
        found: group-major presence bitmap of length
            ``len(group_windows) * num_chunks``; bit ``g * num_chunks + j`` set
            iff chunk ``j`` is available for object group ``g``.
        num_chunks: number of LMCache chunks in the request.
        group_windows: per-object-group cross-chunk window sizes.

    Returns:
        ``(hit_length, retain_mask)`` over the group-major layout.
    """
    return fold_unfold_ranked(found, num_chunks, 1, group_windows)


def _fold_python(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Reference (pure-Python) implementation of :func:`fold`, for testing only.

    Kept as the oracle the native :func:`fold` is validated against in the test
    suite; it is not used at runtime. Behavior matches :func:`fold`.

    Args:
        found: Group-major / chunk-major / rank-minor presence bitmap.
        num_chunks: Number of LMCache chunks in the request.
        num_ranks: Number of kv_rank shards per chunk.
        group_windows: Per-object-group cross-chunk window size in chunks;
            ``<= 0`` means full attention.

    Returns:
        A bitmap of size ``num_chunks``; bit ``j`` set iff every group can
        serve a length-``j + 1`` prefix.
    """
    group_stride = num_chunks * num_ranks

    # For each group, ``run`` is the count of consecutive present chunks ending
    # at the current chunk, so a length-L prefix is servable iff the last
    # ``min(window, L)`` chunks are present. ``servable[j]`` (bit ``j``, prefix
    # length ``j + 1``) stays True only if every group can serve that length.
    servable = [True] * num_chunks
    for group_idx, window in enumerate(group_windows):
        gbase = group_idx * group_stride
        effective_window = num_chunks if window <= 0 else window
        run = 0
        for prefix_len in range(1, num_chunks + 1):
            cbase = gbase + (prefix_len - 1) * num_ranks
            present = all(found.test(cbase + r) for r in range(num_ranks))
            run = run + 1 if present else 0
            if servable[prefix_len - 1] and run < min(effective_window, prefix_len):
                servable[prefix_len - 1] = False

    servable_lengths = Bitmap(num_chunks)
    for j in range(num_chunks):
        if servable[j]:
            servable_lengths.set(j)
    return servable_lengths


def _unfold_python(
    hit_length: int,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Reference (pure-Python) implementation of :func:`unfold`, for testing only.

    Kept as the oracle the native :func:`unfold` is validated against in the
    test suite; it is not used at runtime. Behavior matches :func:`unfold`.

    Args:
        hit_length: Model-wide prefix hit length in chunks (clamped to
            ``num_chunks``).
        num_chunks: Number of LMCache chunks in the request.
        num_ranks: Number of kv_rank shards per chunk.
        group_windows: Per-object-group cross-chunk window size in chunks;
            ``<= 0`` means full attention.

    Returns:
        Retain mask of length ``len(group_windows) * num_chunks * num_ranks``.
    """
    hit_length = min(hit_length, num_chunks)
    num_groups = len(group_windows)
    group_stride = num_chunks * num_ranks
    retain_mask = Bitmap(num_groups * group_stride)
    if hit_length <= 0:
        return retain_mask
    for group_idx, window in enumerate(group_windows):
        lo, hi = unfold_range(hit_length, window)
        gbase = group_idx * group_stride
        for j in range(lo, hi):
            cbase = gbase + j * num_ranks
            for r in range(num_ranks):
                retain_mask.set(cbase + r)
    return retain_mask


def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge bitmaps with a bitwise OR into a ``num_keys``-sized bitmap.

    Always returns a ``num_keys``-sized bitmap (empty input -> all zeros), so
    downstream ``&`` operations never hit a size mismatch.

    Args:
        bitmaps: Per-source presence bitmaps to union.
        num_keys: Size of the merged bitmap.

    Returns:
        The bitwise-OR of all inputs as a ``num_keys``-sized bitmap.
    """
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


def select_retained(
    found: Bitmap,
    num_keys: int,
    policy: TrimPolicy = TrimPolicy.PREFIX,
) -> Bitmap:
    """Select the retained subset of ``found`` for the non-windowed selections.

    ``PREFIX`` (LONGEST) keeps the leading contiguous run and drops everything
    from the first gap on; any other policy keeps every set bit, gaps included.
    The windowed hybrid fold is handled by :func:`fold_unfold_ranked`, not here.

    Args:
        found: Bitmap of found keys, over key indices ``0..num_keys-1``.
        num_keys: Total number of requested keys.
        policy: Selection to apply (see :class:`TrimPolicy`).

    Returns:
        Bitmap of the retained key indices.
    """
    if policy is TrimPolicy.PREFIX:
        return Bitmap(num_keys, found.count_leading_ones())
    return found
