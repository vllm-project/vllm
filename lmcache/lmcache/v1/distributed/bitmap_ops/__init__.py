# SPDX-License-Identifier: Apache-2.0
"""Bitmap arithmetic for multi-object-group prefix-cache hit computation.

See :mod:`~lmcache.v1.distributed.bitmap_ops.fold` and the
package ``README.md`` for the fold -> highest-set-bit -> unfold design.
"""

# First Party
from lmcache.v1.distributed.bitmap_ops.fold import (
    FULL_ATTENTION_WINDOW,
    fold,
    fold_unfold,
    fold_unfold_ranked,
    highest_set_bit,
    merge_bitmaps,
    select_retained,
    unfold,
    unfold_range,
)

__all__ = [
    "FULL_ATTENTION_WINDOW",
    "highest_set_bit",
    "fold",
    "fold_unfold",
    "fold_unfold_ranked",
    "merge_bitmaps",
    "select_retained",
    "unfold",
    "unfold_range",
]
