# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dsa_attn``'s fold, host side: a logical top-k token ``tok`` to its flat KV
row, the converter's ``bt[tok // bs] * bsr + tok % bs``, at any block size
(the packed DSA block is no power of 2: e.g. 1152). The block size is a
launch's (the engine sets it after the builds), so the kernel divides by a
multiply: ``tok // bs = (tok * m) >> s``, ``m`` and ``s`` from ``fold_word``.

Exact for every 0 <= tok < 2^31: with L = ceil(log2 bs), s = 31 + L and
m = ceil(2^s / bs) = (2^s + e) / bs, 0 <= e < bs <= 2^L, tok * m / 2^s is
tok / bs plus tok e / (bs 2^s) < 2^31 2^L / (bs 2^(31 + L)) = 1 / bs, and
tok / bs's fraction is at most (bs - 1) / bs: the floor is tok // bs. m is
2^31 for a power of 2, else 2^s / bs < 2^32 - 1 (bs > 2^(L - 1), L <= 26): m
fits 32 bits and |tok m| < 2^63, one i64 multiply and an arithmetic shift.
A negative tok gives a junk block, masked as before (``tok >= 0``).

No flydsl here: the CPU tests check the math (``fold_slot``).
"""

BS_BITS = 26  # the block size's bits in the fold word (bs < 2^26)
BS_POS = 32  # the block size's first bit; m the bits below it
LG_POS = BS_POS + BS_BITS  # L's first bit (5 bits: L <= 26; bit 63 clear)


def fold_word(bs: int) -> int:
    """The fold's i64 ``bsm`` word: m | bs << 32 | L << 58 (m < 2^32, bs <
    2^26, L = ceil(log2 bs) <= 26: 63 bits, the word >= 0)."""
    assert 0 < bs < 1 << BS_BITS, bs
    lg = (bs - 1).bit_length()
    m = -(-(1 << (31 + lg)) // bs)
    assert m < 1 << BS_POS, (bs, m)
    return m | bs << BS_POS | lg << LG_POS


def fold_unpack(word: int) -> tuple[int, int, int]:
    """(m, bs, s) of a ``fold_word``, as the kernel unpacks it."""
    m = word & ((1 << BS_POS) - 1)
    bs = (word >> BS_POS) & ((1 << BS_BITS) - 1)
    return m, bs, (word >> LG_POS) + 31


def fold_slot(tok: int, bt, bsr: int, word: int, invalid: int = 0) -> int:
    """``dsa_attn``'s fold of token ``tok`` (an int32) in integers, op for op:
    the slot in block table ``bt`` (``len(bt)`` blocks) of pitch ``bsr``;
    ``invalid`` for a negative or out-of-table token."""
    m, bs, s = fold_unpack(word)
    prod = tok * m
    assert -(1 << 63) <= prod < 1 << 63  # the kernel's i64 product
    blk = prod >> s
    if tok < 0 or blk >= len(bt):
        return invalid
    return bt[blk] * bsr + (tok - blk * bs)
