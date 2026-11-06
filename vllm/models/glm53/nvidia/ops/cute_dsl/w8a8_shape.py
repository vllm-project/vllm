# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counts, workspace layout and ``Flags`` map of the GLM-5.3 W8A8 MoE row tiers."""

from dataclasses import dataclass

GRID_CTAS = 148
LOW_CTAS = 128
EXPERTS = 256
TOP_K = 8
HIDDEN = 6144
SHARD = 256
K_BLOCK = 128
K_BLOCKS = HIDDEN // K_BLOCK
SLICE = 64
SLICES = SHARD // SLICE
TILE_M = 2 * SLICE
A_STAGE = TILE_M * K_BLOCK
DOWN_GROUPS = 3
DOWN_ROW_BLOCK = 128
DOWN_CTAS = (HIDDEN // DOWN_ROW_BLOCK) * DOWN_GROUPS
OWNER_ROWS = 48
TIER_KSPLIT = {16: 2}
TIERS = tuple(TIER_KSPLIT)


def _align(offset: int, alignment: int = 1024) -> int:
    return -(-offset // alignment) * alignment


@dataclass(frozen=True)
class TierShape:
    """Counts and offsets of the tier with row capacity ``rmax`` (host and kernel)."""

    rmax: int
    seam: bool = False

    @property
    def ksplit(self) -> int:
        return TIER_KSPLIT[self.rmax]

    @property
    def pairs(self) -> int:
        return TOP_K * self.rmax

    @property
    def max_union(self) -> int:
        return min(self.pairs, EXPERTS)

    @property
    def slots(self) -> int:
        return 1 + self.max_union

    @property
    def up_partials(self) -> int:
        return 0

    @property
    def hq(self) -> int:
        return _align(self.slots * self.ksplit * 2 * SHARD * self.rmax * 4)

    @property
    def token_scale(self) -> int:
        return _align(self.hq + self.slots * self.rmax * SHARD)

    @property
    def down_partials(self) -> int:
        return _align(self.token_scale + self.slots * self.rmax * 2 * 4)

    @property
    def image(self) -> int:
        return _align(self.down_partials + DOWN_GROUPS * self.rmax * HIDDEN * 4)

    @property
    def image_scales(self) -> int:
        return _align(self.image + self.rmax * HIDDEN)

    @property
    def scores(self) -> int:
        return _align(self.image_scales + K_BLOCKS * self.rmax * 4)

    @property
    def corrected(self) -> int:
        return self.scores + self.rmax * EXPERTS * 4

    @property
    def work_bytes(self) -> int:
        return _align(self.corrected + self.rmax * EXPERTS * 4)

    @property
    def flag_router(self) -> int:
        return 0

    @property
    def flag_image(self) -> int:
        return GRID_CTAS

    @property
    def flag_counter(self) -> int:
        return 2 * GRID_CTAS

    @property
    def flag_ready(self) -> int:
        return self.flag_counter + 2 * self.slots

    @property
    def flag_down(self) -> int:
        return self.flag_ready + 2 * self.slots

    @property
    def flag_union(self) -> int:
        return _align(self.flag_down + HIDDEN // DOWN_ROW_BLOCK, 4)

    @property
    def flag_words(self) -> int:
        return _align(self.flag_union + 4 + self.slots, 128)

    @property
    def mailbox_bytes(self) -> int:
        return (8 if self.seam else 7) * 2 * 128 * self.rmax * 128

    @property
    def ring_stage(self) -> int:
        return A_STAGE + self.rmax * K_BLOCK

    @property
    def down_scales(self) -> int:
        return 2 * A_STAGE + 2 * self.rmax * K_BLOCK

    @property
    def down_stage(self) -> int:
        return _align(2 * A_STAGE + 2 * self.rmax * K_BLOCK + self.rmax * 2 * 4, 1024)
