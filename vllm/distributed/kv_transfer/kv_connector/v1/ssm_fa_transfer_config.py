# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSM+FA transfer configuration for hetero-TP NIXL transfers.

Single source of truth for descriptor sizes and transfer targets when
Prefill and Decode engines use different tensor-parallel sizes with
hybrid SSM+Attention models (Mamba + FlashAttention).

One instance per (D rank, P engine) handshake pair.  Queried by:
  - Local handle creation  (add_remote_agent → Path C)
  - Remote descriptor registration (register_remote_blocks)
  - Transfer loop (_read_blocks_for_req)

Design reference:
  my_wip/ssm-fa-nixl-support/notes/design/4p1d_hma_fix_20260326.md

Key insight:  FA and Mamba require *different* numbers of P ranks when
P is replicated for FA but always TP-sharded for Mamba.

  Config   fa_reads  mamba_reads  same?
  2p1d     2         2            yes
  4p1d     2         4            NO
  4p2d     1         2            NO
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _physical_head_range(tp_size: int, num_heads: int, rank: int) -> range:
    """Physical KV head range stored in a rank's KV cache tensor.

    When ``tp_size <= num_heads``: sharded, K/TP contiguous heads per rank.
    When ``tp_size > num_heads``: 1 physical head per rank.  Heads are
    distributed **contiguously** (matching vLLM's GQA weight partitioning):
    consecutive ranks share a head before moving to the next one.
    """
    if tp_size <= num_heads:
        assert num_heads % tp_size == 0
        per_rank = num_heads // tp_size
        return range(rank * per_rank, (rank + 1) * per_rank)
    else:
        h = rank * num_heads // tp_size
        return range(h, h + 1)


def _range_overlap(a: range, b: range) -> range:
    start = max(a.start, b.start)
    stop = min(a.stop, b.stop)
    return range(start, max(start, stop))


@dataclass
class HeteroTPTransferConfig:
    """Precomputed transfer plan for one (D rank, P engine) pair.

    All descriptor sizes are computed here.  The guarantee is:
        local_entry_size == remote_entry_size   (for NIXL)

    Attributes that start with ``fa_`` concern FlashAttention KV cache.
    Attributes that start with ``mamba_`` concern Mamba conv/SSM state.
    """

    # ---- Input parameters (from handshake) ----
    tp_ratio: int
    K: int  # total_num_kv_heads (before TP sharding)
    d_tp: int  # D engine's tensor_parallel_size
    p_tp: int  # P engine's tensor_parallel_size
    d_rank: int  # this D worker's TP rank
    has_mamba: bool
    use_mla: bool

    # Per-layer block lengths (bytes, K+V combined for blocks_first).
    # Uniform across layers for current models.
    d_block_len: int  # D's block_len_per_layer (representative)
    p_block_len: int  # P's block_len_per_layer (from handshake)
    is_blocks_first: bool  # kv_topo.is_kv_layout_blocks_first

    # ---- Derived: computed in __post_init__ ----
    #
    # Physical heads per rank (what the KV tensor actually stores)
    d_physical_heads: int = field(init=False)
    p_physical_heads: int = field(init=False)

    # How many distinct P ranks D needs for FA data
    physical_fa_num_reads: int = field(init=False)

    # Which P ranks contribute unique FA heads (ordered by head index)
    fa_read_targets: list[int] = field(init=False)

    # All P ranks needed for mamba (always abs_tp for tp_ratio < 0)
    mamba_num_reads: int = field(init=False)

    # All P ranks this D rank communicates with (FA ∪ mamba)
    transfer_targets: list[int] = field(init=False)

    # FA descriptor entry size (K or V side, for blocks_first layout)
    # Guaranteed: fa_entry_size is the SAME for local handle AND remote desc.
    fa_entry_size: int = field(init=False)

    # Replication flags
    is_d_replicated: bool = field(init=False)
    is_p_replicated: bool = field(init=False)

    # Pre-built set for fast lookup
    _fa_target_set: frozenset[int] = field(init=False, repr=False)
    # Map: P rank → index in fa_read_targets (for head slot offset)
    _fa_target_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        K = self.K
        self.is_d_replicated = self.d_tp > K
        self.is_p_replicated = self.p_tp > K

        self.d_physical_heads = max(1, K // self.d_tp)
        self.p_physical_heads = max(1, K // self.p_tp)

        abs_tp = -self.tp_ratio if self.tp_ratio < 0 else 1

        # ---- Mamba range (computed first so FA can prefer ranks in it) ----
        mamba_range: range | None = None
        if self.has_mamba and self.tp_ratio < 0:
            mamba_range = range(self.d_rank * abs_tp, (self.d_rank + 1) * abs_tp)

        # ---- FA read targets ----
        if self.use_mla or self.tp_ratio >= 0:
            self.physical_fa_num_reads = 1
            self.fa_read_targets = (
                [0]
                if self.use_mla
                # OLD: [self.d_rank % self.p_tp if self.tp_ratio > 0 ...]
                # Must match kv_topo.get_target_remote_ranks which uses
                # d_rank // tp_ratio.  The old d_rank % p_tp gave wrong
                # targets for 2p4d (D_TP=4, P_TP=2, K=2) and similar.
                else [
                    self.d_rank // self.tp_ratio if self.tp_ratio > 0 else self.d_rank
                ]
            )
        else:
            d_needs = _physical_head_range(self.d_tp, K, self.d_rank)
            # When mamba range exists, prefer P ranks within it so that
            # FA targets are a subset of mamba transfer_targets (avoids
            # orphaned FA targets outside the transfer loop).
            search_range = mamba_range if mamba_range is not None else range(self.p_tp)
            seen: set[tuple[int, int]] = set()
            targets: list[int] = []
            for p in search_range:
                p_has = _physical_head_range(self.p_tp, K, p)
                ov = _range_overlap(d_needs, p_has)
                if len(ov) > 0:
                    key = (ov.start, ov.stop)
                    if key not in seen:
                        seen.add(key)
                        targets.append(p)
            if not targets:
                # Fallback: search globally (should not happen in practice)
                for p in range(self.p_tp):
                    p_has = _physical_head_range(self.p_tp, K, p)
                    ov = _range_overlap(d_needs, p_has)
                    if len(ov) > 0:
                        key = (ov.start, ov.stop)
                        if key not in seen:
                            seen.add(key)
                            targets.append(p)
            self.fa_read_targets = targets
            self.physical_fa_num_reads = len(targets)

        self._fa_target_set = frozenset(self.fa_read_targets)
        self._fa_target_index = {r: i for i, r in enumerate(self.fa_read_targets)}

        # ---- Mamba targets ----
        if mamba_range is not None and abs_tp > self.physical_fa_num_reads:
            self.mamba_num_reads = abs_tp
            self.transfer_targets = list(mamba_range)
        else:
            self.mamba_num_reads = self.physical_fa_num_reads
            self.transfer_targets = list(self.fa_read_targets)

        # ---- FA entry size ----
        # For blocks_first: block_len_per_layer includes K+V; // 2 gives K (or V).
        # Use min(D, P) because D indexes into P when tp_ratio > 0,
        # and P is the natural unit when tp_ratio < 0.
        effective_block_len = min(self.d_block_len, self.p_block_len)
        if self.is_blocks_first:
            self.fa_entry_size = effective_block_len // 2
        else:
            self.fa_entry_size = effective_block_len

        self._validate()

    def _validate(self) -> None:
        """Cross-check internal consistency."""
        # OLD: Both-replicated guard — was needed because d_rank % p_tp
        # gave wrong FA targets.  Now fixed: fa_read_targets uses
        # d_rank // tp_ratio (matches kv_topo routing) and
        # fa_rank_offset uses relative head index.
        # if self.is_d_replicated and self.is_p_replicated and self.tp_ratio > 0:
        #     raise NotImplementedError(
        #         f"Both-replicated hetero-TP with D_TP ({self.d_tp}) > "
        #         f"P_TP ({self.p_tp}) > K ({self.K}) is not yet supported. "
        #         f"The FA target selection (d_rank % p_tp) does not account "
        #         f"for head placement when both sides replicate."
        #     )
        if self.is_d_replicated and self.is_p_replicated and self.tp_ratio > 0:
            logger.info(
                "Both-replicated hetero-TP: D_TP=%d > P_TP=%d > K=%d. "
                "Using d_rank // tp_ratio routing with relative head offset.",
                self.d_tp,
                self.p_tp,
                self.K,
            )

        # FA targets must be a subset of transfer_targets
        tt_set = set(self.transfer_targets)
        for t in self.fa_read_targets:
            if t not in tt_set:
                logger.error(
                    "FA target P rank %d is NOT in transfer_targets %s. "
                    "This will cause missed FA reads!",
                    t,
                    self.transfer_targets,
                )

        # For tp_ratio < 0 with blocks_first: D_K_half / reads should == P_K_half
        if (
            self.is_blocks_first
            and self.tp_ratio < 0
            and self.physical_fa_num_reads > 0
        ):
            d_k_half = self.d_block_len // 2
            p_k_half = self.p_block_len // 2
            expected_local = d_k_half // self.physical_fa_num_reads
            if expected_local != p_k_half:
                logger.warning(
                    "FA size mismatch: D_K_half=%d / reads=%d = %d, "
                    "but P_K_half=%d.  This may indicate a head count or "
                    "HMA inflation inconsistency.",
                    d_k_half,
                    self.physical_fa_num_reads,
                    expected_local,
                    p_k_half,
                )

    # ---- Query methods ----

    def should_skip_fa(self, p_rank: int) -> bool:
        """Whether to skip FA groups for this P rank (mamba-only transfer)."""
        return self.has_mamba and p_rank not in self._fa_target_set

    def fa_head_slot(self, p_rank: int) -> int:
        """Index into D's FA block for this P rank's head data.

        For P ranks in fa_read_targets, returns 0, 1, ..., reads-1.
        For P ranks NOT in fa_read_targets (replicated duplicates),
        returns the slot of the matching FA target with the same head.
        """
        if p_rank in self._fa_target_index:
            return self._fa_target_index[p_rank]
        # Duplicate head: find which fa_target has the same physical head
        p_head = _physical_head_range(self.p_tp, self.K, p_rank)
        for target in self.fa_read_targets:
            t_head = _physical_head_range(self.p_tp, self.K, target)
            if _range_overlap(p_head, t_head):
                return self._fa_target_index[target]
        return 0  # fallback

    @property
    def indexes_into_remote(self) -> bool:
        """Whether D indexes into a sub-slice of P's FA block.

        True only when both sides shard (neither replicates) and D_TP > P_TP.
        When either side replicates, offset logic must account for physical
        head placement rather than a simple tp_ratio slice.
        """
        return (
            not self.is_d_replicated
            and not self.is_p_replicated
            and not self.use_mla
            and self.tp_ratio > 0
        )

    def fa_rank_offset(self, remote_kv_block_len: int) -> int:
        """Byte offset into P's FA block for this D rank.

        When D is replicated (D_TP > K), multiple D ranks share a head.
        Computes offset *relative to the target P rank's first head*
        so it works regardless of how many heads P has.
        When neither side replicates, falls back to tp_rank % tp_ratio.
        Returns 0 when D does not index into P's block.
        """
        if self.use_mla or self.tp_ratio <= 0:
            return 0
        if self.is_d_replicated:
            # OLD: head_idx = self.d_rank * self.K // self.d_tp
            #      return head_idx * remote_kv_block_len
            # Bug: used absolute head index, which is out-of-bounds
            # when P has fewer heads than head_idx (e.g. 2p4d P1
            # has 1 head but head_idx=1 gives offset=block_len).
            d_head = self.d_rank * self.K // self.d_tp
            p_rank = self.fa_read_targets[0]
            p_start = p_rank * self.K // self.p_tp
            return (d_head - p_start) * remote_kv_block_len
        return self.d_rank % self.tp_ratio * remote_kv_block_len

    @property
    def needs_split_handles(self) -> bool:
        """Whether per-P-rank split handles are needed.

        True when FA and mamba have different read counts, requiring
        different splitting factors in the local handle.
        """
        return (
            self.tp_ratio < 0
            and self.has_mamba
            and not self.use_mla
            and len(self.transfer_targets) > 1
        )

    def describe(self) -> str:
        """One-line summary for logging."""
        return (
            f"HeteroTPTransferConfig("
            f"tp_ratio={self.tp_ratio}, K={self.K}, "
            f"d_tp={self.d_tp}, p_tp={self.p_tp}, d_rank={self.d_rank}, "
            f"physical_fa_reads={self.physical_fa_num_reads}, "
            f"mamba_reads={self.mamba_num_reads}, "
            f"fa_targets={self.fa_read_targets}, "
            f"transfer_targets={self.transfer_targets}, "
            f"fa_entry_size={self.fa_entry_size}, "
            f"d_block_len={self.d_block_len}, p_block_len={self.p_block_len})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_transfer_config(
    *,
    tp_ratio: int,
    total_num_kv_heads: int,
    d_tp: int,
    p_tp: int,
    d_rank: int,
    has_mamba: bool,
    use_mla: bool,
    d_block_len: int,
    p_block_len: int,
    is_blocks_first: bool,
) -> HeteroTPTransferConfig:
    """Create a transfer config for one (D rank, P engine) handshake."""
    cfg = HeteroTPTransferConfig(
        tp_ratio=tp_ratio,
        K=total_num_kv_heads,
        d_tp=d_tp,
        p_tp=p_tp,
        d_rank=d_rank,
        has_mamba=has_mamba,
        use_mla=use_mla,
        d_block_len=d_block_len,
        p_block_len=p_block_len,
        is_blocks_first=is_blocks_first,
    )
    logger.info("Created %s", cfg.describe())
    return cfg
