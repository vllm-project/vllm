# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External-store cache-hit coordinator for MooncakeStoreConnector."""

from collections.abc import Sequence
from typing import cast

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    chunk_hashes_for_block_size,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    replay_boundary,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by a ``(group_id, hash)`` exists set."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        # ``exists=None`` is used on the recv side where hit_length is already
        # determined and we just want each spec's manager to apply its own mask.
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        # Dummy ID 1 for present block for duck-typing.
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        # Mirrors BlockPool.get_cached_block: hit only when every group_id
        # (groups sharing a spec) has the hash cached.
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None

    def contains(self, group_id: int, block_hash: BlockHash) -> bool:
        """Return whether a group has a loadable key for this hash."""
        return (
            self._exists is not None and (group_id, bytes(block_hash)) in self._exists
        )


class MooncakeStoreCoordinator:
    """Mirror of ``HybridKVCacheCoordinator.find_longest_cache_hit`` over an
    ``ExternalCachedBlockPool``."""

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
        dcp_world_size: int = 1,
    ) -> None:
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0 for g in kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.mamba_group_ids = {
            group_id
            for group_id, group in enumerate(kv_cache_groups)
            if isinstance(_unwrap_spec(group.kv_cache_spec), MambaSpec)
        }
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.enable_partial_hash_hits = partial_hash_hits_enabled(
            kv_cache_groups, hash_block_size, dcp_world_size
        )
        self.use_eagle = use_eagle
        # Mirror vLLM core's KVCacheCoordinator.retention_interval.
        self.retention_interval = retention_interval
        self._verify_and_split_kv_cache_groups()

    def align_lookup_length(self, length: int) -> int:
        alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        return length // alignment * alignment

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Mirrors KVCacheCoordinator.verify_and_split_kv_cache_groups but
        dispatches via spec_manager_map (we don't allocate managers).
        """
        attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for idx, group in enumerate(attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls
                    group.group_ids.append(i)
                    if g.is_eagle_group and not group.use_eagle:
                        attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, g.is_eagle_group)
                )
        # Full attention first (matches upstream convergence ordering).
        attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))
        # Conservatively flag all groups when use_eagle is set but none is flagged.
        if self.use_eagle and not any(g.use_eagle for g in attention_groups):
            attention_groups = [g._replace(use_eagle=True) for g in attention_groups]
        self.attention_groups = attention_groups
        # Per-group eagle bits. SpecGroup carries use_eagle for the whole
        # merged spec group, so the per-group store/lookup masks agree with
        # the merged-group hit check, which applies the eagle drop to every
        # group sharing the spec.
        self.eagle_group_ids = {
            gid for g in attention_groups if g.use_eagle for gid in g.group_ids
        }

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[bool], ...], int]:
        """Returns ``(load_mask_per_group, hit_length)``. ``mask[g][i]`` is True iff
        group ``g`` populates chunk ``i`` locally (e.g. SWA and Mamba tail-only);
        recv-side callers skip False slots.

        ``apply_eagle`` controls whether the per-spec ``use_eagle`` last-block
        pop is applied. Lookup callers want it (the drafter requires recomputing
        the last block); per-chunk mask callers must not, because ``token_len``
        already reflects the eagle-pruned hit length and a second pop would
        leave the trailing block unloaded.
        """
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes, max_length, cached_block_pool, apply_eagle=apply_eagle
        )
        masks = tuple(
            [blk is not cached_block_pool.null_block for blk in blocks]
            for blocks in blocks_per_group
        )
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: Sequence[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        """Per-group load masks: ``mask[g][i]`` is True iff group ``g``'s
        spec would populate chunk ``i`` locally at length ``token_len``
        (e.g. SWA / Mamba tail-only).
        """
        # ``apply_eagle=False`` because ``token_len`` is already the
        # eagle-pruned hit length returned by ``client.lookup``. Re-applying
        # the pop here would shorten the mask by one extra block; the recv
        # thread would then silently skip the trailing chunk yielded by
        # ``db.process_tokens`` and leave that block uninitialized in the
        # local KV pool.
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(self.hash_block_size),
            apply_eagle=False,
        )
        return masks

    def store_mask(
        self,
        aligned_token_len: int,
        start_token: int = 0,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group store masks for the suffix starting at ``start_token``.

        ``mask[g][i]`` is True iff the i-th chunk of group ``g`` *after*
        ``start_token`` should be written to the store so a future cache hit
        can consume it. ``None`` is the all-True sentinel for the suffix.

        Reuses the engine's ``SingleTypeKVCacheManager.reachable_block_mask``
        so the store retains exactly the blocks the local prefix cache would.

        Mamba groups are always all-False: the normal save resolves blocks
        positionally from the connector's append-only block-ID snapshot, but
        an align-mode mamba block table is not append-only (interior state
        blocks are nulled/freed, and speculative decoding relocates the spec
        blocks in place), so a positional read may hit a null, freed, or live
        speculative-state block and persist wrong bytes under a valid prefix
        hash. Mamba state is persisted only through the connector-pinned exact
        block hand-off path
        (``SchedulerOutput.kv_connector_block_state.boundary_state_offloads``).
        """
        return self._reachable_masks(
            aligned_token_len,
            start_token,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
            exclude_mamba=True,
        )

    def lookup_mask(
        self,
        aligned_token_len: int,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group lookup masks.

        ``mask[g][i]`` is True iff chunk ``i`` of group ``g`` should be
        looked up as an aligned hit boundary. ``None`` is the all-True
        sentinel.
        """
        return self._reachable_masks(
            aligned_token_len,
            0,
            retention_interval=None,
            num_prompt_tokens=None,
        )

    def _reachable_masks(
        self,
        aligned_token_len: int,
        start_token: int,
        *,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
        exclude_mamba: bool = False,
    ) -> tuple[list[bool] | None, ...]:
        mask_alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        assert aligned_token_len % mask_alignment == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"{mask_alignment}"
        )
        # Model-level, so it is computed once and shared by every group; the
        # store's retention must name the position the engine resumes at.
        reachable_boundaries = (
            ()
            if num_prompt_tokens is None
            else (
                replay_boundary(
                    num_prompt_tokens, self.lcm_block_size, bool(self.eagle_group_ids)
                ),
            )
        )
        masks: list[list[bool] | None] = []
        for g_idx, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            end_chunk = aligned_token_len // spec.block_size
            start_chunk = min(end_chunk, max(0, cdiv(start_token, spec.block_size)))
            if exclude_mamba and isinstance(spec, MambaSpec):
                masks.append([False] * (end_chunk - start_chunk))
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None
            use_eagle = g_idx in self.eagle_group_ids
            mask = manager_cls.reachable_block_mask(
                start_block=start_chunk,
                end_block=end_chunk,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                use_eagle=use_eagle,
                retention_interval=retention_interval,
                reachable_boundaries=reachable_boundaries,
            )
            if mask is not None:
                assert len(mask) == end_chunk - start_chunk
            masks.append(mask)
        return tuple(masks)

    def block_hashes_for_spec(
        self, block_hashes: Sequence[BlockHash], spec: KVCacheSpec
    ) -> Sequence[BlockHash]:
        return chunk_hashes_for_block_size(
            block_hashes, self.hash_block_size, spec.block_size
        )

    def _find_hit_blocks(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Mirrors HybridKVCacheCoordinator.find_longest_cache_hit but
        dispatches via spec_manager_map (we don't allocate managers).

        When ``apply_eagle`` is False, ignore each group's ``use_eagle`` —
        used by ``load_mask`` to avoid popping a second block on top of the
        one already removed by the lookup.
        """
        alignment_tokens = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls, group_eagle = self.attention_groups[0]
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,  # type: ignore[arg-type]
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=apply_eagle and group_eagle,
                alignment_tokens=alignment_tokens,
            )
            num_groups = len(self.kv_cache_groups)
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
            for gid, blks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[gid] = blks
            return tuple(blocks_by_group), hit_length

        num_groups = len(self.kv_cache_groups)
        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, group_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = (
                    apply_eagle and group_eagle and idx not in eagle_verified
                )
                _max_length = curr_hit_length
                # No eagle peek margin for a recurrent (Mamba) group: its finder
                # never drops a block, so a widened bound would match past the
                # attention-verified hit and resume from speculative state (#43559).
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and spec.block_size > self.hash_block_size
                        else spec.block_size
                    )
                    _max_length = min(curr_hit_length + eagle_margin, max_length)
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,  # type: ignore[arg-type]
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=alignment_tokens,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for gid, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[gid] = blocks
                    hit_length_by_group[gid] = _new_hit_length

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full-attention hit_blocks to final converged length;
        # other specs already trim themselves inside their hit logic. cdiv keeps
        # the partial tail block when hit_length is not block-aligned.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(hit_length, group.spec.block_size)
            for group_id in group.group_ids:
                full_blks = hit_blocks_by_group[group_id]
                assert full_blks is not None
                del full_blks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

        return (
            tuple(blks if blks is not None else [] for blks in hit_blocks_by_group),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def partial_hash_hits_enabled(
    kv_cache_groups: list[KVCacheGroupSpec],
    hash_block_size: int,
    dcp_world_size: int = 1,
) -> bool:
    """Match core's DCP-aware Mamba partial-hit condition."""
    return any(
        isinstance(spec := _unwrap_spec(g.kv_cache_spec), MambaSpec)
        and spec.mamba_cache_mode == "align"
        and (
            (dcp_world_size == 1 and spec.block_size > hash_block_size)
            or (dcp_world_size > 1 and spec.block_size >= hash_block_size)
        )
        for g in kv_cache_groups
    )
