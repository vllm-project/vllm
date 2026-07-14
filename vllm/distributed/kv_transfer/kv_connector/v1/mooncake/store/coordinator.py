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
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
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
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.use_eagle = use_eagle
        # Mirror vLLM core's KVCacheCoordinator.retention_interval.
        self.retention_interval = retention_interval
        self.eagle_group_ids = {
            i for i, g in enumerate(kv_cache_groups) if g.is_eagle_group
        }
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_groups)))
        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Mirrors KVCacheCoordinator.verify_and_split_kv_cache_groups but
        dispatches via spec_manager_map (we don't allocate managers).
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []
        for i, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))
        # Full attention first (matches upstream convergence ordering).
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(self.kv_cache_groups[gid].is_eagle_group for gid in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))

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
        """
        return self._reachable_masks(
            aligned_token_len,
            start_token,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
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
    ) -> tuple[list[bool] | None, ...]:
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"lcm_block_size ({self.lcm_block_size})"
        )
        masks: list[list[bool] | None] = []
        for g_idx, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            end_chunk = aligned_token_len // spec.block_size
            start_chunk = min(end_chunk, max(0, cdiv(start_token, spec.block_size)))
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None
            use_eagle = g_idx in self.eagle_group_ids
            reachable_boundaries = (
                () if num_prompt_tokens is None else (num_prompt_tokens - 1,)
            )
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

        When ``apply_eagle`` is False, ignore ``eagle_attn_group_indices`` —
        used by ``load_mask`` to avoid popping a second block on top of the
        one already removed by the lookup.
        """
        eagle_indices = self.eagle_attn_group_indices if apply_eagle else set()
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls = self.attention_groups[0]
            hashes = self.block_hashes_for_spec(block_hashes, spec)
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=hashes,  # type: ignore[arg-type]
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=(0 in eagle_indices),
                alignment_tokens=spec.block_size,
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
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = idx in eagle_indices and idx not in eagle_verified
                _max_length = curr_hit_length
                if drop_eagle_block:
                    _max_length = min(curr_hit_length + spec.block_size, max_length)
                hashes = self.block_hashes_for_spec(block_hashes, spec)
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=hashes,  # type: ignore[arg-type]
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.lcm_block_size,
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
        # other specs already trim themselves inside their hit logic.
        spec0, group_ids0, _ = self.attention_groups[0]
        if isinstance(spec0, FullAttentionSpec):
            num_blocks = hit_length // spec0.block_size
            for gid in group_ids0:
                full_blks = hit_blocks_by_group[gid]
                assert full_blks is not None
                del full_blks[num_blocks:]
                hit_length_by_group[gid] = hit_length

        return (
            tuple(blks if blks is not None else [] for blks in hit_blocks_by_group),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
