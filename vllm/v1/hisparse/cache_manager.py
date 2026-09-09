# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import HiSparseHotSpec, KVCacheSpec
from vllm.v1.request import Request


class _HiSparseAuxiliaryManager(SingleTypeKVCacheManager):
    """Base for ephemeral groups whose host source owns prefix caching."""

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundary: int,
    ) -> None:
        return None

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0


class HiSparseHotManager(_HiSparseAuxiliaryManager):
    """Allocate a hot region only after a request acquires CPU-only history."""

    def __init__(self, kv_cache_spec: HiSparseHotSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.blocks_per_request = kv_cache_spec.blocks_per_request
        self.hot_required: set[str] = set()

    def require_hot(self, request_id: str) -> None:
        self.hot_required.add(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        self.require_hot(request_id)

    def has_hot(self, request_id: str) -> bool:
        return len(self.req_to_blocks.get(request_id, ())) == self.blocks_per_request

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert not new_computed_blocks
        host_import = total_computed_tokens > num_local_computed_tokens
        resumes_host_prefix = (
            num_local_computed_tokens > 0 and request_id not in self.num_cached_block
        )
        if host_import or resumes_host_prefix or request_id in self.hot_required:
            return self.get_num_required_blocks(request_id)
        return 0

    def get_num_required_blocks(self, request_id: str) -> int:
        return max(
            self.blocks_per_request - len(self.req_to_blocks.get(request_id, ())),
            0,
        )

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        self.num_cached_block[request_id] = 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        # Cold admissions can bypass add_local_computed_blocks.
        self.num_cached_block[request_id] = 0
        if request_id not in self.hot_required:
            return []
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = self.blocks_per_request - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self.hot_required.discard(request_id)
        return super().pop_blocks_for_free(request_id)


class HiSparseResidentManager(_HiSparseAuxiliaryManager):
    """Track reclaimable resident pages for otherwise host-backed KV."""

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del num_tokens_main_model
        assert not new_computed_blocks
        if total_computed_tokens > num_local_computed_tokens:
            if total_computed_tokens % self.block_size != 0:
                raise ValueError(
                    "A host-only HiSparse import must end on a cache-block boundary."
                )
            imported_pages = total_computed_tokens // self.block_size
            return max(cdiv(num_tokens, self.block_size) - imported_pages, 0)
        existing = len(self.req_to_blocks.get(request_id, ()))
        host_pages = cdiv(num_local_computed_tokens, self.block_size)
        required = cdiv(num_tokens, self.block_size)
        if apply_admission_cap:
            assert self._max_admission_blocks_per_request is not None
            required = min(required, self._max_admission_blocks_per_request)
        return max(required - max(existing, host_pages), 0)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Represent imported host history with null resident pages."""
        assert num_external_computed_tokens > 0
        num_tokens = num_local_computed_tokens + num_external_computed_tokens
        blocks = self.req_to_blocks[request_id]
        tail_page = cdiv(num_tokens, self.block_size) - 1
        if len(blocks) <= tail_page:
            blocks.extend([self._null_block] * (tail_page + 1 - len(blocks)))

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        req_blocks = self.req_to_blocks[request_id]
        assert not req_blocks
        num_host_pages = cdiv(num_local_computed_tokens, self.block_size)
        req_blocks.extend([self._null_block] * num_host_pages)
        self.num_cached_block[request_id] = 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def adopt_resident_page(
        self, request_id: str, block_idx: int, block: KVCacheBlock
    ) -> bool:
        """Point a null prefix page at a pinned GPU copy of its contents."""
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks) or not blocks[block_idx].is_null:
            return False
        blocks[block_idx] = block
        return True

    def get_resident_page(self, request_id: str, block_idx: int) -> KVCacheBlock | None:
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks):
            return None
        block = blocks[block_idx]
        return None if block.is_null else block
