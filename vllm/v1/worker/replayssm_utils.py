# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from itertools import product as iprod
from typing import Any

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace


@triton.jit
def _copy_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_copies_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    copy_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return

    src_block = tl.load(block_copies_ptr + 2 * copy_index)
    dst_block = tl.load(block_copies_ptr + 2 * copy_index + 1)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < page_size_el
    values = tl.load(
        ptr + src_block.to(tl.int64) * block_stride_el.to(tl.int64) + cols,
        mask=mask,
    )
    tl.store(
        ptr + dst_block.to(tl.int64) * block_stride_el.to(tl.int64) + cols,
        values,
        mask=mask,
    )


class ReplaySSMBlockCopier:
    """Fuse ReplaySSM copy-on-write across canonical and auxiliary caches."""

    def __init__(
        self,
        caches: Iterable[torch.Tensor],
        num_blocks: int,
    ) -> None:
        self.caches = list(caches)
        self.num_blocks = num_blocks
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None
        ) = None
        if not self.caches or not current_platform.is_cuda_alike():
            return

        device = self.caches[0].device
        if any(cache.device != device for cache in self.caches):
            return

        seen_views: set[tuple[torch.device, int]] = set()
        seen_segments: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []
        for cache in self.caches:
            view_key = (cache.device, cache.data_ptr())
            if view_key in seen_views:
                continue
            seen_views.add(view_key)

            kernel_blocks_per_block, remainder = divmod(cache.shape[0], num_blocks)
            if remainder != 0:
                return
            el = cache.element_size()
            kernel_block_stride = cache.stride(0) * el
            logical_block_stride = kernel_block_stride * kernel_blocks_per_block
            outer_dims = [
                dim
                for dim in range(1, cache.ndim)
                if cache.stride(dim) * el > kernel_block_stride
            ]
            outer_strides = [cache.stride(dim) * el for dim in outer_dims]
            inner_dims = [dim for dim in range(1, cache.ndim) if dim not in outer_dims]
            kernel_page_size = el + sum(
                (cache.shape[dim] - 1) * cache.stride(dim) * el for dim in inner_dims
            )
            if (
                cache.data_ptr() % 4 != 0
                or logical_block_stride % 4 != 0
                or kernel_page_size % 4 != 0
            ):
                return

            for outer in iprod(*(range(cache.shape[dim]) for dim in outer_dims)):
                outer_offset = sum(
                    index * stride
                    for index, stride in zip(outer, outer_strides, strict=True)
                )
                for virtual_index in range(kernel_blocks_per_block):
                    addr = (
                        cache.data_ptr()
                        + outer_offset
                        + virtual_index * kernel_block_stride
                    )
                    segment_index = seen_segments.get(addr)
                    if segment_index is not None:
                        if (
                            seg_block_strides[segment_index]
                            != logical_block_stride // 4
                        ):
                            return
                        seg_page_sizes[segment_index] = max(
                            seg_page_sizes[segment_index], kernel_page_size // 4
                        )
                        continue
                    seen_segments[addr] = len(seg_addrs)
                    seg_addrs.append(addr)
                    seg_block_strides.append(logical_block_stride // 4)
                    seg_page_sizes.append(kernel_page_size // 4)

        if not seg_addrs:
            return
        max_page_size = max(seg_page_sizes)
        block_size = min(1 << (max_page_size - 1).bit_length(), 1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
            (max_page_size + block_size - 1) // block_size,
            block_size,
            len(seg_addrs),
        )

    def copy(self, block_copies: Sequence[KVCacheBlockCopy]) -> None:
        if not block_copies:
            return
        if self._meta is None:
            copy_kv_cache_blocks_inplace(self.caches, self.num_blocks, block_copies)
            return

        copies_np = np.asarray(block_copies, dtype=np.int64)
        sources = set(copies_np[:, 0])
        destinations = list(copies_np[:, 1])
        if sources.intersection(destinations) or len(set(destinations)) != len(
            destinations
        ):
            copy_kv_cache_blocks_inplace(self.caches, self.num_blocks, block_copies)
            return

        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            max_chunks,
            block_size,
            num_segments,
        ) = self._meta
        copies = async_tensor_h2d(copies_np, device=seg_addrs.device)
        grid = (len(block_copies), num_segments, max_chunks)
        _copy_kv_blocks_kernel[grid](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            copies,
            BLOCK_SIZE=block_size,
        )


def get_replayssm_block_copy_tensors(
    forward_context: Mapping[str, Any],
) -> list[torch.Tensor]:
    """Collect FlashInfer ReplaySSM state for scheduler block copies.

    The normal KV-cache list contains packed convolution, SSM, and ring state.
    Exclude those ring aliases; the group-shared trackers remain separate.
    The copier deduplicates trackers shared by layers in the same cache group.
    """
    extra_tensors: list[torch.Tensor] = []
    for layer in forward_context.values():
        if not getattr(layer, "use_flashinfer_replayssm", False):
            continue
        canonical_storages = {
            cache.untyped_storage().data_ptr() for cache in layer.kv_cache
        }
        extra_tensors.extend(
            cache
            for cache in layer.replayssm_cache
            if cache.untyped_storage().data_ptr() not in canonical_storages
        )
        # Group-shared trackers appear once per layer; the block-copy helper
        # deduplicates them by (device, data_ptr()).
        extra_tensors.extend(
            (layer._replayssm_ring_start, layer._replayssm_prev_num_accepted)
        )
    return extra_tensors
