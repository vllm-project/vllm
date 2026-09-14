# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] GPU worker 侧的通用工具集，负责四件彼此独立的事：
# [CN]   1. KV cache 块的批量清零（Triton 内核 + 段地址表）
# [CN]   2. KV cache 的分配、视图切分与绑定到前后文
# [CN]   3. 隐含eu块大小在所有后端之间的协商（虚拟块拆分）
# [CN]   4. 批形态判定（是否均匀解码）等小工具
# [CN] 共同主题：这些都是"框架与内核之间"的胶水，
# [CN] 既不属于调度器也不属于某个后端。

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import numpy as np
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.mamba_mixer2 import share_replayssm_ring_trackers
from vllm.model_executor.layers.utils import warmup_rocm_skinny_gemm_workspaces
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


# [CN] logits 里出现 NaN 通常意味着数值溢出或权重损坏，必须立刻炸掉
# [CN] 而不是让它静默传播成一串垃圾 token。

def raise_if_nan_logits(num_nans_in_logits: Mapping[str, int]) -> None:
    if not any(num_nans_in_logits.values()):
        return

    # [CN] 只筛出真正含 NaN 的请求再报：一次把全部坏请求列出来，
    # [CN] 便于判断是单条请求问题还是全局性问题。

    corrupted_requests = {
        req_id: num_nans
        for req_id, num_nans in num_nans_in_logits.items()
        if num_nans > 0
    }
    # [CN] 用 RuntimeError 而非断言：断言可能被 -O 优化掉，这种错误必须稳定抛出。

    raise RuntimeError(f"NaNs detected in logits: {corrupted_requests}")


# [CN] 清零内核。之所以要专门写内核而不是 tensor.zero_()：
# [CN] 需要清零的是"一批离散块号"且分布在不同逻辑布局里，
# [CN] 通用索引操作会产生大量小 kernel launch。

@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  Layer-compact
    layouts have one segment per layer buffer; dimensions physically outside
    the block dim (separate head groups under LHBNC) and virtual block splits
    each get their own segment.

    Segments may have different block strides and page sizes (e.g. packed
    KV views or models with multiple KV cache groups like MLA + DSA
    indexer). Each segment's block stride determines where a logical block
    begins, while its page size determines how many elements are cleared.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped directly onto a 3-D grid as
    (block_index, seg_index, chunk_index).
    """
    # [CN] 三维网格 = (块号, 段号, 分块号)：一个 program 只清一段里的一小块。

    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    # [CN] 各段页大小不同，尾部 chunk 会越界；提前返回比加掩码更省事
    # [CN] （掩码存储本身也有开销）。

    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    # [CN] 统一按 int32 粒度写：所有 KV dtype 都保证 4 字节对齐，
    # [CN] 这样一套内核能通吃 fp16/bf16/fp8。

    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(
        ptr + block_offset + cols,
        tl.zeros([BLOCK_SIZE], dtype=tl.int32),
        mask=cols < page_size_el,
    )


# [CN] 一次构造、反复使用：构造方法里把所有段的绝对地址算好，
# [CN] 之后每步只需拿新块号去 launch，不必再遍历层。

class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    # [CN] 构造期做重活的原因：段地址在 KV cache 分配完成后就不再变化，
    # [CN] 每步重算等于把固定不变的推导搬进热路径。

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        """Precompute the absolute-address table for the Triton zeroing kernel.

        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Per-layer views are standardized ``[B, H, N, C]`` with blocks at dim 0; dims
        physically outside B (separate head groups under LHBNC) each get their own
        segment. A segment's page spans everything inside its block, so under BHLNC it
        also covers the block's other layers -- safe, since block IDs are global pool
        indices and a newly allocated block owns its whole tile.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each virtual block is represented as an independent segment so its
        physical block stride and zeroed page span remain independent.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        self.device = device
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None
        ) = None

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        # Overlaid layers (packed layouts) share a base address but may have
        # different page sizes; keep the widest span per address so newly
        # allocated blocks are fully zeroed for every overlaying group.
        # [CN] 按"绝对地址"去重：打包布局下多个 KV group 叠在同一地址上，
        # [CN] 只需登记一次，但要保留其中最宽的页跨度。

        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        # [CN] 只处理 AttentionSpec；Mamba 之类没有按块的 KV 概念，直接跳过。

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            # [CN] 组数可能多于 kernel 块表长度（例如某些组被跳过），越界即忽略。

            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            # [CN] 虚拟块拆分的前提：逻辑块大小必须是内核块大小的整数倍。

            assert spec.block_size % kernel_bs == 0
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                dp = kv.data_ptr()

                assert kv.shape[0] % num_blocks == 0, (
                    f"{layer_name}: {kv.shape[0]} kernel blocks is not a "
                    f"multiple of {num_blocks} logical blocks"
                )
                # [CN] ratio = 每个调度块拆成多少个内核块。这就是"虚拟块"。

                ratio = kv.shape[0] // num_blocks

                el = kv.element_size()
                block_stride_bytes = kv.stride(0) * el
                assert block_stride_bytes % 4 == 0
                assert kv.shape[0] % ratio == 0
                # [CN] 步长比块步长还大的维 = 排在块维之外（LHBNC 布局下的独立头组），
                # [CN] 这些维必须各自成为独立段，否则会清错地方。

                outer_dims = [
                    d
                    for d in range(1, kv.ndim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                inner_dims = [d for d in range(1, kv.ndim) if d not in outer_dims]
                # [CN] 用步长而非累乘算页跨度：非连续视图（被 view 过的张量）也适用。

                kernel_page_bytes = el + sum(
                    (kv.shape[d] - 1) * kv.stride(d) * el for d in inner_dims
                )
                assert kernel_page_bytes % 4 == 0
                logical_block_stride_bytes = block_stride_bytes * ratio
                # [CN] 对每个外侧组合展开一个段：外侧维通常很小（如 2 个头组）。

                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    assert (dp + off_bytes) % 4 == 0
                    # [CN] 每个虚拟块单独成段，使其块步长与清零点互不干扰。

                    for virtual_index in range(ratio):
                        addr = dp + off_bytes + virtual_index * block_stride_bytes
                        # [CN] 命中已有段：合并时只能放宽页大小，不能改步长。

                        if (idx := seen_ptrs.get(addr)) is not None:
                            assert (
                                seg_block_strides[idx]
                                == logical_block_stride_bytes // 4
                            )
                            seg_page_sizes[idx] = max(
                                seg_page_sizes[idx], kernel_page_bytes // 4
                            )
                            continue
                        seen_ptrs[addr] = len(seg_addrs)
                        seg_addrs.append(addr)
                        seg_block_strides.append(logical_block_stride_bytes // 4)
                        seg_page_sizes.append(kernel_page_bytes // 4)

        # [CN] 一个段都没有（纯 Mamba 模型）时置 None，
        # [CN] zero_block_ids 会直接短路返回。

        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        # [CN] tile 取不小于最大页的最小 2 的幂，但封顶 1024：
        # [CN] 再大寄存器压力就压不住了。

        blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            len(seg_addrs),
        )

    # [CN] 每步只对新分配的块调用。已清过的块会被真正数据覆盖，不必重复清。

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            max_chunks,
            blk_size,
            n_segs,
        ) = self._meta
        n_blocks = len(block_ids)
        # [CN] 块号在 CPU 侧产生，用 pinned 中转异步上传，避免显式同步。

        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        # [CN] 网格直接映射三者，无需在工作里做除法取模，省指令。

        grid = (n_blocks, n_segs, max_chunks)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            idx,
            BLOCK_SIZE=blk_size,
        )

    # [CN] 预热：Triton JIT 编译要花百毫秒级，必须先于第一个真实请求完成。

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


@dataclass
# [CN] "同一份 KV cache spec + 同一后端"的一组层。分组的意义在于
# [CN] 它们可以共用一套元数据，是混合注意力下的基本调度单元。

class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    # [CN] ubatch 场景下每个 ubatch 各自一份 builder：
    # [CN] 否则它们内部的持久化 buffer 会互相踩踏。

    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    # [CN] 注意 builder 拿到的 spec 可能是"改过块大小"的副本，
    # [CN] 原始的框架级 spec 保持不变。

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        if kernel_block_size is None:
            kv_cache_spec_builder = self.kv_cache_spec
        elif (
            isinstance(self.kv_cache_spec, MLAAttentionSpec)
            # [CN] MLA 有独立的 storage_block_size（物理存储粒度），
            # [CN] 与内核逻辑块大小可能不同，必须优先用它。

            and self.kv_cache_spec.storage_block_size is not None
        ):
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                self.kv_cache_spec.storage_block_size
            )
        else:
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                kernel_block_size
            )
        builder_cls = self.backend.get_builder_cls()
        # [CN] 只有声明需要的 builder 才传块表宽度：多数后端不需要这个参数。

        builder_kwargs = {}
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks, self.kv_cache_spec.block_size, kernel_block_size
            )
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]
        if kernel_block_size is not None:
            for builder in self.metadata_builders:
                builder.set_kernel_block_size(kernel_block_size)

    # [CN] 按 ubatch 下标取，默认第 0 个。

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]

    @property
    def supports_draft_decode_metadata_update(self) -> bool:
        return self.get_metadata_builder().supports_draft_decode_metadata_update

    # [CN] 用组内第一个层名定位元数据：同组所有层的元数据是同一份。

    def update_draft_decode_metadata(
        self,
        attn_metadata: Mapping[str, Any],
    ) -> None:
        # [CN] 同组共一份元数据，取首层即可代表全组。

        metadata = attn_metadata[self.layer_names[0]]
        self.get_metadata_builder().update_draft_decode_metadata(metadata)


# [CN] 多后端共存时求"大家都接受的块大小"，且必须是调度块大小的因子。
# [CN] 这是混合注意力能否跑起来的关键前提。

def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """
    Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.
    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        for backend in backends:
            is_supported = False
            # [CN] 逐个候选比对：只要有一个后端接受就算对这块大小 OK。

            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # Simple proof:
    # If the supported size b is in MultipleOf(x_i) format for all attention
    # backends i, and b a factor of kv_manager_block_size, then
    # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
    # return kv_manager_block_size in case 1.
    # [CN] 只收集定值型 int 尺寸：MultipleOf 无法作为候选答案直接返回。

    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    # [CN] 从大到小试。注释里给了证明：若某 b 对所有后端都满足倍数关系
    # [CN] 且整除调度块大小，那么调度块大小本身也满足条件，会命中 case 1 返回，
    # [CN] 所以这里只需找"int 型"候选，不会漏解。

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    # [CN] 走到这里说明这组后端的块大小约束互相冲突，配置无法调和。

    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


# [CN] 一次性分配整个 KV cache 的大块内存，再切成各层的 [B,H,N,C] 视图。
# [CN] 单块分配而非每层一块：减少碎片，也让 GPU 内存分析更容易。

def allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate the KV cache and view it as ``[B, H, N, C]`` per layer.

    Every KVCacheTensor places its layers in the same backing allocation: layer ``l`` of
    block ``b`` starts at ``offset + l * layer_stride + b * block_stride``. Cache
    groups overlay each other, so tensors may address the same bytes.
    """
    if not kv_cache_config.kv_cache_tensors:
        return {}

    # [CN] 断言所有组共用一个底层分配——这是 overlay（显存复用）的前提。

    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
    raw_size = sizes.pop()
    # wvSplitKrc's process-lifetime static workspaces (csrc/rocm/skinny_gemms.cu)
    # are created lazily on the first qualifying GEMM. Force that now, before
    # the giant backing allocation below: if one landed in this segment's
    # rounding tail it would pin the whole segment at engine shutdown.
    # [CN] ROCm 上必须先做一件事再分配：wvSplitKrc 的静态工作区是首次
    # [CN] GEMM 时才惰性创建的，若落在本次分配的尾部，
    # [CN] 引擎关闭时会把整段显存钉住不放。

    if current_platform.is_rocm():
        warmup_rocm_skinny_gemm_workspaces(device)
        # Pad to the page granularity MoRIIO needs to register the shared
        # backing as a single RDMA memory region. Other platforms keep the
        # exact-size allocation: NIXL and SimpleCPUOffload rely on
        # storage.nbytes() matching the logical KV size (see #53974).
        # [CN] 对齐到 4K 页：MoRIIO 需要把共享底层注册为单个 RDMA 内存区。
        # [CN] 其它平台保持精确大小，因为 NIXL / SimpleCPUOffload 依赖字节数严格相等。

        page_size = 4096
        buf_size = ((raw_size + page_size - 1) // page_size) * page_size
    else:
        buf_size = raw_size
    # [CN] 用 int8 作为字节容器再 view 成目标 dtype，
    # [CN] 而不是按目标 dtype 分配（那样大小不好精确控制）。

    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)

    # [CN] 返回 层名 -> 视图张量 的字典；共享 KV 的多层指向同一视图。

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        # [CN] 用一个代表层就能确定整个 tensor 的 spec（同一 tensor 内布局一致）。

        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[layer_name]

        # [CN] 块总数是调优后的结果，分配时必须原样使用。

        num_blocks = kv_cache_config.num_blocks
        # [CN] 默认不分虚拟块；只有注意力 group 才可能有内核块大小。

        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]
        if isinstance(spec, MLAAttentionSpec) and spec.storage_block_size is not None:
            kernel_block_size = spec.storage_block_size

        # [CN] 由 kv_cache_interface 统一负责"按布局切视图"，
        # [CN] 本函数不关心 NHD/HND 之类细节。

        views = create_kv_cache_views(
            buf,
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches


# [CN] 为每个 KV cache group 定出内核级块大小。
# [CN] 返回的列表按下标与 group id 对齐，是后续分配的依据。

def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        # [CN] encoder-only 不占自回归 KV，不参与块协商（continue 会让后续
        # [CN] group 的下标错位，需依赖调用方按下标一致处理）。

        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        # [CN] 只有注意力后端支持虚拟块拆分，需要走"求公共块大小"协商。

        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        # [CN] Mamba 的状态不是按 token 块组织的，块大小原样沿用。

        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


# [CN] 多模态嵌入的形状自检。放在这里是因为错误信息要明确指出
# [CN] 是模型 embed_multimodal 实现有误，而不是框架的问题。

def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


# [CN] 按 gpu_memory_utilization 算出目标占用并检查当前是否够。
# [CN] 提前失败优于运行到一半 OOM。

def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    # [CN] 向上取整：宁可多算一点，也不要因为浮点误差低估占用。

    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    # [CN] 返回值会被用作后续可用显存预算的上限。

    return requested_memory


# [CN] 把"共享别人 KV"的层也登记进对应 group：
# [CN] 否则后续分配注意力元数据时会漏掉这些层。

def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
    """
    if not shared_kv_cache_layers:
        return

    # [CN] 先建层名 -> group 的反查表，供后面按目标层定位 group。

    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    # [CN] 复用目标层所在的 group，只是把层名追加进去（不额外分配显存）。

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        # [CN] 登记为"runner 专属层"：这些层不参与 KV 块清零，
        # [CN] 因为它们的 KV 是被共享层写、由被共享层负责清零。

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


# [CN] 把分配好的 KV cache 同时接到 ModelRunner 与 forward context 上。
# [CN] 两处都需要：前者用于图捕获/分发，后者用于运行时按层取用。

def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    # [CN] 只允许绑定一次：重复绑定说明生命周期管理有问题。

    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    # [CN] 先按层号分组再排序：保证 runner 侧的列表顺序与模型层顺序一致。

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    # [CN] 记录最终顺序：Mamba 的环形状态共享需要按真实层次推理。

    ordered_layer_names: list[str] = []
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # [CN] 同层号多个注意力层（encoder-decoder 的 self/cross），
        # [CN] 顺序无法自动确定，交由平台钩子校验/处理。

        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            # [CN] 交给平台裁决：某些平台支持同块多层，某些不支持。

            current_platform.check_runner_kv_caches_multi_layer()
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])
            ordered_layer_names.append(layer_name)

    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    # [CN] 各层自己解包 raw 分配（Mamba 要切成 conv/ssm 多份），
    # [CN] 所以字典里每层只需存一个张量，便于 KV connector 注册。

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)

    # [CN] Mamba2 ReplaySSM 需要各层共享同一份环形缓冲追踪器，最后统一关联。

    share_replayssm_ring_trackers(ordered_layer_names, forward_context, kv_cache_groups)


# [CN] 显式解绑：模型对象的生命周期可能长于 runner
# [CN] （如 LLMEngine 的 finalizer 持有引用），不解绑显存释放不掉。

def clear_layer_kv_caches(layers: Iterable[Any]) -> None:
    """Detach the KV/state cache tensors installed by bind_kv_cache().

    The model object can outlive the runner (e.g. LLMEngine's finalizer keeps
    it reachable until engine deletion), so dropping the runner's references
    alone does not release the KV cache memory on teardown paths.
    """
    # [CN] 遍历模型所有层而不是只遍历已绑定层：更稳健，无 kv_cache 属性即跳过。

    for layer in layers:
        # [CN] 用 hasattr 而非 isinstance 判断：这里拿到的是任意 nn.Module。

        if not hasattr(layer, "kv_cache"):
            continue
        kv_cache = layer.kv_cache
        # [CN] 张量与列表两种形态都要还原，保持与构造期占位一致。

        layer.kv_cache = torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
        # [CN] 还要清掉逐 token/逐头量化产生的 scale 视图，
        # [CN] 它们可能持有对底层存储的独立引用。

        # Clean up quantized KV cache scale views
        # (int8_per_token_head, fp8_per_token_head)
        if hasattr(layer, "impl"):
            if hasattr(layer.impl, "_k_scale_cache"):
                layer.impl._k_scale_cache = None
            if hasattr(layer.impl, "_v_scale_cache"):
                layer.impl._v_scale_cache = None


# [CN] 原地拷贝 KV 块（用于前缀复用、抢占恢复等），
# [CN] 靠 view + 索引赋值实现，不申请额外显存。

def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    if not kv_cache_block_copies:
        return

    # [CN] 先在 CPU 拼好 (src, dst) 对再一次性上传：
    # [CN] 比逐对发起拷贝少很多次 kernel launch。

    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    indices: torch.Tensor | None = None
    seen: set[tuple[torch.device, int]] = set()
    # [CN] 二级去重：整段接管过的 storage 不必再按视图拷第二次。

    copied_storages: set[tuple[torch.device, int]] = set()
    for cache in kv_caches:
        # Layers sharing KV (cross-layer sharing) alias the same view; copy it
        # once. data_ptr distinguishes per-layer views of a shared allocation.
        # [CN] 按 data_ptr 去重：跨层共享的 KV 是同一个视图，拷一次就够。

        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)

        if indices is None:
            indices = async_tensor_h2d(indices_np, device=cache.device)
        assert cache.device == indices.device
        src, dst = indices.unbind(dim=1)

        # [CN] 反推虚拟块倍率；必须整除，否则说明 spec 与分配不一致。

        kernel_blocks_per_block, remainder = divmod(cache.shape[0], num_blocks)
        assert remainder == 0, (
            f"{cache.shape[0]} kernel blocks not divisible by "
            f"{num_blocks} scheduler blocks"
        )
        storage = cache.untyped_storage()
        storage_key = (cache.device, storage.data_ptr())
        scheduler_block_stride = (
            cache.stride(0) * cache.element_size() * kernel_blocks_per_block
        )
        # [CN] 整个 storage 恰好等于所有块：整段视为一块农村经济基于
        # [CN] 直接用 set_ 接管 storage，避免逐层重复拷贝。

        if storage.nbytes() == num_blocks * scheduler_block_stride:
            if storage_key in copied_storages:
                continue
            copied_storages.add(storage_key)
            blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
            blocks.set_(storage)
            blocks = blocks.view(num_blocks, -1)
        else:
            # Fold virtual block splitting into the shape so that dim 0 counts
            # scheduler blocks; unflatten of dim 0 is always a view.
            blocks = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))
        # [CN] 索引赋值底层会用高效的 batched copy，无需手写内核。

        blocks[dst] = blocks[src]


# [CN] 纯形状判定。注意提示分块的形状可能与解码批相同，
# [CN] 所以不能单独用它判断"这是不是一个解码批"。

def is_uniform_query_len(num_reqs: int, num_tokens: int, max_query_len: int) -> bool:
    """Whether every request in the batch has the same query length.

    Shape test only; use ``get_uniform_decode_token_count`` to classify a
    scheduled batch, since a prompt chunk can have a decode batch's shape.
    """
    return num_reqs > 0 and num_tokens == max_query_len * num_reqs


# [CN] 在上一函数基础上补一条：批里没有 prefill 才算真解码批。

def get_uniform_decode_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int, has_prefill: bool
) -> int | None:
    """Per-request token count of a uniform decode batch, or None."""
    if not has_prefill and is_uniform_query_len(num_reqs, num_tokens, max_query_len):
        return max_query_len
    return None


# [CN] 序列并行 + 张量并行时残差是分片的；
# [CN] SP 只在全图编译模式下支持，故这里也顺便校验编译配置。

def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled. SP is only supported in
    full-graph compilation mode.
    """
    # [CN] SP 是编译期的 pass 开关，未开则残差根本没被分片。

    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    # [CN] TP=1 时无所谓分片：直接短路。

    if tp == 1:
        return False

    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    # [CN] 启用 SP 时会提前把 token 数补齐到 TP 的整数倍，此处兜底校验。

    assert num_input_tokens % tp == 0

    return True


@dataclass
# [CN] 每条请求的视觉编码器耗时统计，最终汇总进请求级 metrics。

class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    # [CN] 转成扁平字典：metrics 上报只接受标量。

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
