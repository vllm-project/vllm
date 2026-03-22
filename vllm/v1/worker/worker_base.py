# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker 基础接口和工具函数模块。

本模块提供了 Worker 基础类和相关的工具函数，负责：
- 定义 Worker 基础接口（WorkerBase）
- 提供 KV 缓存块零初始化功能
- 管理注意力组（AttentionGroup）
- 处理 KV 缓存规格和块大小选择
- 支持多模态编码器输出验证
- 内存请求和验证
- KV 缓存共享层管理
- KV 缓存绑定到模型层
- 序列并行残余张量检查

主要类：
- KVBlockZeroer: KV 缓存块零初始化器
- AttentionGroup: 注意力组数据类
- WorkerBase: Worker 基础接口（在 utils.py 中定义）

主要函数：
- select_common_block_size: 选择所有后端支持的公共块大小
- prepare_kernel_block_sizes: 准备内核块大小列表
- sanity_check_mm_encoder_outputs: 验证多模态编码器输出
- request_memory: 计算和验证所需内存
- add_kv_sharing_layers_to_kv_cache_groups: 添加 KV 共享层
- bind_kv_cache: 绑定 KV 缓存到模型层
- is_residual_scattered_for_sp: 检查序列并行的残余张量是否分散
"""

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    PAGE_SIZE_EL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """零初始化 KV 缓存块的 Triton 内核。

    在所有段上一次性零初始化 KV 缓存块。

    每个段是一个块数据的连续区域。对于块在外层的后端（block_dim=0），
    每个缓冲区有一个段。对于 K/V 在外层的后端（block_dim=1），
    每个缓冲区有两个段（一个用于 K，一个用于 V）。

    seg_addrs_ptr 包含每个段的起始绝对字节地址（int64），
    允许段存在于不同的 CUDA 分配中。

    程序映射为 (block_index, seg_index, chunk_index)。

    Args:
        seg_addrs_ptr: 段地址指针
        block_ids_ptr: 块 ID 指针
        n_blocks: 块数量
        N_SEGS: 段数量（编译时常量）
        PAGE_SIZE_EL: 每页元素数（编译时常量）
        BLOCK_SIZE: 块大小（编译时常量）
    """
    pid = tl.program_id(0)
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    block_index = pid // work_per_block
    if block_index >= n_blocks:
        return
    remainder = pid % work_per_block
    seg_index = remainder // chunks
    chunk_index = remainder % chunks
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    offset = (
        block_id.to(tl.int64) * PAGE_SIZE_EL + chunk_index.to(tl.int64) * BLOCK_SIZE
    )
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


class KVBlockZeroer:
    """通过 Triton 内核管理 KV 缓存块零初始化的类。

    调用 :meth:`init_meta` 一次（在 KV 缓存分配后）以预计算段地址，
    然后每步调用 :meth:`zero_block_ids` 以零初始化新分配的块。

    Attributes:
        device: 设备
        pin_memory: 是否使用锁页内存
        _meta: 预计算的元数据（段地址、页大小、块大小、段数量）
        _id_cap: 块 ID 容量
        _ids_pinned: 锁页内存中的块 ID 缓冲区
        _ids_gpu: GPU 上的块 ID 缓冲区
    """

    def __init__(self, device: torch.device, pin_memory: bool):
        """初始化 KV 块零初始化器。

        Args:
            device: 设备
            pin_memory: 是否使用锁页内存
        """
        self.device = device
        self.pin_memory = pin_memory
        self._meta: tuple[torch.Tensor, int, int, int] | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        """一次性预计算 :meth:`zero_block_ids` 的元数据。

        为 Triton 零初始化内核构建绝对地址表。
        每个条目是 GPU 上段起始的绝对字节地址，
        因此不同 CUDA 分配中的段可以正常工作。

        来自调度器的块 ID 引用逻辑块，其大小可能与内核块大小不同
        （虚拟块分割）。PAGE_SIZE_EL 考虑了这个比率，使得
        ``block_id * PAGE_SIZE_EL`` 落在正确的偏移量上。

        只处理 AttentionSpec 层；跳过 Mamba 层。

        Args:
            attn_groups_iter: 注意力组迭代器
            kernel_block_sizes: 内核块大小列表
            cache_dtype: 缓存数据类型
            runner_only_attn_layers: 仅运行器的注意力层集合
            static_forward_context: 静态前向传播上下文
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        page_size_el: int | None = None

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if type(spec) is not FullAttentionSpec:
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            ratio = spec.block_size // kernel_bs
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache[0]
                if isinstance(kv, list):
                    continue
                dp = kv.data_ptr()
                if dp in seen_ptrs:
                    continue
                seen_ptrs.add(dp)

                el = kv.element_size()
                cur_bytes = kv.stride(block_dim) * el
                assert cur_bytes % 4 == 0
                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio
                if page_size_el is None:
                    page_size_el = cur_page_el
                else:
                    assert page_size_el == cur_page_el, (
                        f"非统一的页大小：{page_size_el} vs {cur_page_el}"
                    )

                block_stride_bytes = cur_bytes
                outer_dims = [
                    d
                    for d in range(block_dim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    seg_addrs.append(dp + off_bytes)

        if not seg_addrs or page_size_el is None:
            self._meta = None
            return

        blk_size = min(largest_power_of_2_divisor(page_size_el), 1024)
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            page_size_el,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """零初始化给定块 ID 的 KV 缓存内存。

        Args:
            block_ids: 要零初始化的块 ID 列表
        """
        if not block_ids or self._meta is None:
            return
        seg_addrs, page_size_el, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            self._ids_gpu = torch.empty(
                self._id_cap, dtype=torch.int64, device=self.device
            )
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        grid = (n_blocks * n_segs * (page_size_el // blk_size),)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )


@dataclass
class AttentionGroup:
    """注意力组数据类。

    将具有相同 KV 缓存规格的注意力层分组在一起。

    Attributes:
        backend: 注意力后端类型
        layer_names: 层名列表
        kv_cache_spec: KV 缓存规格
        kv_cache_group_id: KV 缓存组 ID
        metadata_builders: 元数据构建器列表（用于 ubatching）
    """
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # 当启用 ubatching 时，我们将为每个 ubatch 有一个元数据构建器
    # 这样如果它们为 cudagraphs 使用内部持久缓冲区，
    # 就不必担心与其他 ubatch 冲突
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        """创建元数据构建器。

        Args:
            vllm_config: vLLM 配置
            device: 设备
            kernel_block_size: 内核块大小（可选）
            num_metadata_builders: 元数据构建器数量
        """
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        self.metadata_builders = [
            self.backend.get_builder_cls()(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        """获取指定 ubatch ID 的元数据构建器。

        Args:
            ubatch_id: ubatch ID，默认为 0

        Returns:
            元数据构建器实例
        """
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """选择一个所有后端支持的块大小，且是 kv_manager_block_size 的因数。

    如果 kv_manager_block_size 被所有后端支持，直接返回它。
    否则，返回支持的最大大小。

    Args:
        kv_manager_block_size: KV 缓存的块大小
        backends: 注意力后端类列表

    Returns:
        选定的块大小

    Raises:
        ValueError: 如果找不到有效的块大小
    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """检查块大小是否被所有后端支持。"""
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"未知的支持大小：{supported_size}")
            if not is_supported:
                return False
        return True

    # 情况 1：如果 kv cache manager 的块大小被所有后端支持，直接返回
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # 情况 2：否则，块大小必须是至少一个后端的 `int` 格式支持大小
    # 按降序遍历所有 `int` 格式支持大小，返回第一个被所有后端支持的
    # 简单证明：
    # 如果支持的大小 b 对所有注意力后端 i 都是 MultipleOf(x_i) 格式，
    # 且 b 是 kv_manager_block_size 的因数，那么
    # kv_manager_block_size 也满足所有 i 的 MultipleOf(x_i)
    # 我们会在情况 1 中返回 kv_manager_block_size
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"没有找到 {kv_manager_block_size} 的公共块大小。")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """生成与每个块大小匹配的内核块大小列表。

    对于支持虚拟块分割的注意力后端，
    使用后端支持的块大小。
    对于其他后端（如 Mamba），使用相同的块大小（不分割）。

    Args:
        kv_cache_config: KV 缓存配置
        attn_groups: 按 KV 缓存组 ID 索引的注意力组列表

    Returns:
        每个缓存组的内核块大小列表
    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # UniformTypeKVCacheSpecs 中的所有层类型相同，
            # 选择一个来分发
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # 这是支持虚拟块分割的注意力后端
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # 这可能是 Mamba 或其他非注意力缓存，不分割
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"未知的 kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """对 [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][] 的结果执行健全性检查。

    Args:
        mm_embeddings: 多模态嵌入
        expected_num_items: 预期的项目数量

    Raises:
        AssertionError: 如果检查失败
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "预期多模态嵌入是 2D 张量的列表/元组，"
        f"或单个 3D 张量，但得到 {type(mm_embeddings)} "
        "相反。这很可能是由于模型 `embed_multimodal` 方法的"
        "实现不正确。"
    )

    assert len(mm_embeddings) == expected_num_items, (
        "预期多模态嵌入数量与输入项目数量匹配："
        f"{expected_num_items}，但得到 {len(mm_embeddings)=} "
        "相反。这很可能是由于模型 `embed_multimodal` 方法的"
        "实现不正确。"
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "预期多模态嵌入是 2D 张量序列，"
        f"但得到形状 {[e.shape for e in mm_embeddings]} "
        "相反。这很可能是由于模型 `embed_multimodal` 方法的"
        "实现不正确。"
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """计算 vLLM 所需的内存量，然后验证当前可用内存是否足够。

    Args:
        init_snapshot: 初始内存快照
        cache_config: 缓存配置

    Returns:
        请求的内存量（字节）

    Raises:
        ValueError: 如果可用内存不足
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"设备上的可用内存 "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) 在启动时 "
            f"小于所需的 GPU 内存利用率 "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB)。请减少 GPU 内存 "
            f"利用率或减少其他进程使用的 GPU 内存。"
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """通过重用 `kv_caches` 中分配的 KV 缓存来设置 KV 缓存共享，
    用于不分配自己 KV 缓存的层，基于 `shared_kv_cache_layers` 中的映射。
    将这些层添加到相应的 KV 缓存组，这对于确保稍后分配注意力元数据是必要的。

    Args:
        shared_kv_cache_layers: 跨层 KV 共享的层配对
            如果注意力层 `layer_name` 在此字典的键中，
            意味着此层将使用 `shared_kv_cache_layers[layer_name]`
            的 KV 缓存执行注意力
        kv_cache_groups: 模型的 KV 缓存组
        runner_only_attn_layers: 仅运行器的注意力层集合（可选）
    """
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """将分配的 KV 缓存绑定到 ModelRunner 和前向上下文，
    以便 KV 缓存可以在前向传播中使用。

    此函数：
      1) 填充 ModelRunner 的 kv cache 列表（`runner_kv_caches`）
      2) 将 `forward_context` 中的每个注意力层与其
         对应的 KV 缓存关联

    Args:
        kv_caches: 以层名为键的已分配 kv_caches
        forward_context: 包含所有 Attention 层的全局前向上下文
        runner_kv_caches: ModelRunner 声明的 kv_cache
        num_attn_module: 注意力模块数量
    """
    # 绑定 kv_caches 到 ModelRunner
    assert len(runner_kv_caches) == 0

    # 将 kv_caches 字典转换为按 layer_index 顺序的张量列表
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # 一种典型情况是 encoder-decoder 模型，如 bart
            # 解码器块中的交叉注意力和自注意力有不同的 layer_name
            # 但相同的 layer_index

            # TODO - 分析 runner_kv_caches 的用途以及正确的
            # 确保它正确反映同一解码器块中的多个注意力层的方法
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                # 我们知道 GPU / CPU 运行器不受此情况影响
                # 一些测试代码依赖于 runner_kv_caches，但
                # 不影响忽略它的方式
                pass
            else:
                raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    # 绑定 kv_caches 到前向上下文
    for layer_name, kv_cache in kv_caches.items():
        # 注意：为通过引擎插槽索引 kv_cache 的层保留列表包装器
        forward_context[layer_name].kv_cache = [kv_cache]


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """检查残余张量是否为序列并行分散。

    当启用序列并行和张量并行时，
    残余张量分散到张量并行秩上。

    这与 SequenceParallelismPass.is_applicable_for_range() 的逻辑相同：
    - 在全图编译模式（没有分割操作或使用 inductor 图分区）中，
      SP 总是应用
    - 否则，SP 仅对 compile_sizes 中的特定形状应用

    Args:
        vllm_config: vLLM 配置
        num_input_tokens: 输入 token 数量

    Returns:
        残余张量是否分散
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        return False

    # 当启用序列并行时，我们总是将 num_input_tokens 填充
    # 为 tensor_parallel_size (tp) 的倍数
    assert num_input_tokens % tp == 0

    if (
        not vllm_config.compilation_config.splitting_ops
        or vllm_config.compilation_config.use_inductor_graph_partition
    ):
        return True
    compile_sizes = vllm_config.compilation_config.compile_sizes
    if compile_sizes is None:
        return False
    return num_input_tokens in compile_sizes
