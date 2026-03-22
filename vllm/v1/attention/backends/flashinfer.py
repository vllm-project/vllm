# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer 后端模块。

本模块实现了基于 FlashInfer 库的注意力后端，负责：
- 实现 FlashInfer 后端类
- 支持 TRT-LLM 注意力 kernel
- 支持 FP8 KV 缓存
- 支持 Cascade 注意力
- 支持分布式上下文并行（DCP）
- 支持 CUDA 图

主要类：
- FlashInferBackend: FlashInfer 后端类
- FlashInferMetadata: FlashInfer 元数据类
- FlashInferMetadataBuilder: 元数据构建器
- FlashInferImpl: 后端实现类

辅助类：
- FIPrefill: FlashInfer 原生预填充元数据
- FIDecode: FlashInfer 原生解码元数据
- TRTLLMPrefill: TRT-LLM 预填充元数据
- TRTLLMDecode: TRT-LLM 解码元数据
- BatchDCPPrefillWrapper: DCP 预填充包装器
"""

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import is_strictly_contiguous
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)

trtllm_gen_workspace_buffer = None


def _get_trtllm_gen_workspace_buffer():
    """获取 TRT-LLM 生成工作空间缓冲区。

    Returns:
        工作空间缓冲区张量
    """
    global trtllm_gen_workspace_buffer
    if trtllm_gen_workspace_buffer is None:
        trtllm_gen_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_gen_workspace_buffer


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Triton kernel 用于 FP8 KV 缓存的反量化。

    为 TRT-LLM 预填充注意力创建模拟 KV 缓存，将 FP8 KV 缓存反量化为 BF16/FP16。

    Args:
        kv_cache_ptr: FP8 KV 缓存指针
        block_tables_prefill_ptr: 预填充块表指针
        block_table_stride: 块表步幅
        mock_kv_cache_ptr: 输出模拟 KV 缓存指针
        k_scale_ptr: K 缩放因子
        v_scale_ptr: V 缩放因子
        src_stride_page: 源页面步幅
        src_stride_kv: 源 KV 步幅
        src_stride_head: 源头步幅
        DST_K_CACHE_STRIDE: 目标 K 缓存步幅
        DST_KV_CACHE_STRIDE: 目标 KV 缓存步幅
        HEAD_STRIDE: 头步幅
        NUM_KV_HEADS: KV 头数量
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    head_offsets = tl.arange(0, HEAD_STRIDE)

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # 从源读取 K（支持非连续页面/kv/头步幅）
        src_k = orig_page_num * src_stride_page + h_off * src_stride_head + head_offsets
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # 将 K 写入连续模拟缓存
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + head_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # 从源读取 V（通过 src_stride_kv 偏移 V 半部分）
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + head_offsets
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # 将 V 写入连续模拟缓存
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + head_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为 TRT-LLM 预填充注意力创建模拟 KV 缓存。

    当 KV 缓存为 FP8 而 query 为 BF16/FP16 时，需要将 FP8 KV 缓存反量化。

    Args:
        kv_cache: FP8 KV 缓存
        block_tables_prefill: 预填充块表
        k_scale: K 缩放因子
        v_scale: V 缩放因子
        dequant_dtype: 反量化数据类型（BF16 或 FP16）

    Returns:
        (mock_kv_cache, mock_block_table) 元组
    """
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    strides = kv_cache.stride()
    assert strides[3] == head_size and strides[4] == 1, (
        "对于 kv 缓存布局，(block_size, head_size) "
        f"维度必须是连续的，得到步幅 {strides}"
    )

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache 只包含此预填充所需的页面
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # 我们简单地按顺序索引此预填充所需的页面
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


class BatchDCPPrefillWrapper:
    """DCP（分布式上下文并行）预填充包装器。

    包装 FlashInfer 的预填充接口以支持 DCP。

    Attributes:
        _dcp_combine: DCP 合并函数
        _context: 上下文预填充包装器
        _new_tokens: 新 token 预填充包装器
    """
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        """初始化 DCP 预填充包装器。

        Args:
            workspace_buffer: 工作空间缓冲区
            dcp_a2a: 是否使用 A2A 通信后端
        """
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ):
        """计划预填充操作。

        Args:
            qo_indptr_cpu: QO 累积长度指针（CPU）
            paged_kv_indptr_cpu: 分页 KV 累积长度指针（CPU）
            paged_kv_indices: 分页 KV 索引
            paged_kv_last_page_len_cpu: 分页 KV 最后一页长度（CPU）
            page_size: 页面大小
            num_qo_heads: QO 头数量
            dcp_world_size: DCP 世界大小
            num_kv_heads: KV 头数量
            head_dim: 头维度
            sm_scale: 缩放因子
            window_left: 左窗口大小
            logits_soft_cap: logits 软上限
            q_data_type: Q 数据类型
            kv_cache_dtype: KV 缓存数据类型
            prefill_fixed_split_size: 预填充固定分割大小
            disable_split_kv: 是否禁用 KV 分割
        """
        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # 这是上下文运行
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # 这是新 token 运行
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        """运行预填充操作。

        Args:
            layer: 注意力层
            prefill_query: 预填充 query
            kv_cache_permute: 置换后的 KV 缓存
            key: Key 张量
            value: Value 张量
            out: 输出张量

        Returns:
            输出张量
        """
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_permute,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


class FlashInferBackend(AttentionBackend):
    """FlashInfer 后端类。

    基于 FlashInfer 库实现的高效注意力后端。
    支持 TRT-LLM kernel、FP8 KV 缓存、Cascade 注意力等特性。

    Class Attributes:
        accept_output_buffer: 是否接受输出缓冲区
        supported_dtypes: 支持的数据类型
        supported_kv_cache_dtypes: 支持的 KV 缓存数据类型
        forward_includes_kv_cache_update: 前向传播是否包含 KV 缓存更新
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """获取支持的内核块大小列表。

        Note:
            在 Blackwell 上，仅支持页面大小 16、32、64。

        Returns:
            支持的块大小列表
        """
        # Note: Not sure for all platforms, but on Blackwell,
        # only support a page size of 16, 32, 64.
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "FLASHINFER"
        """
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        """获取注意力实现类。

        Returns:
            FlashInferImpl 类
        """
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            FlashInferMetadataBuilder 类
        """
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """获取 KV 缓存形状。

        Args:
            num_blocks: 块数量
            block_size: 块大小
            num_kv_heads: KV 头数量
            head_size: 头大小
            cache_dtype_str: 缓存数据类型

        Returns:
            KV 缓存形状元组
        """
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """获取 KV 缓存步幅顺序。

        Args:
            include_num_layers_dimension: 是否包含层数维度

        Returns:
            步幅顺序元组

        Raises:
            ValueError: 如果缓存布局未知
        """
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        """获取 FlashInfer 的 FP8 数据类型。

        Args:
            kv_cache_dtype: KV 缓存数据类型

        Returns:
            对应的 FP8 数据类型

        Raises:
            ValueError: 如果数据类型未识别
        """
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """获取支持的头大小列表。

        Returns:
            支持的头大小列表 [64, 128, 256]
        """
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持指定的计算能力。

        Args:
            capability: 设备计算能力

        Returns:
            如果计算能力在 7.5 到 12.1 之间则返回 True
        """
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """检查是否支持 sink。

        FlashInfer 在 TRTLLM 注意力可用时（SM100）支持 sink。

        Returns:
            是否支持 sink
        """
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
            supports_trtllm_attention,
        )

        # 尊重显式禁用标志（例如 --attention-config.use_trtllm_attention=0）
        if force_use_trtllm_attention() is False:
            return False

        # 检查此平台是否支持 TRTLLM
        return supports_trtllm_attention()

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        """获取所需的 KV 缓存布局。

        Args:
            能力为 10.x 的设备返回 "HND"

        Returns:
            KV 缓存布局类型或 None
        """
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefill:
    """FlashInfer 原生预填充途径（非 TRTLLM）的元数据。

    Attributes:
        wrapper: 预填充包装器
    """
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


@dataclass
class FIDecode:
    """FlashInfer 原生解码途径（非 TRTLLM）的元数据。

    Attributes:
        wrapper: 解码包装器
    """
    wrapper: BatchDecodeWithPagedKVCacheWrapper


@dataclass
class TRTLLMPrefill:
    """TRTLLM 预填充途径的元数据。

    Attributes:
        block_tables: 仅对应预填充请求的块表切片，形状 [num_prefills, max_num_blocks_per_seq]
        seq_lens: 仅对应预填充请求的序列长度切片，形状 [num_prefills]
        cum_seq_lens_q: Q 的累积序列长度
        cum_seq_lens_kv: KV 的累积序列长度
        max_q_len: 预填充请求中的最大 query 长度
        max_seq_len: KV 缓存的最大序列长度
    """
    block_tables: torch.Tensor
    """
    仅对应预填充请求的块表切片。
    形状：[num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    仅对应预填充请求的序列长度切片。
    形状：[num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    预填充请求中的最大 query 长度。
    """

    max_seq_len: int
    """KV 缓存的最大序列长度。"""


@dataclass
class TRTLLMDecode:
    """TRTLLM 解码途径的元数据。

    Attributes:
        block_tables: 仅对应解码请求的块表切片，形状 [num_decodes, max_num_blocks_per_seq]
        seq_lens: 仅对应解码请求的序列长度切片，形状 [num_decodes]
        max_seq_len: KV 缓存的最大序列长度
    """
    block_tables: torch.Tensor
    """
    仅对应解码请求的块表切片。
    形状：[num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    仅对应解码请求的序列长度切片。
    形状：[num_decodes]
    """

    max_seq_len: int
    """KV 缓存的最大序列长度。"""


@dataclass
class FlashInferMetadata:
    """FlashInfer 注意力元数据类。

    存储 FlashInfer 注意力前向传播所需的元数据信息。

    Attributes:
        num_actual_tokens: 批次中的实际 token 总数（不包括 padding）
        slot_mapping: 用于写入 K/V 到缓存的张量，形状 [num_actual_tokens]
        q_data_type: Q 数据类型
        num_decodes: 解码请求数
        num_decode_tokens: 解码 token 数
        num_prefills: 预填充请求数
        num_prefill_tokens: 预填充 token 数
        prefill: 预填充部分的元数据，如果 num_prefill_tokens == 0 则为 None
        decode: 解码部分的元数据，如果 num_decode_tokens == 0 则为 None
        use_cascade: 如果为 True，整个批次是 cascade 注意力调用，prefill 和 decode 都为 None
        cascade_wrapper: Cascade 注意力包装器
    """
    num_actual_tokens: int
    """批次中的实际 token 总数（不包括 padding）。"""

    slot_mapping: torch.Tensor
    """用于写入 K/V 到缓存的张量。形状：[num_actual_tokens]"""

    q_data_type: torch.dtype
    """Query 数据类型。"""

    num_decodes: int
    """解码请求数。"""

    num_decode_tokens: int
    """解码 token 数。"""

    num_prefills: int
    """预填充请求数。"""

    num_prefill_tokens: int
    """预填充 token 数。"""

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    保存批次预填充部分的元数据。
    如果 num_prefill_tokens == 0 则为 None。
    """

    decode: FIDecode | TRTLLMDecode | None
    """
    保存批次解码部分的元数据。
    如果 num_decode_tokens == 0 则为 None。
    """

    # --- 特殊情况：Cascade 注意力 ---

    use_cascade: bool
    """
    如果为 True，整个批次是 cascade 注意力调用，
    prefill 和 decode 字段都为 None。
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None
    """Cascade 注意力包装器。"""


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    """FlashInfer 元数据构建器类。

    负责构建 FlashInfer 注意力运行所需的元数据对象。
    支持 TRT-LLM kernel、FP8 KV 缓存、Cascade 注意力等特性。

    Class Attributes:
        reorder_batch_threshold: 重排序批次阈值
    """
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # 预填充/追加使用的包装器
        self._decode_wrapper = None  # 解码使用的包装器（通用形状）

        if vllm_is_batch_invariant():
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # 对于完整的 cudagraph 捕获，每个批次大小需要一个 `decode_wrapper`
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP 可能在测试中未初始化
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        self.cache_dtype = self.cache_config.cache_dtype
        if self.cache_dtype.startswith("fp8"):
            self.kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.cache_dtype
            )
        else:
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # 当不支持 TRTLLM 注意力或设置了 --attention-config.disable_flashinfer_q_quantization=1 时，
        # 使用模型 dtype 作为 q dtype。否则，如果 kv 缓存为 fp8，则尝试使用 fp8 q，
        # 如果构建 attn metadata 时未使用 TRTLLM 注意力 kernel，则回退到模型 dtype
        can_use_trtllm = can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)

        if (
            can_use_trtllm
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        ):
            self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        # 在所有情况下都优先使用 TRTLLM 注意力进行解码。
        # 这允许我们使用 AttentionCGSupport.UNIFORM_BATCH 模式。
        self.use_trtllm_decode_attention = can_use_trtllm
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=can_use_trtllm)

        self._cascade_wrapper = None  # Cascade 注意力包装器

        # 所有注意力层共享的全局超参数
        # TODO: trtllm-gen backend 丢弃这个
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl)
        )
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not can_use_trtllm:
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        # 准备持久化缓冲区
        # 由于我们在 ModelRunnerV2 中没有显式同步，我们不固定重用的 CPU 缓冲区，
        # 以避免步骤 N 异步复制到 GPU 和步骤 N+1 缓冲区更新之间的竞争条件。
        self.pin_memory = (
            not envs.VLLM_USE_V2_MODEL_RUNNER and is_pin_memory_available()
        )
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # CUDA graph 模式下 paged_kv_indptr.cpu 的可变额外缓冲区
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32
    ) -> CpuGpuBuffer:
        """创建 CPU/GPU 缓冲区。

        Args:
            *size: 缓冲区大小
            dtype: 数据类型

        Returns:
            CpuGpuBuffer 对象
        """
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """获取 FlashInfer 注意力的 cudagraph 支持级别。

        这取决于我们是否可以使用 TRTLLM 注意力进行解码，因为如果不可用，
        我们只能使用 UNIFORM_SINGLE_TOKEN_DECODE。要检查这一点，我们必须使用
        kv_cache_spec 中的 KV 头数量调用 can_use_trtllm_attention。
        我们检查所有可用的 KV cache specs，只有当所有都支持 TRTLLM 注意力时
        才返回 UNIFORM_BATCH。

        Args:
            vllm_config: vLLM 配置
            kv_cache_spec: KV 缓存规格

        Returns:
            注意力 CUDA 图支持级别
        """
        # 对于 UniformTypeKVCacheSpecs，检查所有包含的 specs
        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        has_trtllm_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer 仅适用于注意力，所以我们不考虑其他类型的 KV spec
                #（例如 Mamba）。这主要用于类型检查。
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
            ):
                has_trtllm_support = False
                break

        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        """获取工作空间缓冲区。

        Returns:
            工作空间缓冲区张量
        """
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if vllm_is_batch_invariant():
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        """设置工作空间缓冲区。

        Args:
            workspace_buffer: 工作空间缓冲区张量
        """
        self._workspace_buffer = workspace_buffer

    def _get_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), get_kv_cache_layout()
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        """获取解码包装器。

        Args:
            batch_size: 批次大小
            use_cudagraph: 是否使用 CUDA 图

        Returns:
            解码包装器
        """
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores 默认启用，因为在最新 GPU 上，
                # 对于所有注意力操作，性能至少与 cuda cores 一样好。
                use_tensor_cores=True,
            )

            # 保存 decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _get_cascade_wrapper(self):
        """获取 Cascade 注意力包装器。

        Returns:
            Cascade 注意力包装器
        """
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout()
            )
        return self._cascade_wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """计算 FlashInfer 注意力的 paged_kv_indptr、paged_kv_indices、paged_kv_last_page_len。

        结果存储在 self.paged_kv_indptr、self.paged_kv_indices、
        self.paged_kv_last_page_len 缓冲区中。

        Args:
            num_blocks_np: 每个序列的块数 numpy 数组
            seq_lens_np: 序列长度 numpy 数组
            block_table_tensor: 块表张量
            num_reqs: 请求数
            page_size: 页面大小

        Returns:
            paged_kv_indices，形状为 [num_actual_pages] 的 GPU 张量
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )
        return paged_kv_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native or TRTLLM)
        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        prefill_use_trtllm = use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type,
            is_prefill=True,
            force_use_trtllm=self.attention_config.use_trtllm_attention,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_use_trtllm = (
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1
        )

        all_uses_trtllm = (num_prefills == 0 or prefill_use_trtllm) and (
            num_decodes == 0 or decode_use_trtllm
        )
        is_only_trtllm_decode = num_prefills == 0 and (
            num_decodes > 0 and decode_use_trtllm
        )

        if not all_uses_trtllm:
            if self.has_sinks:
                raise NotImplementedError(
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

            # The q quantization is not supported for non-trtllm attention,
            # fall back to model dtype.
            self.q_data_type = self.model_config.dtype

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not is_only_trtllm_decode
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
        seq_lens_np = seq_lens_cpu.numpy() if seq_lens_cpu is not None else None
        num_blocks_np = (
            (seq_lens_np + (page_size - 1)) // page_size
            if seq_lens_np is not None
            else None
        )

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # Compute paged_kv_indices if necessary
        needs_paged_kv_indices = use_cascade or not is_only_trtllm_decode
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                prefill_wrapper = self._get_prefill_wrapper()
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                if self.use_dcp:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                else:
                    assert isinstance(
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                        num_qo_heads=self.num_qo_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=self.page_size,
                        causal=True,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=self.model_config.dtype,
                        fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_use_trtllm:
                assert num_decode_tokens % num_decodes == 0, (
                    "TRTLLM decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                attn_metadata.decode = TRTLLMDecode(
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                fast_plan_decode(
                    decode_wrapper,
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                    indices=paged_kv_indices,
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.kv_cache_dtype,
                    o_data_type=self.model_config.dtype,
                    fixed_split_size=self.decode_fixed_split_size,
                    disable_split_kv=self.disable_split_kv,
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


class FlashInferImpl(AttentionImpl):
    """FlashInfer 注意力实现类。

    基于 FlashInfer 库实现的高效注意力后端。
    支持 TRT-LLM kernel、FP8 KV 缓存、Cascade 注意力等特性。

    Class Attributes:
        can_return_lse_for_decode: 是否可以为解码返回 LSE
    """
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        """初始化 FlashInfer 注意力实现。

        Args:
            num_heads: 注意力头数量
            head_size: 头大小
            scale: 缩放因子
            num_kv_heads: KV 头数量
            alibi_slopes: ALIBI 斜率列表
            sliding_window: 滑动窗口大小
            kv_cache_dtype: KV 缓存数据类型
            logits_soft_cap: Logits 软化上限
            attn_type: 注意力类型
            kv_sharing_target_layer_name: KV 共享目标层名称
            sinks: 注意力 sink 张量
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)
        vllm_config = get_current_vllm_config_or_none()
        self.supports_quant_query_input = (
            self.support_trtllm_attn
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持融合输出量化。

        Args:
            quant_key: 量化键

        Returns:
            是否支持
        """
        return (
            self.support_trtllm_attn
            and self.kv_cache_dtype.startswith("fp8")
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """在加载后处理权重。

        FlashInfer 要求 attention sinks 为 float32 类型。

        Args:
            act_dtype: 激活数据类型（未使用）
        """
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用 FlashInfer 进行前向传播。

        Args:
            layer: 注意力层
            query: 形状 = [num_tokens, num_heads, head_size]
            key: 形状 = [num_tokens, num_kv_heads, head_size]
            value: 形状 = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV 缓存张量，可能的形状：
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: 注意力元数据
            output: 输出张量
            output_scale: 输出缩放因子（用于融合量化）
            output_block_scale: 输出块缩放因子（用于 NVFP4）

        Returns:
            形状 = [num_tokens, num_heads * head_size] 的输出张量
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # 性能分析运行
            return output.fill_(0)

        # 确保 query 数据类型与注意力元数据中的预期数据类型匹配
        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if self.kv_cache_dtype.startswith("fp8"):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if self.kv_cache_dtype.startswith("fp8"):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_use_trtllm = isinstance(attn_metadata.decode, TRTLLMDecode)

        # 当提供 output_scale 时发生 attn+quant 融合
        if output_scale is None:
            assert output_block_scale is None, (
                "当未发生融合时不应提供 output_block_scale"
            )
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE, (
                "当发生 attn+quant 融合时 Query 必须是 FP8。"
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_use_trtllm
            ), "必须使用 TRT-LLM 注意力"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale 不应在 fp8 输出时提供"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "nvfp4 输出需要提供 output_block_scale"
                )
            else:
                raise ValueError(f"不支持的输出数据类型：{output.dtype}")

            # TRTLLM 注意力 kernel 要求缩放因子作为主机标量传递，
            # 在未启用 cuda graph 的 warmup 运行中将 o scale 存储为主机标量
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # 重要提示！
        # NOTE(woosuk): 使用分段 CUDA 图时，此方法在 eager-mode PyTorch 中执行。
        # 因此，我们需要小心此方法中的任何 CPU 开销。
        # 例如，`view` 和 `slice`（或 `[:n]`）操作即使在不调用任何 GPU 操作的情况下也慢得惊人。
        # 尽可能减少此方法中的 PyTorch 操作。
        # 在此方法中进行任何更改时，请基准测试性能以确保不会引入任何开销。

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 当 kv_cache_dtype 为 fp8 时，FlashInfer api 要求数据为 fp8_e4m3 或 fp8_e5m2
        if self.kv_sharing_target_layer_name is None and self.kv_cache_dtype.startswith(
            "fp8"
        ):
            torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.kv_cache_dtype
            )
            kv_cache = kv_cache.view(torch_dtype)

        # 输入和输出可能因 CUDA graphs 而被填充
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade 注意力（罕见情况）
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        # 使用 spec decoding 时，num_decodes 可能小于 num_decode_tokens，
        # 因为一些 decode 请求可能有多个 query token
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # 获取 KV 缓存步幅顺序并置换
        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)

        use_dcp = self.dcp_world_size > 1

        # 常规注意力（常见情况）
        # 解码请求在前，预填充请求在后。
        if num_prefill_tokens > 0:
            # 提取预填充 query
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            if not prefill_use_trtllm:
                # 使用 FlashInfer 原生预填充
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                if use_dcp:
                    # 使用 DCP（分布式上下文并行）
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    assert prefill_wrapper._context._window_left == self.window_left
                    assert prefill_wrapper._context._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale
                    assert not prefill_wrapper._context._causal
                    assert prefill_wrapper._new_tokens._window_left == self.window_left
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale
                    assert prefill_wrapper._new_tokens._causal

                    prefill_wrapper.run(
                        layer,
                        prefill_query,
                        kv_cache_permute,
                        key[num_decode_tokens:],
                        value[num_decode_tokens:],
                        out=output[num_decode_tokens:],
                    )
                else:
                    # 不使用 DCP
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal
                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],
                    )
            else:
                # 使用 TRT-LLM 预填充
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query 可能是非连续或退化的步幅
                # 首先确保内存连续性，然后用 reshape 修复退化的步幅
                # contiguous() 本身在维度大小为 1 时无法修复退化的步幅
                prefill_query = prefill_query.contiguous().reshape(prefill_query.shape)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # 此路径需要 VLLM_KV_CACHE_LAYOUT = HND 启用
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    # NVFP4 输出处理
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                if (
                    attn_metadata.q_data_type != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM 预填充注意力不支持 BF16 Q 和 fp8 kv cache
                    # 所以要启用 fp8 kv cache 的预填充注意力，我们可以构建一个模拟块
                    # 和模拟 kv cache，其中包含预填充中涉及的 BF16 KV
                    #
                    # 内部 (block_size, head_size) 维度必须是连续的；
                    # 外部维度可能有非规范步幅（例如跨层统一分配）。
                    # 外部维度上的退化步幅会破坏 TMA 描述符
                    # (参见 flashinfer-ai/flashinfer#2232)
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV 缓存内部维度 (block_size, head_size) 必须是 "
                        f"连续的，得到步幅 {kv_strides}"
                    )
                    # 执行 FP8 KV 缓存反量化
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_permute,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type,
                    )
                else:
                    # 不需要反量化
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                # 运行 TRT-LLM 批处理上下文注意力
                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                )

        if num_decode_tokens > 0:
            # 提取解码 query
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            if not decode_use_trtllm:
                # 使用 FlashInfer 原生解码
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if use_dcp:
                    # 使用 DCP（分布式上下文并行）
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                    )
                    # 合并 DCP 输出
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    # 不使用 DCP
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[:num_decode_tokens],
                    )
            else:
                # 使用 TRT-LLM 解码
                # decode_query 可能是非连续或退化的步幅
                assert isinstance(attn_metadata.decode, TRTLLMDecode)
                # 首先确保内存连续性，然后用 reshape 修复退化的步幅
                # contiguous() 本身在维度大小为 1 时无法修复退化的步幅
                decode_query = decode_query.contiguous().reshape(decode_query.shape)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # 此路径需要 VLLM_KV_CACHE_LAYOUT = HND 启用
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                # kv_cache 外部维度可能是非连续的（例如
                # 跨层统一分配），但内部维度
                # (block_size, head_size) 必须是连续的且
                # 步幅必须是规范的以避免 TMA 描述符
                # 失败（参见 flashinfer-ai/flashinfer#2232）
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV 缓存内部维度 (block_size, head_size) 必须是 "
                    f"连续的，得到步幅 {kv_strides}"
                )

                if output.dtype == FP4_DTYPE:
                    # NVFP4 输出处理
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # 计算每个请求的 query 长度
                if num_decode_tokens % attn_metadata.num_decodes != 0:
                    # 当 dummy_run 强制注意力初始化 q_len = 0 时会触发
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                # 运行 TRT-LLM 批处理解码注意力
                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=kv_cache_permute,
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    q_len_per_req=q_len_per_req,
                )
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """执行 KV 缓存更新。

        Args:
            layer: 注意力层
            key: Key 张量
            value: Value 张量
            kv_cache: KV 缓存张量
            slot_mapping: 槽位映射
        """
        if self.kv_sharing_target_layer_name is None:
            # 重新塑造输入键和值并将其存储在缓存中。
            # 如果与早期注意力层共享 KV 缓存则跳过此步骤。
            # NOTE(woosuk): 这里 key 和 value 被填充而 slot_mapping 没有填充。
            # 但是，我们不需要做 key[:num_actual_tokens] 和 value[:num_actual_tokens]，
            # 因为 reshape_and_cache_flash op 使用 slot_mapping 的形状来确定实际 token 数。
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """用于 CUDA 图捕获/重放的快速 plan 函数。

    这是 BatchDecodeWithPagedKVCacheWrapper::plan 的快速版本，
    用于 CUDA 图捕获/重放，而无 CUDA 图版本则回到原始 plan。

     modifications for cudagraph:
    - 仅 indptr 和 last_page_len 缓冲区的主机到设备复制
    - 避免 indices 缓冲区的设备到设备复制

    代码部分 inspirated from FlashInfer repo 的原始 plan
    和 SGlang repo 中 FlashInfer 的 fast_decode_plan 实现。

    Args:
        self: 解码包装器
        indptr_cpu: 累积长度指针（CPU）
        indices: 索引张量
        last_page_len_cpu: 最后一页长度（CPU）
        num_qo_heads: QO 头数量
        num_kv_heads: KV 头数量
        head_dim: 头维度
        page_size: 页面大小
        pos_encoding_mode: 位置编码模式
        window_left: 左窗口大小
        logits_soft_cap: logits 软上限
        q_data_type: Q 数据类型
        kv_data_type: KV 数据类型
        o_data_type: 输出数据类型
        data_type: 数据类型
        sm_scale: 缩放因子
        rope_scale: RoPE 缩放
        rope_theta: RoPE theta
        non_blocking: 是否非阻塞
        fixed_split_size: 固定分割大小
        disable_split_kv: 是否禁用 KV 分割
    """
    # 如果是第一次调用，使用原始 plan 进行 warm up
    # 如果我们为动态形状运行，则始终运行原始 plan
    # 对于固定形状（cudagraph），此 warm up 是为了解码包装器生成_cached_module
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "这里应该仅使用 cudagraph"

    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
