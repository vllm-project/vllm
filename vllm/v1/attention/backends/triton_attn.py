# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""高性能纯 Triton 注意力层模块。

本模块实现了基于纯 Triton kernel 的注意力后端，负责：
- 实现 Triton 注意力后端类
- 支持级联注意力
- 支持 FP8 KV 缓存
- 支持多种注意力类型（解码器、编码器、交叉注意力）
- 支持 ALIBI、滑动窗口、sink 等特性
- 支持 CUDA 图

主要类：
- TritonAttentionBackend: Triton 注意力后端类
- TritonAttentionMetadata: Triton 注意力元数据类
- TritonAttentionMetadataBuilder: 元数据构建器
- TritonAttentionImpl: 后端实现类
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# 常量
MIN_LAUNCH_GRID_SIZE_2D = 128  # 2D kernel 的最小启动网格大小
NUM_PAR_SOFTMAX_SEGMENTS = 16  # 并行分块 softmax 段数


@dataclass
class TritonAttentionMetadata:
    """Triton 注意力元数据类。

    存储 Triton 注意力前向传播所需的元数据信息。

    Attributes:
        num_actual_tokens: 实际 token 数（不包括 padding）
        max_query_len: 最大 query 长度
        query_start_loc: query 起始位置
        max_seq_len: 最大序列长度
        seq_lens: 序列长度
        block_table: 块表
        slot_mapping: 槽位映射
        seq_threshold_3D: 3D kernel 的序列阈值
        num_par_softmax_segments: 并行 softmax 段数
        softmax_segm_output: softmax 段输出
        softmax_segm_max: softmax 段最大值
        softmax_segm_expsum: softmax 段指数和
        use_cascade: 是否使用级联注意力
        common_prefix_len: 公共前缀长度
        cu_prefix_query_lens: 前缀 query 累积长度
        prefix_kv_lens: 前缀 KV 长度
        suffix_kv_lens: 后缀 KV 长度
        scheduler_metadata: 调度器元数据
        prefix_scheduler_metadata: 前缀调度器元数据
        mm_prefix_range: 多 token 前缀范围
    """
    # NOTE(sang): context_len、query_len 和 seq_len 的定义：
    # |---------- N-1 次迭代 --------|
    # |---------------- N 次迭代 ---------------------|
    # |- tokenA -|......................|-- 新 token ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int
    """实际 token 数（不包括 padding）。"""

    max_query_len: int
    """最大 query 长度。"""

    query_start_loc: torch.Tensor
    """query 起始位置张量。"""

    max_seq_len: int
    """最大序列长度。"""

    seq_lens: torch.Tensor
    """序列长度张量。"""

    block_table: torch.Tensor
    """块表张量。"""

    slot_mapping: torch.Tensor
    """槽位映射张量。"""

    seq_threshold_3D: int
    """3D kernel 的序列阈值。"""

    num_par_softmax_segments: int
    """并行 softmax 段数。"""

    softmax_segm_output: torch.Tensor
    """softmax 段输出张量。"""

    softmax_segm_max: torch.Tensor
    """softmax 段最大值张量。"""

    softmax_segm_expsum: torch.Tensor
    """softmax 段指数和张量。"""

    # 用于级联注意力
    use_cascade: bool
    """是否使用级联注意力。"""

    common_prefix_len: int
    """公共前缀长度。"""

    cu_prefix_query_lens: torch.Tensor | None
    """前缀 query 累积长度张量。"""

    prefix_kv_lens: torch.Tensor | None
    """前缀 KV 长度张量。"""

    suffix_kv_lens: torch.Tensor | None
    """后缀 KV 长度张量。"""

    # 可选的 AOT 调度
    scheduler_metadata: torch.Tensor | None = None
    """调度器元数据张量。"""

    prefix_scheduler_metadata: torch.Tensor | None = None
    """前缀调度器元数据张量。"""

    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    """多 token 前缀范围字典。"""

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """将 mm_prefix_range 字典转换为适合 Triton kernel 的填充张量。

        Returns:
            形状为 (num_seqs, max_ranges, 2) 的张量，空范围用 0 填充。
            空范围的 start==end==0，kernel 通过 is_valid 检查跳过。
        """
        # TODO(Isotr0py): 移至模型 runner 的注意力元数据准备
        # 以避免重复计算。
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # 收集范围，为空序列使用 [(0,0)] 以确保统一维度
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # 如果所有范围都是平凡的（只有 (0,0) 占位符），返回 None
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # 为每个序列创建 2D 张量，形状为 (num_ranges, 2)
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(
            range_tensors, layout=torch.jagged
        ).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    """Triton 注意力元数据构建器类。

    负责构建 Triton 注意力运行所需的元数据对象。
    支持 CUDA 图和级联注意力。

    Class Attributes:
        _cudagraph_support: CUDA 图支持级别
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Triton 注意力元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # 检查是否为解码启用了 CUDA 图
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # 2D kernel 的启动网格定义为 (num_q_blocks, num_heads_kv)。
        # num_q_blocks 的下限是序列数。
        # 为了确保达到最小启动网格大小，序列数必须至少等于下面的阈值。
        # 如果未达到此阈值（即批次大小不够大），将选择 3D kernel。
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # 如果需要则修改阈值。
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA 图已启用但没有指定捕获大小。"

            # 选择最接近 self.seq_threshold_3D 的 CUDA 图捕获大小
            # 作为阈值。这确保每个捕获的图覆盖正确的执行路径。
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        # 分配并行 softmax 段所需的缓冲区
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        """为 CUDA 图捕获构建元数据。

        Args:
            common_attn_metadata: 通用注意力元数据

        Returns:
            构建的 TritonAttentionMetadata 对象
        """
        attn_metadata = self.build(0, common_attn_metadata)
        # 当执行完整图捕获时，将 seq_lens 设置为
        # max_model_len 会导致图捕获非常慢，所以这里设置为 1。
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        """构建 Triton 注意力元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建

        Returns:
            构建的 TritonAttentionMetadata 对象
        """
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # 检查是否使用级联注意力
        use_cascade = common_prefix_len > 0

        if use_cascade:
            # 为级联注意力构建前缀张量
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        # 构建注意力元数据
        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    """Triton 注意力后端类。

    基于纯 Triton kernel 实现的高效注意力后端。
    支持多种数据类型、FP8 KV 缓存、级联注意力等特性。
    """
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
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

        Returns:
            支持的块大小列表 [MultipleOf(16)]
        """
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """检查是否支持指定的块大小。

        Args:
            block_size: 块大小

        Returns:
            如果块大小是 16 的倍数则返回 True
        """
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """获取后端名称。

        Returns:
            后端名称 "TRITON_ATTN"
        """
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        """获取注意力实现类。

        Returns:
            TritonAttentionImpl 类
        """
        return TritonAttentionImpl

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

        Raises:
            ValueError: 如果块大小不是 16 的倍数
        """
        if block_size % 16 != 0:
            raise ValueError("块大小必须是 16 的倍数。")
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
        """
        # `stride_order` 表示从 `get_kv_cache_shape` 到我们想要的实际内存布局的排列。
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)

        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        """检查是否使用级联注意力。

        Returns:
            False（级联注意力由 metadata builder 处理）
        """
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        """获取元数据构建器类。

        Returns:
            TritonAttentionMetadataBuilder 类
        """
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """检查是否支持指定的头大小。

        Args:
            head_size: 头大小

        Returns:
            如果头大小 >= 32 则返回 True
        """
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """检查是否支持多 token 前缀。

        Returns:
            True（Triton 支持多 token 前缀）
        """
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        """检查是否支持 sink。

        Returns:
            True（Triton 支持 sink）
        """
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Triton 注意力支持所有注意力类型。

        Args:
            attn_type: 注意力类型

        Returns:
            True
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        """检查是否支持 ALIBI 平方根。

        Returns:
            True（Triton 支持 ALIBI 平方根）
        """
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """检查是否支持指定的计算能力。

        Args:
            capability: 设备计算能力

        Returns:
            True（Triton 支持所有计算能力）
        """
        return True


class TritonAttentionImpl(AttentionImpl):
    """Triton 注意力实现类。

    基于 Triton kernel 实现的 Paged Attention。
    支持解码器、编码器、交叉注意力等多种注意力类型。
    """
    def fused_output_quant_supported(self, quant_key: QuantKey):
        """检查是否支持融合输出量化。

        Args:
            quant_key: 量化键

        Returns:
            如果 quant_key 是 kFp8StaticTensorSym 则返回 True
        """
        return quant_key == kFp8StaticTensorSym

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
        use_alibi_sqrt: bool = False,
    ) -> None:
        """初始化 Triton 注意力实现。

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
            use_alibi_sqrt: 是否使用 ALIBI 平方根
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
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # 在 flash-attn 中，logits_soft_cap 设置为 0 表示没有软化上限
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks 的形状必须与层的头数量相同。"
                f"Sinks 形状：{sinks.shape}, num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.supports_quant_query_input = current_platform.is_cuda()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用 Triton Paged Attention 实现进行前向传播。

        Args:
            layer: 注意力层
            query: 形状 = [num_tokens, num_heads, head_size]
            key: 形状 = [num_tokens, num_kv_heads, head_size]
            value: 形状 = [num_tokens, num_kv_heads, head_size]
            kv_cache: 形状 = [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: 注意力元数据
            output: 输出张量
            output_scale: 输出缩放因子
            output_block_scale: 输出块缩放因子

        Returns:
            形状 = [num_tokens, num_heads * head_size] 的输出张量
        """
        assert output is not None, "必须提供输出张量。"

        if output_block_scale is not None:
            raise NotImplementedError(
                "TritonAttentionImpl 暂不支持融合 block_scale 输出量化"
            )

        if attn_metadata is None:
            # 性能分析运行
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # 重要提示！
        # NOTE(woosuk): 使用分段 CUDA 图时，此方法在 eager-mode PyTorch 中执行。
        # 因此，我们需要小心此方法中的任何 CPU 开销。
        # 例如，`view` 和 `slice`（或 `[:n]`）操作即使在不调用任何 GPU 操作的情况下也慢得惊人。
        # 尽可能减少此方法中的 PyTorch 操作。
        # 在此方法中进行任何更改时，请基准测试性能以确保不会引入任何开销。

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 以不同方式处理编码器注意力 - 不需要 KV 缓存
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # 对于编码器注意力，我们使用直接的 Q、K、V 张量而不进行缓存
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # 对于解码器和交叉注意力，照常使用 KV 缓存
        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "目前不支持非 1.0 的 q_scale。"
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        # 调用 unified attention kernel
        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # 不支持
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """编码器注意力的前向传播，不使用 KV 缓存。

        Args:
            query: 形状 = [num_encoder_tokens, num_heads, head_size]
            key: 形状 = [num_encoder_tokens, num_kv_heads, head_size]
            value: 形状 = [num_encoder_tokens, num_kv_heads, head_size]
            output: 形状 = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: 编码器注意力元数据
            layer: 注意力层

        Returns:
            输出张量
        """
        # 对于编码器注意力，如果需要则处理 FP8 量化
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "编码器注意力不支持量化"
            )

        # 使用编码器专用元数据获取序列信息
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # 直接在 Q、K、V 张量上调用 flash 注意力
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # 编码器注意力是双向的
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """执行 KV 缓存更新。

        Args:
            layer: 注意力层
            key: Key 张量
            value: Value 张量
            kv_cache: KV 缓存张量
            slot_mapping: 槽位映射
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # 对于编码器注意力，我们使用直接的 Q、K、V 张量而不进行缓存
            return
        # 对于解码器和交叉注意力，照常使用 KV 缓存
        key_cache, value_cache = kv_cache.unbind(1)

        # 重新塑造输入键和值并将其存储在缓存中。
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # triton kernel 不支持 uint8 kv_cache
            # （因为一些显式转换（如 float8_e4m3fnuz）不支持）
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        """检查是否支持融合 RoPE KV 缓存。

        Returns:
            如果 ROCm aiter 操作可用则返回 True
        """
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """执行 RoPE 和 KV 缓存更新。

        Args:
            layer: 注意力层
            query: Query 张量
            key: Key 张量
            value: Value 张量
            positions: 位置张量
            cos_sin_cache: RoPE cos/sin 缓存
            is_neox: 是否使用 Neox 风格
            kv_cache: KV 缓存张量
            layer_slot_mapping: 层槽位映射
        """
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        # 调用 ROCm Triton RoPE 和缓存 kernel
        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
