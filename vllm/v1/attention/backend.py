# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] V1 注意力后端的抽象契约层。整个 V1 注意力体系由四组角色构成：
# [CN]   AttentionBackend      —— 后端的"能力说明书"：支持什么 dtype / head_size /
# [CN]                            block_size / 是否 MLA / 是否稀疏，供 selector 筛选
# [CN]   AttentionMetadataBuilder —— 每层的元数据构造器，把 CommonAttentionMetadata
# [CN]                            翻译成各后端内核要的具体布局
# [CN]   AttentionImpl / MLAAttentionImpl —— 真正跑 forward 的内核封装
# [CN]   AttentionMetadata     —— 传给 forward 的只读数据容器
# [CN] 分层理由：一个模型的不同层可以落在不同后端上（混合注意力），
# [CN] 所以"选后端"是以层为粒度做的，元数据也按层构造。

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import torch
from typing_extensions import deprecated

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        KVCacheLayout,
        KVCacheSpec,
        KVQuantMode,
    )

from vllm.v1.kv_cache_interface import KVCacheLayout, get_kv_quant_mode


# [CN] 继承 str 而非纯 Enum：torch.compile 会把 Enum 当非常量守卫，
# [CN] str 子类能被常量折叠，避免每次 forward 触发 Dynamo 重编译。

class AttentionType(str, Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # [CN] 标准自回归解码注意力：Q 只看自己及之前的 K/V。

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


# [CN] "倍数约束"标记：内核要求 block_size 是 base 的整数倍，而不是某个定值。
# [CN] hybrid_blocks 特性下框架级块大小只需满足倍数关系即可。

class MultipleOf:
    base: int

    def __init__(self, base: int):
        self.base = base


# [CN] 后端能力声明基类。注意这里全是 staticmethod / classmethod：
# [CN] selector 需要在"还没实例化任何后端"的阶段就能问出能力，
# [CN] 因此不能依赖实例状态，只查类属性。

class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # [CN] 模型激活 dtype 白名单（这里是 fp16/bf16，fp32 注意力内核基本不提供）。

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # Does attention's forward() include kv cache update?
    # [CN] 关键语义开关：True 表示该后端的 forward() 内部已经顺带写了 KV cache，
    # [CN] 外层就不该再单独调一次 cache 写入；False 则外层必须显式写。
    # [CN] 搞反的后果是 KV 被写两遍（覆盖错位）或一遍都没写（历史丢失）。

    forward_includes_kv_cache_update: bool = True

    @staticmethod
    # [CN] 返回内核支持的块大小列表；元素可以是定值 int，也可以是 MultipleOf。
    # [CN] 默认 MultipleOf(1) 即"任意块大小都行"。

    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImplBase"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    # [CN] 延迟到调用时才返回 builder 类：避免在 import 本模块时
    # [CN] 就把所有后端的 CUDA 扩展拉进来（import 期即失败）。

    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @classmethod
    # [CN] 返回 (module, qualname)：用于日志与缓存 key，比 repr 稳定
    # [CN] （不受实例地址、动态子类名影响）。

    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    # [CN] 空列表 = 不限制。这是刻意的"宽松默认"，
    # [CN] 让新后端不必为了兼容而穷举所有 head_size。

    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    # [CN] None 表示用户没指定（走 auto），一律放行，由后端自己挑。

    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True

        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True

        for supported_size in supported_kernel_block_sizes:
            if isinstance(supported_size, MultipleOf):
                supported_size = supported_size.base
            # With hybrid_blocks feature, the framework-level block size
            # only needs to be a multiple of the kernel's requirement,
            # even if the kernel requires a fixed block_size.
            # [CN] 取模而非相等：框架块大小可以是内核块大小的整数倍，
            # [CN] hybrid_blocks 会把大块切成若干内核块来喂。

            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    # [CN] 后端对 KV cache spec 的"事后微调"钩子（例如要求 K/V 打包在一起）。
    # [CN] 这是过渡 API：终态是让后端直接构造并返回 spec，这个钩子就会删掉。

    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Adjust the layer's KV cache spec for this backend's kernels. Used when the
        kernels want KV packed in a specific way.

        NOTE: temporary compatibility API. Today the Attention layer builds the spec
        from the model config and the backend only gets to adjust it post-hoc; the end
        state is for the backend to build and return the spec directly, at which point
        this hook goes away.

        (see: https://github.com/vllm-project/vllm/issues/42449)
        """
        return spec

    @classmethod
    # [CN] 协商顺序：默认块大小若已被支持就原样用；否则退到后端支持列表里
    # [CN] 最小的那个（保守选小，避免一次性浪费太多显存）。

    def get_preferred_block_size(cls, default_block_size: int) -> int:
        supported_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_sizes:
            return default_block_size

        if cls.supports_block_size(default_block_size):
            return default_block_size

        return min(s.base if isinstance(s, MultipleOf) else s for s in supported_sizes)

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    # [CN] attention sink：始终把开头若干 token 留在窗口内，
    # [CN] 用于滑动窗口下防止长文本质量崩塌。

    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    # [CN] 多模态前缀部分 token 走全注意力（其余走稀疏/滑窗）的能力。

    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    # [CN] KV 量化时每个 head 独立缩放因子（而非整层共享一个 scale）。

    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    # [CN] 与 ENCODER_ONLY 不同：这里指"仍在标准 paged KV 解码路径上，
    # [CN] 但去掉因果掩码"的双向注意力（如 PrefixLM、bidirectional 模型）。

    def supports_non_causal(cls) -> bool:
        """Check if backend supports non-causal (bidirectional) attention
        for decoder models.

        Unlike ENCODER_ONLY attention type which implies a different
        execution model, this refers to non-causal attention within the
        standard paged-KV-cache decoder path.
        """
        return False

    @classmethod
    # [CN] 批不变性：同一请求在不同批次组成下得到逐位相同的结果。
    # [CN] 只有少数内核（如某些 FlashInfer 配置）能保证，默认关闭。

    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return True

    @classmethod
    # [CN] 是否容忍"设备侧 query 长度与 CPU 侧不一致"。
    # [CN] 自适应验证会在设备上裁剪草稿长度，CPU 侧记录的是分配前的值，
    # [CN] 二者会分叉。SSM 类后端因为按 CPU 边界规划循环状态，必须投反对票。

    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        """Whether this backend can run a batch whose device query_start_loc disagrees
        with the CPU one; backends that plan off the CPU query lengths must opt out.

        Currently only verification requests are affected: adaptive verification trims
        their drafts on device. On the CPU the draft budget is evenly distributed across
        requests, so the total draft budget, the decode/prefill split point and the CPU
        prefill query lengths all stay correct.

        SSM backends opt out: their recurrent-state planning is built from the CPU
        per-request boundaries, which the trimmed batch no longer matches.
        """
        return not cls.is_ssm()

    @classmethod
    # [CN] PCP = Prefill Context Parallelism（预填充沿序列切分）。
    # [CN] 用 try/except 兜 NotImplementedError：未实现的后端直接判不支持，
    # [CN] 不必强制每个子类都写一遍这个 classmethod。

    def supports_pcp(cls) -> bool:
        try:
            return cls.get_impl_cls().supports_pcp
        except NotImplementedError:
            return False

    @classmethod
    # [CN] DCP = Decode Context Parallelism（解码沿 KV 切分，需要 all-reduce lse）。

    def supports_dcp(cls) -> bool:
        try:
            return cls.get_impl_cls().supports_dcp
        except NotImplementedError:
            return False

    @classmethod
    # [CN] 不直接问 impl 而是查 builder 的类属性：
    # [CN] 因为这是"元数据构造能力"而非"内核计算能力"。

    def supports_non_causal_dcp(cls) -> bool:
        builder_cls = cls.get_builder_cls()
        return bool(getattr(builder_cls, "supports_non_causal_multi_token_dcp", False))

    @classmethod
    # [CN] 默认只吃 DECODER。encoder-decoder / encoder-only 必须显式重写，
    # [CN] 否则会在 validate_configuration 阶段被拦下。

    def supports_attn_type(cls, attn_type: str) -> bool:
        """Check if backend supports a given attention type.

        By default, only supports decoder attention.
        Backends should override this to support other attention types.
        """
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return True

    @classmethod
    # [CN] 兜底的"组合校验"钩子：单维度都通过但仍不兼容时（例如
    # [CN] 某 head_size + 某 dtype 的组合内核没编），在这里返回人话原因。
    # [CN] 返回 None 表示没问题，返回字符串即拒绝理由。

    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
        has_sliding_window: bool = False,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
        use_kv_connector: bool = False,
        use_pcp: bool = False,
        use_adaptive_verification: bool = False,
        use_dcp: bool = False,
    ) -> list[str]:
        # [CN] 收集式校验而非短路抛错：一次把所有不匹配项列出来，
        # [CN] 用户能一眼看到需要改哪几个配置，而不是改一个再报下一个。

        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("head_size not supported")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("dtype not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("kv_cache_dtype not supported")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("block_size not supported")
        if use_mm_prefix and not cls.supports_mm_prefix():
            invalid_reasons.append(
                "partial multimodal token full attention not supported"
            )
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("MLA not supported")
            else:
                invalid_reasons.append("non-MLA not supported")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("attention sinks not supported")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
            else:
                invalid_reasons.append("non-sparse not supported")
        if use_per_head_quant_scales and not cls.supports_per_head_quant_scales():
            invalid_reasons.append("per-head quant scales not supported")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("compute capability not supported")
        if not cls.supports_attn_type(attn_type):
            invalid_reasons.append(f"attention type {attn_type} not supported")
        if has_sliding_window and not cls.supports_sliding_window():
            invalid_reasons.append("sliding window not supported")
        if use_non_causal and not cls.supports_non_causal():
            invalid_reasons.append("non-causal attention not supported")
        if use_mla and use_non_causal and use_dcp and not cls.supports_non_causal_dcp():
            invalid_reasons.append("non-causal MLA attention with DCP not supported")
        if use_batch_invariant and not cls.supports_batch_invariance():
            invalid_reasons.append("batch invariance not supported")
        if use_kv_connector and not cls.supports_kv_connector():
            invalid_reasons.append("KV connector not supported")
        if use_pcp and not cls.supports_pcp():
            invalid_reasons.append("PCP not supported")
        if use_dcp and not cls.supports_dcp():
            invalid_reasons.append("DCP not supported")
        if (
            use_adaptive_verification
            and not cls.supports_device_cpu_query_lens_mismatch()
        ):
            invalid_reasons.append(
                "device-cpu query lens mismatch not supported, "
                "this is needed for adaptive verification"
            )
        # [CN] 组合钩子放在最后：先做完所有便宜的单维度检查，
        # [CN] 避免为明显不合格的配置去跑可能昂贵的组合判断。

        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    # [CN] 返回"按偏好排序"的布局元组（如 NHD 优先于 HND）；
    # [CN] 返回 None 表示不在乎布局，不参与全局布局协商投票。

    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        """Layouts this backend's kernels can consume, most preferred first, or
        None when the kernels consume any layout and express no preference."""
        return None

    @classmethod
    # [CN] SSM（Mamba 类）后端没有 KV cache，只有循环状态，走完全不同的分支。

    def is_ssm(cls) -> bool:
        return False


# [CN] 空标记基类：只为类型约束（T = TypeVar bound=AttentionMetadata），
# [CN] 各后端自己定义具体字段，基类故意不预设任何形状。

class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


@dataclass
# [CN] 全批次、跨层、跨后端共享的"原材料"。每个 AttentionMetadataBuilder
# [CN] 拿它加工出自己的后端专用元数据。
# [CN] 设计要点：多数张量同时保留 GPU 版和 CPU 版——GPU 版喂内核，
# [CN] CPU 版给需要 Python 侧循环/分支的 builder 用（避免隐式同步）。

class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    # [CN] 形状 (batch_size + 1,)：每个请求在拼平的 query 张量里的起始偏移。
    # [CN] 长度是 reqs+1 而不是 reqs，是为了能直接相减得到每条的 query 长度。

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    # [CN] 形状 (batch_size,)：每条请求"已计算 token 数"，含本次要算的 query。
    # [CN] 注意不是 prompt 长度，而是 KV cache 里的有效上下文长度。

    seq_lens: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    # [CN] 请求条数（可能含 padding 的空请求，故与 seq_lens 真实条数可不同）。

    num_reqs: int
    """Number of requests"""
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
    num_actual_tokens: int
    """Total number of tokens in batch"""
    # [CN] 批次内最长 query。很多内核需要它来决定 tile / 分配 workspace。

    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    # [CN] 形状 (batch_size, max_num_blocks)：逻辑块号 -> 物理块号的映射表。

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    # [CN] 既可以是标量 bool（整批统一），也可以是每条请求一个 bool 的张量
    # [CN] （混合批里部分请求走双向、部分走因果）。

    causal: bool | torch.Tensor = True

    # Needed by FastPrefillAttentionBuilder
    # [CN] 只为 FastPrefillAttentionBuilder 服务：只有部分位置需要算 logits，
    # [CN] 这里给出被 padding 对齐后的索引与真实条数。

    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None
    max_logits_per_req: int | None = None

    # Needed by CrossAttentionBuilder
    # [CN] 交叉注意力专用：encoder 侧序列长度，decoder-only 模型恒为 None。

    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None

    # [CN] DCP 下本 rank 持有的那一段 KV 的长度；非 DCP 场景为 None。

    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """Sequence lengths of the local rank in decode context parallelism world"""

    # [CN] 可选。给了位置信息，builder 就能预先算好依赖位置的稀疏元数据
    # [CN] （DeepSeek V4 的 C128A 层需要），省掉内核里重算一遍。

    positions: torch.Tensor | None = None
    """(num_actual_tokens,) token positions.  Optional; set when the caller
    has positions available so that builders can pre-compute position-dependent
    sparse metadata for DeepSeek V4 C128A layers."""

    # [CN] 区分"真 prefill"和"短 extend"：后者 query 长度 >1 但请求其实
    # [CN] 已经在解码了（如投机解码的验证步），部分后端要走不同内核。

    is_prefilling: torch.Tensor | None = None
    """(batch_size,) bool tensor: True if request is still in prefill phase
    (num_computed_tokens < num_prompt_tokens). Used by some backends to
    distinguish actual decodes from short extends."""

    # [CN] CPU 侧的 seq_lens 上界。prefill 行与同步场景是精确值；
    # [CN] 异步投机解码的 decode 行是"假设草稿全中"的乐观值，
    # [CN] 因此不能给需要精确上下文长度的内核用。

    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    """(batch_size,) CPU upper bound on seq_lens. Precise for prefill rows
    and for all rows outside async spec decode; optimistic for async-spec
    decode rows (assumes every draft was accepted). Not safe for kernels
    that need exact per-row context lengths on decode rows."""

    # [CN] PrefixLM 的双向区间：同一请求内可有多段（多图），但段之间不得重叠。

    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    """PrefixLM bidirectional ranges for multimodal tokens. Maps
    request index to list of (start, end) token position ranges
    where bidirectional attention should apply. None for text-only
    batches or non-PrefixLM models. A request's ranges must not overlap."""

    # [CN] R-SWA：逻辑位置小于该长度的 token 全局可见，之后的 token 额外
    # [CN] 只看固定滑窗。窗口大小不由这里给，后端从模型配置里读。

    rswa_prefix_lens: torch.Tensor | None = None
    """(batch_size,) per-request prefix length (prompt/image token count) for
    Reference Sliding Window Attention (R-SWA). Tokens with logical index below
    this stay globally visible; later (generated) tokens additionally see a
    fixed sliding window. None disables R-SWA. The attention backend copies this
    into its own persistent buffer and reads ``rswa_window`` from model config."""

    # [CN] Mamba2 ReplaySSM 解码的环形缓冲原点：被抢占恢复的请求会重新锚定，
    # [CN] 跳过 prompt 边界，否则写位置会算错。

    replayssm_decode_base_cpu: torch.Tensor | None = None
    """(batch_size,) CPU ring origin for Mamba2 ReplaySSM decode: num_computed
    at the current decode run's last full-state write. write_pos counts from
    here, so a preemption-resumed request re-anchors past the prompt boundary."""

    # WARNING: Deprecated fields. Will be removed in a future release (v0.15.0)
    # [CN] 下划线前缀 = 已废弃字段，计划 v0.15.0 移除。
    # [CN] 保留是因为它们还会被 unpadded() 显式搬运。

    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    _num_computed_tokens_cache: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None

    # [CN] 从 seq_lens 反推批大小：seq_lens 恒为 (batch_size,) 形状。

    def batch_size(self) -> int:
        return self.seq_lens.shape[0]

    # [CN] "naive" 是因为假设每个 query 恰好到下一条起点为止；
    # [CN] padding 请求会算出 0 长度，这正是想要的行为。

    def naive_query_lens(self) -> torch.Tensor:
        """Naive because it assumes that query ends where the next query starts."""
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    def replace(self, **kwargs) -> "CommonAttentionMetadata":
        return replace(self, **kwargs)

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    # [CN] 被 @deprecated 标记：隐式 H2D/D2H 同步会打断全异步调度流水线，
    # [CN] 是 V1 里最典型的性能陷阱之一。

    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
    async scheduling. If a CPU copy is needed, it can be derived from 
    query_start_loc_cpu and seq_lens.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    # [CN] num_computed = seq_lens - query_lens：本次之前已算好的 token 数。
    # [CN] 同样被废弃，理由同上（隐式同步）。

    def num_computed_tokens_cpu(self) -> torch.Tensor:
        if self._num_computed_tokens_cpu is None:
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu

    # [CN] 与上面的 CPU 版本对应，但纯粹在设备上算，不产生同步。

    def compute_num_computed_tokens(self) -> torch.Tensor:
        """Compute num_computed_tokens on device (seq_lens - query_lens)."""
        if self._num_computed_tokens_cache is None:
            query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
            self._num_computed_tokens_cache = self.seq_lens - query_lens
        return self._num_computed_tokens_cache

    # [CN] 把"每个 token 属于哪条请求"做成 repeat_interleave 的索引表，
    # [CN] 结果缓存在 _token_to_req_indices_cache 里跨步复用。

    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        """Build or reuse the per-token request index mapping."""
        num_tokens = self.num_actual_tokens
        if self._token_to_req_indices_cache is not None:
            assert self._token_to_req_indices_cache.device == buffer.device
            assert self._token_to_req_indices_cache.dtype == torch.int32
            assert self._token_to_req_indices_cache.shape[0] >= num_tokens
            return self._token_to_req_indices_cache[:num_tokens]

        # Built from the device query_start_loc: adaptive verification decides the
        # per-request draft split on device, so the CPU copy carries the right total
        # but not the right per-request boundaries. Padding requests have a query
        # length of zero and drop out of the repeat.
        # [CN] 用 CPU 侧总数（正确）配设备侧 query_lens（含设备裁剪）：
        # [CN] 自适应验证只改了"每条分多少"，没改"总共多少"。

        num_mapped_tokens = int(self.query_start_loc_cpu[-1])
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)
        token_to_req_indices = torch.repeat_interleave(
            torch.arange(query_lens.shape[0], dtype=torch.int32, device=buffer.device),
            query_lens,
            output_size=num_mapped_tokens,
        )
        buffer[:num_mapped_tokens].copy_(token_to_req_indices)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]
        return self._token_to_req_indices_cache[:num_tokens]

    # TODO(lucas): remove once we have FULL-CG spec-decode support
    # [CN] 去掉 padding 请求/token，回填成真实大小的元数据。
    # [CN] 只因 FULL-CG 尚不支持投机解码而存在，待支持后即可删除。

    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            _seq_lens_cpu=self._seq_lens_cpu[:num_actual_reqs]
            if self._seq_lens_cpu is not None
            else None,
            _num_computed_tokens_cpu=self._num_computed_tokens_cpu[:num_actual_reqs]
            if self._num_computed_tokens_cpu is not None
            else None,
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal[:num_actual_reqs]
            if isinstance(self.causal, torch.Tensor)
            else self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            max_logits_per_req=self.max_logits_per_req,
            encoder_seq_lens=maybe_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=maybe_slice_reqs(self.encoder_seq_lens_cpu),
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
            is_prefilling=maybe_slice_reqs(self.is_prefilling),
            rswa_prefix_lens=maybe_slice_reqs(self.rswa_prefix_lens),
            replayssm_decode_base_cpu=maybe_slice_reqs(self.replayssm_decode_base_cpu),
        )


M = TypeVar("M")


# [CN] CUDA graph 支持力度枚举，数值即"强弱"：越大越宽松。
# [CN] 这里刻意不考虑 cascade attention —— 它目前从不被 CUDA graph 支持。

class AttentionCGSupport(Enum):
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    # [CN] 连"prefill 与 decode 混在一个批里"都能图捕获，最强档。

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    # [CN] 只要求批内 query 长度一致。投机解码的 decode 步恰好满足：
    # [CN] 每条都是 1 + num_speculative_tokens。

    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    # [CN] 最弱的"支持"：只有纯 query_len==1 的解码批能进图。

    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""


# [CN] 每层一个实例（不是每批一个），构造期拿到 spec / 层名 / 全局配置，
# [CN] 之后每步调 build() 产出当步元数据。这样可以把持久 buffer
# [CN] （slot_mapping、block_table 等）缓存下来跨步复用。

class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    # [CN] 非 None 表示该后端要求把批重排：query 长度 <= 该阈值的请求
    # [CN] 被拉到批首（形成 decode 段），其余放后面（prefill 段）。
    # [CN] None = 不重排。

    reorder_batch_threshold: int | None = None
    # Does this backend/builder support updating the block table in existing
    # metadata
    # [CN] 多个 KV cache group 的元数据几乎一样、只是 block table 不同时，
    # [CN] 支持就地换表就能只 build 一次，省掉重复构造。

    supports_update_block_table: bool = False
    # Whether the builder constructor requires the block-table width.
    requires_block_table_width: ClassVar[bool] = False
    # Whether all step-dependent draft decode metadata can be updated in place,
    # allowing one metadata build to be reused across autoregressive draft steps.
    # [CN] 草稿模型的自回归多步能否复用同一份元数据（只就地改依赖步号的部分）。

    supports_draft_decode_metadata_update: bool = False

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: "KVCacheSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.kernel_block_size: int | None = None

    # [CN] 由框架在块大小协商完成后回灌给 builder：
    # [CN] 构造期可能还不知道最终块大小。

    def set_kernel_block_size(self, kernel_block_size: int) -> None:
        self.kernel_block_size = kernel_block_size

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],
        vllm_config: "VllmConfig",
        kv_cache_spec: "KVCacheSpec",
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class."""
        return cls._cudagraph_support

    # [CN] 阈值不是写死的常量，而要按投机配置动态抬高：
    # [CN] 草稿步的 query 长度是 1+K，若阈值仍是 1 会把草稿误判成 prefill。

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                max_num_queries_for_spec = (
                    1
                    + (2 if speculative_config.parallel_drafting else 1)
                    * speculative_config.num_speculative_tokens
                )
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    max_num_queries_for_spec,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    # [CN] 元数据构造主入口。部分后端（MLA）要求先调 reorder_batch 再 build。
    # [CN] fast_build=True 表示"构造速度优先于执行速度"，
    # [CN] 适用于投机解码这种一份元数据只用几层/几步的场景。

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.

        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """
        Update the block table for the attention metadata.
        Faster when theres multiple kv-cache groups that create virtually the
        same metadata but just with different block tables.

        Only needs to be implemented if supports_update_block_table is True.
        """
        raise NotImplementedError

    # [CN] 图捕获期的构造：默认走 build 且 common_prefix_len 强制为 0
    # [CN] （图上不能依赖"这批请求恰好有公共前缀"这种运行时事实）。

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    # [CN] 草稿模型专用。draft_index 语义取决于解码形态：
    # [CN] 链式投机 = 第 i 个 token；树状投机 = 树第 i 层。

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    # [CN] 就地更新步相关草稿元数据。注意：CUDA graph 回放不会执行 Python，
    # [CN] 所以实现必须是"可捕获的操作"，且状态存持久 buffer 而非局部变量。

    def update_draft_decode_metadata(self, metadata: M) -> None:
        """Update step-dependent draft decode metadata in place.

        The fused draft loop may call this method during full CUDA graph
        capture. CUDA graph replay does not run this Python method, so
        implementations must emit capture-safe operations and keep replayed
        tensor state in persistent storage.
        """
        raise NotImplementedError

    # [CN] 级联注意力判定：公共前缀足够长时才划算。默认一律不用。

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        return False


# [CN] 用 Protocol 而非 ABC：注意力层只需"结构上长得像"（有这些 scale
# [CN] 属性和 forward），不必真的继承某个基类，便于各模型自定义层。

class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _k_scale_cpu: torch.Tensor
    _v_scale: torch.Tensor
    _v_scale_cpu: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...


# [CN] 标准 impl 与 MLA impl 的公共部分。刻意不定义 forward：
# [CN] 两类后端的 forward 签名根本不同（MLA 有 mha/mqa 两条路径）。

class AttentionImplBase(ABC, Generic[T]):
    """Base class for attention implementations.

    Contains common attributes and initialization logic shared by both
    standard AttentionImpl and MLAAttentionImpl. Does not define a forward
    method - subclasses define their own forward interfaces.
    """

    # Whether this impl uses a sparse (top-k) attention path. Used by MLA to
    # route between the dense-MHA prefill and sparse-MQA paths.
    # [CN] 走 top-k 稀疏路径的标记。MLA 靠它在"密集 MHA prefill"和
    # [CN] "稀疏 MQA"之间做路由。

    is_sparse: ClassVar[bool] = False

    # Whether this impl provides a dense-MHA prefill path (forward_mha). Sparse
    # impls without one run the top-k MQA path for all requests.
    # [CN] 没有密集 MHA prefill 的稀疏 impl，连 prefill 也得走 top-k MQA。

    supports_dense_mha_prefill: ClassVar[bool] = True

    # Required attributes that all impls should have
    num_heads: int
    head_size: int
    scale: float

    # Whether the attention impl can return the softmax lse for decode.
    # Some features like decode context parallelism require the softmax lse.
    # [CN] 能否返回 softmax_lse。DCP 必须靠 lse 才能跨分片合并 softmax。

    can_return_lse_for_decode: bool = False

    # Base of the logarithm used by this backend when returning softmax lse.
    # True  => natural log (lse = ln(sum(exp(qk))))
    #          -- e.g. Triton MLA, FlashAttention, FlashMLA, Cutlass MLA
    # False => base 2      (lse = log2(sum(exp(qk))))
    #          -- e.g. FlashInfer trtllm-gen MLA
    # The DCP combine kernel (cp_lse_ag_out_rs / dcp_a2a_lse_reduce in
    # vllm/v1/attention/ops/dcp.py) branches on this via its IS_BASE_E
    # constexpr; getting it wrong silently corrupts the cross-shard
    # softmax denominator.
    # [CN] lse 的对数底：True=自然对数，False=以 2 为底（FlashInfer trtllm-gen）。
    # [CN] DCP 合并内核按这个常量分支，弄错会静默算错跨分片 softmax 分母。

    lse_base_on_e: bool = True

    # Whether the attention impl supports Prefill Context Parallelism.
    supports_pcp: bool = False
    # Whether the attention impl supports Decode Context Parallelism.
    # [CN] 注意默认 True 而 supports_pcp 默认 False：DCP 的实现代价
    # [CN] （通信 lse）比 PCP（切序列重排）低得多。

    supports_dcp: bool = True
    # Whether the attention impl(or ops) supports MTP
    # when cp_kv_cache_interleave_size > 1
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # some attention backends might not always want to return lse
    # even if they can return lse (for efficiency reasons)
    # [CN] "能返回"不等于"要返回"：只在真的开了 DCP 时才强制返回，
    # [CN] 避免给单卡场景白添 lse 计算开销。

    need_to_return_lse_for_decode: bool = False

    # Whether this attention implementation supports pre-quantized query input.
    # When True, the attention layer will quantize queries before passing them
    # to this backend, allowing torch.compile to fuse the quantization with
    # previous operations. This is typically supported when using FP8 KV cache
    # with compatible attention kernels (e.g., TRT-LLM).
    # Subclasses should set this in __init__.
    # TODO add support to more backends:
    # https://github.com/vllm-project/vllm/issues/25584
    # [CN] 允许注意力层在调用前就把 Q 量化：量化算子因此能被 torch.compile
    # [CN] 与前一个算子融合（FP8 KV + TRT-LLM 内核这条路径才有）。

    supports_quant_query_input: bool = False

    dcp_world_size: int
    dcp_rank: int

    pcp_world_size: int
    pcp_rank: int

    total_cp_world_size: int
    total_cp_rank: int

    # [CN] 用 __new__ 而不是 __init__：保证所有子类都跑这段，
    # [CN] 子类即便忘了调 super().__init__() 也能拿到正确的 CP 秩信息。

    def __new__(cls, *args, **kwargs):
        # use __new__ so that all subclasses will call this
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.total_cp_world_size = self.dcp_world_size
        self.total_cp_rank = self.dcp_rank

        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass


# [CN] 标准（非 MLA）注意力实现：统一 forward 签名，
# [CN] 输出张量由调用方预分配后传入（output），支持原地写。

class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""

    kv_cache_dtype: str

    @property
    # [CN] 从 kv_cache_dtype 字符串解析出结构化的量化模式（scale 布局等）。

    def kv_quant_mode(self) -> "KVQuantMode":
        """Return the KV cache quantization mode for this layer."""
        return get_kv_quant_mode(self.kv_cache_dtype)

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # [CN] 统一签名。output / output_scale / output_block_scale 均由调用方
    # [CN] 预分配：一是省显存分配，二是让融合 pass 能把量化直接挂上去。

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    # [CN] 供 AttnFusionPass 查询：只把输出量化融合到"确实支持"的 impl 上。

    def fused_output_quant_supported(self, quant_key: "QuantKey") -> bool:
        """
        Does this attention implementation support fused output quantization.
        This is used by the AttnFusionPass to only fuse output quantization
        onto implementations that support it.

        Args:
            quant_key: QuantKey object that describes the quantization op

        Returns:
            is fusion supported for this type of quantization
        """
        return False

    # [CN] QK-Norm + RoPE + KV 写入三段融合的能力声明。

    def fused_qk_norm_rope_kvcache_supported(self):
        """
        Does this attention implementation support fused QKNorm+RoPE+KVCache fusion.
        This is used by the QkNormRopeKvCachePattern to only fuse the QKNorm ops
        with the RoPE ops and the KV cache update for implementations that support it.
        """
        return False

    def fused_rope_kvcache_supported(self):
        """
        Does this attention implementation support RoPE+KVCache fusion.
        This is used by the RopeKVCacheFusionPass to only fuse the RoPE ops
        with the KV cache update for implementations that support it.
        """
        return False

    # [CN] 被融合自定义算子回调的真正实现：结果写进预分配的 q_out/k_out，
    # [CN] V 在图层面就已从 QKV 切好，所以这里不处理 V。

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        """
        If `fused_qk_norm_rope_kvcache_supported` returns True, this method
        will be called by the fused custom op. Applies QK-norm + RoPE and
        writes K/V to the KV cache. Results are written to the pre-allocated
        q_out and k_out tensors; V is split from QKV at the graph level.
        """
        raise NotImplementedError

    # [CN] 由 torch.ops.vllm.fused_rope_and_unified_kv_cache_update 回调，
    # [CN] 就地做 RoPE 并写 KV cache。

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
        """
        If `fused_rope_kvcache_supported` returns True, this method will be called
        by torch.ops.vllm.fused_rope_and_unified_kv_cache_update
        to perform the inplace RoPE and KV cache update.
        """
        raise NotImplementedError


# [CN] MLA 专用基类：prefill 走 MHA（可展开成多头），decode 走 MQA
# [CN] （低秩 KV 只存一份），两条路径签名完全不同，故分开定义。

class MLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """MLA attention implementation with forward_mqa and forward_mha methods."""

    supports_pcp: bool = True

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        raise NotImplementedError

    # [CN] MHA 形态的 prefill：把低秩 KV 展开成真正的多头做密集注意力。

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        """MHA-style prefill forward pass."""
        raise NotImplementedError

    @abstractmethod
    # [CN] MQA 形态的 decode：q 可以是单张量，也可以是 (nope, rope) 二元组
    # [CN] （取决于是否做吸收矩阵乘法）。返回 (输出, 可选 lse)。

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MQA-style decode forward pass."""
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        """
        Does this attention implementation support fused output quantization.
        Since MLA quantization is done manually in forward_impl (common code),
        all MLA backends support it by default.
        """
        return quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
        )

    # [CN] MLA 的 KV 是"压缩 KV + RoPE 的 K 位置编码"两份拼起来的，
    # [CN] 所以用专门的 concat_and_cache_mla 而不是普通写入。
    # [CN] numel()==0 直接返回：某些层可能根本没分到 cache。

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )


# [CN] 动态造子类：同一个后端 + 不同 builder 组合出多个变体，
# [CN] 用于"后端能力相同但元数据布局不同"的场景。

def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]:
    """
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(
        name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls}
    )


# [CN] 更通用的版本：可覆盖任意属性，不只是 builder 类。

def subclass_attention_backend_with_overrides(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    overrides: dict[str, Any],
) -> type[AttentionBackend]:
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore
    return type(name, (attention_backend_cls,), overrides)
