# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
import functools
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    FA4_HD256_PAGE_SIZE,
    flash_attn_supports_kv_cache_dtype,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
    uses_fa4_hd256_kernel,
)
from vllm.v1.attention.backends.utils import (
    fill_mm_prefix_query_ranges,
    get_dcp_local_seq_lens,
    get_num_attention_heads_from_layers,
)
from vllm.v1.attention.ops.dcp import (
    cp_lse_ag_out_rs,
    dcp_a2a_lse_reduce,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.worker.cp_utils import (
    run_split_fa2_dcp_context_attention,
    should_skip_dcp_context_attention,
    should_split_fa2_dcp_context_attention,
    split_dcp_context_queries,
)

logger = init_logger(__name__)


# [CN] FlashAttention 后端：vLLM V1 里覆盖面最广的注意力后端。
# [CN] 三件套结构：本类只声明"能力"（支持什么），
# [CN] FlashAttentionMetadataBuilder 负责每步构造 metadata，
# [CN] FlashAttentionImpl 负责真正的前向计算。
# [CN] 能力声明是静态的 classmethod，被 v1/attention/selector.py 拿去跟
# [CN] 模型配置比对，决定这一类模型能不能用 FA。

class FlashAttentionBackend(AttentionBackend):
    # [CN] FA 只吃半精度：fp32 输入直接不合法，不是慢而是不支持。

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # [CN] KV 缓存允许的 dtype 清单；fp8 的两个别名都列了，区别只在元数据写法。

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    # [CN] V 的头维可以与 Q/K 不同（如 MLA 的解压形态）。它是类属性而非实例属性，
    # [CN] 因为块大小协商发生在 Impl 构造之前，只能用类级别信息推断。

    head_size_v: int | None = None

    # [CN] FA4 的 Blackwell 专用内核（head_size=256）强制页大小固定，
    # [CN] 这里判断当前模型是否落在该内核上，是就返回它要求的页大小。
    @classmethod
    def _get_fa4_hd256_block_size(cls) -> int | None:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return None

        head_size = vllm_config.model_config.get_head_size()
        if (
            uses_fa4_hd256_kernel(head_size, cls.head_size_v)
            and get_flash_attn_version(
                head_size=head_size,
                head_size_v=cls.head_size_v,
                supports_fa4_hd256=True,
            )
            == 4
        ):
            return FA4_HD256_PAGE_SIZE
        return None

    # [CN] 正常情况下 FA 对块大小没有硬性要求，只要求 16 的倍数即可；
    # [CN] 只有走 hd256 内核时才退化成"唯一允许值"。
    @classmethod
    def get_supported_kernel_block_sizes(cls) -> list[int | MultipleOf]:
        # [CN] 走 hd256 内核时块大小被钉死，滑动窗口层会自动挑这个唯一值。

        if block_size := cls._get_fa4_hd256_block_size():
            # Sliding-window specs select the smallest advertised size.
            return [block_size]
        return [MultipleOf(16)]

    # [CN] FA 的写缓存是独立一步（do_kv_cache_update），不在注意力前向里顺带做。
    # [CN] 这个标志告诉调度器：前向与写缓存是两次独立的算子调用。

    forward_includes_kv_cache_update: bool = False

    # [CN] 在调度块大小与内核要求之间取较大者：XPU 上 64 是性能拐点。
    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if block_size := cls._get_fa4_hd256_block_size():
            return max(default_block_size, block_size)
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    # [CN] 后端名字是 selector 的一级索引，必须与其他后端不重名。
    # [CN] 用户可以通过 VLLM_ATTENTION_BACKEND 直接指定它来覆盖自动选择。
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    # [CN] FA 原生支持滑动窗口（window_size 参数直接传给内核）。
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    # [CN] 该后端能保证同一批数据不管怎么切分，结果逐位一致。
    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    # [CN] 非因果（双向）注意力：encoder 与 PrefixLM 场景需要。
    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    # [CN] FA 是少数同时覆盖 decoder / encoder / encoder-only / encoder-decoder
    # [CN] 四种类型的后端，代价是 forward 里要按类型分叉。
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    # [CN] 每头独立量化 scales 只有 FA3 起才支持：FA2 的 descale 是每张量一份。
    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    # [CN] 返回实现类而非实例：真正的实例化由 Attention 层自己完成。
    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    # [CN] 头维必须 8 字节对齐（内核向量化要求）；超过 256 只有 FA4 才接得住。
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    # [CN] 量化 KV 的支持情况不能只看清单，还要问 FA 运行时：
    # [CN] fp8 在不同架构上（SM90 vs SM100）由不同版本的内核承担。
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype not in cls.supported_kv_cache_dtypes:
            return False
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_kv_cache_dtype(kv_cache_dtype)
        return True

    # [CN] mm_prefix（PrefixLM 双向段）只有 FA4 的 mask_mod 机制才能表达。
    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return is_fa_version_supported(4)

    # [CN] attention sink 是 GPT-OSS 一类模型需要的额外可学习 logits，
    # [CN] 只有 FA3+ 的 s_aux 参数支持。
    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    # [CN] 硬门槛 SM80（A100）：Hopper 以前的架构 FA 根本不编译。
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    # [CN] 联合判定入口：很多限制不是单看某一项，而是几项的组合。
    # [CN] 比如 fp8 KV 要 FA3+SM90 或 FA4+SM100，单独看 dtype 或架构都不够。
    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # [CN] sink 需要 Hopper 以上；A100 上带 sink 的模型必须换后端。

        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        if (
            kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
            and not flash_attn_supports_kv_cache_dtype(
                kv_cache_dtype,
                head_size=head_size,
                head_size_v=head_size,
                has_sinks=has_sink,
                kv_cache_block_size=block_size,
                supports_fa4_hd256=True,
            )
        ):
            return "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"
        if (
            use_mm_prefix
            and get_flash_attn_version(
                head_size=head_size,
                has_sinks=has_sink,
                kv_cache_block_size=block_size,
                supports_fa4_hd256=True,
            )
            != 4
        ):
            return (
                "mm_prefix (PrefixLM bidirectional attention) requires "
                "FlashAttention v4, which does not resolve for this "
                "head_size"
            )
        return None


# [CN] 每一步前向需要递给 FA 内核的全部变长信息，全部是 GPU 张量。
# [CN] 关键约定：所有长度都是"上界"，真正有效长度另由 seqused_k 之类给出，
# [CN] 这样 CUDA graph 重放时形状不变、只换内容。
@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # [CN] 不含 padding 的真实 token 数。切片 query[:num_actual_tokens]
    # [CN] 就是这个值的作用：张量本身按最大尺寸补齐过。

    num_actual_tokens: int  # Number of tokens excluding padding.
    # [CN] 本步最长请求的 query 长度；decode 步恒为 1，prefill 步可能很大。

    max_query_len: int
    # [CN] 累计 query 偏移（cu_seqlens_q），FA varlen 接口的标配。

    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    # [CN] 是否走级联注意力：批内请求共享一段足够长的前缀时才划算。

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # [CN] DCP：把上下文 KV 沿序列维切到多个 rank，各自算完再合并。
    # [CN] 这里记的是"本 rank 负责的那段上下文"的长度信息。

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # [CN] FA2 下 DCP 的上下文注意力无法统一处理混合批，
    # [CN] 必须预知 decode / prefill 各有几条、各占多少 token。

    # Split counts for FA2 DCP context attention. num_prefill_* tracks
    # context-bearing extend rows; pure prefills do not attend to DCP context.
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # [CN] AOT（ahead-of-time）调度：FA3 可以在 host 侧一次性算好 tile 划分，
    # [CN] 把结果作为一段 int32 buffer 传进内核，省掉设备侧的动态调度开销。

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    # [CN] 既可以是标量布尔（整批一致），也可以是每请求一张量（PrefixLM 场景）。
    # [CN] 张量形态是 FA4 才有的 per-sequence causal。

    causal: bool | torch.Tensor = True

    # [CN] 多模态 PrefixLM：图像 token 段内要双向可见，段外仍然因果。
    # [CN] 这里存每个 query token 所属双向段的绝对 [start, end]，不在段内则 (-1,-1)。

    # PrefixLM bidirectional range containing each scheduled query token.
    # Shape: (num_actual_tokens, 2) int32, absolute [start, end] bounds;
    # (-1, -1) for query tokens outside every multimodal range.
    mm_prefix_query_range_tensor: torch.Tensor | None = None

    # [CN] R-SWA：prompt 全局可见、生成部分只在窗口内可见。
    # [CN] 三个字段分工：长度表、标量窗口值、以及预分配好的 CUDA 窗口张量。

    # Reference Sliding Window Attention (R-SWA) fields.
    # rswa_prefix_lens:  per-request prompt lengths [num_reqs], int32, CUDA.
    # rswa_window:       sliding window size (scalar int, for logic checks).
    # rswa_window_tensor: [1] int32 CUDA tensor — pre-allocated in build() so
    #   no CPU→CUDA copy is needed inside forward() during CUDA graph capture.
    # Only populated when the model uses R-SWA (Unlimited-OCR).
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
    rswa_window_tensor: torch.Tensor | None = None


# [CN] 收集模型里所有 FA 层用到的滑窗配置。只统计 FlashAttentionImpl，
# [CN] 因为别的后端有自己的 builder，混进来会让"是否唯一"的判断失真。

def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model.

    Only inspects FlashAttentionImpl layers. Other backends (e.g.
    TurboQuant, MLA) use their own metadata builders and are skipped.
    """
    # [CN] 用集合而不是列表：要判断"模型里是否只有一种滑窗配置"。

    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        if not isinstance(layer.impl, FlashAttentionImpl):
            continue
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


# [CN] 因果滑窗写作 (w, 0)：左边看 w 个、右边看 0 个。
# [CN] 一旦注意力变成非因果（双向），右边也必须对称地放开 w，
# [CN] 否则双向查询会看不到它右边的键。

def _maybe_symmetrize_window(
    window: tuple[int, int] | None,
    causal: bool | torch.Tensor,
) -> tuple[int, int] | None:
    """Make a causal sliding window ``(w, 0)`` symmetric ``(w, w)`` when attention
    is non-causal, so bidirectional queries attend in both directions. Leaves
    full-attention ``(-1, -1)`` and already-symmetric windows untouched.
    """
    non_causal = isinstance(causal, torch.Tensor) or causal is False
    if window is not None and window[0] >= 0 and window[1] == 0 and non_causal:
        return (window[0], window[0])
    return window


# [CN] MetadataBuilder：每步调度后被调用一次，产出 FlashAttentionMetadata。
# [CN] 设计约束：build() 里不能有同步（GPU->CPU 拷贝），
# [CN] 否则会打断全异步调度；因此大量信息改用 CPU 侧上界推算。

class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    # [CN] FA3 全场景可用全图；FA2 因为对 max_query_len=1 有特殊打包处理，
    # [CN] 混合 prefill-decode 批会算错，只能降级到 UNIFORM_BATCH。

    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    # [CN] 允许在不重建 metadata 的前提下换掉 block_table / slot_mapping，
    # [CN] 这是投机解码多步复用同一份 metadata 的前提。

    supports_update_block_table: bool = True

    # [CN] 类级别常量直接返回：这个后端不因模型/KV 配置改变图的可用性。
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "KVCacheSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    # [CN] 调用 FA 的 host 侧调度器，把 tile 划分结果序列化成 int32 张量。
    # [CN] 只有 FA3 支持，其余情况直接返回 None 让内核自己动态调度。

    def _get_scheduler_metadata(
        self,
        *,
        aot_schedule: bool,
        batch_size: int,
        cu_query_lens: torch.Tensor,
        max_query_len: int,
        seqlens: torch.Tensor,
        max_seq_len: int,
        causal: bool | torch.Tensor,
        max_num_splits: int,
    ) -> torch.Tensor | None:
        if not aot_schedule:
            return None

        # [CN] 量化缓存时内核按 fp8 解释数据，调度器也要按 fp8 估算 tile。

        cache_dtype = self.cache_config.cache_dtype
        if is_quantized_kv_cache(cache_dtype):
            qkv_dtype = current_platform.fp8_dtype()
        else:
            qkv_dtype = self.kv_cache_dtype
        # [CN] 注意 num_heads_q 乘了 DCP world size：调度器要按全局头数规划。

        return get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            num_heads_q=self.num_heads_q * self.dcp_world_size,
            num_heads_kv=self.num_heads_kv,
            headdim=self.headdim,
            cache_seqlens=seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_query_lens,
            page_size=self.block_size,
            causal=causal,
            window_size=_maybe_symmetrize_window(self.aot_sliding_window, causal),
            num_splits=max_num_splits,
        )

    # [CN] CUDA graph 要求 buffer 地址固定：新调度结果必须写回预分配的张量。
    # [CN] 尾部清零不可省——否则残留的旧调度会让多余的线程块写坏输出。

    def _store_scheduler_metadata(
        self, scheduler_metadata: torch.Tensor | None
    ) -> torch.Tensor | None:
        # [CN] 只有全图模式才需要把结果搬进固定 buffer；非图模式直接用新张量。

        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            assert self.scheduler_metadata is not None
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            # [CN] 不清零的话，多余的线程块会按残留调度去写输出缓冲，结果是脏数据。

            self.scheduler_metadata[n:] = 0
            return self.scheduler_metadata[:n]
        return scheduler_metadata

    # [CN] 构造期就完成所有能在编译期确定的决策，build() 里只做每步变化的部分。

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # [CN] 把四个子配置提升为实例属性：后续每步都要读，避免层层下钻。

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        # [CN] 优先从真实层对象读头数；拿不到才回退到模型配置的静态值。

        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.parallel_config)
        # [CN] KV 头数取自 KV cache spec 而非模型配置：TP 切分后的值才准确。

        self.num_heads_kv = kv_cache_spec.num_kv_heads
        # [CN] 从 spec 而非配置读 dtype：spec 已经把量化决策固化下来了。

        self.kv_cache_dtype = kv_cache_spec.dtype
        # [CN] 头维也从 spec 读：量化后可能与模型声明值不同。

        self.headdim = kv_cache_spec.head_size
        # [CN] 块大小是协商后的最终值，构造 Impl 时还拿不到，故只在 builder 侧存。

        self.block_size = kv_cache_spec.block_size

        # [CN] 0 表示让 FA 自己决定 SplitKV 份数。非 graph 场景这样最省显存。

        self.max_num_splits = 0  # No upper bound on the number of splits.
        # [CN] AOT 调度是 FA3 独占能力，FA2/FA4 走不到这条路径。

        self.aot_schedule = get_flash_attn_version() == 3

        # [CN] builder 侧也要独立判断一次 hd256，因为它影响块大小与缓冲分配。

        self.fa4_hd256 = uses_fa4_hd256_kernel(self.headdim) and (
            get_flash_attn_version(
                head_size=self.headdim,
                kv_cache_block_size=self.block_size,
                supports_fa4_hd256=True,
            )
            == 4
        )

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            # [CN] DCP 进程组可能尚未初始化（单测场景），这里吞掉 AssertionError。

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        # Fused draft decode reuses the captured metadata object across draft
        # steps. For DCP, build-time host-side decisions such as
        # skip_dcp_context_attention() can change the metadata shape/control
        # path (for example max_dcp_context_kv_len), and those Python-side
        # fields are not refreshed in-place between graph replays. Keep the
        # fused path disabled until DCP gets a full replay-safe refresh model.
        # [CN] 融合草稿解码会跨多个草稿步复用同一个 metadata 对象；
        # [CN] DCP 下有些字段是 host 侧布尔/整数，graph 重放时不会原地刷新，
        # [CN] 所以 DCP 开启时干脆禁用这条快路径。

        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1

        # [CN] CP 交错粒度：KV 在 rank 间按这个粒度轮转分布，
        # [CN] 目的是让长序列的相邻片段不至于全落在一个 rank 上。

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        # [CN] 全图模式（FULL cudagraph）下所有 buffer 必须预先按最大尺寸分配。

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # [CN] 超过这个 token 数的批不进 graph，只能走 eager 路径。

        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        # [CN] 为 AOT 调度预分配固定大小的 metadata buffer：
        # [CN] 1 个 int32 给 tile 计数信号量，每个请求再给 4 个槽位。

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # [CN] DCP 下按最大并发请求数预分配上下文长度表，避免 build 里临时分配。

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # [CN] AOT 调度要求滑窗值对所有层一致，且只能在层构造完成后的
        # [CN] 第一个 build() 里才拿得到，因此延迟到这里初始化。

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

        # [CN] R-SWA 的窗口大小是模型静态配置，可以提前固化成 [1] 的 CUDA 张量，
        # [CN] 这样 forward 里就不需要再做一次 CPU->GPU 拷贝。

        # R-SWA: persistent CUDA-graph-safe buffers owned by this builder.
        # [CN] None 表示模型不用 R-SWA，后续所有相关分支都会跳过。

        self.rswa_window: int | None = self.model_config.rswa_window
        self.persistent_rswa_prefix_lens: torch.Tensor | None = None
        self.persistent_rswa_window_tensor: torch.Tensor | None = None
        # [CN] 只有启用 R-SWA 的模型才预分配这两个常驻缓冲。

        if self.rswa_window is not None:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self.persistent_rswa_prefix_lens = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=self.device
            )
            self.persistent_rswa_window_tensor = torch.tensor(
                [self.rswa_window], dtype=torch.int32, device=self.device
            )

        # [CN] 三段式缓冲：pinned 的 CPU 暂存 + numpy 视图 + GPU 目标。
        # [CN] 用 numpy 视图填充是为了绕开 PyTorch 的逐元素 Python 开销。

        # mm_prefix: persistent staging + device buffers owned by this builder,
        # sized by scheduled query tokens so build() never allocates.
        self.mm_prefix_query_ranges_cpu: torch.Tensor | None = None
        self.mm_prefix_query_ranges_np: np.ndarray | None = None
        self.mm_prefix_query_ranges_gpu: torch.Tensor | None = None
        # [CN] 按最大 token 数预分配，保证 build 阶段永不分配新显存。

        if self.model_config.is_mm_prefix_lm:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.mm_prefix_query_ranges_cpu = torch.empty(
                (max_num_tokens, 2), dtype=torch.int32, pin_memory=PIN_MEMORY
            )
            self.mm_prefix_query_ranges_np = self.mm_prefix_query_ranges_cpu.numpy()
            self.mm_prefix_query_ranges_gpu = torch.empty(
                (max_num_tokens, 2), dtype=torch.int32, device=self.device
            )

    # [CN] 主入口。分支优先级：DCP > 级联 > 普通。
    # [CN] 三条路径产出不同形状的 metadata，但都写进同一个 dataclass。

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        # [CN] 请求数是标量（来自调度器），张量本身按上限补齐，所以必须单独传。

        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        # [CN] 作为 max_seqlen_k 上界传进内核；真实长度另由 seqused_k 给出。

        max_seq_len = common_attn_metadata.max_seq_len
        # [CN] 已经在设备上算好的累计偏移，build 全程不需要回读主机。

        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        # [CN] 块表由调度器统一维护，所有后端共用同一份，避免重复构造。

        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        # [CN] 因果性由调度器决定：PrefixLM 场景下它可能是逐请求的张量。

        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        # [CN] 两处关闭 AOT：投机解码的草稿步（步数少，调度开销收不回来），
        # [CN] 以及 batch invariance（调度结果随形状变化，会破坏逐位一致）。

        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        # [CN] 只在首次 build 时探测滑窗配置；若模型里存在多种滑窗配置，
        # [CN] 就无法给 AOT 一个统一值，只能整体关闭 AOT。

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                # [CN] 首次 build 时层已构造完毕，这时才能枚举出所有滑窗配置。

                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                # [CN] 混用多种滑窗配置时无法给出统一值，只能整体放弃 AOT。

                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        # [CN] SplitKV 的中间缓冲是 [num_splits, heads, tokens, head_size]，
        # [CN] 只在 CUDA graph 下才愿意为它预分配，其余情况交给启发式。

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        # [CN] 批不变模式强制单分片：SplitKV 的归约顺序会随分片数变化。

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        # [CN] 是否真的划算由 use_cascade_attention() 的启发式决定，
        # [CN] 这里只是把调度器算出的公共前缀长度转成开关。

        use_cascade = common_prefix_len > 0
        # [CN] 先置 0：这个值既当长度用，也当"是否跳过上下文注意力"的开关用。

        max_dcp_context_kv_len = 0
        # [CN] 非 DCP 路径保持 None，下游靠它判断要不要走两段式注意力。

        dcp_context_kv_lens = None
        num_decode_reqs = 0
        num_prefill_reqs = 0
        num_decode_tokens = 0
        num_prefill_tokens = 0

        # [CN] 三条级联专用字段先置空：只有走级联分支才会被填上设备张量。

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            # [CN] 上下文 KV 长度 = 总长 - 本次 query 长度。这一步纯设备侧完成，无同步。

            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            # [CN] 按交错粒度把全局上下文长度映射到本 rank 负责的那一段。

            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            # [CN] 尾部清零：补齐的请求若留旧值，内核会读到不存在的 KV 长度。

            self._dcp_context_kv_lens[num_reqs:] = 0
            # [CN] 只暴露前 num_reqs 项，尾部补齐区对内核不可见。

            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # [CN] 用 CPU 侧的"长度上界"判断能否完全跳过上下文注意力。
            # [CN] 上界足够小就说明本 rank 那段上下文是空的，可以省一次内核。

            skip_dcp_context_attention = False
            # [CN] 用 CPU 侧上界做判断：真实值在设备上，取回来会强制同步。

            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                query_lens_cpu = (
                    common_attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]
                    - common_attn_metadata.query_start_loc_cpu[:num_reqs]
                )
                context_kv_lens_cpu = (
                    common_attn_metadata.seq_lens_cpu_upper_bound[:num_reqs]
                    - query_lens_cpu
                )
                skip_dcp_context_attention = should_skip_dcp_context_attention(
                    context_kv_lens_cpu
                )

            # [CN] 纯 decode 步（max_query_len==1）不需要拆分计数，省掉这次 CPU 计算。

            if max_query_len > 1:
                (
                    num_decode_reqs,
                    num_prefill_reqs,
                    num_decode_tokens,
                    num_prefill_tokens,
                ) = split_dcp_context_queries(
                    common_attn_metadata.query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu_upper_bound,
                    max_query_len,
                    num_actual_tokens,
                )

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            # [CN] 跳过上下文注意力后，AOT 调度结果也必须置空，否则形状对不上。

            if skip_dcp_context_attention:
                max_dcp_context_kv_len = 0
                scheduler_metadata = None
            else:
                # [CN] 分区总数 = rank 数 × 交错粒度，用来推算单 rank 的最大 KV 长度上界。

                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_dcp_context_kv_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

                scheduler_metadata = self._get_scheduler_metadata(
                    aot_schedule=aot_schedule,
                    batch_size=num_reqs,
                    cu_query_lens=query_start_loc,
                    max_query_len=max_query_len,
                    seqlens=dcp_context_kv_lens,
                    max_seq_len=max_dcp_context_kv_len,
                    causal=False,
                    max_num_splits=max_num_splits,
                )
        # [CN] 级联：把整批 query 拼成"一批"去算共享前缀，再逐请求算各自后缀，
        # [CN] 最后按 log-sum-exp 合并。前缀部分 causal=False 是必须的。

        elif use_cascade:
            # [CN] 级联前缀视为一个整体序列，所以偏移表只有 [0, num_actual_tokens]。

            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            # [CN] 后缀长度用设备张量直接相减，避免把 seq_lens 拷回主机再算。

            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = self._get_scheduler_metadata(
                aot_schedule=aot_schedule,
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
                max_num_splits=max_num_splits,
            )
            scheduler_metadata = self._get_scheduler_metadata(
                aot_schedule=aot_schedule,
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
                max_num_splits=max_num_splits,
            )
        else:
            scheduler_metadata = self._get_scheduler_metadata(
                aot_schedule=aot_schedule,
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                # [CN] 普通路径：一次算完，causal 直接取调度器给的值。

                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
                max_num_splits=max_num_splits,
            )
        # [CN] 三条分支汇合后统一做一次"写回固定 buffer"的处理。

        scheduler_metadata = self._store_scheduler_metadata(scheduler_metadata)

        # [CN] FA 的 dynamic_causal 只认 int32，传 bool 张量会在内核里被解释错。

        if isinstance(causal, torch.Tensor) and causal.dtype != torch.int32:
            causal = causal.to(torch.int32)

        # [CN] 一次性构造 dataclass：字段很多但都是引用赋值，没有数据拷贝。

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            num_decode_reqs=num_decode_reqs,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )

        # Compute mm_prefix range tensor if the batch contains
        # multimodal tokens with bidirectional ranges.  Built for every FA
        # group; Gemma4 nulls the field for its non-sliding layers.
        # [CN] mm_prefix 的区间是在 CPU 侧算好再异步拷上去的：
        # [CN] 上界对 prefill 行是精确值，decode 行只会偏大，而偏大只会让它
        # [CN] 更彻底地落在所有区间之外，因此用上界是安全的。

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None and self.mm_prefix_query_ranges_np is not None:
            # The upper bound is exact for prefill rows, which is where
            # mm_prefix ranges live; decode rows only ever get an optimistic
            # (larger) context, moving them further past every range.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None, (
                "mm_prefix requires seq_lens_cpu_upper_bound"
            )
            # [CN] 在 pinned CPU 缓冲上填充区间，返回真实写入的行数。

            num_mm_tokens = fill_mm_prefix_query_ranges(
                self.mm_prefix_query_ranges_np,
                mm_ranges,
                common_attn_metadata.query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu_upper_bound,
            )
            # [CN] 为 0 时不挂字段：下游靠 is not None 判断，省掉一次空张量上传。

            if num_mm_tokens > 0:
                assert self.mm_prefix_query_ranges_cpu is not None
                assert self.mm_prefix_query_ranges_gpu is not None
                mm_query_ranges = self.mm_prefix_query_ranges_gpu[:num_mm_tokens]
                mm_query_ranges.copy_(
                    self.mm_prefix_query_ranges_cpu[:num_mm_tokens],
                    non_blocking=True,
                )
                attn_metadata.mm_prefix_query_range_tensor = mm_query_ranges

        # R-SWA: copy prefix lengths into persistent buffers (outside the
        # compiled region) so forward() never allocates during CUDA graph
        # capture.  rswa_window is a static model config scalar read here.
        if (
            self.rswa_window is not None
            and common_attn_metadata.rswa_prefix_lens is not None
        ):
            assert self.persistent_rswa_prefix_lens is not None
            assert self.persistent_rswa_window_tensor is not None
            # [CN] 拷贝而非引用：源张量每步都可能换，常驻缓冲必须保持地址不变。

            src = common_attn_metadata.rswa_prefix_lens
            rswa_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            rswa_prefix_lens.copy_(src[:num_reqs], non_blocking=True)
            attn_metadata.rswa_prefix_lens = rswa_prefix_lens
            attn_metadata.rswa_window = self.rswa_window
            attn_metadata.rswa_window_tensor = self.persistent_rswa_window_tensor

        return attn_metadata

    # [CN] 浅拷贝后只换两张表：metadata 里其余字段在多步投机中保持不变。

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    # [CN] 草稿步的序列长度变了，AOT 调度结果必须重算并写回原 buffer。

    def update_draft_decode_metadata(self, metadata: FlashAttentionMetadata) -> None:
        # [CN] 没有 AOT 调度就无需刷新，直接复用上一步的 metadata 对象。

        if metadata.scheduler_metadata is None:
            return

        # [CN] 草稿步只会有 decode 请求，优先用专门的计数更准确。

        num_reqs = metadata.num_decode_reqs or metadata.seq_lens.shape[0]

        # [CN] 草稿步刷新目前只在非 DCP 下成立，这里用断言把假设钉死。

        assert self.dcp_world_size == 1
        assert not metadata.use_cascade

        scheduler_metadata = self._get_scheduler_metadata(
            aot_schedule=True,
            batch_size=num_reqs,
            cu_query_lens=metadata.query_start_loc,
            max_query_len=metadata.max_query_len,
            seqlens=metadata.seq_lens,
            max_seq_len=metadata.max_seq_len,
            causal=metadata.causal,
            max_num_splits=metadata.max_num_splits,
        )

        metadata.scheduler_metadata = self._store_scheduler_metadata(scheduler_metadata)

    # [CN] hd256 专用内核要求页对齐，而级联的前缀长度未必页对齐，直接禁用。

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.fa4_hd256:
            # Cascade may use a non-page-aligned prefix length.
            return False
        return use_cascade_attention(*args, **kwargs)


# [CN] 真正执行注意力的地方。所有张量形状约定：
# [CN] q/k/v 均为 [num_tokens, num_heads, head_size]，KV 缓存为
# [CN] [num_blocks, num_kv_heads, block_size, 2*head_size]（K 和 V 拼在最后一维）。

class FlashAttentionImpl(AttentionImpl):
    # [CN] FA 能顺带返回 softmax_lse，这是 DCP 合并与级联合并的前提。

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
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        # [CN] 滑窗统一表示成 (左, 右)：(-1,-1) 表示全局注意力。
        # [CN] 因果 decoder 是 (w-1, 0)，encoder-only 是 (w-1, w-1) 因为双向。
        # [CN] 注意 w-1：FA 的语义是"距离"，所以窗口大小 w 对应距离 w-1。

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        # [CN] FA 用 0 表示不加软帽，所以要把 None 归一成 0，而不是跳过参数。

        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # [CN] GQA 分组比：大于 1 时 FA 内部会启用打包 GQA 或 FlashDecoding。

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        vllm_config = get_current_vllm_config_or_none()
        # [CN] encoder 类注意力不存在"历史 KV"，也就不参与缓存分配。

        uses_kv_cache = attn_type not in (
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )
        # The final KV cache block size is unavailable during construction.
        # [CN] 版本不是装机版本而是"本层能用的版本"：alibi、sink、softcap
        # [CN] 各自会淘汰掉某些版本，最终取交集。

        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
            has_sinks=sinks is not None,
            requires_softcap=bool(self.logits_soft_cap),
            supports_fa4_hd256=True,
        )
        # [CN] hd256 是 FA4 在 Blackwell 上的一条特化路径，需要额外的页对齐约束。

        self.fa4_hd256 = self.vllm_flash_attn_version == 4 and uses_fa4_hd256_kernel(
            head_size
        )
        # [CN] hd256 内核缺少 seqused_k 支持，带滑窗的 encoder 输入只能回退 FA2。

        if self.fa4_hd256 and not uses_kv_cache and sliding_window is not None:
            # The hd256 kernel requires seqused_k for local attention.
            logger.warning_once(
                "FA4's Blackwell head_size=256 kernel does not support local "
                "attention on encoder inputs, defaulting to FA version 2."
            )
            self.vllm_flash_attn_version = 2
            self.fa4_hd256 = False
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )
        # Cache the batch invariant result for use in forward passes
        # [CN] 构造期固化环境变量：前向里读 os.environ 太慢。

        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(
            self.kv_cache_dtype
        ) and not flash_attn_supports_kv_cache_dtype(
            self.kv_cache_dtype,
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
            head_size_v=head_size,
            has_sinks=sinks is not None,
            requires_softcap=bool(self.logits_soft_cap),
            supports_fa4_hd256=True,
        ):
            raise NotImplementedError(
                f"FlashAttention does not support {self.kv_cache_dtype}"
                " kv-cache on this device."
            )

        # [CN] sink 必须逐头给定，且头数要与本层 query 头数一致。

        self.sinks = sinks
        # [CN] sink 是逐头的可学习参数，用于让注意力始终保留若干"锚点"位置。
        # [CN] 它在内核里以额外的一列 logits 参与 softmax，所以会改变 LSE 的语义，
        # [CN] 这也是级联路径里必须让 sink 只进前缀段的原因。

        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        # [CN] 查询侧量化是较新的能力，老版本只能传 fp16/bf16 的 q。

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        # [CN] DCP 两种通信后端：a2a（all-to-all 后本地归约）与默认的
        # [CN] all-gather + reduce-scatter。选哪个由并行配置决定。

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        # [CN] 两种合并函数签名相同、语义等价，选择只影响通信模式与显存占用。

        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        # [CN] DCP 的中间输出缓冲要按模型 dtype 预借，这里先占位。

        self._dcp_dtype: torch.dtype | None = None
        self._dcp_max_num_tokens: int = 0
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype
            self._dcp_max_num_tokens = (
                vllm_config.scheduler_config.max_num_batched_tokens
            )

    # [CN] 热路径。分段 CUDA graph 下这里跑的是 eager PyTorch，
    # [CN] 所以任何 view / slice 都有可观的 CPU 开销，改动必须实测。

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # [CN] 融合输出量化尚未实现：FA 内核不支持直接写出带 scale 的量化结果。

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        # [CN] 显存 profiling 的空跑：直接清零输出即可，不碰任何内核。

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # [CN] 本地变量化：后续热路径里反复读属性会多一次 Python 属性查找。

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        # [CN] 后续所有切片都以它为界，张量尾部是给 CUDA graph 预留的补齐区。

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        # [CN] 编码器分支不读 KV 缓存，直接拿当步的 K/V 做一次双向注意力。

        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        # [CN] 把拼在一起的 K/V 缓存拆开：转置是为了让 num_kv_heads 变成第 2 维，
        # [CN] 得到 [num_blocks, block_size, num_kv_heads, head_size] 的视图。

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        # [CN] 大小为 1 的维度（如 TP 下 num_kv_heads=1）会产生退化的 stride 0，
        # [CN] 而 FA3/4 的 TMA 要求 stride 至少 16 字节对齐，必须显式修正。

        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        # [CN] 只有真的修过 stride 才打日志：正常路径下这一步是零开销。

        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttention): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        # [CN] fp8 缓存底层还是 uint8 存储，这里只是换一个 dtype 视图给内核看。

        # [CN] 前向里同样要换 dtype 视图，与 build 侧保持一致。

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # [CN] 主路径（非级联）：绝大多数步走这里，级联是罕见分支。

        if not attn_metadata.use_cascade:
            # [CN] 非级联路径的这四个量直接来自 metadata，不做任何加工。

            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            # [CN] descale 形状是 (序列数, KV 头数)：用 expand 广播而不是真的复制数据。

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            # [CN] DCP 单独走 _forward_with_dcp：它要算两次注意力再合并。

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                # [CN] 非 DCP 路径下 causal 可能来自调度器的每请求张量，直接透传。

                causal = attn_metadata.causal
                is_dynamic_causal = isinstance(causal, torch.Tensor)

                # The layer's own window wins over the group's: one KV cache
                # group can hold both windowed and global layers (e.g. Gemma-3
                # with the hybrid KV cache manager disabled), and the group spec
                # cannot describe both.
                # [CN] 层自己的窗口优先于 KV cache group 的窗口：同一个组里可能既有
                # [CN] 局部层又有全局层（如关闭混合缓存管理器的 Gemma-3），
                # [CN] 组级 spec 无法同时描述两者。

                window = _maybe_symmetrize_window(self.sliding_window, causal)
                sliding_window_size: list[int] | None = (
                    list(window) if window is not None else None
                )

                # [CN] mm_prefix 只在纯因果、非动态 causal、且 FA4 时才启用：
                # [CN] 它依赖 FA4 的 mask_mod 扩展点。

                mm_prefix_query_ranges = attn_metadata.mm_prefix_query_range_tensor
                mm_mask_mod = None
                mm_aux = None
                if (
                    mm_prefix_query_ranges is not None
                    and not is_dynamic_causal
                    and causal is True
                    and self.vllm_flash_attn_version == 4
                ):
                    # Triton convention: 1 + window_size[0]. Global layers store
                    # (-1, -1) → sw stays None.
                    layer_window = self.sliding_window
                    sw_val = (
                        1 + layer_window[0]
                        if layer_window is not None and layer_window[0] >= 0
                        else None
                    )
                    # Gemma4: also clamp the bidirectional block to the
                    # sliding window when the layer opts in
                    # (mm_prefix_clamp_sliding_window flag from PR #47217).
                    mm_clamp_sw = 0
                    if (
                        getattr(layer, "mm_prefix_clamp_sliding_window", False)
                        and sw_val is not None
                    ):
                        mm_clamp_sw = sw_val
                    mm_mask_mod = _make_mm_prefix_mask_mod(
                        sliding_window=mm_clamp_sw,
                        sliding_window_left=sw_val,
                    )
                    mm_aux = [mm_prefix_query_ranges, attn_metadata.query_start_loc]
                    # mm_prefix is (causal ∧ window) ∨ bidirectional-range —
                    # not ⊆ causal. FA #155 stopped auto-clearing causal/local
                    # when mask_mod is set, so the caller must disable them or
                    # the built-in causal path shorts out / clips the mask_mod.
                    causal = False
                    sliding_window_size = None

                # R-SWA: use CuTE-DSL mask_mod on FA4 for exact token-level
                # mask without block-size approximation.  The mask_mod encodes
                # "causal AND (kv < prefix_len OR q - kv < rswa_window)", which
                # supersedes any FA-layer sliding_window_size parameter.
                # [CN] R-SWA 用 CuTE-DSL 写 mask_mod，能做到逐 token 精确掩码，
                # [CN] 避免 FA 内置滑窗按块近似带来的误差。

                rswa_mask_mod_fn = None
                rswa_aux = None
                if (
                    attn_metadata.rswa_prefix_lens is not None
                    and self.vllm_flash_attn_version == 4
                    and not is_dynamic_causal
                ):
                    rswa_mask_mod_fn = _make_rswa_mask_mod()
                    rswa_aux = [
                        attn_metadata.rswa_prefix_lens.to(torch.int32),
                        attn_metadata.rswa_window_tensor,  # pre-allocated CUDA tensor
                    ]
                    # mask_mod fully expresses R-SWA; disable FA's own window.
                    sliding_window_size = None

                dynamic_causal = None
                # [CN] 动态 causal 是 FA4 独有；此时若有窗口，因果性由窗口自己表达，
                # [CN] 必须把 causal 置 False，否则两套掩码叠加会算错。

                if isinstance(causal, torch.Tensor):
                    if self.vllm_flash_attn_version != 4:
                        raise NotImplementedError(
                            "Per-sequence causal requires FA4. Current version: "
                            f"FA{self.vllm_flash_attn_version}"
                        )
                    dynamic_causal = causal
                    has_window = (
                        sliding_window_size is not None and sliding_window_size[1] >= 0
                    )
                    causal = not has_window

                num_splits = attn_metadata.max_num_splits
                # [CN] hd256 的三条硬约束：长度页对齐、块表宽度精确、禁止 SplitKV。

                if self.fa4_hd256:
                    # hd256 requires page-aligned lengths, exact-width block
                    # tables, and no SplitKV.
                    # [CN] 向上取整到整页：hd256 内核按页粒度取数，长度必须是页的整数倍。

                    num_pages = cdiv(max_seqlen_k, FA4_HD256_PAGE_SIZE)
                    max_seqlen_k = num_pages * FA4_HD256_PAGE_SIZE
                    # [CN] 块表宽度必须与实际页数严格相等，多一列都会被内核当成有效块。

                    block_table = block_table[:, :num_pages]
                    num_splits = 1

                # [CN] 统一的 FA 入口。注意 mask_mod 与 aux_tensors 是"或"的关系：
                # [CN] 两个特性互斥，不会同时出现。

                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    dynamic_causal=dynamic_causal,
                    num_splits=num_splits,
                    s_aux=self.sinks,
                    mask_mod=rswa_mask_mod_fn or mm_mask_mod,
                    aux_tensors=rswa_aux or mm_aux,
                )
                return output

        # Cascade attention (rare case).
        # [CN] 级联注意力：前缀算一次、后缀算一次，最后按 LSE 合并。
        # [CN] 只在批内请求共享长前缀时才划算，启发式见 use_cascade_attention。

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    # [CN] 写缓存独立成一步：这里只做离散散射写，不经过 TMA，
    # [CN] 所以不需要前面那套 stride 修正。

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # [CN] 编码器层永远不写缓存，写缓存这一步对它必须是空操作。

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        # [CN] 不需要对 key/value 做 [:num_actual_tokens] 切片：
        # [CN] 内核是以 slot_mapping 的形状为准来判断真实 token 数的。

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # [CN] DCP 前向：先对本 rank 的上下文 KV 算一次，再对本地 query 算一次，
    # [CN] 最后用各自的 softmax_lse 做数值稳定的合并。

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        if attn_metadata.max_dcp_context_kv_len == 0:
            flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                out=output,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_k=max_seqlen_q,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window)
                if self.sliding_window is not None
                else None,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
            )
            return output

        # [CN] 沿 head 维 all-gather：每个 rank 拿全量 query 头，但只喂自己那段 KV。

        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        # [CN] DCP 上下文段也要遵守滑窗：窗口值先转成 FA 要求的 [左, 右] 列表。

        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        n = query_across_dcp.shape[0]
        num_reqs = cu_seqlens_q.shape[0] - 1
        num_decodes = attn_metadata.num_decode_reqs
        num_context_prefills = attn_metadata.num_prefill_reqs
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_context_prefill_tokens = attn_metadata.num_prefill_tokens
        split_dcp_context = should_split_fa2_dcp_context_attention(
            self.vllm_flash_attn_version,
            max_seqlen_q,
            num_reqs,
            num_decodes,
            num_context_prefills,
        )
        dcp_context_out_tokens = max(n, self._dcp_max_num_tokens)
        dcp_context_out_spec = (
            (
                dcp_context_out_tokens,
                self.num_heads * self.dcp_world_size,
                self.head_size,
            ),
            self._dcp_dtype,
        )
        # [CN] 从 workspace 管理器借缓冲而不是现分配：graph 捕获期间禁止分配。

        (dcp_context_out_workspace,) = current_workspace_manager().get_simultaneous(
            dcp_context_out_spec,
        )
        dcp_context_out = dcp_context_out_workspace[:n]

        # [CN] FA2 下混合 decode/prefill 批无法一次算完上下文注意力，
        # [CN] 必须拆成两段分别处理——这是纯 workaround。

        if split_dcp_context:
            # TODO: Remove this DCP + FA2 mixed decode/prefill workaround once
            # FA4 supports this Qwen3.5 shape.
            assert attn_metadata.dcp_context_kv_lens is not None
            assert attn_metadata.max_dcp_context_kv_len is not None
            assert self.vllm_flash_attn_version is not None
            context_attn_out, context_lse = run_split_fa2_dcp_context_attention(
                flash_attn_varlen_func,
                query_across_dcp,
                key_cache,
                value_cache,
                dcp_context_out,
                cu_seqlens_q,
                max_seqlen_q,
                attn_metadata.dcp_context_kv_lens,
                attn_metadata.max_dcp_context_kv_len,
                self.scale,
                self.alibi_slopes,
                sliding_window_size,
                block_table,
                self.logits_soft_cap,
                self.vllm_flash_attn_version,
                q_descale,
                k_descale,
                v_descale,
                attn_metadata.max_num_splits,
                self.num_heads,
                self.dcp_world_size,
                num_decodes,
                num_context_prefills,
                num_decode_tokens,
                num_context_prefill_tokens,
            )
        else:
            context_attn_out, context_lse = flash_attn_varlen_func(
                q=query_across_dcp,
                k=key_cache,
                v=value_cache,
                out=dcp_context_out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=attn_metadata.dcp_context_kv_lens,
                max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
                softmax_scale=self.scale,
                causal=False,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
            )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        # [CN] 合并两步：先通信归约输出，同时把 LSE 一并归约好留给后面用。

        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        # [CN] 第二次注意力只针对本 rank 自己那段新 query，kv 长度就是 query 长度。

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # [CN] 合并前形状必须一致；形状不对说明两个 rank 的切分不一致。

        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        # [CN] 结果直接写进 output：合并是原地操作，不额外占显存。

        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    # [CN] 编码器注意力：不碰 KV 缓存，直接对 Q/K/V 做一次双向 FA。

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        # [CN] 编码器路径不支持量化：q/k/v 直接来自计算图，没有 descale 的落点。

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # [CN] 编码器里 query 与 key 长度相同，所以两张偏移表可以共用。

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        # [CN] encoder 的 query 与 key 同长，两张偏移表因此可以复用同一个张量。

        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        # [CN] 编码器没有历史 KV，最大 key 长度就等于最大 query 长度。

        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)  # type: ignore[operator]
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),  # type: ignore[operator]
            v_descale=layer._v_scale.expand(descale_shape),  # type: ignore[operator]
            # The hd256 kernel does not support SplitKV.
            num_splits=1 if self.batch_invariant_enabled or self.fa4_hd256 else 0,
            s_aux=self.sinks,
        )

        return output


# [CN] 用 CuTE-DSL 生成 mask_mod。加 functools.cache 是性能关键：
# [CN] FA4 用 repr(闭包单元) 参与编译键，不缓存的话每次前向都会
# [CN] 因为函数对象地址变化而触发一次完整 JIT 重编译。
@functools.cache
def _make_mm_prefix_mask_mod(
    sliding_window: int = 0,
    sliding_window_left: int | None = None,
):
    """Build a CuTE-DSL mask_mod implementing
    ``(causal AND sliding_window) OR mm_prefix``.

    Cached so identical ``(sliding_window, sliding_window_left)`` reuse the
    same function object. FA4's ``hash_callable`` mixes ``repr()`` of closure
    cells into the compile key; the nested ``_load_q_range`` would otherwise
    get a new address each call and force a full JIT recompile every forward.

    The FA4 kernel passes *local* ``q_idx`` (0-based within the current
    prefill chunk) while ``kv_idx`` is absolute (0-based over the full
    KV cache).  We recover the absolute Q position via
    ``q_abs = q_idx + seqlen_k - seqlen_q`` (the context-length offset)
    so that causal, sliding-window, and mm_prefix range comparisons all
    use consistent absolute positions.  This matches the Triton
    reference path (``compute_kv_seq_mask``).

    ``aux_tensors[0]`` holds the absolute ``[start, end]`` bounds of the
    mm_prefix range containing each scheduled query token (``(-1, -1)`` when
    none), and ``aux_tensors[1]`` is ``cu_seqlens_q``, used to turn the local
    ``q_idx`` into a packed row index.  Because mm_prefix ranges never overlap,
    ``r_start <= kv_idx <= r_end`` is exactly "query and key share a range", so
    no key-side lookup is needed and ``kv_idx`` is never used as an index.
    The ``(-1, -1)`` sentinel falls out for free: ``kv_idx <= -1`` is false for
    every valid key.

    ``sliding_window_left`` enforces the sliding window on the causal
    term (None = full causal, no window).  ``sliding_window`` clamps the
    bidirectional block to the window (0 = unclamped; >0 = Gemma4 local
    layers via ``mm_prefix_clamp_sliding_window``).
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32  # type: ignore[attr-defined]

    from vllm.vllm_flash_attn.cute.utils import (  # type: ignore[import-untyped]
        scalar_to_ssa,
        ssa_to_scalar,
    )

    @cute.jit
    def _load_q_range(q_idx, seqlen_info, aux_tensors, batch_idx):
        """Load the mm_prefix range bounds for this query row.

        Both loads depend only on the query index, so they hoist out of the
        unrolled per-element mask loop.  ``ssa_to_scalar`` reads lane 0, so one
        call must not span query rows (hence the ``__vec_size__`` pin below).
        Clamping keeps ``token_idx`` in bounds for partial tiles and padded
        (``seqlen_q == 0``) rows; neither produces output.
        """
        # [CN] 从 aux 张量取区间表：aux 是 FA4 给 mask_mod 传自定义数据的通道。

        q_ranges = aux_tensors[0]
        cu_seqlens_q = aux_tensors[1]
        # [CN] batch_idx 是 SSA 张量，取第 0 个元素得到当前序列的批内序号。

        b = batch_idx[0]
        q_local = cutlass.min(ssa_to_scalar(q_idx), seqlen_info.seqlen_q - Int32(1))
        token_idx = cutlass.max(cu_seqlens_q[b] + q_local, Int32(0))
        return (
            scalar_to_ssa(q_ranges[token_idx, 0], Int32),
            scalar_to_ssa(q_ranges[token_idx, 1], Int32),
        )

    # [CN] 有滑窗时掩码 = (因果 ∧ 窗口) ∨ 双向段，需要额外的窗口比较。

    if sliding_window_left is not None:

        @cute.jit
        def mm_prefix_mask_mod(
            batch_idx: cute.TensorSSA,
            head_idx: cute.TensorSSA,
            q_idx: cute.TensorSSA,
            kv_idx: cute.TensorSSA,
            seqlen_info,
            aux_tensors,
        ):
            # [CN] 关键换算：局部 q 下标 + (k 长 - q 长) = 绝对 token 位置。

            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            sw = scalar_to_ssa(Int32(sliding_window_left), Int32)
            keep = (kv_idx <= q_abs) & ((q_abs - kv_idx) < sw)
            r_start, r_end = _load_q_range(q_idx, seqlen_info, aux_tensors, batch_idx)
            mm = (kv_idx >= r_start) & (kv_idx <= r_end)
            if sliding_window > 0:
                mm = mm & ((q_abs - kv_idx) < sw)
            keep = keep | mm
            return keep

    else:

        @cute.jit
        def mm_prefix_mask_mod(
            batch_idx: cute.TensorSSA,
            head_idx: cute.TensorSSA,
            q_idx: cute.TensorSSA,
            kv_idx: cute.TensorSSA,
            seqlen_info,
            aux_tensors,
        ):
            ctx_off = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
            q_abs = q_idx + ctx_off
            keep = kv_idx <= q_abs
            r_start, r_end = _load_q_range(q_idx, seqlen_info, aux_tensors, batch_idx)
            keep = keep | ((kv_idx >= r_start) & (kv_idx <= r_end))
            return keep

    # [CN] 允许内核整块跳过全掩码的 KV 块，省掉无效的数据加载。

    mm_prefix_mask_mod.use_fast_sampling = True
    mm_prefix_mask_mod.__vec_size__ = 1  # _load_q_range takes lane 0 of q_idx
    return mm_prefix_mask_mod


# [CN] R-SWA 的掩码语义：因果 AND (在 prompt 前缀内 OR 落在滑窗内)。
# [CN] 注意 FA 传进来的 q/kv 下标都是"本序列内"的局部下标，
# [CN] 要还原绝对位置必须加上 (seqlen_k - seqlen_q) 这个偏移。

def _make_rswa_mask_mod():
    """Build a CuTE-DSL mask_mod for Reference Sliding Window Attention (R-SWA).

    FA4 varlen + paged-KV convention (verified from cute/mask.py apply_mask):
      q_idx  = LOCAL query-token offset (0 .. seqlen_q - 1) within this sequence.
      kv_idx = LOCAL KV-token position (0 .. seqlen_k - 1) within this sequence.

    To recover the ABSOLUTE token position (needed for causal and the sliding
    window distance), use the standard offset:
      abs_q = q_idx + (seqlen_k - seqlen_q)

    R-SWA keep condition:
      abs_q >= kv_idx                    (causal: KV at or before the query)
      AND (kv_idx < prefix_len           (global prefix is always visible)
           OR  abs_q - kv_idx < window)  (generated tokens: sliding window)

    aux_tensors[0]: prefix_lens [num_reqs] int32 — per-request prefill length.
    aux_tensors[1]: rswa_window [1]        int32 — decode sliding window size.

    use_fast_sampling=True lets FA4 skip fully-masked KV blocks (gap blocks)
    without loading their data.
    """
    import cutlass.cute as cute
    from cutlass import Int32  # type: ignore[attr-defined]

    from vllm.vllm_flash_attn.cute.utils import (  # type: ignore[import-untyped]
        scalar_to_ssa,
    )

    @cute.jit
    def rswa_mask_mod(
        batch_idx: cute.TensorSSA,
        head_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ):
        b = batch_idx[0]
        prefix_len = scalar_to_ssa(aux_tensors[0][b], Int32)
        window = scalar_to_ssa(aux_tensors[1][0], Int32)
        # Convert local q offset to absolute token position.
        offset = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
        abs_q = q_idx + offset
        causal = kv_idx <= abs_q
        in_prefix = kv_idx < prefix_len
        in_window = (abs_q - kv_idx) < window
        return causal & (in_prefix | in_window)

    # [CN] R-SWA 的掩码大片连续为假，快采样能显著省掉无效访存。

    rswa_mask_mod.use_fast_sampling = True
    return rswa_mask_mod


# [CN] 级联注意力的收益判断。前面几个 return False 都是"早退"：
# [CN] 绝大多数批在第一个 common_prefix_len < 256 就返回了。

def use_cascade_attention(
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
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    # [CN] 256 是拍出来的经验阈值，放在最前面是为了让绝大多数批立刻返回。

    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    # [CN] 这三项都会让前缀段与后缀段的掩码不一致，级联的数学前提不成立。

    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    # [CN] query_lens 是 numpy 数组，长度即请求数，取长度不触发设备同步。

    num_reqs = len(query_lens)
    # [CN] 请求太少时级联省下的带宽抵不上多一次内核启动的开销。

    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    # [CN] 级联与 DCP 的序列切分语义冲突，直接禁用。

    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    # [CN] GQA 比 >1 时 FA 会启用 FlashDecoding，此时级联未必更快，
    # [CN] 需要一个粗略的 CTA 占用模型来比较。

    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    # [CN] 只有 GQA 且全批都是 decode（query 长度全 1）时 FA 才启用 FlashDecoding。

    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    # [CN] 下面这套性能模型用的是默认 tile 尺寸，实际内核可能取 64 或 256。

    q_tile_size = 128
    kv_tile_size = 128
    # [CN] 前缀被切成多少个 KV tile，决定了级联的串行深度。

    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    # [CN] 级联的并行度只跟头数和 token 数有关，与前缀长度无关。

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    # [CN] 波次数 = CTA 数 / SM 数：这就是性能模型里"时间"的代理量。

    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    # [CN] FlashDecoding 的并行度随请求数、KV 头数和前缀 tile 数一起放大。

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    # [CN] 用"波次 × 前缀 tile 数"粗估级联耗时，与 FlashDecoding 的 CTA 数对比。

    return cascade_time < flash_decoding_time


# [CN] 级联实现：前缀与后缀两次 FA，都要求 return_softmax_lse，
# [CN] 再用 merge_attn_states 按 log-sum-exp 做数值稳定合并。

def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    # [CN] ALiBi 的位置偏置依赖绝对下标，切分前后缀两段会让偏置算错。

    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    # [CN] 块大小从缓存张量形状反推，不额外存一份配置，避免不一致。

    block_size = key_cache.shape[-3]
    # [CN] 不对齐的前缀无法用"块表前 N 块"表达，级联的共享假设就不成立。

    assert common_prefix_len % block_size == 0
    # [CN] 前缀长度必须块对齐，否则无法用块表前 N 项直接索引共享前缀。

    num_common_kv_blocks = common_prefix_len // block_size
    # [CN] 前缀不足一块说明整段前缀都在同一块内，级联没有意义。

    assert num_common_kv_blocks > 0
    # [CN] 前缀段的"序列数"恒为 1，形状退化为 (1, num_kv_heads)。

    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # [CN] 前缀段：整批 query 当成"一个序列"去 attend 共享前缀，故 causal=False。
    # [CN] sink 只在这里并入，它已经被写进 prefix_lse，后续合并自然生效。

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # [CN] 后缀段：正常逐请求因果注意力，块表要跳过前缀占用的前 N 块。

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    # [CN] 合并公式按 lse 加权：out = (o1*e^{l1} + o2*e^{l2}) / (e^{l1}+e^{l2})。
    # [CN] 用 lse 而不是 softmax 概率是为了避免下溢。

    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
