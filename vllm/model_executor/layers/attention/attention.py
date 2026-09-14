# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 模型侧的注意力层（Attention）与后端之间的桥接层。
# [CN] 关键设计：Attention 层本身不含任何内核代码，它只做三件事——
# [CN]   1. 初始化时按 head_size/dtype/块大小等特征选出 AttentionBackend；
# [CN]   2. 把 KV cache 需求描述成 KVCacheSpec 交给框架统一分配显存；
# [CN]   3. 前向时把 query/key/value 转交给"自定义算子"执行。
# [CN] 第 3 步是重点：注意力被注册成 torch.compile 的**不透明自定义算子**
# [CN] （unified_attention_with_output），这样 Dynamo 不会试图拆开它，
# [CN] 从而既保住了图捕获，又允许算子内部走任意后端实现。

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import MLAAttention

logger = init_logger(__name__)


# [CN] KV 共享（某层复用更早层的 KV cache）的合法性校验。
# [CN] 必须在编译期就报错：运行期才发现会变成难以定位的显存踩踏。

def validate_kv_sharing_target(
    current_layer_name, target_layer_name, static_forward_context
):
    error_msg = (
        f"Specified KV sharing target layer for {current_layer_name} "
        f"is not valid: target layer {target_layer_name} "
    )

    # [CN] 自引用：等于要求"先算完自己再用自己"，必然读不到有效 KV。

    if current_layer_name == target_layer_name:
        raise ValueError(error_msg + "cannot be the same as the current layer.")

    # [CN] 不在静态上下文里，说明要么顺序不对、要么根本不是注意力层。
    # [CN] 下面靠层号大小区分这两种情况，好给出准确的错误信息。

    if target_layer_name not in static_forward_context:
        from vllm.model_executor.models.utils import extract_layer_index

        # If target layer name is not in the static fwd context, it means either
        # a) the target layer does not come BEFORE the current layer, or
        # b) the target layer is not an Attention layer that exists in the model
        current_layer_idx = extract_layer_index(current_layer_name)
        target_layer_idx = extract_layer_index(target_layer_name)
        if current_layer_idx <= target_layer_idx:
            raise ValueError(error_msg + "must come before the current layer.")
        else:
            raise ValueError(error_msg + "is not a valid Attention layer in the model.")

    # Currently KV sharing is only supported between layers of the same type
    # [CN] 只允许同类型层之间共享：全注意力与滑窗层的 KV 生命周期不同，
    # [CN] 混共享会让滑窗外的 token 被错误地截断或保留。

    target_layer_attn_type = static_forward_context[target_layer_name].attn_type
    expected = static_forward_context[current_layer_name].attn_type
    if target_layer_attn_type != expected:
        raise ValueError(
            error_msg + f"must be the same type as the current layer ({expected})."
        )


# [CN] 只有真正的量化方法才有 scale 权重可加载；
# [CN] UnquantizedLinearMethod 是"未量化"的占位实现，checkpoint 里没有 scale。

def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


# [CN] 在页预算内挑最大的内核块大小。背景：跳过量化的层会被填充到
# [CN] 统一的大页上，块越小则每块浪费的填充字节越多，所以要尽量取大。
# [CN] 页预算为 None 时不存在填充，取最小块即可（由 unify 按整数倍放大）。

def _largest_kernel_block_within(
    attn_backend: "type[AttentionBackend]",
    per_token_bytes: int,
    page_budget: int | None,
    fallback: int,
) -> int:
    """Largest supported kernel block size whose page fits in ``page_budget``.

    A padded spec (e.g. skip-quant layer) that pads its page up to a large shared page
    wastes ``page_budget - block*per_token`` bytes per block. Picking the largest kernel
    block whose natural page still fits under ``page_budget`` minimizes that waste.
    Falls back to the smallest supported block when ``page_budget`` is None (no padding
    — the block is handled by ``unify``'s integer scaling instead) or nothing fits.
    """
    from vllm.v1.attention.backend import MultipleOf

    # [CN] 优先用定值列表；只有全是 MultipleOf 时才退化为取它们的 base。

    sizes = attn_backend.get_supported_kernel_block_sizes()
    candidates = [s for s in sizes if isinstance(s, int)]
    if not candidates:
        candidates = [s.base for s in sizes if isinstance(s, MultipleOf)]
    if not candidates:
        return fallback
    smallest = min(candidates)
    if not page_budget or per_token_bytes <= 0:
        return smallest
    fitting = [b for b in candidates if b * per_token_bytes <= page_budget]
    return max(fitting) if fitting else smallest


# [CN] 把 q/k/v/prob 的 scale 统一置 1.0（即"不缩放"）。
# [CN] register_buffer=True 用于首次创建；False 用于加载后重置脏值。

def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer & AITER
    # [CN] 同时保留纯 Python float 副本与 CPU 张量副本：
    # [CN] FlashInfer / AITER 等后端的 scale 必须在主机侧而不能在设备上。

    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._k_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
    layer._v_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
    layer._prob_scale_float = 1.0


# [CN] Attention 与 MLAAttention 共用的量化初始化，避免两份重复实现。

def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
    """

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would
    # be an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e., should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    # [CN] 必须先注册成 buffer 才能进 state_dict —— 否则 .to(device) 搬不动它，
    # [CN] 内核访问留在 CPU 上的 scale 会直接 IMA。

    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    # [CN] 输出 scale 其实是"注意力之后那个量化算子的输入 scale"，
    # [CN] 此刻尚不知道，留待后续阶段填。

    layer._o_scale_float = None

    quant_method = (
        resolve_quant_method(quant_config, layer, prefix=prefix)
        if quant_config
        else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    # [CN] 有真量化方法才去 create_weights 建 k_scale/v_scale 参数，
    # [CN] 这样它们能被 checkpoint 直接加载。

    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if layer.kv_cache_dtype == "fp8_e5m2":
            # A compressed-tensors checkpoint stores fp8 KV scales only when it
            # declares a kv_cache_scheme; weight-only ones declare none and must
            # keep fp8_e5m2, the only fp8 KV dtype usable on Ampere.
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
                CompressedTensorsConfig,
                CompressedTensorsKVCacheMethod,
            )

            if not isinstance(quant_method, CompressedTensorsKVCacheMethod) or (
                cast(CompressedTensorsConfig, quant_method.quant_config).kv_cache_scheme
                is not None
            ):
                raise ValueError(
                    "fp8_e5m2 kv-cache is not supported with fp8 checkpoints."
                )
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


# [CN] 模型里实际使用的注意力层。它同时是 nn.Module（参与权重加载）
# [CN] 和 AttentionLayerBase（提供 scale 属性给后端读取）。

class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    # [CN] 构造期完成"选后端 + 建 impl + 注册进静态上下文"三件大事。
    # [CN] 注意 KV cache 张量此时还是空占位，真正绑定在 bind_kv_cache。

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        mm_prefix_clamp_sliding_window: bool = False,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        # [CN] 滑窗优先取"每层配置"，其次取 cache 配置的模型级值，都没有则关闭。

        sliding_window: int | None
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        # llm-compressor models declare an FP8 KV-cache scheme in their
        # checkpoint config. Honor it only when the user did not explicitly
        # pick a kv_cache_dtype; an explicit choice (e.g. bfloat16) must win.
        # The "auto" case is normally resolved upstream in
        # resolve_kv_cache_dtype_string, but we re-apply here defensively in
        # case anything bypassed that path.
        # [CN] llm-compressor 的 checkpoint 里会声明 FP8 KV 方案。
        # [CN] 只在用户没显式指定 kv_cache_dtype（即 auto）时才采纳，
        # [CN] 显式选择（如 bfloat16）必须优先。

        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None and kv_cache_dtype == "auto":
            kv_cache_dtype = "fp8"
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"

        # Check if per-head quant scales are required based on kv_cache_scheme
        # [CN] strategy == "attn_head" 表示每个注意力头一套独立 scale，
        # [CN] 这会影响后端选型（不是所有内核都支持逐头 scale）。

        use_per_head_quant_scales = (
            kv_cache_scheme is not None
            and kv_cache_scheme.get("strategy") == "attn_head"
        )

        # Skip quantization for specified layers
        # [CN] 按层号或按"是否滑窗"跳过量化：某些层对量化极敏感，
        # [CN] 保留 fp16/bf16 能显著缓解精度损失。

        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            skip = False
            # Check attention type
            if (
                sliding_window is not None
                and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
            ):
                skip = True
            # Check layer index
            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                skip = True
            if skip:
                kv_cache_dtype = "auto"
            logger.debug(
                "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
                prefix,
                kv_cache_dtype,
                sliding_window,
            )

        # [CN] 字符串 dtype -> torch.dtype。模型配置参与决策是因为
        # [CN] auto 要按模型原生 dtype 推。

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        if num_kv_heads is None:
            num_kv_heads = num_heads
        # [CN] GQA/MQA 的前提：Q 头数必须是 KV 头数的整数倍，否则无法分组。

        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        # [CN] 从透传参数里嗅探是否带 attention sink：sinks 是可选张量，
        # [CN] 存在即代表这个模型（如 Gemma 系列）需要 sink 语义。

        self.has_sink = extra_impl_args.get("sinks") is not None

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        # [CN] PrefixLM 形态的多模态模型：前缀 token 走双向注意力。

        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        # [CN] 模型初始化期间，全局默认 dtype 就是模型权重/激活的 dtype，
        # [CN] 直接取它比层层传参可靠。

        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                use_per_head_quant_scales=use_per_head_quant_scales,
                attn_type=attn_type,
                has_sliding_window=sliding_window is not None,
            )
        else:
            self.attn_backend = attn_backend
        # [CN] ALiBi 有两种斜率用法（直接减 / 先开方再减），后端必须明确表态
        # [CN] 支持哪一种，不能靠数值兜底。

        backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
        use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
        if use_alibi_sqrt and not backend_supports_alibi_sqrt:
            raise ValueError(
                f"use_alibi_sqrt is not supported by backend "
                f"{self.attn_backend.get_name()}."
            )
        self.use_alibi_sqrt = bool(use_alibi_sqrt)
        if backend_supports_alibi_sqrt:
            extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and envs.VLLM_BATCH_INVARIANT
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
            )
            cache_config.enable_prefix_caching = False

        # [CN] 分块注意力带回看（chunked local attention）目前只有 Triton 后端实现。

        if extra_impl_args.get("chunk_lookback", -1) > -1:
            assert self.attn_backend.get_name() == "TRITON_ATTN", (
                f"Chunked attention with lookback requires the Triton backend, "
                f"but got {self.attn_backend.get_name()}."
            )

        # [CN] FlexAttention 需要显式 tile 尺寸；批不变性要求 tile 不超过缓存块，
        # [CN] 否则一个 tile 跨块会导致归约顺序随批次变化。

        if self.attn_backend.get_name() == "FLEX_ATTENTION":
            block_m = vllm_config.attention_config.flex_attn_block_m
            block_n = vllm_config.attention_config.flex_attn_block_n

            if envs.VLLM_BATCH_INVARIANT and cache_config is not None:
                if block_m is not None and block_m > cache_config.block_size:
                    raise ValueError(
                        f"flex_attn_block_m ({block_m}) must be "
                        f"<= cache block size ({cache_config.block_size}) for "
                        f"batch invariance"
                    )
                if block_n is not None and block_n > cache_config.block_size:
                    raise ValueError(
                        f"flex_attn_block_n ({block_n}) must be "
                        f"<= cache block size ({cache_config.block_size}) for "
                        f"batch invariance"
                    )

            if block_m is not None:
                extra_impl_args.setdefault("block_m", block_m)
            if block_n is not None:
                extra_impl_args.setdefault("block_n", block_n)

        # [CN] 到这里才真正实例化 impl：前面所有校验都通过后，
        # [CN] 避免为一个注定要失败的后端付构造代价。

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an AttentionImpl subclass
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        # [CN] 反查枚举：用于需要离散后端标识的地方（序列化、日志、快速分支）。

        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        # [CN] 平台开关：CUDA/ROCm/CPU 走"不透明自定义算子"以获得可控的图行为；
        # [CN] 其它平台直接 Python 调用，交给 torch.compile 自行处理。

        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = vllm_config.compilation_config
        # [CN] 层名必须唯一：它是自定义算子在图上定位该层的唯一 key，
        # [CN] 重名会让两层拿到同一份元数据。

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window
        # (read by the Triton backend impl). Default False for all other models.
        self.mm_prefix_clamp_sliding_window = mm_prefix_clamp_sliding_window

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        # [CN] 空占位张量。use_direct_call=True 的路径不会读它
        # [CN] （那时 KV cache 从 forward context 里取）。

        self.kv_cache = torch.tensor([])

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        # for attn backends supporting query quantization
        # [CN] 只有当后端支持"预量化 Q"且 KV 确实是 fp8/nvfp4 时才启用，
        # [CN] 目的是让量化算子能与前序算子融合，省掉解码期的额外开销。

        self.query_quant = None
        if (
            self.impl.supports_quant_query_input
            and (
                self.kv_cache_dtype.startswith("fp8")
                or self.kv_cache_dtype.startswith("nvfp4")
            )
            and not self.kv_cache_dtype.endswith("per_token_head")
        ):
            is_per_head = (
                hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
            )
            # [CN] 逐头量化时，一个量化块 = 单个 KV 头对应的所有 Q 头，
            # [CN] 故块大小是 head_size * GQA 分组数。

            block_size = self.head_size * self.num_heads // self.num_kv_heads
            self.query_quant = QuantFP8(
                static=True,
                group_shape=GroupShape(-1, block_size)
                if is_per_head
                else GroupShape.PER_TENSOR,
            )

    # [CN] 前向本身极薄： reshape -> 派发到自定义算子 -> reshape 回来。
    # [CN] 刻意把 reshape 放在算子外面，减少图内非 CUDA graph 区域的 CPU 开销。

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if output_dtype is None:
            output_dtype = query.dtype
        # [CN] 用普通 torch 算子做 Q 量化（而非自定义算子），
        # [CN] 这样 torch.compile 能把它和前面的算子融合掉。

        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"} or (
                self.kv_cache_dtype.startswith("nvfp4")
            )

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        # [CN] 默认输出形状是 (num_tokens, num_heads * head_size_v)；
        # [CN] MLA 等非标准后端的输出形状与 Q 不同，由模型显式给出。

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        # [CN] 统一成 3D 再进算子：兼容上游传入的 2D [tokens, hidden] 形态。

        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)
        kv_cache_dummy_dep = None
        # [CN] 两条等价路径：直接调 Python 函数（便于调试/非 CUDA 平台）
        # [CN] 与走 torch.ops.vllm.* 自定义算子（图友好）。

        if self.use_direct_call:
            # Skip this if sharing KV cache with an earlier attention layer.
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                # [CN] 后端若不自带 KV 写入，这里补一次。共享 KV 的层要跳过
                # [CN] （它的 KV 由被共享的那一层写）。

                kv_cache_dummy_dep = unified_kv_cache_update(
                    key, value, self.layer_name
                )
            unified_attention_with_output(
                query,
                key,
                value,
                output,
                self.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        else:
            # Skip this if sharing KV cache with an earlier attention layer.
            encoded = _encode_layer_name(self.layer_name)
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    key, value, encoded
                )
            torch.ops.vllm.unified_attention_with_output(
                query,
                key,
                value,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        # [CN] 算子内按 3D 写，返回前摊回 2D，与模型其余部分对接。

        return output.view(-1, hidden_size)

    # [CN] 打印时优先展示 impl 的实际值而非构造参数：
    # [CN] 后端可能调整过（如对齐 head_size）。

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    # [CN] 权重加载后回调。若本次没加载量化权重（先加载过 dummy 权重的话
    # [CN] scale 可能是脏值），必须把 scale 重置回 1.0。

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            resolve_quant_method(self.quant_config, self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    # [CN] 向框架声明"这一层需要什么样的 KV cache"。
    # [CN] 注意块大小在这里重取——模型加载后它可能已被更新过。

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Encoder-only attention is prefill-only and keeps no autoregressive KV
        # cache. In hybrid models (e.g. Qwen3.5 / ColQwen3.5: GatedDeltaNet
        # linear_attention interleaved with full_attention) the runner iterates
        # every attention module to build the KV-cache spec, so an ENCODER_ONLY
        # full_attention layer reaches here; it contributes no KV cache group.
        # [CN] encoder-only 只做预填充、不留自回归 KV，因此不贡献 KV cache group。
        # [CN] 混合模型（GatedDeltaNet + full attention）遍历时会走到这里，必须显式返回 None。

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        # Should not be called for enc-dec attention.
        assert self.attn_type == AttentionType.DECODER
        # [CN] 字符串 -> 结构化量化模式，决定 scale 的存储布局。

        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        # [CN] 滑窗层自选块大小，与用户 --block-size 解耦：
        # [CN] 后者只约束主注意力。MLA 不支持滑窗，直接断言。

        if self.sliding_window is not None:
            assert not self.attn_backend.is_mla(), (
                "MLA is not supported for sliding window"
            )
            # SW chooses its own block_size, decoupled from the user's
            # ``--block-size`` (which only constrains primary attention).
            # When this SW layer is a padded spec (skip-quant: its page is
            # padded up to ``skip_page_size_padded``), pick the largest kernel
            # block that still fits the shared page so we waste fewer padding
            # bytes per block. Otherwise (page_size_padded is None) the smallest
            # block is fine — ``unify`` scales it up by an integer ratio.
            shared_page = vllm_config.cache_config.skip_page_size_padded
            # The backend owns its packing
            sw_per_token = self.attn_backend.customize_spec(
                SlidingWindowSpec(
                    block_size=1,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size_v,
                    dtype=self.kv_cache_torch_dtype,
                    kv_quant_mode=quant_mode,
                    sliding_window=self.sliding_window,
                )
            ).real_page_size_bytes
            # [CN] 先构造一个 block_size=1 的临时 spec 问后端"每 token 占多少字节"，
            # [CN] 再据此在共享页预算内反推最大可用块。

            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, shared_page, block_size
            )
            # [CN] 带上 page_size_padded：告诉框架这一层要被填充到多大的统一页。

            return SlidingWindowSpec(
                block_size=sw_block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
                page_size_padded=shared_page,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
            )


# [CN] 从 ForwardContext 里取出"这一层"需要的四件套。
# [CN] 之所以封装成函数：自定义算子内部拿不到 self，
# [CN] 只能靠层名去全局上下文里反查层实例与元数据。

def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: AttentionMetadata
    # [CN] 混合注意力下元数据是 {层名: 元数据} 的字典，按层名取。

    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    # [CN] 投机解码下是 list[dict]，第 0 项是主模型（非投机）的元数据。

    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    # [CN] slot_mapping 是字典而非单张量：不同 KV cache group 的槽位不同。

    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


# [CN] 独立出来的 KV 写入算子。返回空张量作为"假依赖"，
# [CN] 供后续注意力算子引用。

def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    # [CN] 零元素张量没有数据搬移成本，但能在图上建立真实的依赖边。

    return key.new_empty(0)


# [CN] fake 实现（meta 内核）：只描述形状/设备/类型，供 Dynamo 做形状推导。

def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


# [CN] 注册为自定义算子。mutates_args 为空列表：
# [CN] KV cache 是通过层实例间接改的，不在算子签名里。

direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


# [CN] 图捕获期间强制走 eager：注意力内部可能含数据相关的控制流，
# [CN] 捕获时会"打断"以保证正确性。

@eager_break_during_capture
# [CN] KV connector 钩子：分布式/分离式部署时先做 KV 拉取再算注意力。

@maybe_transfer_kv_layer
# [CN] 所有非 MLA 注意力最终都汇聚到这一个自定义算子。
# [CN] 它不含任何后端逻辑，只负责把上下文取出来转交给 impl.forward。

def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    # [CN] 这个参数只用于建立依赖，函数体里刻意不使用它。
    # [CN] 若不显式 del，编译器可能把它当无用参数优化掉，依赖就断了。

    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    # [CN] output 由外部预分配并传入：省分配、也便于融合 pass 挂载输出量化。

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


# [CN] mutates_args 声明 output 与 output_block_scale 被就地修改，
# [CN] 否则 torch.compile 可能误判并插入多余拷贝。

direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
)
