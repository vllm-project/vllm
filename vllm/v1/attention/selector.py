# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：注意力后端选择器 —— 把「一层的静态特征」翻译成「一个具体的 Backend 类」
# [CN] 职责：依据 head_size / dtype / kv_cache_dtype / block_size / use_mla 等能力开关，
# [CN]       选出最合适的 AttentionBackend 子类，并做懒加载（不在 import 期就拉 CUDA 后端）。
# [CN] 链路：Attention.__init__ -> get_attn_backend(...) -> <Backend 类>
# [CN]       之后由 backend.py 的 metadata builder 在每次 forward 前构造 attn_metadata
# [CN] 三个核心对象：
# [CN]   AttentionSelectorConfig  —— 选择的输入「特征向量」，全部 bool 开关集中在此
# [CN]   get_attn_backend()       —— 每构建一层 attention 就被调用一次（结果有缓存）
# [CN]   get_attn_spec_kind()     —— 层特征 -> KVCacheSpecKind，供 backend_per_kind 覆盖用
# [CN] 设计要点：
# [CN]   1. 结果用 functools.cache 缓存。几百层 Transformer 实际只落在极少数不同特征组合上，
# [CN]      因此昂贵的 platform 查询与类路径解析只发生少数几次。
# [CN]   2. 只有用户显式设过 block_size 才把它放进特征向量，否则传 None 让后端自选 ——
# [CN]      这正是「同一个模型换后端后 block_size 不一样」的根源。
# [CN]   3. backend_per_kind 优先级高于全局 backend，可以按 MLA / SLIDING_WINDOW 分别指定。
# [CN] 易错点：本函数不做 KV cache layout 决策。layout 必须在全部层的 backend 选出之后，
# [CN]         由 get_kv_cache_spec() 统一协商 —— 单次选择看不到同伴，无从定 layout。

from functools import cache
from typing import TYPE_CHECKING, NamedTuple, cast, get_args

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpecKind


# [CN] 选择阶段的输入特征向量。选 NamedTuple 而非 dataclass 是因为配合 functools.cache
# [CN] 需要 hashable：字段顺序稳定且可直接当 dict key 用。
# [CN] 新增能力开关必须带默认 False，否则所有历史调用点都要跟着改签名。

class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    has_sliding_window: bool = False
    use_non_causal: bool = False
    use_batch_invariant: bool = False
    use_kv_connector: bool = False
    use_pcp: bool = False
    use_adaptive_verification: bool = False
    use_dcp: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"has_sliding_window={self.has_sliding_window}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant}, "
            f"use_kv_connector={self.use_kv_connector}, "
            f"use_adaptive_verification={self.use_adaptive_verification}, "
            f"use_pcp={self.use_pcp}, "
            f"use_dcp={self.use_dcp})"
        )


# [CN] 层特征 -> KVCacheSpecKind。它镜像 get_kv_cache_spec_kind()，差别是后者从已构造好的
# [CN] KVCacheSpec 反推，而这里只有静态信号 —— 用途是让用户能用 kind 作 backend_per_kind 的键。
# [CN] 注意 SINK_FULL_ATTENTION 故意不在这里推导：它只由 StaticSinkAttention 这类专用层产出，
# [CN] 而带 sink 的普通 Attention（例如 gpt-oss）产出的仍是 FULL/SLIDING_WINDOW。
# [CN] 把「有 sink」当成 kind 会变的话，backend_per_kind 配不出来还以为是配置没生效。

def get_attn_spec_kind(
    use_mla: bool,
    has_sliding_window: bool,
    attn_type: str,
) -> "KVCacheSpecKind":
    """Derive the KV-cache group kind a layer belongs to from its signals.

    Mirrors ``get_kv_cache_spec_kind`` (which derives the kind from the
    produced ``KVCacheSpec``) so users can target groups by kind when
    setting ``AttentionConfig.backend_per_kind``.

    ``SINK_FULL_ATTENTION`` is intentionally not derived here: it is produced
    only by the ``StaticSinkAttention`` layer, whereas a plain ``Attention``
    layer with attention sinks (e.g. gpt-oss) still yields a
    ``FullAttentionSpec``/``SlidingWindowSpec``. Sinks therefore do not change
    the kind.

    Args:
        use_mla: Whether the layer uses multi-head latent attention.
        has_sliding_window: Whether the layer applies a sliding window.
        attn_type: The layer's ``AttentionType``.

    Returns:
        The ``KVCacheSpecKind`` the layer maps to.
    """
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

    if attn_type == AttentionType.ENCODER_ONLY:
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if attn_type == AttentionType.ENCODER_DECODER:
        return KVCacheSpecKind.CROSS_ATTENTION
    if use_mla:
        if has_sliding_window:
            return KVCacheSpecKind.SLIDING_WINDOW_MLA
        return KVCacheSpecKind.MLA_ATTENTION
    if has_sliding_window:
        return KVCacheSpecKind.SLIDING_WINDOW
    return KVCacheSpecKind.FULL_ATTENTION


# [CN] 主入口：把散落的开关整合成 AttentionSelectorConfig，处理两级覆盖后交给带缓存的解析器。

def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
    has_sliding_window: bool = False,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    # [CN] kv_cache_dtype 只允许 CacheDType 这组字面量：在这里早失败，
    # [CN] 好过让后端在 kernel launch 时才因为不支持的 dtype 崩掉。

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    # [CN] 只有用户显式指定过 block_size 才把它当作选择条件，否则传 None 交给后端自选。

    block_size: int | None
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    # [CN] KV connector（PD 分离场景）会改变后端的能力要求 —— 需要能导出/注入外部 KV，
    # [CN] 因此必须进特征向量参与选择。

    kv_transfer_config = vllm_config.kv_transfer_config
    use_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )

    speculative_config = vllm_config.speculative_config
    use_adaptive_verification = (
        speculative_config is not None
        and speculative_config.enable_adaptive_verification
    )
    # [CN] 自适应校验下 drafter 永远跑满整块，只有 verifier 才看到被裁剪过的 query 长度，
    # [CN] 所以同一个模型里 drafter 头要把这个开关关掉 —— 靠 model_tag 区分当前正在编谁。

    if use_adaptive_verification:
        from vllm.compilation.backends import model_tag

        # The drafter always runs full-length blocks; only the verifier sees
        # the trimmed, device-decided query lengths.
        use_adaptive_verification = model_tag != "dspark_head"

    # [CN] 默认 DECODER：绝大多数层不显式传 attn_type，只有 encoder-only / encoder-decoder 才传。

    attn_type = attn_type or AttentionType.DECODER
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type,
        has_sliding_window=has_sliding_window,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        use_kv_connector=use_kv_connector,
        use_pcp=vllm_config.parallel_config.prefill_context_parallel_size > 1,
        use_adaptive_verification=use_adaptive_verification,
        use_dcp=vllm_config.parallel_config.decode_context_parallel_size > 1,
    )

    # [CN] 按 KV cache kind 的覆盖优先于全局 backend，未列出来的 kind 回落到全局值。

    # A per-KV-group override (keyed by KVCacheSpecKind) takes precedence over
    # the global backend; kinds not present in the map fall back to it.
    attention_config = vllm_config.attention_config
    backend = attention_config.backend
    if attention_config.backend_per_kind:
        kind = get_attn_spec_kind(
            use_mla=use_mla,
            has_sliding_window=has_sliding_window,
            attn_type=attn_type,
        )
        backend = attention_config.backend_per_kind.get(kind.value, backend)

    # The KV cache layout is resolved across all of the model's backends at once
    # in get_kv_cache_spec(); a single selection cannot see its peers.
    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


# [CN] 解析结果缓存：同特征只解析一次，避免成百上千层重复走 platform 查询 + 类路径 import。

@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)
    return backend


# [CN] Mamba / 短卷积 / 线性注意力走独立通道：它们没有真正的 KV cache（只有循环 state），
# [CN] 因此不与上面共用 AttentionSelectorConfig，而是按 mamba_type 查实现类。

def get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


# [CN] 同样带缓存。额外检查 batch_invariant 模式下后端是否支持：不支持则直接抛错而非静默降级
# [CN] —— batch invariance 是正确性承诺，偷偷换 backend 会让结果与承诺不符。

@cache
def _cached_get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, MambaAttentionBackendEnum)

    mamba_attn_backend = mamba_type.get_class()
    if envs.VLLM_BATCH_INVARIANT and not mamba_attn_backend.supports_batch_invariance():
        raise RuntimeError(
            "VLLM batch_invariant mode is not supported for "
            f"{mamba_attn_backend.get_name()}."
        )
    return mamba_attn_backend
