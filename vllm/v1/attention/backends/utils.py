# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] 文件总览：attention 后端的「公共工具箱」—— 本文件不实现任何 kernel，
# [CN] 只负责把调度结果翻译成各 kernel 需要的 metadata，以及做跨后端的协商。
# [CN] 四大块内容：
# [CN]   1. KV cache layout 协商（各后端支持集求交 → 全局唯一 layout）；
# [CN]   2. 批次切分与重排（decode / short_extend / long_extend / prefill 四区）；
# [CN]   3. metadata 构造（local attention 虚拟批、KV sharing 快路径、多模态前缀）；
# [CN]   4. 杂项适配（投机解码 reshape、mamba 块表、DCP 本地 seq len）。
# [CN] 贯穿全文件的性能主线：**一切 metadata 尽量在 CPU 上算、然后异步上传**，
# [CN] 因为 GPU→CPU 的同步会把整个流水线打穿；文件里反复出现的 async_tensor_h2d
# [CN] 与 *_cpu 后缀就是这条原则的痕迹。
# [CN] 三个最容易看错的点：
# [CN]   1. 切分函数（split_decodes_and_prefills 等）都**假设批次已排序**，
# [CN]      顺序由 reorder_batch_to_split_decodes_and_prefills 保证，不能单独调用。
# [CN]   2. local attention 不是靠 mask 实现的，而是把长序列拆成「虚拟 batch」，
# [CN]      因此返回的 num_reqs 会比真实请求数大很多。
# [CN]   3. mamba / 混合模型的 block table 语义与 attention 不同，别混用。
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields, make_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
)

import numpy as np
import torch
from typing_extensions import runtime_checkable

from vllm.config import CacheConfig, VllmConfig, get_layers_from_vllm_config
from vllm.config.cache import _layout_from_name
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d, np_to_pinned_tensor
from vllm.v1.kv_cache_interface import KVCacheLayout, KVCacheSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)

# [CN] 两个哨兵值：PAD_SLOT_ID=-1 表示 padding 槽位；NULL_BLOCK_ID=0 表示空块。
# [CN] 注意 NULL_BLOCK_ID 取 0 意味着 block 0 永远不能被当作有效数据块使用。
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0

_LN_2 = math.log(2.0)


# [CN] FlashInfer 返回的 log-sum-exp 是**以 2 为底**的，这里换算为自然对数，
# [CN] 才能与 PyTorch 侧的 log_softmax 结果直接比较。
def log2_lse_to_ln(lse: torch.Tensor) -> torch.Tensor:
    """Convert a base-2 log-sum-exp tensor to natural-log units."""
    return lse * _LN_2


# [CN] 把「每个请求的多模态前缀双向区间」pad 成 (num_seqs, max_ranges, 2) 给
# [CN] Triton kernel 用；空区间填 (0,0)，kernel 侧靠 is_valid 跳过。
def compute_mm_prefix_range_tensor(
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
    num_seqs: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert mm_prefix_range dict to padded tensor for Triton kernel.

    Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
    Empty ranges have start==end==0, which kernel skips via is_valid check.
    """
    if mm_prefix_range is None:
        return None

    # [CN] 缺失的请求补 [(0,0)]，空 list 也视为 [(0,0)]，保证下游 max() 不炸。
    range_lists = [
        mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
    ]

    # [CN] 整批都没有有效区间时直接返回 None —— 让调用方跳过整个 mask_mod，
    # [CN] 而不是上传一张全 0 的张量去让 kernel 空跑。
    if all(r == [(0, 0)] for r in range_lists):
        return None

    max_ranges = max(len(r) for r in range_lists)
    padded = []
    for r in range_lists:
        padded_r = list(r) + [(0, 0)] * (max_ranges - len(r))
        padded.append(padded_r)
    # [CN] 异步上传（pinned → GPU），不引入同步点。
    padded = async_tensor_h2d(padded, dtype=torch.int32, device=device)
    return padded.view(num_seqs, max_ranges, 2)


# [CN] 与上一个函数同目标但更省：不按 seq 展开，而是**按本次调度的 query token**
# [CN] 展开成 (max_num_batched_tokens, 2)。
# [CN] 能这么做的关键前提是「mm_prefix 区间互不重叠」——于是「q 与 k 同区间」等价于
# [CN] 「k 落在 q 自己区间内」，kernel 完全不需要 key 侧查表。
# [CN] 规模从 num_seqs * max_seq_len 降到 max_num_batched_tokens，省一个量级。
def fill_mm_prefix_query_ranges(
    out: np.ndarray,
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> int:
    """Map each scheduled query token to the mm_prefix range containing it.

    Writes into ``out``, a caller-owned ``(max_num_batched_tokens, 2)`` int32
    staging buffer, and returns the number of rows written (0 if no range
    covers any scheduled query token, in which case ``out`` is untouched and
    the caller should skip the mask_mod entirely). Row ``i`` holds the absolute
    ``[start, end]`` bounds of the bidirectional range that query token ``i``
    belongs to, or ``(-1, -1)`` when it is outside every range.

    mm_prefix ranges never overlap, so "query and key share a range" is
    equivalent to "the key lies inside the query's own range". The kernel
    therefore needs no key-side lookup, and this metadata is sized by scheduled
    query tokens rather than by context length -- bounded by
    ``max_num_batched_tokens`` instead of ``num_seqs * max_seq_len``.

    Ranges are absolute prompt positions and may extend past the tokens
    scheduled so far under chunked prefill; the portion outside the current
    chunk is simply not recorded. Degenerate ranges (``start >= end``) are
    skipped to match the Triton path's ``start < end`` validity check.

    ``seq_lens_cpu`` only needs to be exact for prefill rows, since mm_prefix
    ranges cover prompt tokens: an over-estimate on a decode row shifts that
    row's query position further past every range, which still matches nothing.
    """
    if mm_prefix_range is None:
        return 0

    query_start_loc = query_start_loc_cpu.numpy()
    num_actual_tokens = int(query_start_loc[-1])
    if num_actual_tokens <= 0:
        return 0
    assert num_actual_tokens <= out.shape[0], (
        f"mm_prefix staging buffer holds {out.shape[0]} tokens, got {num_actual_tokens}"
    )

    # Resolve every span before touching `out`, so batches whose ranges all
    # fall outside the scheduled tokens skip the fill entirely.
    spans: list[tuple[int, int, int, int]] = []
    for req_idx, req_ranges in mm_prefix_range.items():
        if not req_ranges:
            continue
        token_start = int(query_start_loc[req_idx])
        query_len = int(query_start_loc[req_idx + 1]) - token_start
        if query_len <= 0:
            continue
        # Absolute position of this request's first scheduled query token.
        context_len = int(seq_lens_cpu[req_idx]) - query_len
        for start, end in req_ranges:
            if start >= end:
                continue
            first = max(start - context_len, 0)
            last = min(end - context_len, query_len - 1)
            if first > last:
                continue
            spans.append((token_start + first, token_start + last + 1, start, end))

    if not spans:
        return 0

    # [CN] 先整体置 (-1,-1) 再按 span 覆盖：默认「不属于任何区间」。
    # [CN] 只在确实存在有效 span 之后才写 out，避免无谓地污染调用方缓冲区。
    out[:num_actual_tokens] = -1
    for row_start, row_end, start, end in spans:
        out[row_start:row_end] = (start, end)
    return num_actual_tokens


_FLASHINFER_LAYOUT_NAMES = {
    "LBNHC": "NHD",
    "LBHNC": "HND",
    "BLHNC": "HND",
    "BLNHC": "NHD",
    "BHLNC": "HND",
}


# [CN] vLLM 的 layout 命名（LBNHC…）与 FlashInfer 的（NHD/HND）互不通用，
# [CN] 这里做映射；断言失败说明该 layout 本就不该走 FlashInfer。
def get_flashinfer_layout_string(layout: KVCacheLayout) -> str:
    """Return the layout name in FlashInfer's convention (NHD/HND)."""
    assert layout.name in _FLASHINFER_LAYOUT_NAMES, (
        f"KV cache layout {layout.name} has no FlashInfer equivalent; FlashInfer "
        "rejects it in supported_kv_cache_layouts"
    )
    return _FLASHINFER_LAYOUT_NAMES[layout.name]


# Preference order when no backend declares a supported set; LBNHC (NHD)
# first to match main's default.
# [CN] 没有任何后端声明支持集时的兜底顺序：LBNHC(NHD) 优先，与主线默认一致。
_DEFAULT_LAYOUT_PREFERENCE = (
    KVCacheLayout.LBNHC,
    KVCacheLayout.LBHNC,
    KVCacheLayout.BLNHC,
    KVCacheLayout.BLHNC,
    KVCacheLayout.BHLNC,
    KVCacheLayout.LHBNC,
)


def _layout_names(layouts: Iterable[KVCacheLayout]) -> list[str]:
    return [layout.name for layout in layouts]


# [CN] 多后端 layout 求交：声明一致的直接沿用顺序；不一致则「被最多后端排在第一位」
# [CN] 的 layout 胜出，同分按枚举序。交集为空是硬错误（说明这组后端不能共存）。
def get_supported_kv_cache_layouts(
    backends: Iterable[type[AttentionBackend]],
) -> list[KVCacheLayout]:
    """Layouts every one of the worker's backends supports, most preferred first.

    Every backend declares the layouts its kernels support, most preferred first
    (``supported_kv_cache_layouts``), or None when any layout works; workers where
    nothing declares follow the default preference. Identical declarations keep
    their order; otherwise the layout the most backends put first wins, ties
    keeping the enum order. An empty intersection is a hard error.
    """
    supported_layouts_lists: list[Sequence[KVCacheLayout]] = [
        layouts
        for backend in backends
        if (layouts := backend.supported_kv_cache_layouts()) is not None
    ] or [_DEFAULT_LAYOUT_PREFERENCE]

    # [CN] 快路径：所有后端声明完全一致时直接沿用其顺序（保留各自的偏好排序）。
    first = supported_layouts_lists[0]
    if all(layouts == first for layouts in supported_layouts_lists[1:]):
        return list(first)

    priorities: dict[KVCacheLayout, int] = defaultdict(int)
    for preferred_layout, *_ in supported_layouts_lists:
        priorities[preferred_layout] += 1
    supported_layouts = set.intersection(*map(set, supported_layouts_lists))
    # [CN] 慢路径：按「被多少后端排在第一位」降序排序，同分按枚举序稳定。
    candidates = sorted(
        (layout for layout in KVCacheLayout if layout in supported_layouts),
        key=lambda layout: priorities[layout],
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            "No KV cache layout satisfies every supported set: "
            f"{list(map(_layout_names, supported_layouts_lists))}."
        )
    return candidates


# [CN] 在 worker 进程里采纳 engine core 已经解析好的 layout；
# [CN] 若本进程已有不同结果则报错 —— 防止同一进程内两处各解一遍导致不一致。
def record_kv_cache_layout(cache_config: CacheConfig, layout_name: str) -> None:
    """Adopt a layout resolved elsewhere (the engine core) in this process."""
    layout = _layout_from_name(layout_name)
    existing = cache_config.kv_cache_layout
    if existing is not None and existing != layout.name:
        raise ValueError(
            f"KV cache layout is already resolved to {existing}; "
            f"cannot change it to {layout.name}."
        )
    cache_config.kv_cache_layout = layout.name


# [CN] 全局 layout 解析，**只在 engine core 跑一次**：
# [CN]   各 worker 上报支持集 → 断言所有 rank 一致 → 得到候选 →
# [CN]   若各 spec 的 (num_heads, num_states, page_size_bytes) 不一致，则必须选
# [CN]   block-compact layout（块内紧密排布才能被 HMA 复用同一页）→
# [CN]   再按 VLLM_KV_CACHE_LAYOUT > kv connector 偏好 > 候选首项的优先级定案。
# [CN] 结果写回 cache_config，并通过 set_kv_cache_layout RPC 下发到各 worker。
def resolve_kv_cache_layout(
    vllm_config: VllmConfig,
    supported_layouts: list[list[str]],
    kv_cache_specs: Iterable[KVCacheSpec] | None = None,
) -> KVCacheLayout:
    """Resolve one KV cache layout for the whole model.

    Runs once in the engine core. Every worker reports the layouts its backends
    support, most preferred first (``get_supported_kv_cache_layouts``); all
    ranks run the same backends, so their lists must agree. Specs mixing HNC
    shapes narrow the candidates to block-compact layouts. An explicit
    ``VLLM_KV_CACHE_LAYOUT`` must be one of the candidates or resolution fails,
    with the legacy ``NHD``/``HND`` names as aliases for ``LBNHC``/``LBHNC``; the
    connector's preference is used when compatible and dropped with a warning
    otherwise. A layout already present on ``cache_config`` wins outright, and
    the result is recorded there (see ``CacheConfig.kv_cache_layout``); it
    reaches workers through the ``set_kv_cache_layout`` RPC and
    ``KVCacheConfig.kv_cache_layout``.
    """
    cache_config = vllm_config.cache_config
    if cache_config.kv_cache_layout is not None:
        return cache_config.get_resolved_kv_cache_layout()

    assert supported_layouts and all(supported_layouts), (
        "No worker reported supported KV cache layouts."
    )
    assert all(names == supported_layouts[0] for names in supported_layouts[1:]), (
        f"Workers disagree on supported KV cache layouts: {supported_layouts}."
    )
    candidates = [_layout_from_name(name) for name in supported_layouts[0]]

    # A block-compact layout means the block is densely packed in memory, so any mix of
    # specs can re-interpret HNC with different sizes as long as the total number of
    # bytes is the same. If not block-compact, each spec must agree on HNC to alias
    # the same page (this aliasing is done by the Hybrid Memory Allocator, HMA).
    # [CN] 非 block-compact layout 要求所有 spec 的 HNC 形状完全一致才能别名同一页；
    # [CN] 形状一多就只能退到 block-compact（按字节数解释），否则抛错。
    hnc_shapes = {
        (spec.num_heads, spec.num_states, spec.page_size_bytes)
        for spec in kv_cache_specs or ()
    }
    if len(hnc_shapes) > 1:
        candidates = [m for m in candidates if m.is_block_compact]
        if not candidates:
            raise ValueError(
                "Specs with mixed HNC shapes need a block-compact layout, but "
                f"none is in every supported set: {supported_layouts}."
            )

    # [CN] 环境变量是**强约束**：不在候选集里就直接报错，而不是静默退化。
    if (requested := envs.VLLM_KV_CACHE_LAYOUT) is not None:
        layout = _layout_from_name(requested)
        if layout not in candidates:
            raise ValueError(
                f"VLLM_KV_CACHE_LAYOUT={requested} does not satisfy every "
                f"supported set; valid layouts: {_layout_names(candidates)}."
            )
    # [CN] kv connector 的偏好是**弱约束**：不兼容时打 warning 并用候选首项。
    elif (connector := get_kv_connector_cache_layout(vllm_config)) is not None:
        layout = _layout_from_name(connector)
        if layout not in candidates:
            logger.warning_once(
                f"KV connector cache layout {connector} does not satisfy every "
                f"supported set; valid layouts: {_layout_names(candidates)}. "
                f"Using {candidates[0].name} instead."
            )
            layout = candidates[0]
    else:
        layout = candidates[0]

    logger.info_once("Using %s KV cache layout.", layout.name)
    cache_config.kv_cache_layout = layout.name
    return layout


# [CN] PerLayerParameters：每个 attention 层的超参快照。
# [CN] 为什么需要它：FlashInfer 系列后端（trtllm-gen 除外）要求**所有层**共享同一组
# [CN] window_left / logits_soft_cap / sm_scale，所以必须先扫一遍再判断能否用。
@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters. Should not be used for
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    # [CN] window_left：滑窗左侧可见范围，-1 表示不限制（非滑窗层）。
    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = False
    # has same params for all layers
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)


# [CN] 扫描指定层，把 sliding_window / logits_soft_cap / scale / sinks 抽成参数表。
# [CN] window_size 为 None 时用 -1 表示「不限制」，这是下游 kernel 的约定值。
def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)

        # Infer hyperparameters from the attention layer
        # [CN] 用 getattr 兜底而不是直接取属性：不同后端的 impl 不一定都有这些字段。
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale
        has_sinks = getattr(impl, "sinks", None) is not None

        per_layer_params[key] = PerLayerParameters(
            window_left, logits_soft_cap, sm_scale, has_sinks
        )

    return per_layer_params


# [CN] 取「每个 TP rank 的 num_heads」，而不是模型全局的 num_heads ——
# [CN] 对逐层头数不一致的模型（如 MLA / 混合架构）必须用前者，否则 plan 期分配出错。
def get_num_attention_heads_from_layers(
    vllm_config: VllmConfig, layer_names: list[str]
) -> int | None:
    """Per-TP-rank ``num_heads`` shared by the named Attention layers.

    Use in metadata builders whose plan-time allocations depend on the
    head count: the model-wide ``get_num_attention_heads()`` is wrong
    for models with non-uniform per-layer head counts. All layers in
    one attention group must agree on ``num_heads``; this is asserted.
    Returns ``None`` when no matching Attention layer is found.
    """
    attn_layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    if not attn_layers:
        return None
    heads = {
        cast(Any, getattr(layer, "impl", layer)).num_heads
        for layer in attn_layers.values()
    }
    assert len(heads) == 1, (
        f"All layers in one attention group must share num_heads; "
        f"got {heads} for {layer_names}."
    )
    return heads.pop()


# [CN] 判断所有层是否共享超参，并把结论写进 has_same_window_lefts /
# [CN] has_same_all_params（compare=False，不参与相等比较，避免自指循环）。
def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters:
    """
    Currently, FlashInfer backend other than trtllm-gen
    only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]

    # [CN] 两个「是否一致」的结论都挂在第一个参数对象上返回给调用方，
    # [CN] 后端据此决定能不能走「所有层共用一份 plan」的快路径。
    global_params.has_same_window_lefts = all(
        params.window_left == global_params.window_left for params in param_sets
    )
    global_params.has_same_all_params = all(
        params == global_params for params in param_sets
    )

    return global_params


#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
# [CN] **本文件最核心也最难的一段**：用标准 flash attention 模拟「滑窗/local attention」。
# [CN] 思路：不做 mask，而是把每个请求按 attn_chunk_size 切成若干「虚拟 batch」，
# [CN] 每个虚拟批只 attend 自己那一段 KV，从而复用现成的 varlen 核。
# [CN] 代价是 num_reqs 被放大（= 各请求 local block 数之和），block_table 也要
# [CN] 按虚拟批重新 gather，所以函数额外返回一个 make_block_table 回调，
# [CN] 供后续 block table 变化时重建。
# [CN] 上方那段 ASCII 图示是本函数最好的说明书，务必先读它再读代码。
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    # [CN] 唯一允许的同步点：这里必须把 seq_lens 真正读到 CPU 才能算切分。
    with gpu_sync_allowed():
        # TODO see https://github.com/vllm-project/vllm/pull/31852
        seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    # [CN] 处理「从 local block 中间开始」的情形（chunked prefill 续算）：
    # [CN] 先算出首个（可能是残缺的）block 里有几个 query token，剩下的用 cdiv 补齐。
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: make a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    # [CN] 经典「分段 arange」技巧：用 cumsum + repeat + arange 纯向量化地造出
    # [CN] [0,1,0,1,2,3,0,1] 这样的块内序号，避免任何 Python 循环。
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    # [CN] 每个虚拟批的 K 长度默认是一个完整 chunk，只有该请求的**最后一个块**
    # [CN] 可能是残缺的（等于 seq_len 落在最后一个 chunk 内的部分）。
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}"
    )
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    # [CN] 为每个虚拟批算出它在原 block_table 里的页下标区间；
    # [CN] clip 到 block_table 宽度内，防止越界（末尾残缺块会多算一页）。
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    # Upload the index tensors to the block_table's device up-front so that the
    # fancy indexing below doesn't implicitly force a synchronous H2D copy.
    # [CN] 刻意先把 numpy 索引转成 torch 再上传：直接用 numpy 索引 torch 张量会触发
    # [CN] PyTorch 的一次性能回归；提前上传也避免 fancy indexing 隐式同步 H2D。
    batch_indices_torch = async_tensor_h2d(batch_indices, device=device)
    block_indices_torch = async_tensor_h2d(block_indices, device=device)

    # Save as a lambda so we can return this for update_block_table
    make_block_table = lambda block_table: block_table[
        batch_indices_torch, block_indices_torch
    ].view(virtual_batches, -1)
    block_table_local = make_block_table(block_table)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    # [CN] 返回的 num_reqs 是**虚拟批数量**，不是真实请求数；
    # [CN] num_actual_tokens 仍沿用原值，因为切分不改变 token 总数。
    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=async_tensor_h2d(query_start_loc_cpu, device=device),
        seq_lens=async_tensor_h2d(seq_lens_cpu, device=device),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
    ), make_block_table


# [CN] KV sharing 快路径：共享 KV 的层只需要**产出 logits 的那些位置**的结果，
# [CN] 于是把 metadata 里「全部 query」换成「logits_indices 对应的位置」，
# [CN] 大幅减少注意力计算量。全 decode（max_query_len==1）时无需变换直接返回。
def make_kv_sharing_fast_prefill_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    # [CN] 全 decode 时「要算的位置」就是全部位置，快路径无收益，直接原样返回。
    if common_attn_metadata.max_query_len == 1:
        # All requests are decode (assume 1 token for now)
        # Skip computing fast prefill path
        return common_attn_metadata

    assert common_attn_metadata.logits_indices_padded is not None
    assert common_attn_metadata.num_logits_indices is not None

    logits_indices_padded = common_attn_metadata.logits_indices_padded
    num_logits_indices = common_attn_metadata.num_logits_indices
    # Get rid of CUDAGraph padding, if any
    logits_indices = logits_indices_padded[:num_logits_indices]
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc = common_attn_metadata.query_start_loc
    # Example inputs
    # num_reqs: 3
    # generation_indices:  [14, 18, 19, 27]
    # query_start_loc: [0, 15, 20, 28]
    # seq_lens:        [41, 31, 40]

    # Find how many decode indices belong to each request
    # request_ids: [0, 1, 1, 2]
    # [CN] 用 bucketize 把每个 logits 下标映射回它所属的请求（O(n log n)、无需同步）。
    request_ids = torch.bucketize(logits_indices, query_start_loc[1:], right=True)

    # Figure out how many tokens are in each request
    # num_decode_tokens: [1, 2, 1]
    # Avoid `torch.bincount` here — on CUDA it forces a sync to determine
    # the output size (even with `minlength`, the kernel must confirm no
    # value exceeds the bound). `scatter_add_` into a preallocated buffer
    # is equivalent and stays async.
    # [CN] 刻意不用 torch.bincount：它在 CUDA 上会为确定输出尺寸强制同步；
    # [CN] 改成 scatter_add_ 进预分配缓冲区，语义等价且全程异步。
    num_decode_tokens = torch.zeros(
        num_reqs, dtype=request_ids.dtype, device=request_ids.device
    )
    num_decode_tokens.scatter_add_(
        0, request_ids.to(num_decode_tokens.dtype), torch.ones_like(request_ids)
    )

    # Calculate new query_start_loc with tokens in generation_indices
    # decode_query_start_loc: [0, 1, 3, 4]
    decode_query_start_loc = torch.empty(
        num_reqs + 1, device=query_start_loc.device, dtype=query_start_loc.dtype
    )

    # [CN] 切片赋值而不是 [0] = 0：标量赋值会触发一次 GPU 同步，fill_ 不会。
    decode_query_start_loc[:1].fill_(0)  # Avoid sync from scalar assignment.
    decode_query_start_loc[1:] = torch.cumsum(num_decode_tokens, dim=0)

    # `num_decode_tokens` is a histogram over `logits_indices`, so its total is
    # just how many there were -- already known as a Python int.
    total_num_decode_tokens = num_logits_indices

    # Largest per-request logits count.
    decode_max_query_len = common_attn_metadata.max_logits_per_req
    assert decode_max_query_len is not None

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=decode_query_start_loc,
        query_start_loc_cpu=decode_query_start_loc.to("cpu", non_blocking=True),
        seq_lens=common_attn_metadata.seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_decode_tokens,
        max_query_len=decode_max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
    )
    return common_attn_metadata


# [CN] 把已排序批次切成 decode / extend / prefill 三段，返回各自请求数与 token 数。
# [CN] 区分 extend 与 prefill 的依据是「seq_len == query_len」（prefill 没有历史）。
# [CN] 注意用 seq_lens_cpu_upper_bound：**上界对 prefill 行是精确值**，
# [CN] 而 decode 行满足 seq_len > query_len，判据在两种情况下都不会误判。
def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_extends: The number of extend requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_extend_tokens: The number of tokens in the extend requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    # Upper bound is exact for prefill rows; decode rows still satisfy
    # seq_len > query_len under the optimistic bound, so `seq_lens ==
    # query_lens` identifies prefills correctly either way.
    assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # [CN] argmax 找第一个 True 的位置 = 第一个非 decode 请求。
    # [CN] 这是「批次已排序」这一前提的具体利用方式：无需排序，只找分界点。
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    # [CN] argmax 找首个 True 的下标；全是 False 时 argmax 返回 0，
    # [CN] 配合后面的 torch.any 判断来区分「0 个」和「第一个就是」。
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    if not torch.any(is_prefill_or_extend):
        return (num_decodes, 0, 0, num_decode_tokens, 0, 0)

    num_prefills_or_extends = num_reqs - num_decodes
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    if not torch.any(is_prefill):
        return (
            num_decodes,
            num_prefills_or_extends,
            0,
            num_decode_tokens,
            num_prefill_or_extend_tokens,
            0,
        )

    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill

    num_prefill_tokens = num_tokens - query_start_loc[first_prefill]
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


# [CN] 上一个函数的两分版（decode vs 非 decode）。
# [CN] require_uniform：full CUDA graph 场景要求所有 decode 的 query 长度一致，
# [CN] 此时长度不等的请求会被判为 prefill，以保证 num_decodes 命中抓图时的尺寸。
def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    The batch is expected to be ordered as:
        decode → short_extend → long_extend → prefill

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.
        treat_short_extends_as_decodes: If True (default), short extends
            (query_len <= threshold but still prefilling) are counted as
            decodes. If False, they are counted as prefills.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # [CN] 既然批次已排序，第一个请求就不是 decode ⇒ 整个批次没有 decode。
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    # [CN] 允许 query_lens==0（padding 行）混在 decode 里：padding 不影响
    # [CN] num_decodes 与图尺寸对齐，把它们算作 decode 反而更安全。
    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if treat_short_extends_as_decodes and torch.all(
            (query_lens == query_lens[0]) | (query_lens == 0)
        ):
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        is_prefill |= common_attn_metadata.is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


# [CN] 按 workspace 上限把 prefill 请求切成若干块，保证每块总 seq len 不超限。
# [CN] 只按 CPU 侧 seq_lens 贪心累加，是纯 CPU 逻辑（不触碰 GPU）。
def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    """
    Split the prefill requests into chunks such that the total sequence length
    of each chunk is less than or equal to the workspace size.

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests on CPU.
        workspace_size: The maximum workspace size (in tokens) per chunk.
        request_offset: The offset to add to the request indices.
    Returns:
        A list of tuples of (reqs_start, reqs_end) representing chunk boundaries.
    """
    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert torch.all(seq_lens_cpu <= workspace_size).item()

    # [CN] 贪心装箱：能塞就塞，塞不下就开新块。要求单个 seq_len 不超过 workspace
    # [CN] （上面已断言），否则会死循环。
    while i < n:
        start, chunk_total = i, 0
        while i < n and (chunk_total + (s := seq_lens_cpu[i].item())) <= workspace_size:
            chunk_total += s
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


# [CN] **切分函数的前置步骤**：把批次原地重排成
# [CN] decode → short_extend → long_extend → prefill 四个连续区域。
# [CN] 四类互斥且全覆盖：无 context=prefill；有 context 且超阈值=long_extend；
# [CN] 有 context 未超阈值但仍在 prefill=short_extend；已完成 prefill=decode。
def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    The batch is reordered into 4 regions:
        decode:        (num_scheduled <= threshold AND is not prefilling)
        short_extend:  (num_scheduled <= threshold AND is chunked prefilling)
        long_extend:   (num_scheduled > threshold AND is chunked prefilling)
        prefill:       (num_computed == 0)   # First chunks

    Returns:
        True if the batch was modified, False otherwise.
    """
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]
    num_prompt_tokens_np = input_batch.num_prompt_tokens[:num_reqs]

    # [CN] 四类判定完全在 numpy 上向量化完成，避免逐请求 Python 循环。
    has_context = num_computed_tokens_np > 0
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np

    # Mutually exclusive categories (exactly one True per request):
    # 1. No context yet -> prefill
    # 2. Has context, above threshold -> long_extend
    # 3. Has context, below threshold, still prefilling -> short_extend
    # 4. Has context, below threshold, done prefilling -> decode
    is_pure_prefill = ~has_context
    is_long_extend = has_context & ~is_below_threshold
    is_short_extend = has_context & is_below_threshold & ~done_prefilling
    is_decode = has_context & is_below_threshold & done_prefilling

    # Desired order: decode → short_extend → long_extend → prefill
    req_regions = np.zeros(num_reqs, dtype=np.int32)  # 0 = decode by default
    req_regions[is_short_extend] = 1
    req_regions[is_long_extend] = 2
    req_regions[is_pure_prefill] = 3

    num_decodes = int(is_decode.sum())
    num_short_extends = int(is_short_extend.sum())
    num_long_extends = int(is_long_extend.sum())
    num_prefills = int(is_pure_prefill.sum())

    # [CN] 构造「理想排布」的目标区域序列，用于和实际区域逐位比较。
    target_regions = np.repeat(
        [0, 1, 2, 3],
        [num_decodes, num_short_extends, num_long_extends, num_prefills],
    ).astype(np.int32)

    # [CN] 只有区域对不上的位置才需要动；全部对上直接返回 False，
    # [CN] 让调用方跳过后续所有 metadata 重建。
    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    # [CN] 用「置换环」做原地交换：沿 src→dst 链反复 swap_states，直到回到自身。
    # [CN] 这样只需要 O(环长) 次交换且不需要额外缓冲；返回值告诉调用方批次是否变过
    # [CN] （没变就不用重做后续 metadata）。
    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True


# [CN] 投机解码下 query 是 (total_tokens, heads, head_dim)，这里按 batch_size
# [CN] 折叠成 (batch, seq_len, heads, head_dim) 以适配批处理核。
def reshape_query_for_spec_decode(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reshapes the query tensor for the specified batch size, so that
    it has shape (batch_size, seq_len, num_heads, head_dim).
    """
    # [CN] 输入必须是「已压平的 token 维」形式，否则无法按 batch 均分。
    assert query.dim() == 3, f"query must be 3D, got {query.dim()}D"
    total_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    assert total_tokens % batch_size == 0, (
        f"{total_tokens=} is not divisible by {batch_size=}"
    )
    seq_len = total_tokens // batch_size
    return query.view(batch_size, seq_len, num_heads, head_dim)


# [CN] 上一步的逆操作：把 batch 与 seq_len 两维合并回 token 维。
def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the attention output tensor, so that
    the batch_size and seq_len dimensions are combined.
    """
    if attn_output.dim() == 3:
        # Already in the correct shape
        return attn_output
    assert attn_output.dim() == 4, f"attn_output must be 4D, got {attn_output.dim()}D"
    total_tokens = attn_output.shape[0] * attn_output.shape[1]
    return attn_output.view(total_tokens, attn_output.shape[2], attn_output.shape[3])


# [CN] 用 make_dataclass 动态生成 metadata 子类，让后端能在不改基类的前提下
# [CN] 追加字段（典型用途是给快路径加 logits_indices）。
def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields
    """
    name: str = name_prefix + metadata_cls.__name__  # type: ignore
    # [CN] 运行时造类而不写死类：让不同后端自由扩展字段而不污染基类定义。
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls,))
    return Wrapped


@runtime_checkable
# [CN] 协议类：只要实现了这两个字段，就被视为「支持 KV sharing 快路径」的 metadata。
class KVSharingFastPrefillMetadata(Protocol):
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None


# [CN] 工厂：在既有后端外面包一层，build 时先把 common metadata 换成
# [CN] 「只算 logits 位置」的版本，再交给原 builder，最后把 logits_indices
# [CN] 补回 metadata —— 实现一个零侵入的快路径后端。
def create_fast_prefill_custom_backend(
    prefix: str,
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    underlying_builder = underlying_attn_backend.get_builder_cls()

    # [CN] 继承原 builder 只覆盖 build：把 common metadata 换成快路径版本后
    # [CN] 交给 super().build，从而对原后端零改动。
    class FastPrefillAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = (
                make_kv_sharing_fast_prefill_common_attn_metadata(common_attn_metadata)
            )
            metadata = super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

            class KVSharingFastPrefillAttentionMetadata(
                metadata.__class__,  #  type: ignore
                KVSharingFastPrefillMetadata,
            ):
                def __init__(self, metadata, common_attn_metadata):
                    # Shallow copy all fields in metadata cls
                    for _field in fields(metadata.__class__):
                        setattr(self, _field.name, getattr(metadata, _field.name))

                    self.logits_indices_padded = (
                        common_attn_metadata.logits_indices_padded
                    )
                    self.num_logits_indices = common_attn_metadata.num_logits_indices

            return KVSharingFastPrefillAttentionMetadata(metadata, common_attn_metadata)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=FastPrefillAttentionBuilder,
    )

    return attn_backend


# [CN] 为 mamba 的 causal_conv1d 预生成块指针（batch_ptr / token_chunk_offset_ptr）。
# [CN] 刻意用 **CPU 上的** query_start_loc 计算，避免 DtoH 同步；
# [CN] 指针按 MAX_NUM_PROGRAMS 预分配并在不足时 resize，避免每步重新分配。
def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor, *, device: torch.device
) -> tuple[dict[int, dict[str, Any]], torch.Tensor, torch.Tensor]:
    # Needed for causal_conv1d. Use the CPU query_start_loc to avoid DtoH sync.
    assert query_start_loc_p_cpu.device.type == "cpu"
    # [CN] diff 直接得到每条序列的 query 长度（前缀和取差分）。
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict[str, Any]] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = np_to_pinned_tensor(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        # [CN] 程序数按 2 倍冗余预留：CUDA 上 program 数由 grid 决定，
        # [CN] 预留不足会导致每次都重新分配指针缓冲。
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32, pin_memory=PIN_MEMORY)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                assert token_chunk_offset_ptr is not None
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)

        assert batch_ptr is not None
        batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)
        assert token_chunk_offset_ptr is not None
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist, non_blocking=True)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr


# [CN] 解码上下文并行（DCP）下每个 rank 只存一部分 KV，本函数算出「本 rank 的
# [CN] 本地 seq len」：先按 interleave 粒度均分 base，再把余数按 rank 补齐。
# [CN] 各 rank 本地长度可能不同，因此不能直接用全局 seq_lens。
def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each dcp rank.
    Only consider dcp now, we can extend the case of cp based on this.
    """
    # [CN] 先降 int32 再做除法：int32 除法在 GPU 上明显快于 int64。
    seq_lens_i32 = seq_lens.to(torch.int32)
    if dcp_rank is None:
        rank_offsets = torch.arange(
            dcp_size,
            dtype=torch.int32,
            device=seq_lens.device,
        ).view(
            *((1,) * seq_lens_i32.dim()),
            dcp_size,
        )
        seq_lens_tiled = seq_lens_i32.unsqueeze(-1)
    else:
        rank_offsets = torch.tensor(dcp_rank, dtype=torch.int32, device=seq_lens.device)
        seq_lens_tiled = seq_lens_i32
    # [CN] base = 每个 rank 至少分到的完整份数；remainder 按 rank 偏移补足剩余部分，
    # [CN] clip 到 [0, interleave_size] 保证不会某个 rank 多拿超过一份。
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    return dcp_local_seq_lens


# [CN] mamba 的三种 cache 模式下 block table 语义不同：
# [CN]   all  = 全量页表，原样返回；none = 每请求固定 1 页，原样返回；
# [CN]   align = 输入是完整页表，输出只取每请求最后 1+speculative 页。
# [CN] gather 下标用 int32 算再转 int64（gather 要求 Long），省一次类型提升开销。
def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
    """
    Get the block table tensor for mamba kernels from the input
    common_attn_metadata.block_table_tensor given different mamba cache modes.

    - "all":   input  (#requests, cdiv(max_model_len, block_size)
                        + num_speculative_blocks);
               output (#requests, cdiv(max_model_len, block_size)
                        + num_speculative_blocks).

    - "none":  input  (#requests, 1 + num_speculative_blocks);
               output (#requests, 1 + num_speculative_blocks).

    - "align": input  (#requests, cdiv(max_model_len, block_size));
               output (#requests, 1 + num_speculative_blocks), which are the last
               1 + num_speculative_blocks of each request.
    """
    # [CN] all / none 两种模式下页表形状本来就对，无需变换。
    if mamba_cache_mode in ("all", "none"):
        return block_table
    else:
        assert isinstance(kv_cache_spec, MambaSpec)
        # NOTE: For 0-length requests in CUDA graph, use a start_index of 0
        # to handle the invalid block table.
        # [CN] align 模式：只取每请求「当前所在的那一页」及其后的投机页。
        # [CN] clamp(min=0) 是给 CUDA graph 里 0 长度请求兜底（seq_len=0 会算出 -1）。
        start_indices = (seq_lens - 1) // kv_cache_spec.block_size
        start_indices.clamp_(min=0)
        # Use int32 for arithmetic to avoid dtype promotion overhead,
        # then convert to int64 for gather (which requires Long indices)
        offsets = torch.arange(
            1 + kv_cache_spec.num_speculative_blocks,
            device=block_table.device,
            dtype=torch.int32,
        )
        indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
        return torch.gather(block_table, 1, indices_to_gather)
