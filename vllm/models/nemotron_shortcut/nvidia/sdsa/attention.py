# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CacheConfig,
    CompilationMode,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.nemotron_shortcut.nvidia.sdsa.backend import (
    SDSA_CORE_HEAD_SIZE,
    SdsaAttentionBackend,
    SdsaIndexerBackend,
    SdsaIndexerMetadata,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.flashinfer_ops import check_requirements
from vllm.models.nemotron_shortcut.nvidia.sdsa.kv_cache_grouping import (
    SDSA_CORE_SUFFIX,
    SDSA_INDEXER_CACHE_SUFFIX,
    SDSA_INDEXER_SUFFIX,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.prefill import (
    INDEXER_FORMATS,
    PANEL_ROWS,
    SdsaConfig,
    SdsaTiles,
    _arena_layout,
    _rows,
    _top_k,
    indexer_pages,
    reserve,
    union_columns,
    union_group,
    write_index_qk,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager

E4M3 = torch.float8_e4m3fn


class SdsaIndexerCache(DeepseekV32IndexerCache):
    """The indexer's key cache, a key's value and scale bytes per key. It shares the
    core's KV cache group, so one block table and one slot mapping locate both.
    Each PAGE-key page holds its keys, then their scales (`indexer_pages`).
    This class is vLLM's indexer key cache (DeepseekV32IndexerCache). Thus vLLM
    sizes, allocates and binds it as it does there. It holds the layer's
    SdsaConfig, which the indexer's metadata builder reads."""

    def __init__(
        self, sdsa_config: SdsaConfig, prefix: str, cache_config: CacheConfig
    ) -> None:
        key_bytes = sum(INDEXER_FORMATS[sdsa_config.indexer_dtype][1:])
        super().__init__(key_bytes, torch.uint8, prefix, cache_config)
        self.sdsa_config = sdsa_config

    def get_attn_backend(self) -> type[AttentionBackend]:
        return SdsaIndexerBackend


# The indexer's custom op. Under VLLM_COMPILE it is a Dynamo splitting op: the
# core reads the selections through its impl, so nothing in one Inductor graph
# orders the core after the indexer; only a split does (a default splitting op).
SDSA_INDEXER_OP = "vllm::nemotron_shortcut_sdsa_indexer"


@eager_break_during_capture
def nemotron_shortcut_sdsa_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    layer.select(query, key)


def nemotron_shortcut_sdsa_indexer_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="nemotron_shortcut_sdsa_indexer",
    op_func=nemotron_shortcut_sdsa_indexer,
    mutates_args=["topk_indices", "valid_counts"],
    fake_impl=nemotron_shortcut_sdsa_indexer_fake,
)


def _split_at_indexer(vllm_config) -> None:
    """Refuse a compilation config that does not split at the indexer op: an
    Inductor graph partition, stock torch.compile, or splitting_ops without
    the op (one full graph) cannot order the core after the indexer."""
    compilation = vllm_config.compilation_config
    if compilation.mode == CompilationMode.NONE:
        return
    splitting_ops = compilation.splitting_ops
    if (
        compilation.mode != CompilationMode.VLLM_COMPILE
        or compilation.use_inductor_graph_partition
        or SDSA_INDEXER_OP not in (splitting_ops or [])
    ):
        raise NotImplementedError(
            f"SDSA needs {SDSA_INDEXER_OP} as a Dynamo splitting op: use "
            "VLLM_COMPILE without Inductor graph partition and with the default "
            "splitting_ops, or enforce eager mode"
        )


@dataclasses.dataclass(frozen=True)
class SdsaBuffers:
    """The selection buffers that the SDSA layers of a model share: each query
    row's Top-K slots (`topk_indices`) and the union core's mask rows (`mask`).
    They are zeroed: trtllm-gen loads the slots past a prefill union's size in
    its last key tile (masked), so they must never point outside the KV cache."""

    topk_indices: torch.Tensor
    mask: torch.Tensor


def allocate_sdsa_buffers(vllm_config: VllmConfig, top_k: int) -> SdsaBuffers:
    """The buffers of a model's SDSA layers, for K = `top_k`. The union core's
    groups are the largest. A request's last group may be partial, hence the
    extra rows, as are the last TP rank's decode rows (`select`), 1 + k per
    request with k drafts."""
    model_config = vllm_config.model_config
    scheduler = vllm_config.scheduler_config
    group = union_group(
        model_config.get_num_attention_heads(vllm_config.parallel_config)
    )
    columns = union_columns(group, top_k, model_config.max_model_len)
    max_tokens = scheduler.max_num_batched_tokens
    decode_rows = 1 + vllm_config.num_speculative_tokens
    spare = (get_tp_group().world_size - 1) * decode_rows
    rows = max_tokens + (group - 1) * scheduler.max_num_seqs + spare
    device = current_platform.device_type
    return SdsaBuffers(
        topk_indices=torch.zeros(rows, top_k, dtype=torch.int32, device=device),
        mask=torch.zeros(max_tokens, columns // 32, dtype=torch.int32, device=device),
    )


class SdsaAttention(nn.Module):
    """The indexer selects each query's Top-K keys into `topk_indices`, an
    int32 buffer of `top_k`-slot rows shared by all SDSA layers: a decode token's
    as key positions, prefill tokens' as their groups' unions of core cache
    slots (SdsaPrefillPlan.run); the core attends to them."""

    def __init__(
        self,
        *,
        num_heads: int,
        sdsa_config: SdsaConfig,
        buffers: SdsaBuffers,
        cache_config: CacheConfig,
        scale: float | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if sdsa_config.indexer_dtype not in INDEXER_FORMATS:
            raise ValueError(
                "SdsaAttention needs the k cache dtype, nvfp4 or bf16, got "
                f"{sdsa_config.indexer_dtype!r}"
            )
        self.sdsa_config = sdsa_config
        top_k = sdsa_config.top_k
        check_requirements()
        vllm_config = get_current_vllm_config()
        _split_at_indexer(vllm_config)
        scheduler = vllm_config.scheduler_config
        max_tokens = scheduler.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.group = union_group(num_heads)
        self.columns = union_columns(self.group, top_k, self.max_model_len)
        tp = get_tp_group().world_size
        self.decode_rows = 1 + vllm_config.num_speculative_tokens
        self.register_buffer("topk_indices", buffers.topk_indices, persistent=False)
        self.register_buffer(
            "valid_counts",
            torch.empty(max_tokens, dtype=torch.int32),
            persistent=False,
        )
        self.mask = buffers.mask
        # A TP rank's decode score rows, each padded to a multiple of 4 floats
        # because gvr_2 loads them as float4.
        self.decode_scores = (
            (cdiv(scheduler.max_num_seqs, tp) * self.decode_rows,
             round_up(self.max_model_len, 4)),
            torch.float32,
        )  # fmt: skip
        self.layer_name = f"{prefix}{SDSA_INDEXER_SUFFIX}"
        context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        context[self.layer_name] = self
        # A rank's decode requests (their tokens, when flattened) are rows 0, 1,
        # 2, ... of its block table slice.
        self.register_buffer(
            "decode_requests",
            torch.arange(scheduler.max_num_seqs * self.decode_rows, dtype=torch.int32),
            persistent=False,
        )
        # The core is built before the indexer cache: vLLM sizes a hybrid
        # model's attention pages with the first attention backend it finds,
        # which must be the core's (NVFP4 K/V pack its pages), not the indexer's.
        indexer_cache_name = f"{prefix}{SDSA_INDEXER_CACHE_SUFFIX}"
        self.core = Attention(
            num_heads,
            SDSA_CORE_HEAD_SIZE,
            scale if scale is not None else SDSA_CORE_HEAD_SIZE**-0.5,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}{SDSA_CORE_SUFFIX}",
            attn_backend=SdsaAttentionBackend,
            topk_indices_buffer=self.topk_indices,
            valid_counts_buffer=self.valid_counts,
            indexer_name=indexer_cache_name,
            sdsa_config=sdsa_config,
        )
        # Calibrated KV cache scales keep FP32 precision; the model's default
        # dtype would round them as they load.
        for name in ("q_scale", "k_scale", "v_scale", "prob_scale"):
            scale_param = getattr(self.core, name, None)
            if isinstance(scale_param, torch.Tensor):
                scale_param.data = scale_param.data.float()
        self.indexer_cache = SdsaIndexerCache(
            sdsa_config, indexer_cache_name, cache_config
        )

    def forward(
        self,
        indexer_query: torch.Tensor,
        indexer_key: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] == 0:
            return torch.empty_like(query)
        torch.ops.vllm.nemotron_shortcut_sdsa_indexer(
            indexer_query,
            indexer_key,
            self.topk_indices,
            self.valid_counts,
            _encode_layer_name(self.layer_name),
        )
        input_shape = query.shape
        output = self.core(query.reshape(input_shape[0], -1), key, value)
        return output.view(input_shape)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the KV cache scales into the core attention, which holds the KV
        cache; vLLM's loader names them after this module (`attn.k_scale`).
        Scales the core lacks are skipped. Without a checkpoint query scale, a
        quantized cache quantizes the query with the key scale. The loader may
        hand the scales over in several calls, in any order, so the query scale
        follows the key scale only while it is still unset (negative): a
        checkpoint query scale loaded before or after replaces it."""
        params = dict(self.core.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name in params:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(f"core.{name}")
            elif not name.endswith("_scale"):
                raise ValueError(f"SdsaAttention has no parameter {name!r}")
        q_scale, k_scale = params.get("q_scale"), params.get("k_scale")
        if (
            is_quantized_kv_cache(self.core.kv_cache_dtype)
            and q_scale is not None
            and k_scale is not None
            and "core.q_scale" not in loaded
            and q_scale.item() < 0 < k_scale.item()
        ):
            q_scale.data.copy_(k_scale.data)
        return loaded

    def select(self, query: torch.Tensor, key: torch.Tensor) -> None:
        """Cache the key and write the query rows in the keys' format, then
        select every row's Top-K keys: the decode rows', as key positions, then
        the prefill rows', as their groups' unions. The profiling run instead
        reserves the scratch memory and gvr_2's tables."""
        from flashinfer.sdsa import indexer_scores, quantize_write_index_qk_nvfp4
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import workspace_bytes

        manager = current_workspace_manager()
        # gvr_2's Top-K scratch, shared by all layers. It is zeroed once and
        # every call leaves it zeroed. It is requested on every call, including
        # the profiling run, because the workspace manager is locked after
        # profiling and then refuses resources it has not seen. It is int32, the
        # type gvr_2 reads, because a byte buffer would be re-viewed on every call.
        topk_workspace = manager.get_persistent(
            "sdsa_topk_workspace",
            (workspace_bytes() // 4,),
            torch.int32,
            zero_init=True,
        )
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # The decode Top-K for every decode row count, compiled now (see
            # reserve): one row per request, and 1 + k.
            (rows, _), _ = self.decode_scores
            scores, top = manager.get_simultaneous(
                self.decode_scores, ((rows, self.sdsa_config.top_k), torch.int32)
            )
            for rows in {1, self.decode_rows}:
                for count in range(1, len(scores) // rows + 1):
                    lengths = self.decode_requests.new_zeros(count)
                    out = top[: count * rows]
                    _top_k(scores[: len(out)], lengths, out, topk_workspace, rows)
            reserve(
                len(self.decode_requests),
                len(self.valid_counts),
                self.max_model_len,
                topk_workspace,
                self.sdsa_config,
            )
            return
        assert isinstance(attn_metadata, dict)
        metadata = attn_metadata[self.indexer_cache.prefix]
        assert isinstance(metadata, SdsaIndexerMetadata)
        # The indexer cache and the core share one block table and slot mapping.
        block_table = metadata.block_table
        num_decodes, rows_per_decode = metadata.num_decodes, metadata.decode_rows
        num_decode_rows, prefill = num_decodes * rows_per_decode, metadata.prefill

        # Cache the step's keys, and write its query tokens to their query rows
        # in the keys' format. NVFP4's scales share the keys' pages and table.
        key_cache, key_scales = indexer_pages(
            self.indexer_cache.kv_cache, self.sdsa_config.indexer_dtype
        )
        slot_mapping = get_forward_context().slot_mapping
        assert isinstance(slot_mapping, dict)
        num_keys = metadata.num_actual_tokens
        slots = slot_mapping[self.indexer_cache.prefix][:num_keys]
        query_slots = metadata.query_slots  # the query tokens' rows
        query = query[: len(query_slots)]
        num_query_rows = num_decode_rows
        if prefill is not None:
            num_query_rows += len(prefill.query_rows)
        # Prefill tiles' padding rows hold no query; zeroed, their scores are 0.
        new_rows = key.new_empty if prefill is None else key.new_zeros
        if key_scales is None:  # BF16, as it is
            packed_query = new_rows(num_query_rows, key_cache.shape[2])
            write_index_qk(
                key[:num_keys],
                slots,
                key_cache.flatten(0, 1),
                query,
                query_slots,
                packed_query,
            )
            query_scales = None
        else:
            packed_query = new_rows(
                num_query_rows, key_cache.shape[2], dtype=torch.uint8
            )
            query_scales = new_rows(num_query_rows, key_scales.shape[2], dtype=E4M3)
            quantize_write_index_qk_nvfp4(
                key[:num_keys],
                slots,
                query,
                query_slots,
                key_cache=key_cache,
                key_scales=key_scales,
                packed_query=packed_query,
                query_scales=query_scales,
            )

        if num_decodes:
            # Decode request i owns rows i * n .. i * n + n - 1 (n = 1 + k). The
            # indexer has one head, so the TP ranks split its requests instead
            # (sequence parallel): each takes `share`, selects their rows' exact
            # Top-K as key positions (none for a request of length 0), and
            # the ranks all-gather the rows with the one-shot custom all-gather
            # of sequence-parallel layers (NCCL where it does not apply), as
            # float32 words: it takes only float dtypes. Its MNNVL kernel turns
            # the word 0x80000000 into 0; the rows hold key positions and -1,
            # never that word. A request of length 0 for a selector is another's.
            tp = get_tp_group()
            share = cdiv(num_decodes, tp.world_size)
            first = tp.rank_in_group * share
            mine = slice(first, min(first + share, num_decodes))
            rows = slice(first * rows_per_decode, mine.stop * rows_per_decode)
            out = self.topk_indices[rows]
            count = mine.stop - first
            if count > 0 and metadata.dense_lens is not None:
                scores = manager.get_simultaneous(self.decode_scores)[0][: len(out)]
                lengths = metadata.dense_lens[mine]
                indexer_scores(
                    packed_query[rows],
                    key_cache,
                    block_table[mine],
                    lengths,
                    requests=self.decode_requests[:count],
                    scores=scores,
                    query_scales=_rows(query_scales, rows),
                    key_scales=key_scales,
                )
                _top_k(scores, lengths, out, topk_workspace, rows_per_decode)
            if count > 0 and metadata.fused_lens is not None:
                query_rows = packed_query[rows], _rows(query_scales, rows)
                keys = key_cache, key_scales
                self._fused_decode(
                    metadata, mine, *query_rows, *keys, out, topk_workspace
                )
            if tp.world_size > 1:  # the last rank's tail is padding
                own = self.topk_indices[
                    first * rows_per_decode : (first + share) * rows_per_decode
                ]
                gathered = sp_all_gather(own.view(torch.float32))[:num_decode_rows]
                self.topk_indices[:num_decode_rows] = gathered.view(torch.int32)
        if prefill is not None:
            prefill.run(
                packed_query[num_decode_rows:],
                _rows(query_scales, slice(num_decode_rows, None)),
                key_cache,
                key_scales,
                out=self.topk_indices[num_decode_rows:],
                valid_counts=self.valid_counts,
                mask=self.mask,
                topk_workspace=topk_workspace,
            )

    def _fused_decode(
        self,
        metadata: SdsaIndexerMetadata,
        mine: slice,
        query: torch.Tensor,
        query_scales: torch.Tensor | None,
        key_cache: torch.Tensor,
        key_scales: torch.Tensor | None,
        out: torch.Tensor,
        workspace: torch.Tensor,
    ) -> None:
        """The fused selector on decode requests `mine` (query: their rows), one
        128-row tile each, into the rows of `out` of the requests it routes."""
        manager = current_workspace_manager()
        count, rows = mine.stop - mine.start, metadata.decode_rows
        fused_lens = metadata.fused_lens
        assert fused_lens is not None  # the caller routes requests here
        tiles = SdsaTiles(
            self.decode_requests[:count],
            metadata.fused_ends[128 * mine.start : 128 * mine.stop],
            fused_lens[mine],
            metadata.block_table[mine],
            max_context=self.max_model_len,
            sample_offsets=metadata.sample_offsets,
        )
        tiled = []  # the query rows and scales in their requests' tiles
        for x in query, query_scales:
            if x is not None:
                tile = x.new_zeros(count, 128, x.shape[1])
                tile[:, :rows] = x.view(count, rows, -1)
                x = tile.flatten(0, 1)
            tiled.append(x)
        query_tiles, scale_tiles = tiled
        assert query_tiles is not None
        panel = PANEL_ROWS // 128
        layout = _arena_layout(
            count,
            tiles.sample_capacity,
            128 * min(count, panel),
            self.sdsa_config,
            False,
        )
        arena = manager.get_simultaneous(*layout)
        samples = tiles.gather_sample(key_cache, key_scales, arena[0], arena[1])
        for begin in range(0, count, panel):
            end = min(begin + panel, count)
            tile_rows = slice(128 * begin, 128 * end)
            keys = tiles.select(
                begin,
                end,
                query_tiles[tile_rows],
                _rows(scale_tiles, tile_rows),
                key_cache,
                key_scales,
                samples,
                arena[2:-1],
                workspace,
                self.sdsa_config,
            )
            keys = keys.view(end - begin, 128, self.sdsa_config.top_k)[:, :rows]
            requests = out.view(count, rows, self.sdsa_config.top_k)[begin:end]
            if metadata.dense_lens is not None:  # only the requests routed here
                routed = fused_lens[mine][begin:end] > 0
                keys = torch.where(routed[:, None, None], keys, requests)
            requests.copy_(keys)
