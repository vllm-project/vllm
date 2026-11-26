# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.nemotron_shortcut.nvidia.sdsa.flashinfer_ops import (
    has_nvfp4_kv_conversion,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.prefill import (
    GROUP_UNION_MASKS_KERNEL,
    PAGE,
    SdsaConfig,
    SdsaPrefillPlan,
    _fused,
    _sample_capacity,
    _sample_offsets,
    union_columns,
    union_group,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    is_quantized_kv_cache,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

SDSA_CORE_HEAD_SIZE = 256


class SdsaIndexerBackend(AttentionBackend):
    """The indexer's key cache: packed NVFP4 keys and their scales."""

    @staticmethod
    def get_name() -> str:
        return "NEMOTRON_SHORTCUT_SDSA_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return SdsaIndexerMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        return [PAGE]

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


@dataclass
class SdsaPagesMetadata(AttentionMetadata):
    """A step's token count and each request's pages, shared by the indexer
    cache and the core: they locate the indexer's keys and map its selected
    keys to core cache slots."""

    num_actual_tokens: int
    block_table: torch.Tensor


class SdsaPagesMetadataBuilder(AttentionMetadataBuilder[SdsaPagesMetadata]):
    # CUDA graphs for pure-decode steps, speculative ones included.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # AttentionMetadataBuilder declares __init__ abstract.
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SdsaPagesMetadata:
        return SdsaPagesMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            block_table=common_attn_metadata.block_table_tensor,
        )


@dataclass
class SdsaIndexerMetadata(SdsaPagesMetadata):
    """Decode requests come first, `decode_rows` tokens each (1 + k with k
    drafts; unequal ones as one-row requests), with their pages in
    `block_table`; the prefill plan covers the rest of the batch. A decode
    request takes the dense or the fused selector (SdsaConfig.decode_fused):
    its length for the one it does not take is 0, and a selector that takes
    none has no lengths. `fused_ends` are its rows' ends in its 128-row tile.
    The step's query tokens go to query rows query_slots: decode tokens to
    their own rows, then prefill tokens to their rows in the plan's tiles."""

    num_decodes: int
    decode_rows: int
    query_slots: torch.Tensor
    token_requests: torch.Tensor
    dense_lens: torch.Tensor | None
    fused_lens: torch.Tensor | None
    fused_ends: torch.Tensor
    sample_offsets: torch.Tensor
    prefill: SdsaPrefillPlan | None


class SdsaIndexerMetadataBuilder(SdsaPagesMetadataBuilder):
    # The JIT warm-up of the union core needs the block table's width.
    requires_block_table_width = True

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        assert self.reorder_batch_threshold is not None
        config, device = self.vllm_config, self.device
        # Decode requests, or their tokens when flattened.
        rows = config.scheduler_config.max_num_seqs * self.reorder_batch_threshold
        self.token_requests = torch.zeros(rows, dtype=torch.int32, device=device)
        self.decode_slots = torch.arange(rows, dtype=torch.int64, device=device)
        self.dense_lens, self.fused_lens = torch.zeros(2, rows).to(self.token_requests)
        self.fused_ends = torch.zeros(rows, 128).to(self.token_requests)
        capacity = _sample_capacity(config.model_config.max_model_len)
        self.sample_offsets = _sample_offsets(capacity, device)
        self.heads = config.model_config.get_num_attention_heads(config.parallel_config)
        # The SDSA layers of a model share one config; their indexer caches,
        # this builder's layers, hold it.
        context = config.compilation_config.static_forward_context
        self.sdsa_config = context[self.layer_names[0]].sdsa_config
        # The JIT warm-up builds the union core's kernels at startup. The
        # config can pick the union core if it picks it for a small step.
        small_step_core = self.sdsa_config.prefill(
            heads=self.heads,
            context=PAGE,
            rows=PAGE,
            kv_cache_dtype=config.cache_config.cache_dtype,
        )
        if small_step_core == "union":
            group = union_group(self.heads)
            max_model_len = config.model_config.max_model_len
            columns = union_columns(group, self.sdsa_config.top_k, max_model_len)
            GROUP_UNION_MASKS_KERNEL.register_warmup(
                group=group,
                top_k=self.sdsa_config.top_k,
                max_model_len=max_model_len,
                table_pages=block_table_width,
                columns=columns,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SdsaIndexerMetadata:
        common = common_attn_metadata
        # vLLM's indexer builder (DeepseekV32IndexerMetadataBuilder) uses this
        # split too. It puts the decode requests first in the batch.
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_tokens, _ = split_decodes_and_prefills(
            common, decode_threshold=self.reorder_batch_threshold
        )
        query_lens = torch.diff(common.query_start_loc_cpu)
        rows = int(query_lens[:num_decodes].max()) if num_decodes else 1
        lengths = common.seq_lens[:num_decodes]
        table = common.block_table_tensor[:num_decodes]
        if ((query_lens[:num_decodes] != rows) & (query_lens[:num_decodes] > 0)).any():
            # Decodes of unequal lengths: each token a one-row decode of its own
            # length, as vLLM's indexer builder flattens them.
            starts = common.query_start_loc[: num_decodes + 1]
            request = torch.arange(num_decodes, device=self.device).repeat_interleave(
                starts.diff(), output_size=num_tokens
            )
            token = torch.arange(num_tokens, device=self.device)
            lengths = lengths[request] - starts[request + 1] + token + 1
            table, num_decodes, rows = table[request], num_tokens, 1
        rule = self.sdsa_config.fused("decode")
        fused = _fused(rule, lengths)
        dense_lens = fused_lens = None
        if rule is not True:
            dense_lens = self.dense_lens[:num_decodes]
            dense_lens.copy_(lengths.masked_fill(fused, 0))
        if rule is not False:
            fused_lens = self.fused_lens[:num_decodes]
            fused_lens.copy_(lengths.masked_fill(~fused, 0))
            column = torch.arange(128, device=self.device)
            ends = (fused_lens[:, None] - rows + 1 + column).clamp(min=0)
            self.fused_ends[:num_decodes] = ends.masked_fill(column >= rows, 0)
        tokens = self.token_requests[: num_decodes * rows]
        torch.arange(len(tokens), out=tokens).floor_divide_(rows)
        prefill = None
        if num_prefills or rule is not False:
            self.sample_offsets.copy_(
                _sample_offsets(len(self.sample_offsets), self.device)
            )
        if num_prefills:
            requests = slice(common.num_reqs - num_prefills, common.num_reqs)
            seq_lens = common.seq_lens_cpu_upper_bound
            assert seq_lens is not None
            prefill = SdsaPrefillPlan(
                first_row=num_decodes * rows,
                query_lens=tuple(query_lens[requests].tolist()),
                context_lens=tuple(seq_lens[requests].tolist()),
                seq_lens=common.seq_lens[requests],
                block_table=common.block_table_tensor[requests],
                sample_offsets=self.sample_offsets,
                heads=self.heads,
                config=self.sdsa_config,
                kv_cache_dtype=self.vllm_config.cache_config.cache_dtype,
            )
        return SdsaIndexerMetadata(
            num_actual_tokens=common.num_actual_tokens,
            block_table=table,
            num_decodes=num_decodes,
            decode_rows=rows,
            query_slots=(
                self.decode_slots[: len(tokens)]
                if prefill is None
                else prefill.query_slots
            ),
            token_requests=tokens,
            dense_lens=dense_lens,
            fused_lens=fused_lens,
            fused_ends=self.fused_ends.view(-1),
            sample_offsets=self.sample_offsets,
            prefill=prefill,
        )


class SdsaAttentionBackend(AttentionBackend):
    """Core attention over the tokens the indexer selected, in FlashAttention's
    cache layout; K/V BF16, FP8 E4M3, or packed NVFP4."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # NVFP4 keeps K and V in separate head slots, each its packed values
        # followed by their block scales.
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=nvfp4_kv_cache_full_dim(spec.head_size),
        )

    # Attention caches the K/V first, with SdsaAttentionImpl.do_kv_cache_update.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        return [PAGE]  # the indexer's page; the Top-K maps keys to slots with it

    @staticmethod
    def get_name() -> str:
        return "NEMOTRON_SHORTCUT_SDSA"

    @staticmethod
    def get_impl_cls() -> type["SdsaAttentionImpl"]:
        return SdsaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SdsaPagesMetadataBuilder"]:
        return SdsaPagesMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == SDSA_CORE_HEAD_SIZE

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The core reads each layer's pages as one pool of token rows, so the
        # layer stays outermost and its pages dense. NVFP4 separates the K and
        # V head slots and swizzles its block scales per head (HND).
        return (KVCacheLayout.LBHNC,)


# The core kernel runs one request per CTA row of its grid (grid z <= 65535).
_CORE_MAX_ROWS = 65535
# The kernel's smaller query tile (Q64). A token's heads may be split into
# requests of at most this many heads each.
_CORE_HEADS_PER_REQUEST = 64
# FlashInfer's recommended trtllm-gen workspace size (decode.py: "The
# recommended size is 128MB"); the launcher's multi-CTA scratch takes it all.
_CORE_WORKSPACE_BYTES = 128 << 20
# NVFP4 K/V converted to FP8 (rows of [K | V]): workspace shared by the decode
# and prefill conversions; one launch converts as many rows as fit.
_NVFP4_FAST_WORKSPACE_BYTES = 512 << 20
# Decode batches of at least this many requests that verify drafts convert
# each request's selected keys once (compact_nvfp4_kv_cache); smaller ones
# convert each row's (gather_nvfp4_kv_cache_by_position, one launch).
_NVFP4_UNION_MIN_DECODES = 16
# Most row groups (requests, or parts of a long prefill) per compaction launch;
# sizes its key bitmap (a bit per key of max_model_len per group).
_NVFP4_MAX_GROUPS = 256
# Rows per trtllm-gen launch over the converted K/V: 32 (decode) and 256
# (prefill), the launches of the earlier per-row conversion. trtllm-gen picks
# its kernel (split-KV or not, hence the summation order) by the launch's rows,
# so these keep the outputs bit-identical to it.
_NVFP4_DECODE_FMHA_ROWS = 32
_NVFP4_PREFILL_FMHA_ROWS = 256
# Conversion launch tuning (tokens per program, warps); no numerical effect.
_NVFP4_CONVERT_TOKENS = 16
_NVFP4_CONVERT_WARPS = 4


def _nvfp4_prefill_launches(
    query_lens: tuple[int, ...],
    context_lens: tuple[int, ...],
    top_k: int,
    region_rows: int,
    max_rows: int,
    max_groups: int,
) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray]:
    """Split prefill requests' tokens into compaction groups and launches.

    A group is a run of one request's tokens whose selected keys get one
    region of the FP8 workspace: all its tokens when the region fits, else
    runs of region_rows // top_k tokens. A group is dense (its region holds
    every key up to its context, by position) when its rows select at least
    as many keys as its context holds, else a union (its region holds each
    selected key once; at most rows * top_k of them). Launches take whole
    groups while their regions fit region_rows, their tokens max_rows and
    they number at most max_groups.

    Returns the launches as (first group, end group, first token, end token,
    largest region), and per group (request, first token, tokens, region
    offset in its launch, dense).

    >>> launches, groups = _nvfp4_prefill_launches(
    ...     (2816, 50), (200_000, 150_000), 2048, 1 << 20, 65535, 256
    ... )
    >>> launches
    [(0, 2, 0, 2866, 200000)]
    >>> groups.tolist()
    [[0, 0, 2816, 0, 1], [1, 2816, 50, 200000, 0]]
    >>> launches, _ = _nvfp4_prefill_launches(
    ...     (3000,), (900_000,), 2048, 300_000, 65535, 256
    ... )
    >>> len(launches), launches[0], launches[-1]
    (21, (0, 1, 0, 146, 299008), (20, 21, 2920, 3000, 163840))
    """
    groups = []
    token = 0
    for request, (tokens, context) in enumerate(zip(query_lens, context_lens)):
        if min(tokens * top_k, context) <= region_rows:
            run = tokens
        else:
            run = max(1, region_rows // top_k)
        run = max(1, min(run, max_rows))
        for first in range(0, tokens, run):
            count = min(run, tokens - first)
            dense = context <= count * top_k
            region = context if dense else count * top_k
            groups.append([request, token + first, count, region, int(dense)])
        token += tokens
    launches, begin, used, rows = [], 0, 0, 0
    for index, group in enumerate(groups):
        count, region = group[2], group[3]
        if index > begin and (
            used + region > region_rows
            or rows + count > max_rows
            or index - begin >= max_groups
        ):
            launches.append((begin, index))
            begin, used, rows = index, 0, 0
        group[3] = used  # the region's offset in its launch
        used += region
        rows += count
        group.append(region)
    launches.append((begin, len(groups)))
    table = np.array(groups, dtype=np.int64).reshape(-1, 6)
    result = []
    for begin, end in launches:
        part = table[begin:end]
        result.append(
            (
                begin,
                end,
                int(part[0, 1]),
                int(part[-1, 1] + part[-1, 2]),
                int(part[:, 5].max()),
            )
        )
    return result, table[:, :5]


class SdsaAttentionImpl(AttentionImpl):
    """MQA over each query's selected tokens. K and V are views of the cache's
    [K | V] rows as page-size-1 pages, so there is no paging and no plan.

    Each phase's core (SdsaConfig) attends in one of three ways. "token": a
    token runs FlashInfer's trtllm-gen head-dim-256 static token-sparse kernel
    as one or more requests, each with an equal share of its heads, whose page
    list is its selected slots, as vLLM's FlashInfer sparse MLA backend
    (FlashInferMLASparseImpl) gives that kernel its selections. "fa4": a token
    is a FlashAttention 4 batch over its selected slots. "union": a group of
    consecutive prefill tokens is an FA4 batch over the union of their
    selections (SdsaPrefillPlan.run), each token's bitmask over it its mask,
    the group's heads in one tile (PackGQA).
    K/V are BF16, FP8 E4M3 or packed NVFP4 with the layer's per-tensor scales;
    for FP8 and NVFP4 the query arrives quantized to FP8 by Attention with the
    layer's q scale, as in vLLM's FlashInfer backend. NVFP4 K/V take the token
    core: their selected tokens are first converted to FP8 scratch by key
    position (FlashInfer's compact_nvfp4_kv_cache, once per request, or
    gather_nvfp4_kv_cache_by_position, per row), divided by 6 to fit E4M3's
    range, which the BMM scales restore.
    """

    supports_quant_query_input = True

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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        *,
        topk_indices_buffer: torch.Tensor,
        valid_counts_buffer: torch.Tensor,
        indexer_name: str,
        sdsa_config: SdsaConfig,
    ) -> None:
        if num_kv_heads != 1:
            raise ValueError("SDSA requires one KV head")
        if head_size != SDSA_CORE_HEAD_SIZE:
            raise NotImplementedError("SDSA core attention serves head size 256")
        # trtllm_fmha_kernel_launcher.cu: num_qo_heads / num_kv_heads <= 128.
        if num_heads > 128:
            raise NotImplementedError("SDSA core attention serves at most 128 heads")
        if kv_cache_dtype not in SdsaAttentionBackend.supported_kv_cache_dtypes:
            raise NotImplementedError(
                f"SDSA core does not support {kv_cache_dtype} K/V"
            )
        if alibi_slopes is not None or sliding_window is not None or sinks is not None:
            raise NotImplementedError(
                "SDSA does not support ALiBi, sliding window, or sinks"
            )
        if logits_soft_cap:
            raise NotImplementedError("SDSA does not support logits soft cap")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("SDSA supports decoder self-attention only")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("SDSA does not support KV sharing")
        self.num_heads = num_heads
        # The fewest core requests per token that each fit the Q64 tile.
        self.requests_per_token = next(
            n
            for n in range(1, num_heads + 1)
            if num_heads % n == 0 and num_heads // n <= _CORE_HEADS_PER_REQUEST
        )
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = 1
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        # The core attends in FP8 for FP8 and NVFP4 caches alike (NVFP4 K/V are
        # converted to FP8); nvfp4_kv adds the packed layout and the conversion.
        self.fp8_kv = is_quantized_kv_cache(kv_cache_dtype)
        self.nvfp4_kv = kv_cache_dtype == "nvfp4"
        if self.nvfp4_kv and not has_nvfp4_kv_conversion():
            raise NotImplementedError(
                "SDSA with NVFP4 K/V needs FlashInfer's conversion by key position "
                "(flashinfer.sdsa.token_sparse.compact_nvfp4_kv_cache and "
                "gather_nvfp4_kv_cache_by_position, FlashInfer 0.7.6 or later)"
            )
        self.topk_indices_buffer = topk_indices_buffer
        self.valid_counts_buffer = valid_counts_buffer
        self.indexer_name = indexer_name
        self.counter_bytes: int | None = None  # needs the device, known in forward
        vllm_config = get_current_vllm_config()
        drafts = vllm_config.num_speculative_tokens
        self.decode_core = sdsa_config.decode(
            heads=num_heads, drafts=drafts, kv_cache_dtype=kv_cache_dtype
        )
        # Key positions are below the context length; the compaction bitmap
        # takes a bit per possible key.
        self.max_keys = vllm_config.model_config.max_model_len + drafts + 1
        self.decode_rows = 1 + drafts

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # (B, 1, N, 2*D) -> ((B, N, 1, D), (B, N, 1, D)), as FlashAttention's;
        # NVFP4 (B, 2, N, C) -> K and V head slots.
        if self.nvfp4_kv:
            key_cache, value_cache = kv_cache.transpose(1, 2).chunk(2, dim=-2)
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SdsaPagesMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("SDSA does not support fused output quantization")
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.token_sparse import token_sparse_kv_indices

        from vllm.models.nemotron_shortcut.nvidia.sdsa.mask import sdsa_mask_mod
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        # Requested before the profiling run returns, so their memory is counted.
        manager = current_workspace_manager()
        workspace = manager.get_persistent(
            "sdsa_core_workspace", (_CORE_WORKSPACE_BYTES,), torch.uint8, zero_init=True
        )
        # Zeroed once; the kernel leaves its split-KV semaphores zeroed.
        if self.counter_bytes is None:
            from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

            self.counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                _CORE_MAX_ROWS, self.num_heads, num_compute_units(query.device.index)
            )
        counters = manager.get_persistent(
            ("sdsa_core_counters", self.num_heads),
            (self.counter_bytes,),
            torch.uint8,
            zero_init=True,
        )
        # The kernel reads a K and a V slot list per request; ours are one list
        # per token, repeated for each of its requests.
        max_tokens, top_k = self.topk_indices_buffer.shape
        layout = [((max_tokens, 2, top_k), torch.int32), ((max_tokens,), torch.int32)]
        # NVFP4 conversion: FP8 [K | V] rows, then one zero row.
        fast_rows = _NVFP4_FAST_WORKSPACE_BYTES // (2 * self.head_size)
        decode_region = self.decode_rows * top_k
        decode_groups = max(1, min(_NVFP4_MAX_GROUPS, fast_rows // decode_region))
        if self.nvfp4_kv:
            from flashinfer.sdsa.token_sparse import (
                compact_nvfp4_bitmap_numel,
                compact_nvfp4_scratch_numel,
            )

            layout.extend(
                [
                    ((fast_rows + 1, 1, 1, 2 * self.head_size), torch.float8_e4m3fn),
                    (
                        (
                            compact_nvfp4_scratch_numel(
                                _NVFP4_MAX_GROUPS, self.max_keys, fast_rows + 1
                            ),
                        ),
                        torch.int32,
                    ),
                ]
            )
            # Zeroed once; every compaction leaves it zeroed.
            bitmap = manager.get_persistent(
                ("sdsa_nvfp4_bitmap", self.max_keys),
                (compact_nvfp4_bitmap_numel(_NVFP4_MAX_GROUPS, self.max_keys),),
                torch.int32,
                zero_init=True,
            )
            # Decode request i is group i, at region i * decode_region of a launch.
            device = query.device
            group_ids = manager.get_persistent_resource(
                ("sdsa_nvfp4_group_ids", max_tokens),
                lambda: torch.arange(max_tokens, dtype=torch.int32, device=device),
            )
            decode_offsets = manager.get_persistent_resource(
                ("sdsa_nvfp4_decode_offsets", decode_groups, decode_region),
                lambda: (
                    torch.arange(decode_groups, dtype=torch.int64, device=device)
                    * decode_region
                ),
            )
        scratch = manager.get_simultaneous(*layout)
        block_tables, counts = scratch[:2]
        if self.nvfp4_kv:
            converted, compact_scratch = scratch[2:]
        if attn_metadata is None:
            return output.zero_()
        num_tokens = attn_metadata.num_actual_tokens
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        indexer = metadata[self.indexer_name]
        assert isinstance(indexer, SdsaIndexerMetadata)
        decode_tokens = min(indexer.num_decodes * indexer.decode_rows, num_tokens)
        prefill = indexer.prefill

        block_scales = None
        if self.nvfp4_kv:
            # (B, 2, N, C): the K and V head slots, values then block scales.
            key_cache, value_cache = kv_cache.chunk(2, dim=1)
            key_cache, key_scales = nvfp4_split_data_scale(key_cache)
            value_cache, value_scales = nvfp4_split_data_scale(value_cache)
            block_scales = (key_scales, value_scales)
            kv_dtype = torch.float8_e4m3fn
        else:
            # (B, 1, N, 2*D) -> token rows [K | V]; K and V as (B*N, 1, 1, D).
            kv_cache = kv_cache.transpose(1, 2)
            if kv_cache.stride(0) != kv_cache.shape[1] * kv_cache.stride(1):
                raise RuntimeError("SDSA core attention requires contiguous KV pages")
            rows = kv_cache.reshape(-1, 1, 1, 2 * self.head_size)
            if self.fp8_kv:
                rows = rows.view(torch.float8_e4m3fn)
            kv_dtype = rows.dtype
            key_cache, value_cache = rows.split(self.head_size, dim=-1)
        if query.dtype != kv_dtype:
            raise TypeError("SDSA core attention needs the query in the K/V dtype")
        bmm1_scale, bmm2_scale = self.scale, 1.0
        if self.fp8_kv:
            bmm1_scale *= layer._q_scale_float * layer._k_scale_float
            bmm2_scale = layer._v_scale_float
        if self.nvfp4_kv:
            # NVFP4's range is 6 * 448: the conversion divides by 6 to fit E4M3.
            bmm1_scale *= 6.0
            bmm2_scale *= 6.0

        def flash(q, out, page_table, lengths, **kwargs) -> None:
            # A 0-dim scale expands to stride 0 everywhere; for one row PyTorch
            # calls that contiguous, so FA4 passes it on and CuTe rejects the 0
            # last stride.
            v_descale = (
                layer._v_scale.reshape(1, 1).expand(len(lengths), 1)
                if self.fp8_kv
                else None
            )
            _flash_attn_fwd(
                q,
                key_cache,
                value_cache,
                page_table=page_table,
                seqused_k=lengths,
                softmax_scale=bmm1_scale,
                v_descale=v_descale,
                out=out,
                **kwargs,
            )

        def per_token(core, tokens, slots, lengths) -> None:
            """Each of `tokens` attends to its lengths[t] slots: as an FA4 batch
            whose page table is slots[t] ("fa4"), or as trtllm-gen requests of
            an equal share of its heads, whose K and V lists are slots[r]
            ("token")."""
            q = query[tokens].reshape(len(lengths), -1, self.head_size)
            out = output[tokens].view_as(q)
            if core == "fa4":
                flash(q[:, None], out[:, None], slots, lengths, num_splits=0)
                return
            trtllm_batch_decode_with_kv_cache(
                q,
                (key_cache, value_cache),
                workspace,
                slots,
                lengths,
                top_k,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out=out,
                backend="trtllm-gen",
                uses_shared_paged_kv_idx=False,
                multi_ctas_kv_counter_buffer=counters,
                sparse_top_k=top_k,
            )

        def converted_attention(tokens, tables, lengths, launch_rows) -> None:
            """trtllm-gen token-sparse attention of `tokens` (len(lengths)
            requests, `split` per token) over the converted FP8 K/V rows, in
            launches of launch_rows requests (0: one launch)."""
            keys, values = converted.chunk(2, dim=-1)
            q = query[tokens].reshape(len(lengths), -1, self.head_size)
            out = output[tokens].view_as(q)
            step = launch_rows or len(lengths)
            for first in range(0, len(lengths), step):
                part = slice(first, first + step)
                trtllm_batch_decode_with_kv_cache(
                    q[part],
                    (keys, values),
                    workspace,
                    tables[part],
                    lengths[part],
                    top_k,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    out=out[part],
                    backend="trtllm-gen",
                    uses_shared_paged_kv_idx=False,
                    multi_ctas_kv_counter_buffer=counters,
                    sparse_top_k=top_k,
                )

        # Each decode token is `split` core requests. The token core splits a
        # token's heads to fit its Q64 tile, only while the split grid has at
        # most one request per SM and fits the table.
        split = self.requests_per_token if self.decode_core == "token" else 1
        if decode_tokens * split > min(
            num_compute_units(query.device.index), max_tokens
        ):
            split = 1
        tokens_per_launch = min(max_tokens, _CORE_MAX_ROWS) // split
        if self.nvfp4_kv and decode_tokens:
            from flashinfer.sdsa.token_sparse import (
                compact_nvfp4_kv_cache,
                gather_nvfp4_kv_cache_by_position,
            )

            # Decode request i owns tokens i * n .. i * n + n - 1 (token_requests).
            rows_per_request = indexer.decode_rows
            requests = -(-decode_tokens // rows_per_request)
            union = rows_per_request > 1 and requests >= _NVFP4_UNION_MIN_DECODES
            logger.info_once(
                "SDSA NVFP4 decode: %s conversion (%d+ requests x %d rows), "
                "FMHA launches of %s rows",
                "union (once per request)" if union else "per-row",
                _NVFP4_UNION_MIN_DECODES if union else 1,
                rows_per_request,
                _NVFP4_DECODE_FMHA_ROWS or "all",
            )
            if union:
                # Each request's selected keys once: its rows overlap.
                per_launch = min(decode_groups, tokens_per_launch // rows_per_request)
                for first in range(0, requests, per_launch):
                    end = min(first + per_launch, requests)
                    tokens = slice(
                        first * rows_per_request,
                        min(end * rows_per_request, decode_tokens),
                    )
                    num_rows = (tokens.stop - tokens.start) * split
                    compact_nvfp4_kv_cache(
                        (key_cache, value_cache),
                        block_scales,
                        self.topk_indices_buffer[tokens],
                        indexer.token_requests[tokens],
                        group_ids[first:end],
                        indexer.block_table,
                        decode_offsets[: end - first],
                        out=converted,
                        block_tables=block_tables,
                        seq_lens=counts,
                        scratch=compact_scratch,
                        bitmap=bitmap,
                        max_keys=self.max_keys,
                        max_capacity=decode_region,
                        zero_row=fast_rows,
                        group_base=first,
                        requests_per_row=split,
                        tokens_per_program=_NVFP4_CONVERT_TOKENS,
                        num_warps=_NVFP4_CONVERT_WARPS,
                    )
                    converted_attention(
                        tokens,
                        block_tables[:num_rows],
                        counts[:num_rows],
                        _NVFP4_DECODE_FMHA_ROWS,
                    )
            else:
                per_launch = min(tokens_per_launch, fast_rows // top_k)
                for start in range(0, decode_tokens, per_launch):
                    tokens = slice(start, min(start + per_launch, decode_tokens))
                    num_rows = (tokens.stop - tokens.start) * split
                    gather_nvfp4_kv_cache_by_position(
                        (key_cache, value_cache),
                        block_scales,
                        self.topk_indices_buffer[tokens],
                        indexer.token_requests[tokens],
                        indexer.block_table,
                        out=converted,
                        block_tables=block_tables,
                        seq_lens=counts,
                        requests_per_row=split,
                        tokens_per_program=_NVFP4_CONVERT_TOKENS,
                        num_warps=_NVFP4_CONVERT_WARPS,
                    )
                    converted_attention(
                        tokens,
                        block_tables[:num_rows],
                        counts[:num_rows],
                        _NVFP4_DECODE_FMHA_ROWS,
                    )
            decode_tokens_left = 0
        else:
            decode_tokens_left = decode_tokens
        for start in range(0, decode_tokens_left, tokens_per_launch):
            stop = min(start + tokens_per_launch, decode_tokens_left)
            num_rows = (stop - start) * split
            token_sparse_kv_indices(
                self.topk_indices_buffer[start:stop],
                indexer.token_requests[start:stop],
                indexer.block_table,
                block_tables=block_tables,
                seq_lens=counts,
                requests_per_token=split,
            )
            slots = block_tables[:num_rows]
            if self.decode_core == "fa4":
                slots = slots[:, 0]
            per_token(self.decode_core, slice(start, stop), slots, counts[:num_rows])
        if prefill is not None and self.nvfp4_kv:
            from flashinfer.sdsa.token_sparse import compact_nvfp4_kv_cache

            # The step's plan, made by its first layer: each request's selected
            # keys once, its tokens' rows holding key positions.
            assert prefill.keep_positions
            if prefill.nvfp4_launches is None:
                launches, groups = _nvfp4_prefill_launches(
                    prefill.query_lens,
                    prefill.context_lens,
                    top_k,
                    fast_rows,
                    min(max_tokens, _CORE_MAX_ROWS),
                    _NVFP4_MAX_GROUPS,
                )
                device = query.device
                token_groups = np.repeat(
                    np.arange(len(groups), dtype=np.int32), groups[:, 2]
                )
                columns = [groups[:, i].copy() for i in (0, 3, 4)]
                prefill.nvfp4_launches = [
                    launches,
                    async_tensor_h2d(token_groups, device, torch.int32),
                    async_tensor_h2d(columns[0].astype(np.int32), device),
                    async_tensor_h2d(columns[1], device, torch.int64),
                    async_tensor_h2d(columns[2].astype(np.int32), device),
                ]
            launches, token_groups, requests, offsets, dense = prefill.nvfp4_launches
            logger.info_once(
                "SDSA NVFP4 prefill: once-per-request conversion (dense when a "
                "chunk's rows x %d >= its context, else union), FMHA launches of "
                "%s rows",
                top_k,
                _NVFP4_PREFILL_FMHA_ROWS or "all",
            )
            for begin, end, first, stop, region in launches:
                num_rows = stop - first
                compact_nvfp4_kv_cache(
                    (key_cache, value_cache),
                    block_scales,
                    prefill.unions[first:stop],
                    token_groups[first:stop],
                    requests[begin:end],
                    prefill.block_table,
                    offsets[begin:end],
                    out=converted,
                    block_tables=block_tables,
                    seq_lens=counts,
                    scratch=compact_scratch,
                    bitmap=bitmap,
                    max_keys=self.max_keys,
                    max_capacity=region,
                    zero_row=fast_rows,
                    group_base=begin,
                    group_dense=dense[begin:end],
                    # Valid counts: TP ranks all-gather only the rows' columns
                    # that short contexts can fill; later ones may be stale.
                    lengths=self.valid_counts_buffer[first:stop],
                    tokens_per_program=_NVFP4_CONVERT_TOKENS,
                    num_warps=_NVFP4_CONVERT_WARPS,
                )
                converted_attention(
                    slice(decode_tokens + first, decode_tokens + stop),
                    block_tables[:num_rows],
                    counts[:num_rows],
                    _NVFP4_PREFILL_FMHA_ROWS,
                )
        elif prefill is not None:
            tokens, unions = slice(decode_tokens, num_tokens), prefill.unions
            sizes = self.valid_counts_buffer[: len(unions)]
            if prefill.core == "union":
                flash(
                    query[tokens].view(-1, self.num_heads, self.head_size),
                    output[tokens].view(-1, self.num_heads, self.head_size),
                    unions,
                    sizes,
                    cu_seqlens_q=prefill.batch_starts,
                    max_seqlen_q=prefill.group,
                    mask_mod=sdsa_mask_mod,
                    aux_tensors=[prefill.mask],
                )
            else:  # each one-token group's union is its token's slots
                for start in range(0, len(unions), _CORE_MAX_ROWS):
                    part = slice(start, start + _CORE_MAX_ROWS)
                    slots, first = unions[part], decode_tokens + start
                    if prefill.core == "token":  # as the K and the V list
                        slots = block_tables[: len(slots)]
                        slots.copy_(unions[part, None, :top_k])
                    tokens = slice(first, first + len(slots))
                    per_token(prefill.core, tokens, slots, sizes[part])
        return output[:num_tokens]
