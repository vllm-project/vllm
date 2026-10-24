# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.utils import (
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import nvfp4_mla_kv_cache_split_views
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
    max_decode_query_len,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


def _trtllm_gen_mla_decode_supports_num_heads(num_heads: int) -> bool:
    """True if trtllm-gen's MLA decode kernel supports this query head count.

    The kernel groups Q heads into CTAs of ``min(num_heads, tileSizeQ)`` and
    requires ``num_heads`` divisible by that, else raises "The
    numHeadsQ/numHeadsKv is not supported" (flashinfer fmhaKernels.cuh).
    ``tileSizeQ`` is 8/16 for ``num_heads <= 8``/``<= 32`` (SwapsMmaAb) else 64;
    treat ``> 32`` as tile 64 (safe bound). E.g. 96/24 -> False, 48/64/128 -> True.
    """
    if num_heads <= 8:
        tile = 8
    elif num_heads <= 32:
        tile = 16
    else:
        tile = 64
    return num_heads % min(num_heads, tile) == 0


def _select_mla_decode_backend(num_heads: int) -> str | None:
    """cute-dsl for head counts trtllm-gen cannot tile, else None (=> auto)."""
    if not _trtllm_gen_mla_decode_supports_num_heads(num_heads):
        logger.warning_once(
            "trtllm-gen MLA decode does not support num_heads=%d "
            "(query/kv head ratio); falling back to the cute-dsl backend.",
            num_heads,
        )
        return "cute-dsl"
    return None


FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
FLASHINFER_MLA_LSE_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

_fi_workspace: torch.Tensor | None = None


def _get_workspace_buffer(return_lse: bool) -> torch.Tensor:
    global _fi_workspace

    buffer_size = (
        FLASHINFER_MLA_LSE_WORKSPACE_BUFFER_SIZE
        if return_lse
        else FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE
    )
    if _fi_workspace is None or _fi_workspace.numel() < buffer_size:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_workspace = torch.zeros(buffer_size, dtype=torch.int8, device="cuda")
    return _fi_workspace


_fi_multi_ctas_kv_counter: torch.Tensor | None = None


def _get_multi_ctas_kv_counter_buffer(
    min_bytes: int, device: torch.device
) -> torch.Tensor:
    """Persistent, zero-initialized trtllm-gen multi-CTA-KV counter buffer.

    trtllm-gen's multi-CTA-KV MLA decode kernel resets these semaphores to zero
    at the end of every launch, so the buffer only needs zeroing once. The
    public ``trtllm_batch_decode_with_kv_cache_mla`` entry point builds a fresh
    runner per call, so without a caller-owned buffer it re-allocates and
    re-zeros this counter on every decode step (a tiny ``FillFunctor<uint8>``
    launch right before the FMHA). Owning it here and passing it in removes that
    per-step launch. ``min_bytes`` is sized to the worst-case batch so the
    buffer is allocated once and never reallocated after CUDA-graph capture.
    """
    global _fi_multi_ctas_kv_counter
    if (
        _fi_multi_ctas_kv_counter is None
        or _fi_multi_ctas_kv_counter.numel() < min_bytes
    ):
        _fi_multi_ctas_kv_counter = torch.zeros(
            min_bytes, dtype=torch.uint8, device=device
        )
    return _fi_multi_ctas_kv_counter


def _cake_nvfp4_mla_decode_unsupported_reason() -> str | None:
    """Why FlashInfer's Cake NVFP4 MLA decode (flashinfer #6242) cannot run here."""
    try:
        from flashinfer.experimental.cake_mla_nvfp4_paged_decode import cake_jit
    except ImportError:
        return (
            "kv_cache_dtype='nvfp4' needs a FlashInfer build with the Cake "
            "NVFP4 MLA decode (flashinfer-ai/flashinfer#6242)"
        )
    if not all(cake_jit.toolkit_supports(p) for p in cake_jit.KERNELS.values()):
        return (
            "kv_cache_dtype='nvfp4' MLA decode JIT-builds kernels that need "
            "nvcc from a CUDA 13.4+ toolkit (CUDA_HOME)"
        )
    return None


def cake_nvfp4_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    k_scale: float,
    max_seq_len: int,
    return_lse: bool = False,
    cum_seq_lens_q: torch.Tensor | None = None,
    max_q_len: int | None = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FlashInfer Cake MLA decode over the page-segmented NVFP4 MLA cache.

    The kernel's QK runs on NVFP4 x NVFP4 block-scaled MMAs, so it takes no FP8
    query: the BF16 query is quantized per row here (NVFP4 NoPE, FP8 RoPE).
    ``concat_and_cache_mla_nvfp4`` stores ``x / k_scale`` for both the NoPE and
    the RoPE, so ``k_scale`` is both of Cake's ``ckv_scale`` and ``kpe_scale``.

    Args:
        q: BF16 ``[B, q_len, H, 576]``, or packed ``[total_q, H, 576]`` with
            ``cum_seq_lens_q`` and ``max_q_len``.
        kv_cache: uint8 NVFP4 MLA cache ``[num_blocks, block_size, 352]``.
        block_table: int32 ``[B, max_blocks]``.
        seq_lens: int32 ``[B]`` (local lengths under DCP).
        sm_scale: softmax scale of the real-unit ``q . k``.
        k_scale: the cache's global scale (a host float).
        max_seq_len: KV bound used to plan the KV splits.
        return_lse: also return the natural-log LSE.
        cum_seq_lens_q: int32 ``[B + 1]`` query offsets of a packed ``q``.
        max_q_len: query-length bound of a packed ``q``.
        cp_world: decode context parallel world size.
        cp_rank: decode context parallel rank.
        kv_len_global: int32 ``[B]`` global KV lengths when ``cp_world > 1``.

    Returns:
        BF16 ``q.shape[:-1] + (512,)`` in real units, and the FP32
        ``q.shape[:-1]`` LSE when ``return_lse``.

    """
    from flashinfer.experimental.cake_mla_nvfp4_paged_decode import cake_backend
    from flashinfer.mla import cake_mla_nvfp4_paged_decode, quantize_mla_nvfp4_query

    q = q.contiguous()
    lead = tuple(q.shape[:-1])
    rows = math.prod(lead)
    query_workspace = torch.empty(
        cake_backend.query_workspace_bytes(rows), dtype=torch.uint8, device=q.device
    )
    q_nope, q_sf, q_rope, q_scale = quantize_mla_nvfp4_query(
        q,
        k_scale,
        k_scale,
        out=cake_backend.mla_nvfp4_query_buffers(query_workspace, lead),
    )
    # Size the split partials for the plan the call below makes.
    if cum_seq_lens_q is not None:
        assert max_q_len is not None
        batch, plan_q_len = cum_seq_lens_q.shape[0] - 1, max_q_len
    else:
        batch, plan_q_len = q.shape[0], q.shape[1]
    plan = cake_backend.plan_mla_nvfp4_paged_decode(
        batch=batch,
        max_q_len=plan_q_len,
        num_heads=q.shape[-2],
        max_seq_len=max_seq_len,
        sm_count=get_device_sm_count(q.device),
        rows=rows,
    )
    workspace = torch.empty(
        cake_backend.workspace_bytes(plan.rows_max, plan.num_split),
        dtype=torch.uint8,
        device=q.device,
    )
    ckv, kpe, ckv_sf = nvfp4_mla_kv_cache_split_views(kv_cache)
    out = torch.empty(
        lead + (ckv.shape[-1] * 2,), dtype=torch.bfloat16, device=q.device
    )
    lse = (
        torch.empty(lead, dtype=torch.float32, device=q.device) if return_lse else None
    )
    cake_mla_nvfp4_paged_decode(
        q_nope,
        q_sf,
        q_rope,
        q_scale,
        ckv,
        ckv_sf,
        kpe,
        block_table,
        seq_lens,
        workspace,
        sm_scale=sm_scale,
        ckv_scale=k_scale,
        out=out,
        lse=lse,
        return_lse=return_lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
    )
    return out, lse


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    flattened_block_table: torch.Tensor | None = None
    flattened_seq_lens: torch.Tensor | None = None
    flattened_row_req: torch.Tensor | None = None
    # Row count the flattened tensors were built for (0 = not built), not a query len.
    query_len: int = 0
    query_start_loc: torch.Tensor | None = None
    max_query_len: int = 1


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[FlashInferMLAMetadata]):
    # Adaptive verification requires ALWAYS from every builder, matching upstream's
    # DeepseekV4FlashMLAMetadataBuilder. The kernels tile ragged queries from the device
    # query offsets (flashinfer #3238), so one k+1 graph replays any 1..k+1 mix.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    # Non-causal DSpark blocks are flattened to single-token rows in forward_mqa.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        dcp_size = parallel_config.decode_context_parallel_size
        interleave_size = parallel_config.cp_kv_cache_interleave_size
        if dcp_size > 1 and interleave_size != 1:
            raise ValueError(
                "FlashInfer MLA native DCP requires "
                "cp_kv_cache_interleave_size=1; got "
                f"{interleave_size}."
            )
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            FlashInferMLAMetadata,
            supports_dcp_with_varlen=True,
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        max_query_len: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashInferMLADecodeMetadata:
        # Promised bound, not measured: a capture dummy puts one token/req, so
        # measuring would bake an undersized graph. Ragged only when max_query_len>1.
        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
        )


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

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
        if kv_cache_dtype == "nvfp4":
            reason = _cake_nvfp4_mla_decode_unsupported_reason()
            if reason is not None:
                return reason
            if (device_capability.major, device_capability.minor) not in (
                (10, 0),
                (10, 3),
            ):
                return "kv_cache_dtype='nvfp4' MLA decode needs SM100 or SM103"
            if dtype != torch.bfloat16 or head_size != 576:
                return (
                    "kv_cache_dtype='nvfp4' MLA decode needs bfloat16 activations, "
                    "kv_lora_rank 512 and qk_rope_head_dim 64"
                )
        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192]
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [64, 128, 192]:
                return (
                    "FlashInfer MLA kernel requires qk_nope_head_dim "
                    f"in [64, 128, 192], but got {qk_nope_head_dim}"
                )
        return None


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    # DCP is the only path that consumes LSE. It uses monolithic CuTeDSL,
    # whose public LSE contract is natural-log.
    lse_base_on_e: bool = True

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
        **mla_args,
    ) -> None:
        super().__init__(
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
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLAImpl"
            )

        if kv_cache_dtype == "nvfp4":
            # The Cake decode quantizes the BF16 query itself (per-row NVFP4).
            self.supports_quant_query_input = False
            from flashinfer.experimental.cake_mla_nvfp4_paged_decode import (
                cake_backend,
            )

            # Cake's two-CTA route, taken above WIDE_MIN_ROWS (token, head) rows
            # per request, zeroes V blocks whose E4M3 scale is below 2^-3.
            q_len = max_decode_query_len(get_current_vllm_config())
            rows = num_heads * self.dcp_world_size * q_len
            if cake_backend.use_wide_route(rows):
                raise NotImplementedError(
                    "kv_cache_dtype='nvfp4' MLA decode supports at most "
                    f"{cake_backend.WIDE_MIN_ROWS} query rows per request, got "
                    f"{rows} ({num_heads} heads x decode context parallel size "
                    f"{self.dcp_world_size} x {q_len} query tokens); use fewer "
                    "speculative tokens, a smaller decode context parallel size, "
                    "more tensor parallelism or --kv-cache-dtype fp8."
                )

        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        # Worst-case decode batch for the persistent trtllm-gen multi-CTA-KV
        # counter buffer (see _get_multi_ctas_kv_counter_buffer). Captured here
        # (config is in scope during construction) so the byte size can be
        # resolved once on the first decode and never grows after CUDA-graph
        # capture. The impl has no _vllm_config, hence get_current_vllm_config().
        _sched = get_current_vllm_config().scheduler_config
        self._mla_counter_max_batch: int = (
            _sched.max_num_batched_tokens or _sched.max_num_seqs
        )
        self._mla_counter_bytes: int | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens

        # Led by the promised bound (baked into the graph), not
        # num_decode_tokens // num_decodes -- that average is a per-request length
        # only for uniform batches.
        multi_token_decode = (
            attn_metadata.decode.max_query_len > 1
            or attn_metadata.num_decode_tokens > attn_metadata.num_decodes
        )
        cum_seq_lens_q: torch.Tensor | None = None
        max_q_len: int | None = None
        row_req: torch.Tensor | None = None

        if not attn_metadata.causal:
            # FlashInfer decode has no causal flag. Flatten each non-causal
            # query block into independent single-token rows.
            q = q.unsqueeze(1)
            if multi_token_decode:
                block_table, seq_lens, row_req = self._flattened_decode_metadata(
                    attn_metadata, q.shape[0]
                )
        elif attn_metadata.decode.max_query_len > 1 and (
            self.dcp_world_size == 1 or self.kv_cache_dtype == "nvfp4"
        ):
            # Causal spec decode: keep q compact and let the kernel tile each
            # request's length (uniform 1+k and adaptive ragged). Mirrors vllm #52157.
            # DCP keeps the uniform reshape below: it needs LSE, which the kernels
            # do not return on the ragged path (flashinfer #3238). The NVFP4 decode
            # returns LSE for a packed query too.
            cum_seq_lens_q = attn_metadata.decode.query_start_loc
            max_q_len = attn_metadata.decode.max_query_len
        # trtllm API requires extra dimension q_len_per_request for MTP
        elif attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
            logger.warning_once(
                """FlashInferMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(attn_metadata.num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.kv_cache_dtype == "nvfp4":
            kv_len_global = None
            if self.dcp_world_size > 1:
                kv_len_global = attn_metadata.decode.dcp_tot_seq_lens
                assert kv_len_global is not None
                if row_req is not None:
                    kv_len_global = kv_len_global[row_req]
            o, lse = cake_nvfp4_mla_decode(
                q,
                kv_c_and_k_pe_cache,
                block_table,
                seq_lens,
                sm_scale=self.scale,
                k_scale=layer._k_scale_float,
                max_seq_len=attn_metadata.max_seq_len,
                return_lse=self.need_to_return_lse_for_decode,
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=max_q_len,
                cp_world=self.dcp_world_size,
                cp_rank=self.dcp_rank,
                kv_len_global=kv_len_global,
            )
            if lse is not None:
                lse = lse.view(-1, lse.shape[-1])
            return o.view(-1, o.shape[-2], o.shape[-1]), lse

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        return_lse = self.need_to_return_lse_for_decode
        workspace_buffer = _get_workspace_buffer(return_lse)
        # Parallel gathers can change the runtime Q heads from TP-local num_heads.
        runtime_num_heads = q.shape[-2]
        extra_kwargs: dict[str, Any] = {}
        decode_backend: str | None
        if self.dcp_world_size > 1:
            causal_seqlens_kv_global = attn_metadata.decode.dcp_tot_seq_lens
            assert causal_seqlens_kv_global is not None
            if row_req is not None:
                causal_seqlens_kv_global = causal_seqlens_kv_global[row_req]
            extra_kwargs.update(
                enable_dcp=True,
                cp_world=self.dcp_world_size,
                cp_rank=self.dcp_rank,
                causal_seqlens_kv_global=causal_seqlens_kv_global,
            )
            decode_backend = "cute-dsl"
        else:
            # trtllm-gen rejects MLA head counts it can't tile (e.g. 96);
            # fall back to cute-dsl for those.
            decode_backend = _select_mla_decode_backend(runtime_num_heads)
        if cum_seq_lens_q is not None:
            # Neither decode backend returns LSE on the ragged path
            # (flashinfer #3238); DCP, the only LSE consumer, took the uniform
            # branch above, so this only guards a future caller wiring the two.
            assert not return_lse, (
                "FlashInferMLA ragged decode cannot return LSE; DCP and adaptive "
                "variable-length decode are mutually exclusive."
            )
            extra_kwargs["cum_seq_lens_q"] = cum_seq_lens_q
            extra_kwargs["max_q_len"] = max_q_len
        if decode_backend:
            extra_kwargs["backend"] = decode_backend
        elif kv_c_and_k_pe_cache.shape[-2] in (32, 64):
            # The auto path can dispatch to trtllm-gen, whose multi-CTA-KV decode
            # kernel self-resets its semaphore counter after each launch (so it
            # only needs zeroing once). Pass a persistent counter buffer to skip
            # the per-step re-allocate + re-zero the public entry point would
            # otherwise do. Guarded to configs where a trtllm-gen runner is
            # eligible (page/block size in {32, 64}); the arg is rejected when
            # only a cute-dsl runner can run.
            if self._mla_counter_bytes is None:
                self._mla_counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    self._mla_counter_max_batch,
                    runtime_num_heads,
                    get_device_sm_count(q.device),
                )
            extra_kwargs["multi_ctas_kv_counter_buffer"] = (
                _get_multi_ctas_kv_counter_buffer(self._mla_counter_bytes, q.device)
            )
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_table,
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            return_lse=return_lse,
            **extra_kwargs,
        )
        if return_lse:
            o, lse = kernel_out
            lse = lse.view(-1, lse.shape[-1])
        else:
            o, lse = kernel_out, None

        # Flatten the output for consistent shape
        o = o.view(-1, o.shape[-2], o.shape[-1])

        return o, lse

    def _flattened_decode_metadata(
        self,
        attn_metadata: FlashInferMLAMetadata,
        num_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand per-request decode tensors to one row per query token.

        Cached across the layers of a group, which all see the same batch.
        """
        decode = attn_metadata.decode
        assert decode is not None
        if decode.query_len != num_rows:
            cu = decode.query_start_loc
            assert cu is not None
            # searchsorted on the device offsets, not repeat_interleave(uniform_len):
            # the latter only lines up for uniform batches and hands ragged rows
            # another request's KV (silent garbage).
            rows = torch.arange(num_rows, device=cu.device, dtype=cu.dtype)
            row_req = torch.searchsorted(cu[1:], rows, right=True).clamp_(
                max=decode.block_table.shape[0] - 1
            )
            decode.flattened_row_req = row_req
            decode.flattened_block_table = decode.block_table[row_req]
            decode.flattened_seq_lens = decode.seq_lens[row_req]
            decode.query_len = num_rows
        assert decode.flattened_block_table is not None
        assert decode.flattened_seq_lens is not None
        assert decode.flattened_row_req is not None
        return (
            decode.flattened_block_table,
            decode.flattened_seq_lens,
            decode.flattened_row_req,
        )
