# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

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
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType

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


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    flattened_block_table: torch.Tensor | None = None
    flattened_seq_lens: torch.Tensor | None = None
    query_len: int = 0


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[FlashInferMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # Non-causal DSpark blocks are flattened to single-token rows in forward_mqa.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
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
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashInferMLADecodeMetadata:
        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

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
        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192]
        from vllm.config import get_current_vllm_config

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

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):
    can_return_lse_for_decode: bool = True
    # trtllm-gen MLA decode emits LSE in log2 (per flashinfer's own
    # reference at flashinfer/trace/templates/attention.py:81:
    # `logsumexp / log(2.0)`). Override the AttentionImplBase default
    # so MLAAttention's DCP combine branches on the correct base
    # (IS_BASE_E=False uses tl.exp2/tl.log2 natively, avoiding an FP
    # multiply per decode step).
    lse_base_on_e: bool = False

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
        query_len = attn_metadata.num_decode_tokens // attn_metadata.num_decodes

        if not attn_metadata.causal:
            # Non-causal DSpark block: flatten to single-token decode rows with
            # per-row context seq_lens (trtllm-gen has no causal flag and would
            # otherwise mask the block causally).
            q = q.unsqueeze(1)
            if query_len > 1:
                block_table, seq_lens = self._prepare_flattened_decode_metadata(
                    attn_metadata,
                    query_len,
                    causal=False,
                )
        elif self.dcp_world_size > 1 and query_len > 1:
            # Causal DCP block: flatten to single-token decode rows with
            # per-row rank-local seq_lens for each query's visible prefix.
            block_table, seq_lens = self._prepare_flattened_decode_metadata(
                attn_metadata,
                query_len,
                causal=True,
            )
            q = q.unsqueeze(1)
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
        # trtllm-gen rejects MLA head counts it can't tile (e.g. 96);
        # fall back to cute-dsl for those.
        decode_backend = _select_mla_decode_backend(runtime_num_heads)
        extra_kwargs = {}
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
        else:
            o, lse = kernel_out, None

        # Flatten the output for consistent shape
        o = o.view(-1, o.shape[-2], o.shape[-1])

        return o, lse

    def _prepare_flattened_decode_metadata(
        self,
        attn_metadata: FlashInferMLAMetadata,
        query_len: int,
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare flattened decode metadata once for all layers in the group."""
        decode = attn_metadata.decode
        assert decode is not None
        if decode.query_len:
            assert decode.query_len == query_len
            assert decode.flattened_block_table is not None
            assert decode.flattened_seq_lens is not None
            return decode.flattened_block_table, decode.flattened_seq_lens

        block_table = decode.block_table.repeat_interleave(query_len, dim=0)
        if causal:
            global_seq_lens = decode.dcp_tot_seq_lens
            assert global_seq_lens is not None
            offsets = torch.arange(
                query_len - 1,
                -1,
                -1,
                device=global_seq_lens.device,
                dtype=global_seq_lens.dtype,
            )
            per_query_global_lens = torch.clamp(
                (global_seq_lens.unsqueeze(1) - offsets).reshape(-1), min=0
            )
            interleave = self.cp_kv_cache_interleave_size
            dcp_span = self.dcp_world_size * interleave
            remainder = torch.clamp(
                per_query_global_lens % dcp_span - self.dcp_rank * interleave,
                min=0,
                max=interleave,
            )
            seq_lens = per_query_global_lens // dcp_span * interleave + remainder
        else:
            seq_lens = decode.seq_lens.repeat_interleave(query_len)

        decode.flattened_block_table = block_table
        decode.flattened_seq_lens = seq_lens
        decode.query_len = query_len
        return block_table, seq_lens
