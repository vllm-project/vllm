# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100 family) block-sparse attention for MiniMax M3.

``MiniMaxM3SparseMSAImpl`` serves BF16 and FP8 KV caches: prefill attends with
``fmha_sm100`` (``build_k2q_csr`` + ``sparse_atten_func``), and decode uses the
CUTLASS ``fmha_sm100`` path by default for regular decode and speculative
verification, with Triton split-K as the fallback.

``MiniMaxM3SparseMSANvfp4Impl`` serves NVFP4 KV caches, reading vLLM's packed
pages directly with only the E4M3 query: decode and prefill both run the CUTLASS
Q8KV4 kernels.
"""

import functools
import os
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    NVFP4_BLOCK_SCALE_SHIFT,
    MSACutlassDecodeMetadata,
    MSACutlassDecodePlanCache,
    is_nvfp4_kv_cache,
    msa_cutlass_sparse_decode,
    nvfp4_kv_cache_views,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
    supports_cutlass_sparse_decode,
    uses_q8kv4_decode,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backend import (
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

if TYPE_CHECKING:
    from vllm.third_party.fmha_sm100.prefill_q8kv4 import PrefillWorkspace

logger = init_logger(__name__)


@functools.cache
def _q8kv4_prefill_blocker(
    num_q_heads: int, num_kv_heads: int, page_size: int, topk_blocks: int
) -> str | None:
    """Why the Q8KV4 kernel cannot prefill this layout on this device, or None."""
    from vllm.third_party.fmha_sm100.q8kv4_prefill_adapter import _plan_blocker

    return _plan_blocker(
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        kv_block_num=topk_blocks,
        causal=True,
        output_maxscore=False,
        device=None,
    )


def _get_msa_cutlass_min_batch_size() -> int | None:
    value = os.getenv("VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH")
    return int(value) if value is not None else None


def _dequantize_query(
    query_fp8: torch.Tensor, q_scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """``query_fp8 * q_scale`` in ``dtype`` (E4M3 is exact in BF16)."""
    out = query_fp8.to(dtype)
    if q_scale != 1.0:
        out.mul_(q_scale)
    return out


class MiniMaxM3SparseMSABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend with NVIDIA MSA-specific decode metadata."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMSAMetadataBuilder"]:
        return MiniMaxM3SparseMSAMetadataBuilder


class MiniMaxM3SparseMSANvfp4Backend(MiniMaxM3SparseMSABackend):
    """MSA backend over HND NVFP4 KV cache pages in per-head K/V slots."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Store each head's K and V as two slots of packed data + block scales
        (see ``nvfp4_kv_cache_views``)."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=nvfp4_kv_cache_full_dim(spec.head_size)
            * get_dtype_size(spec.dtype),
            kv_head_major_slots=True,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        # Every head's K/V slots are one contiguous run of the page.
        return (KVCacheLayout.LBHNC,)


class MiniMaxM3SparseCutlassBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting CUTLASS MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MSA"


class MiniMaxM3SparseTritonBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting Triton MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MSA"


@dataclass
class MiniMaxM3SparseMSADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    msa_cutlass: MSACutlassDecodeMetadata | None = None


@dataclass
class MiniMaxM3SparseMSAPrefillMetadata(MiniMaxM3SparsePrefillMetadata):
    # The builder's Q8KV4 prefill split workspace; None off Rubin.
    workspace: "PrefillWorkspace | None" = None


class MiniMaxM3SparseMSAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Prepare MSA plans only for decode shapes supported by ``fmha_sm100``."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_q_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.topk_blocks = config.sparse_attention_config["sparse_topk_blocks"]
        # AttentionSpec stores every FP8 mode as uint8, so retain the configured
        # format to distinguish E4M3 (supported) from E5M2 before planning.
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.decode_backend = vllm_config.attention_config.minimax_m3_msa_decode_backend
        self.q8kv4_decode = uses_q8kv4_decode(
            kv_cache_dtype=self.kv_cache_dtype,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.msa_cutlass_plan_cache = MSACutlassDecodePlanCache()
        self.msa_cutlass_min_batch_size = _get_msa_cutlass_min_batch_size()
        # Rubin only: Q8KV4 prefill split partials striped over the memory
        # locality domains, reserved for the largest batch.
        self.prefill_workspace: "PrefillWorkspace | None" = None  # noqa: UP037
        if is_nvfp4_kv_cache(
            self.kv_cache_dtype
        ) and current_platform.is_device_capability(107):
            from vllm.third_party.fmha_sm100.prefill_q8kv4 import PrefillWorkspace

            self.prefill_workspace = PrefillWorkspace(
                device, block_scale_shift=NVFP4_BLOCK_SCALE_SHIFT
            )
            self.prefill_workspace.split_buffers(
                self.topk_blocks,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.num_q_heads,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        prefill = metadata.prefill
        if prefill is not None and self.prefill_workspace is not None:
            metadata.prefill = MiniMaxM3SparseMSAPrefillMetadata(
                **{f.name: getattr(prefill, f.name) for f in fields(prefill)},
                workspace=self.prefill_workspace,
            )
        decode = metadata.decode
        if decode is None:
            return metadata

        msa_cutlass = None
        if should_prepare_decode_metadata(
            metadata.num_decodes,
            decode.decode_query_len,
            decode_backend=self.decode_backend,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=SPARSE_BLOCK_SIZE,
            topk_blocks=self.topk_blocks,
            min_batch_size=self.msa_cutlass_min_batch_size,
        ):
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            msa_cutlass = prepare_decode_metadata(
                decode.block_table,
                decode.seq_lens,
                seq_lens_cpu[: metadata.num_decodes],
                decode.decode_query_len,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                page_size=SPARSE_BLOCK_SIZE,
                topk_blocks=self.topk_blocks,
                q8kv4=self.q8kv4_decode,
                plan_cache=self.msa_cutlass_plan_cache,
            )
        metadata.decode = MiniMaxM3SparseMSADecodeMetadata(
            seq_lens=decode.seq_lens,
            block_table=decode.block_table,
            decode_query_len=decode.decode_query_len,
            msa_cutlass=msa_cutlass,
        )
        return metadata


class MiniMaxM3SparseMSABaseImpl(MiniMaxM3SparseImpl):
    """What the MSA impls share: CUTLASS decode plans and the prefill CSR."""

    use_cutlass_decode: bool
    # The prefill kernels ``fmha_sm100.warmup`` prebuilds for this impl.
    prefill_backend: str

    def should_use_msa_decode(self, layer_name: str) -> bool:
        if not self.use_cutlass_decode:
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        main_md = attn_metadata[layer_name]
        if not isinstance(main_md, MiniMaxM3SparseMetadata):
            return False
        decode = main_md.decode
        return (
            isinstance(decode, MiniMaxM3SparseMSADecodeMetadata)
            and decode.msa_cutlass is not None
        )

    def _build_prefill_csr(
        self,
        topk: torch.Tensor,
        nd: int,
        num_tokens: int,
        p: MiniMaxM3SparsePrefillMetadata,
    ):
        from vllm.third_party.fmha_sm100.sparse import build_k2q_csr

        # [H, prefill, MK] transposed view; build_k2q_csr consumes the strided
        # view directly (topK stays innermost-contiguous).
        return build_k2q_csr(
            topk[nd:num_tokens].transpose(0, 1),
            p.cu_seqlens_q,
            p.cu_seqlens_k,
            SPARSE_BLOCK_SIZE,
            total_k=0,
            max_seqlen_k=p.max_seq_len,
            max_seqlen_q=p.max_query_len,
            total_rows=p.total_kv_blocks,
            qhead_per_kv=self.num_heads // self.num_kv_heads,
            return_schedule=True,
        )


class MiniMaxM3SparseMSAImpl(MiniMaxM3SparseMSABaseImpl):
    """MSA block-sparse attention over a BF16 or FP8 KV cache, with guarded
    CUTLASS sparse decode."""

    prefill_backend = "cute_dsl"

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "cutlass",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        self.use_cutlass_decode = supports_cutlass_sparse_decode(
            decode_backend=msa_decode_backend,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        )
        logger.info_once(
            "MiniMax M3 MSA sparse decode selected %s",
            "CUTLASS" if self.use_cutlass_decode else "Triton",
        )

    def _bf16_query(
        self, q: torch.Tensor, q_scale: float, dtype: torch.dtype
    ) -> torch.Tensor:
        # With CUTLASS decode the query is E4M3; the other kernels read BF16.
        if not self.use_cutlass_decode:
            return q
        return _dequantize_query(q, q_scale, dtype)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer [total_q, H, MK]; the
        # kernels want [H, tokens, MK], so slice tokens on dim 0 then transpose.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        q_scale_float = getattr(layer, "_q_scale_float", 1.0)
        if self.use_fp8_kv:
            kv_cache = kv_cache.view(self.kv_cache_fp8_dtype)
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        # Decode [:nd]: CUTLASS for planned shapes, otherwise Triton.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
            msa_metadata = d.msa_cutlass
            if self.use_cutlass_decode and msa_metadata is not None:
                msa_cutlass_sparse_decode(
                    q[:nd],
                    kv_cache,
                    topk[:nd],
                    out[:nd],
                    msa_metadata,
                    scale=self.scale,
                    q_scale_float=q_scale_float,
                    k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                    v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                )
            else:
                minimax_m3_sparse_attn_decode(
                    self._bf16_query(q[:nd], q_scale_float, output.dtype),
                    kv_cache,
                    topk[:nd].transpose(0, 1),
                    d.block_table,
                    d.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[:nd],
                    d.decode_query_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        # Prefill [nd:]: MSA sparse FMHA over the selected blocks.
        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.sparse import sparse_atten_func

            p = main_md.prefill
            assert p is not None
            k2q_row_ptr, k2q_q_indices, schedule = self._build_prefill_csr(
                topk, nd, num_tokens, p
            )
            # An E4M3 query attends as is, its scale folded into softmax_scale.
            qp = q[nd:]
            softmax_scale = (
                self.scale * q_scale_float if self.use_cutlass_decode else self.scale
            )
            k_cache, v_cache = kv_cache.split(self.head_size, dim=-1)
            sparse_atten_func(
                qp,
                k_cache,
                v_cache,
                k2q_row_ptr,
                k2q_q_indices,
                topK=self.topk_blocks,
                blk_kv=SPARSE_BLOCK_SIZE,
                causal=True,
                softmax_scale=softmax_scale,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                max_seqlen_q=p.max_query_len,
                max_seqlen_k=p.max_seq_len,
                page_table=p.block_table,
                seqused_k=p.seq_lens,
                schedule=schedule,
                out=out[nd:],
            )
        return output


class MiniMaxM3SparseMSANvfp4Impl(MiniMaxM3SparseMSABaseImpl):
    """MSA block-sparse attention over an NVFP4 KV cache, SM100 family only.

    Decode and prefill run the CUTLASS Q8KV4 kernels, which read the E4M3
    query; there is no BF16-query kernel and no Triton fallback.
    """

    use_cutlass_decode = True
    prefill_backend = "q8kv4"

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "nvfp4",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "cutlass",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise ValueError("MiniMax M3 NVFP4 KV cache requires an SM100-family GPU")
        # NVFP4 decodes only on CUTLASS, whatever msa_decode_backend says.
        if not supports_cutlass_sparse_decode(
            decode_backend="cutlass",
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        ):
            raise ValueError(
                "MiniMax M3 NVFP4 KV cache requires CUTLASS MSA sparse decode "
                f"(num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
                f"block_size={self.block_size}, topk_blocks={self.topk_blocks})"
            )
        blocker = _q8kv4_prefill_blocker(
            self.num_heads, self.num_kv_heads, self.block_size, self.topk_blocks
        )
        if blocker is not None:
            raise ValueError(
                "MiniMax M3 NVFP4 KV cache requires the CUTLASS Q8KV4 prefill, "
                f"which cannot serve this layout or device: {blocker}"
            )
        logger.info_once("MiniMax M3 NVFP4 MSA: CUTLASS Q8KV4 decode and prefill")
        # NVFP4_BLOCK_SCALE_SHIFT assumes unit global scales; checked on the
        # first forward, once the checkpoint's KV scales are loaded.
        self._check_global_scales = True

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert query.dtype == torch.float8_e4m3fn
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q_fp8 = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        q_scale_float = getattr(layer, "_q_scale_float", 1.0)
        k_scale = getattr(layer, "_k_scale", None)
        v_scale = getattr(layer, "_v_scale", None)
        if self._check_global_scales:
            k_global = getattr(layer, "_k_scale_float", 1.0)
            v_global = getattr(layer, "_v_scale_float", 1.0)
            if k_global != 1.0 or v_global != 1.0:
                raise ValueError(
                    "MiniMax M3 reads its NVFP4 KV cache on the Q8KV4 kernels with "
                    f"block_scale_shift={NVFP4_BLOCK_SCALE_SHIFT}, which needs unit "
                    f"global scales; got k_scale={k_global}, v_scale={v_global}"
                )
            self._check_global_scales = False

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
            msa_metadata = d.msa_cutlass
            if msa_metadata is None:
                raise RuntimeError(
                    "MiniMax M3 NVFP4 KV cache has no Triton decode fallback; "
                    f"CUTLASS decode was not planned for {main_md.num_decodes} "
                    f"requests with query length {d.decode_query_len}"
                )
            msa_cutlass_sparse_decode(
                q_fp8[:nd],
                kv_cache,
                topk[:nd],
                out[:nd],
                msa_metadata,
                scale=self.scale,
                q_scale_float=q_scale_float,
                k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                k_scale=k_scale,
                v_scale=v_scale,
            )

        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.prefill_q8kv4 import run_prefill

            p = main_md.prefill
            assert p is not None
            k2q_row_ptr, _, schedule = self._build_prefill_csr(topk, nd, num_tokens, p)
            k_data, k_sf, v_data, v_sf = nvfp4_kv_cache_views(kv_cache)
            run_prefill(
                q_fp8[nd:],
                (k_data, v_data),
                (k_sf, v_sf),
                kv_indices=p.block_table,
                kv_indptr=None,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                k2q_row_ptr=k2q_row_ptr,
                schedule=schedule,
                sm_scale=self.scale * q_scale_float,
                topk=self.topk_blocks,
                seqused_k=p.seq_lens,
                k_global_scale=k_scale,
                v_global_scale=v_scale,
                block_scale_shift=NVFP4_BLOCK_SCALE_SHIFT,
                out=out[nd:],
                workspace=(
                    p.workspace
                    if isinstance(p, MiniMaxM3SparseMSAPrefillMetadata)
                    else None
                ),
            )
        return output
