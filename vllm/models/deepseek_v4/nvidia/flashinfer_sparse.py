# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


def _as_sparse_sm120_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dtype == torch.float8_e4m3fn:
        kv_cache = kv_cache.view(torch.uint8)
    if kv_cache.dim() == 4:
        return kv_cache
    return kv_cache.unsqueeze(-2)


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4FlashMLABackend):
    """FlashInfer backend using the DSv4 sparse metadata/cache layout."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

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
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major != 12:
            return "FLASHINFER_MLA_SPARSE_DSV4 requires SM12x"
        if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):
            return "kv_cache_dtype not supported"
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        return None


class _DeepseekV4FlashInferMLAAttentionBase(DeepseekV4Attention):
    """Shared DeepSeek V4 FlashInfer sparse MLA attention helpers."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        # Per-tensor FP8 cache path scales.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # FlashInfer expects scalar scale arguments for this path.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        pass

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        raise NotImplementedError

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Output may be padded to backend-supported head counts.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _reserve_sm120_decode_workspace(self) -> None:
        _get_flashinfer_dsv4_workspace(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def _prepare_sm120_query(
        self, q: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def _forward_sm120_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_sm120_query(q, output)
        swa_cache = _as_sparse_sm120_cache(self.swa_cache_layer.kv_cache)
        extra_cache = _as_sparse_sm120_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache,
            workspace_buffer=_get_flashinfer_dsv4_workspace(q.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            backend="auto",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    def _forward_sm120_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_sm120_query(q, output)
        swa_kv_paged = _as_sparse_sm120_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = _as_sparse_sm120_cache(compressed_k_cache)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            q_chunk = q[query_start:query_end]
            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=_get_flashinfer_dsv4_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                backend="auto",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )


class DeepseekV4FlashInferSM120Attention(_DeepseekV4FlashInferMLAAttentionBase):
    """DeepSeek V4 sparse MLA attention through FlashInfer."""

    use_fp8_ds_mla_layout: ClassVar[bool] = True

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
            "(SM120 kernel requires h_q in {16, 32, 64, 128})."
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )

    def _reserve_empty_forward_workspace(self) -> None:
        self._reserve_sm120_decode_workspace()

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_sm120_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_sm120_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )
