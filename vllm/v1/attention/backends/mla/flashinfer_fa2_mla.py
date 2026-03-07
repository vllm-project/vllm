# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer FA2 MLA Backend.

Uses the FlashInfer BatchMLAPagedAttentionWrapper with the FA2 backend for
bf16 MLA decode. This backend requires plan()/run() API with CSR-format
page indices and supports returning LSE for DCP.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from flashinfer import BatchMLAPagedAttentionWrapper

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
)
from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_per_layer_parameters,
    infer_global_hyperparameters,
)

logger = init_logger(__name__)

FLASHINFER_FA2_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferFA2MLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_FA2_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferFA2MLAImpl"]:
        return FlashInferFA2MLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferFA2MLAMetadataBuilder"]:
        return FlashInferFA2MLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Supported on SM 8.x+ except SM 10.x (which has dedicated backends).
        return capability.major != 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim != 128:
                return (
                    f"FlashInfer FA2 MLA kernel requires "
                    f"qk_nope_head_dim == 128, but got {qk_nope_head_dim}"
                )
        return None

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return None


@dataclass
class FlashInferFA2MLADecodeMetadata(MLACommonDecodeMetadata):
    wrapper: BatchMLAPagedAttentionWrapper | None = None


class FlashInferFA2MLAMetadataBuilder(
    MLACommonMetadataBuilder[MLACommonMetadata[FlashInferFA2MLADecodeMetadata]],
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)

        self._mla_dims = get_mla_dims(self.model_config)

        max_num_pages = (
            self.vllm_config.cache_config.num_gpu_blocks
            if self.vllm_config.cache_config.num_gpu_blocks
            else self.vllm_config.scheduler_config.max_num_seqs * 1024
        )

        self._workspace_buffer = torch.zeros(
            FLASHINFER_FA2_MLA_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=self.device,
        )
        # Max kv_indices buffer shared by all wrappers (kv_indices supports
        # variable-length copy in plan(), so one buffer is fine).
        self._kv_indices_buf = torch.zeros(
            max_num_pages, dtype=torch.int32, device=self.device
        )

        # Per-batch-size wrappers for CUDA graph compatibility.
        # Each wrapper has pre-allocated buffers sized exactly for its
        # batch size so plan()'s in-place copy works correctly.
        self._wrappers: dict[int, BatchMLAPagedAttentionWrapper] = {}

        # Pre-compute constant plan() parameters.
        self._num_heads = self.num_heads * self.dcp_world_size
        # Derive sm_scale from the model's attention layers so it includes
        # any model-specific corrections (e.g. YaRN mscale for DeepSeek).
        global_params = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl)  # type: ignore[type-abstract]
        )
        self._sm_scale = global_params.sm_scale

    def _get_wrapper(self, batch_size: int) -> BatchMLAPagedAttentionWrapper:
        wrapper = self._wrappers.get(batch_size)
        if wrapper is None:
            wrapper = BatchMLAPagedAttentionWrapper(
                self._workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indptr=torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indices=self._kv_indices_buf,
                kv_len_arr=torch.zeros(
                    batch_size, dtype=torch.int32, device=self.device
                ),
                backend="auto",
            )
            self._wrappers[batch_size] = wrapper
        return wrapper

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashInferFA2MLADecodeMetadata:
        num_decodes = seq_lens_device.shape[0]

        # Compute number of pages per request
        num_blocks = (
            seq_lens_device.to(torch.int32) + self.page_size - 1
        ) // self.page_size

        # Build CSR-format kv_indptr
        kv_indptr = torch.empty(num_decodes + 1, dtype=torch.int32, device=self.device)
        kv_indptr[0] = 0
        torch.cumsum(num_blocks, dim=0, out=kv_indptr[1:])

        # Flatten block_table into kv_indices via Triton kernel.
        # The full pre-allocated buffer is passed rather than slicing to
        # total_pages, since kv_indptr delimits the valid range and plan()
        # copies only the entries within that range.
        _copy_page_indices_kernel[(num_decodes,)](
            self._kv_indices_buf,
            block_table_tensor,
            block_table_tensor.stride(0),
            kv_indptr,
            BLOCK_SIZE=1024,
        )

        kv_lens = seq_lens_device.to(torch.int32)

        # Build qo_indptr
        tokens_per_req = num_decode_tokens // num_decodes
        qo_indptr = torch.arange(
            0,
            num_decodes * tokens_per_req + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=self.device,
        )

        # Get or create wrapper for this batch size.
        wrapper = self._get_wrapper(num_decodes)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            self._kv_indices_buf,
            kv_lens,
            self._num_heads,
            self._mla_dims.kv_lora_rank,
            self._mla_dims.qk_rope_head_dim,
            self.page_size,
            False,  # causal=False for decode
            self._sm_scale,
            self.model_config.dtype,
            self.model_config.dtype,
        )

        return FlashInferFA2MLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            wrapper=wrapper,
        )


class FlashInferFA2MLAImpl(
    MLACommonImpl[MLACommonMetadata[FlashInferFA2MLADecodeMetadata]]
):
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
                "FlashInferFA2MLAImpl does not support one of the "
                "following: alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferFA2MLAImpl"
            )

        # FA2 uses bf16 queries; no query quantization needed
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata[FlashInferFA2MLADecodeMetadata],
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.wrapper is not None

        # Verify the wrapper's sm_scale (set during plan()) matches the
        # model layer's scale. These must agree for correct attention output.
        assert attn_metadata.decode.wrapper._sm_scale == self.scale, (
            f"FlashInfer FA2 MLA wrapper sm_scale "
            f"({attn_metadata.decode.wrapper._sm_scale}) does not match "
            f"model scale ({self.scale})."
        )

        # Split query into nope and rope components
        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope = q[..., : self.kv_lora_rank]
            q_pe = q[..., self.kv_lora_rank :]

        # Split cache into compressed KV and rope key components
        ckv_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        kpe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

        output, lse = attn_metadata.decode.wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            return_lse=True,
            # Return LSE in base-e to match the DCP output-merging kernel.
            return_lse_base_on_e=True,
        )

        return output, lse
