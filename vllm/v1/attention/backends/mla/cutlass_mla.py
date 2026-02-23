# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import ClassVar

import torch

import vllm._custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


class CutlassMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # enable full CUDA Graph support for decode-only capture
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


class CutlassMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MLA"

    @staticmethod
    def get_impl_cls() -> type["CutlassMLAImpl"]:
        return CutlassMLAImpl

    @staticmethod
    def get_builder_cls() -> type["CutlassMLAMetadataBuilder"]:
        return CutlassMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10


class SM100Workspace:
    def __init__(self, initial_workspace_size):
        self._workspace_buf = torch.empty(
            initial_workspace_size, device="cuda", dtype=torch.uint8
        )

        self._block_size = 128  # Forced to 128

        # Pre-compute sm_count to avoid recomputing it. Use device 0 as a proxy
        # (assumes all devices are similar)
        properties = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self._sm_count = properties.multi_processor_count

    def get_buf(self):
        return self._workspace_buf

    def ensure_size(self, attn_metadata: MLACommonMetadata, num_kv_splits: int):
        batch_size = attn_metadata.num_reqs
        max_seq_len = attn_metadata.max_query_len

        workspace_size = ops.sm100_cutlass_mla_get_workspace_size(
            max_seq_len * self._block_size,
            batch_size,
            self._sm_count,
            num_kv_splits=num_kv_splits,
        )

        if self._workspace_buf.shape[0] < workspace_size:
            self._workspace_buf.resize_(workspace_size)


g_sm100_workspace = SM100Workspace(128 * 1024 * 1024)  # 128MB

MAX_HEADS = 128


class CutlassMLAImpl(MLACommonImpl[MLACommonMetadata]):
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
        # Remove q_pad_num_heads from mla_args if present - we always use MAX_HEADS
        mla_args.pop("q_pad_num_heads", None)
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
            q_pad_num_heads=MAX_HEADS,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "CutlassMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "CutlassMLAImpl"
            )

        # TODO: Currently, num_kv_splits is limited to 16 to avoid hanging
        #       issues. In case the code hangs, use:
        #       FORCE_NUM_KV_SPLITS=1
        force_num_kv_splits = os.environ.get("FORCE_NUM_KV_SPLITS", None)
        if force_num_kv_splits:
            logger.debug_once("Forcing num_kv_splits to %d", int(force_num_kv_splits))
            self._num_kv_splits = int(force_num_kv_splits)
        else:
            self._num_kv_splits = -1  # => Auto-detect

        # Share workspace buffer across all executions
        self._workspace = g_sm100_workspace

    def _sm100_cutlass_mla_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        workspace: torch.Tensor,
        sm_scale: float,
        num_kv_splits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
        assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
        assert kv_c_and_k_pe_cache.ndim == 3, (
            "kv_c_and_k_pe_cache must be a 3D tensor, but got {}".format(
                kv_c_and_k_pe_cache.ndim
            )
        )

        B_q, H, D_q_nope = q_nope.shape
        B_q_2, H_2, D_q_pe = q_pe.shape
        assert (B_q == B_q_2) and (H == H_2)

        _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

        D_latent = 512
        D_rope = 64
        assert D_q_nope == D_latent
        assert D_q_pe == D_rope
        assert D_ckv == D_latent + D_rope

        MAX_HEADS = 128
        assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"

        assert len(page_table.shape) == 2
        B_block_table, block_num = page_table.shape
        assert B_block_table == B_q
        assert block_num > 0, f"block num must be greater than 0, got {block_num}"
        assert block_num % (128 / PAGE_SIZE) == 0

        assert q_nope.dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn), (
            f"q_nope.dtype needs to be fp16 or bf16 or e4m3 but got {q_nope.dtype}."
        )
        assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
        assert seq_lens.dtype == torch.int32, (
            f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
        )
        assert page_table.dtype == torch.int32, (
            f"page_table.dtype needs to be int32 but got {page_table.dtype}."
        )

        dtype = (
            torch.bfloat16
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else q_nope.dtype
        )
        out = q_nope.new_empty((B_q, MAX_HEADS, D_latent), dtype=dtype)
        lse = (
            torch.empty((B_q, MAX_HEADS), dtype=torch.float32, device=q_nope.device)
            if self.need_to_return_lse_for_decode
            else torch.Tensor()
        )

        ops.sm100_cutlass_mla_decode(
            out,
            lse,
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            seq_lens,
            page_table,
            workspace,
            sm_scale,
            num_kv_splits,
        )

        if H < MAX_HEADS:
            # Extract the subsets of the outputs
            lse = lse[:, :H] if self.need_to_return_lse_for_decode else lse
            out = out[:, :H]

        return out, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        # Adjust workspace size (if necessary)
        self._workspace.ensure_size(attn_metadata, self._num_kv_splits)

        # Run MLA
        o, lse = self._sm100_cutlass_mla_decode(
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            attn_metadata.decode.seq_lens,
            attn_metadata.decode.block_table,
            self._workspace.get_buf(),
            self.scale,
            self._num_kv_splits,
        )

        return o, (lse if self.need_to_return_lse_for_decode else None)
