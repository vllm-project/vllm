# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# num_kv_splits selection (shared by forward_mqa and the workspace reservation
# so the two cannot drift). Both are hardware dependent.
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2
# Bounds fp32 attn_logits scratch and gathered block tables for routed prefill;
# rows are independent, so this chunk size cannot change results.
_BATCH_INVARIANT_PREFILL_ROWS_PER_LAUNCH = 256


def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    # Power of 2 to avoid excessive kernel instantiations, capped by an SM-based
    # maximum (occupancy multiplier allows multiple blocks per SM
    # for latency hiding).
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)


@dataclass
class TritonMLAMetadata(MLACommonMetadata):
    # Per-request metadata for all requests in the step; decodes first.
    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # forward_mqa flattens a uniform multi-token block to one decode row per
    # query token, so causal and non-causal blocks both take the decode path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=TritonMLAMetadata,
        )
        # DCP local sequence lengths are not advanced between draft steps.
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
        self._reserve_attn_logits_workspace()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MLACommonMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        assert isinstance(metadata, TritonMLAMetadata)
        metadata.block_table = common_attn_metadata.block_table_tensor
        metadata.seq_lens = common_attn_metadata.seq_lens
        return metadata

    def update_draft_decode_metadata(self, _metadata: MLACommonMetadata) -> None:
        pass

    def _reserve_attn_logits_workspace(self) -> None:
        """Pre-size the shared workspace for the decode split-KV attn logits.

        Reserving at the worst case (max_model_len -> max num_kv_splits,
        max_num_seqs decode tokens) before warmup/cudagraph capture means the
        per-call ``get_simultaneous`` in ``forward_mqa`` never has to grow the
        buffer at runtime (which would raise once the workspace is locked).
        """
        if not is_workspace_manager_initialized():
            return
        # forward_mqa flattens each request's block to query_len decode rows,
        # and query_len is bounded by the reorder threshold.
        B = (
            self.vllm_config.scheduler_config.max_num_seqs
            * self.reorder_batch_threshold
        )
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        max_splits = _compute_num_kv_splits(
            self.model_config.max_model_len,
            current_platform.num_compute_units(),
        )
        lse_dim = self.mla_dims.kv_lora_rank + 1
        current_workspace_manager().get_simultaneous(
            ((B, q_num_heads, max_splits, lse_dim), torch.float32),
        )


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        # DSpark non-causal blocks are flattened to single-token decode rows in
        # TritonMLAImpl.forward_mqa (decode_attention_fwd has no causal flag /
        # no intra-block masking). Enables the non-causal AMD MLA path.
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    supports_batch_invariant_mqa_prefill: bool = True

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
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton MLA backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 "
                    f"(fp8e4nv) requires SM89+. Re-run with "
                    f"--kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported by the Triton MLA "
                    f"backend on {dev} (compute capability {cap_str}); "
                    f"bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        if (
            envs.VLLM_BATCH_INVARIANT
            and attn_metadata.num_prefills > 0
            and q.shape[0] > attn_metadata.num_decode_tokens
        ):
            # MLAAttention routed the prefill rows here as well.
            return self._forward_mqa_all_rows(
                q, kv_c_and_k_pe_cache, attn_metadata, layer
            )

        assert attn_metadata.decode is not None

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = _compute_num_kv_splits(
                attn_metadata.max_seq_len, self._sm_count
            )

        # NOTE: the +1 stores the LogSumExp (LSE) that the stage2 kernel uses to
        # merge partial attention outputs across splits. The scratch is served
        # from the shared workspace (reserved at max in the metadata builder), so
        # there is no per-call allocation on the decode hot path. Fall back to a
        # direct allocation when the workspace manager is not initialized (e.g.
        # unit tests without a GPUModelRunner).
        logits_shape = (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1)
        if is_workspace_manager_initialized():
            (attn_logits,) = current_workspace_manager().get_simultaneous(
                (logits_shape, torch.float32),
            )
        else:
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=q.device
            )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        # decode_attention_fwd has no causal flag: it launches one program per
        # row of q and reads that row's KV extent from seq_lens, so intra-block
        # causality is expressed as per-row extents. Deriving query_len from the
        # tensors the kernel indexes keeps the three row counts in step.
        num_decodes = seq_lens.shape[0]
        query_len, remainder = divmod(B, num_decodes) if num_decodes else (1, 0)
        assert remainder == 0, (
            f"non-uniform decode block: {B} query rows over {num_decodes} requests"
        )
        if query_len > 1:
            block_table = block_table.repeat_interleave(query_len, dim=0)
            if attn_metadata.causal:
                # Per-row extents are offsets off the global sequence length;
                # under DCP seq_lens is this rank's local slice instead.
                assert self.dcp_world_size == 1, (
                    "causal multi-token decode is not supported with DCP"
                )
                # Row t attends the prefix plus block tokens 0..t. Clamp holds
                # padding rows at extent 0; the kernel skips them either way
                # (split_kv_end == split_kv_start).
                offsets = torch.arange(
                    1 - query_len, 1, device=seq_lens.device, dtype=seq_lens.dtype
                )
                seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
            else:
                # Non-causal draft block: every row sees the same prefix.
                seq_lens = seq_lens.repeat_interleave(query_len)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse

    def _forward_mqa_all_rows(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run all rows as independent decode rows for batch-invariant prefill."""
        assert isinstance(attn_metadata, TritonMLAMetadata)
        assert attn_metadata.block_table is not None
        assert attn_metadata.seq_lens is not None

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        query_start_loc = attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        row_to_req = torch.repeat_interleave(
            torch.arange(query_lens.shape[0], device=q.device),
            query_lens,
            output_size=B,
        )
        row_seq_lens = (
            attn_metadata.seq_lens[row_to_req]
            - query_start_loc[1:][row_to_req]
            + torch.arange(B, device=q.device, dtype=query_start_loc.dtype)
            + 1
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        rows_per_launch = _BATCH_INVARIANT_PREFILL_ROWS_PER_LAUNCH
        attn_logits = torch.empty(
            (
                min(B, rows_per_launch),
                q_num_heads,
                1,
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        for start in range(0, B, rows_per_launch):
            end = min(start + rows_per_launch, B)
            row_count = end - start
            block_table = attn_metadata.block_table[row_to_req[start:end]]

            decode_attention_fwd(
                q[start:end],
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o[start:end],
                lse[start:end],
                block_table,
                row_seq_lens[start:end],
                attn_logits[:row_count],
                1,
                self.scale,
                PAGE_SIZE,
                k_scale=layer._k_scale,
                v_scale=layer._k_scale,
                is_mla=True,
            )

        return o, lse
