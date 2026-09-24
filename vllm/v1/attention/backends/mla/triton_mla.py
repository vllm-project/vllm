# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    # Power of 2 to avoid excessive kernel instantiations, capped by an SM-based
    # maximum (occupancy multiplier allows multiple blocks per SM
    # for latency hiding).
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)


def reserve_triton_mla_decode_workspace(
    max_rows: int,
    num_heads: int,
    max_seq_len: int,
    kv_lora_rank: int,
) -> None:
    """Reserve split-KV scratch before warmup locks the workspace manager."""
    if not is_workspace_manager_initialized():
        return
    max_splits = _compute_num_kv_splits(
        max_seq_len, current_platform.num_compute_units()
    )
    current_workspace_manager().get_simultaneous(
        ((max_rows, num_heads, max_splits, kv_lora_rank + 1), torch.float32),
    )


def triton_mla_decode_forward(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    scale: float,
    kv_lora_rank: int,
    k_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the generic split-KV MLA decode over independently bounded rows.

    A caller may flatten a multi-token block into one row per query token and
    express causality entirely through ``seq_lens``. This is also useful for
    DCP: each rank computes a local output/LSE and the common MLA layer performs
    the cross-rank LSE merge.
    """
    num_rows, num_heads = q.shape[:2]
    output = torch.zeros(
        num_rows, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device
    )
    lse = torch.empty(num_rows, num_heads, dtype=q.dtype, device=q.device)
    num_kv_splits = (
        1
        if envs.VLLM_BATCH_INVARIANT
        else _compute_num_kv_splits(max_seq_len, current_platform.num_compute_units())
    )
    logits_shape = (
        num_rows,
        num_heads,
        num_kv_splits,
        kv_lora_rank + 1,
    )
    if is_workspace_manager_initialized():
        (attn_logits,) = current_workspace_manager().get_simultaneous(
            (logits_shape, torch.float32),
        )
    else:
        attn_logits = torch.empty(logits_shape, dtype=torch.float32, device=q.device)

    paged_kv = kv_cache.unsqueeze(2)
    decode_attention_fwd(
        q,
        paged_kv,
        paged_kv[..., :kv_lora_rank],
        output,
        lse,
        block_table,
        seq_lens,
        attn_logits,
        num_kv_splits,
        scale,
        paged_kv.size(1),
        k_scale=k_scale,
        v_scale=k_scale,
        is_mla=True,
    )

    # Empty local shards are the identity element of the cross-rank LSE merge.
    # Mask after the kernel as well as initializing output to zero: the stage-2
    # reducer has no useful value to produce for a sequence of length zero.
    empty = seq_lens == 0
    output.masked_fill_(empty[:, None, None], 0)
    lse.masked_fill_(empty[:, None], float("-inf"))
    return output, lse


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # forward_mqa flattens a uniform multi-token block to one decode row per
    # query token, so causal and non-causal blocks both take the decode path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # DCP local sequence lengths are not advanced between draft steps.
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
        self._reserve_attn_logits_workspace()

    def update_draft_decode_metadata(self, _metadata: MLACommonMetadata) -> None:
        pass

    def _reserve_attn_logits_workspace(self) -> None:
        """Pre-size the shared workspace for the decode split-KV attn logits.

        Reserving at the worst case (max_model_len -> max num_kv_splits,
        max_num_seqs decode tokens) before warmup/cudagraph capture means the
        per-call ``get_simultaneous`` in ``forward_mqa`` never has to grow the
        buffer at runtime (which would raise once the workspace is locked).
        """
        # forward_mqa flattens each request's block to query_len decode rows,
        # and query_len is bounded by the reorder threshold.
        B = (
            self.vllm_config.scheduler_config.max_num_seqs
            * self.reorder_batch_threshold
        )
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        reserve_triton_mla_decode_workspace(
            B,
            q_num_heads,
            self.model_config.max_model_len,
            self.mla_dims.kv_lora_rank,
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
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
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
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]

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

        return triton_mla_decode_forward(
            q,
            kv_c_and_k_pe_cache,
            block_table,
            seq_lens,
            attn_metadata.max_seq_len,
            self.scale,
            self.kv_lora_rank,
            layer._k_scale,
        )
