# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd,
    stage1_head_tiles,
    stage1_workgroups_per_cu,
)

from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
)

# num_kv_splits selection (shared by forward_mqa and the workspace reservation
# so the two cannot drift). Both are hardware dependent.
_MIN_WORK_PER_SPLIT = 512


def _compute_num_kv_splits(
    max_seq_len: int,
    sm_count: int,
    grid_units: int = 1,
    workgroups_per_cu: int = 2,
) -> int:
    # Power of 2 to avoid excessive kernel instantiations.
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * workgroups_per_cu
    # Splitting past what it takes to fill the device buys no parallelism and
    # costs a proportionally longer stage-2 reduction. grid_units must be the
    # product of the stage-1 grid's non-split dimensions -- batch rows times
    # head tiles -- which after the non-causal fold is neither the query-token
    # count nor the row count alone. Load-bearing under full cudagraphs, where
    # max_seq_len is the capture-time bound (max_model_len) rather than the
    # live one.
    occupancy_splits = triton.next_power_of_2(
        triton.cdiv(max_splits, max(1, grid_units))
    )
    return min(ideal_splits, max_splits, occupancy_splits)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    # forward_mqa folds a non-causal DSpark block into the query-head dim, so
    # no intra-block causal masking is required.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Only the non-causal DSpark draft group serves multi-token blocks via
        # the decode path; raise its reorder threshold to the spec block length
        # so full-cudagraph capture admits it. Causal usage stays single-token.
        if getattr(self, "non_causal_multi_token_decode", False):
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self._reserve_attn_logits_workspace()

    def _reserve_attn_logits_workspace(self) -> None:
        """Pre-size the shared workspace for the decode split-KV attn logits.

        Reserving at the worst case (max_model_len -> max num_kv_splits,
        max_num_seqs decode tokens) before warmup/cudagraph capture means the
        per-call ``get_simultaneous`` in ``forward_mqa`` never has to grow the
        buffer at runtime (which would raise once the workspace is locked).
        """
        if not is_workspace_manager_initialized():
            return
        # Decode reorder threshold is 1, so decode tokens <= max_num_seqs.
        B = self.vllm_config.scheduler_config.max_num_seqs
        # o/lse/attn_logits are still allocated per query token even though the
        # kernel sees them folded, so cover max_num_seqs * block_len rows.
        if getattr(self, "non_causal_multi_token_decode", False):
            B *= self.reorder_batch_threshold
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        # Defaults deliberately: grid_units=1 and two workgroups per CU give
        # the largest split count any launch can ask for, so the reservation
        # bounds every runtime shape.
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
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

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
        # DSpark non-causal blocks are folded into the query-head dim in
        # TritonMLAImpl.forward_mqa (decode_attention_fwd has no causal flag /
        # no intra-block masking). Enables the non-causal AMD MLA path.
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
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

        self._out_dtype = get_current_vllm_config().model_config.dtype
        self._sm_count = current_platform.num_compute_units()

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
        q_num_heads = q.shape[1]
        # The layer hands us a quantized query when the cache is quantized; its
        # scale multiplies the scores alongside the key's, and the output stays
        # in the model dtype.
        is_fp8_q = q.dtype in _FP8_DTYPES
        q_scale = layer._q_scale if is_fp8_q else None
        out_dtype = self._out_dtype if is_fp8_q else q.dtype
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=out_dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=out_dtype, device=q.device)

        # Non-causal DSpark block: every query token attends to the same
        # committed KV prefix and never to a sibling block token. Hand the block
        # to the kernel as extra query heads rather than as extra decode rows:
        # the two are the same problem, but the kernel rereads the whole KV span
        # once per head tile, so the row form multiplies KV traffic by the block
        # length.
        query_len = attn_metadata.max_query_len
        folded_rows = 0
        if not attn_metadata.causal and query_len > 1:
            folded_rows = attn_metadata.num_decodes * query_len

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # Size the splits against the stage-1 grid the launch will
            # actually build: the fold trades decode rows for query heads, so
            # rows alone understate the parallelism by the head-tile count.
            if folded_rows:
                rows, heads = folded_rows // query_len, q_num_heads * query_len
            else:
                rows, heads = B, q_num_heads
            # MLA carries a single KV head, so kv_group_num == heads.
            num_kv_splits = _compute_num_kv_splits(
                attn_metadata.max_seq_len,
                self._sm_count,
                rows * stage1_head_tiles(heads, heads, is_mla=True),
                stage1_workgroups_per_cu(True, heads),
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
        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        if folded_rows:
            # q rows are token-major within a request, so the fold is a view.
            # Rows past the real queries are cudagraph padding: leave them zero.
            n = folded_rows // query_len
            h = q_num_heads * query_len
            decode_attention_fwd(
                q[:folded_rows].view(n, h, -1),
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o[:folded_rows].view(n, h, -1),
                lse[:folded_rows].view(n, h),
                block_table[:n],
                seq_lens[:n],
                attn_logits[:folded_rows].view(n, h, num_kv_splits, -1),
                num_kv_splits,
                self.scale,
                PAGE_SIZE,
                k_scale=layer._k_scale,
                v_scale=layer._k_scale,
                q_scale=q_scale,
                is_mla=True,
            )
            return o, lse

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
            q_scale=q_scale,
            is_mla=True,
        )

        return o, lse
