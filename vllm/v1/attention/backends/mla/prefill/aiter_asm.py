# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER ASM FP8 backend for MLA prefill on AMD gfx950 (MI350).

Dispatches through aiter.mla_prefill_ps_asm_fwd -> aiter.mla_reduce_v1.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions, MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )

logger = init_logger(__name__)

# Q-side tile size baked into the gfx950 mla_prefill_ps_asm_fwd kernel.
_FP8_PREFILL_TILE_Q = 256
# K-side tiling granularity required by the PS scheduler.
_KVLEN_GRANULARITY = 128


class AiterAsmPrefillBackend(MLAPrefillBackend):
    """FP8 MLA prefill backend built on AITER persistent-scheduling ASM on gfx950.

    Persistent metadata buffers are prepared once per forward and then re-used across
    layers.

    Requires FP8 Q/KV cache and gfx950.
    """

    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_mla_dimensions: ClassVar[list[MLADimensions]] = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
    ]
    requires_fp8_query_quantization = True

    # Optimizations.
    # Cache of the kv_indices arange buffer, reused per layer and chunk
    # Key: (device, length, dtype).
    _KV_INDICES_BUFFERS: dict[tuple, torch.Tensor] = {}
    # Cache of scalar 1.0 dequant scale passed as q/k/v_scale. Reused per layer/chunk.
    # Key: device.
    _ONE_SCALE_BUFFERS: dict[str, torch.Tensor] = {}

    @staticmethod
    def get_name() -> str:
        return "AITER_ASM"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return False
        return device_capability.major == 9 and device_capability.minor == 5

    @classmethod
    def is_available(cls) -> bool:
        # Requires:
        # 1. gfx950
        # 2. AITER with MLA PS kernels
        # 3. AITER with the aiter#3606 fix, which corrects the LSE bug required
        #    for correct non-causal attn and adds the `max_kvlen` kwarg to
        #    get_ps_metadata_info_v1. We use that kwarg as a proxy for whether
        #    the fix is available.
        import inspect

        try:
            from vllm.platforms.rocm import on_gfx950
        except Exception:  # noqa: BLE001
            return False
        if not on_gfx950():
            return False
        try:
            from aiter import (  # noqa: F401
                get_ps_metadata_info_v1,
                get_ps_metadata_v1,
                mla_prefill_ps_asm_fwd,
                mla_reduce_v1,
            )
        except Exception:  # noqa: BLE001
            return False

        try:
            params = inspect.signature(get_ps_metadata_info_v1).parameters
        except (ValueError, TypeError):
            return False
        return "max_kvlen" in params

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        if selector_config.cache_dtype not in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            invalid_reasons.append(
                f"cache_dtype {selector_config.cache_dtype!r} is unsupported "
                "(AITER_ASM requires plain per-tensor FP8: fp8, fp8_e4m3, or "
                "fp8_e5m2; per-token-head and nvfp4 variants are not supported)"
            )
        if selector_config.dcp_world_size > 1:
            # Decode context parallel does not support scaled/fp8 KV
            invalid_reasons.append(
                "decode context parallelism (DCP) is not supported with the "
                "FP8 KV cache required by AITER_ASM"
            )
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        from aiter import (
            get_ps_metadata_info_v1,
            get_ps_metadata_v1,
            mla_prefill_ps_asm_fwd,
            mla_reduce_v1,
        )

        self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
        self._mla_reduce_v1 = mla_reduce_v1
        self._get_ps_metadata_v1 = get_ps_metadata_v1
        self._get_ps_metadata_info_v1 = get_ps_metadata_info_v1

        # Persistent-scheduling buffers. Populated once per forward for new tokens
        # and every chunk, respectively. The PS kernels then use these.
        self._new_tokens_ps: dict | None = None
        self._context_ps: list[dict] = []

        # Used to size the shared KV indices buffer, which is at
        # most max_num_batched_tokens long
        self._max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

        # Need to reserve worst case workspace that accounts for chunked context since
        # the warmup only accounts for new tokens.
        self._reserve_workspace(max_num_seqs=vllm_config.scheduler_config.max_num_seqs)

    def _reserve_workspace(self, max_num_seqs: int) -> None:
        """Grow the shared scratch workspace to the prefill worst case."""
        from vllm.utils.math_utils import cdiv
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if (
            not is_workspace_manager_initialized()
            or not torch.accelerator.is_available()
        ):
            return

        # Realistic estimate of max number of partial tiles.
        # The reduce_partial_map_size from get_ps_metadata_info_v1 is a much looser
        # upper bound that reach TB scale at large context sizes, so it's unusable.
        # The PS scheduler can emit one partial tile per QO tile OR per CU. Where
        #  1. the QO tiles can be spread either over the max num batched tokens, or
        #     over all requests (max num seqs)
        #  2. the CU count is a property of gfx950.
        qo_tile_cnt = (
            cdiv(self._max_num_batched_tokens, _FP8_PREFILL_TILE_Q) + max_num_seqs - 1
        )
        from vllm.platforms import current_platform

        cu_num = current_platform.num_compute_units()
        assert cu_num == 256
        max_num_partial_tiles = qo_tile_cnt + cu_num

        max_partial_q = max_num_partial_tiles * _FP8_PREFILL_TILE_Q
        max_total_q = self._max_num_batched_tokens
        # logits, attn_lse, final_lse
        current_workspace_manager().get_simultaneous(
            ((max_partial_q, self.num_heads, self.v_head_dim), torch.float32),
            ((max_partial_q, self.num_heads), torch.float32),
            ((max_total_q, self.num_heads), torch.float32),
        )

    def _get_kv_indices_buf(self, device: torch.device, length: int) -> torch.Tensor:
        """Return a [0, 1, ..., length-1] int32 view into a shared arange buffer.

        This avoids an arange and memcopy per layer and chunk.
        """
        size = max(length, self._max_num_batched_tokens)
        key = (str(device), size)
        buf = type(self)._KV_INDICES_BUFFERS.get(key)
        if buf is None:
            buf = torch.arange(size, device=device, dtype=torch.int32)
            type(self)._KV_INDICES_BUFFERS[key] = buf
        return buf[:length]

    def _get_one_scale(self, device: torch.device) -> torch.Tensor:
        """Return the shared scalar 1.0 dequant scale for this device.

        Passed as q_scale/k_scale/v_scale to the FP8 ASM kernel, which only
        reads it. This removes a fill op from the critical path.
        """
        key = str(device)
        buf = type(self)._ONE_SCALE_BUFFERS.get(key)
        if buf is None:
            buf = torch.ones((), dtype=torch.float32, device=device)
            type(self)._ONE_SCALE_BUFFERS[key] = buf
        return buf

    def _build_ps_metadata_for_chunk(
        self,
        qo_indptr_cpu: torch.Tensor,
        kv_indptr_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        is_causal: bool,
        device: torch.device,
        max_qlen: int,
        max_kvlen: int | None,
    ) -> dict:
        """Build persistent-scheduling metadata buffers for this chunk.

        Instead of allocating a single worst-case buffer using the worst case size
        and re-using that in every chunk, we build correctly sized buffers for
        the new-token chunk and each context chunk. The buffers are only read by the
        kernel so it's safe to share them across layers.

        Note: this is an expensive call because get_ps_metadata_v1 involves several
        host-to-device syncs/copies. Same for num_partial_tiles. Hence we build
        it once per forward and re-use across layers.

        max_kvlen=None means causal: i.e. num K tokens == num Q tokens.
        """
        assert is_causal == (max_kvlen is None)
        num_head_k = self.num_heads

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = self._get_ps_metadata_info_v1(
            batch_size=qo_indptr_cpu.numel() - 1,
            num_head_k=num_head_k,
            max_qlen=max_qlen,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            max_kvlen=max_kvlen,
            kvlen_granularity=_KVLEN_GRANULARITY,
        )

        work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        work_info = torch.empty(*work_info_size, dtype=work_info_dtype, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        reduce_final_map = torch.empty(
            *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_dtype, device=device
        )

        # Prefill is non-absorbed
        gqa_ratio = 1
        self._get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_head_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=1,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            kvlen_granularity=_KVLEN_GRANULARITY,
            block_size=1,
            is_causal=is_causal,
        )

        # The actual number of partial tiles emitted by the scheduler.
        # Required for correctly sizing the (partial) logits/attn_lse buffers that we
        # reduce over.
        num_partial_tiles = int(reduce_indptr[-1].item())
        assert num_partial_tiles > 0

        return {
            "work_indptr": work_indptr,
            "work_info": work_info,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
            "num_partial_tiles": num_partial_tiles,
            "max_q_len": max_qlen,
        }

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)

        qo_indptr = prefill_metadata.query_start_loc  # device int32 [bs+1]
        device = qo_indptr.device

        # Use CPU buffers to avoid host-device sync
        qo_indptr_cpu = prefill_metadata.query_start_loc_cpu.to(torch.int32)
        q_seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)
        total_q = int(qo_indptr_cpu[-1].item())
        max_query_len = prefill_metadata.max_query_len

        # 1. Prep buffers for new-tokens chunk (causal)
        ps = self._build_ps_metadata_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=qo_indptr_cpu,
            seq_lens_cpu=q_seq_lens_cpu,
            is_causal=True,
            device=device,
            max_qlen=max_query_len,
            max_kvlen=None,
        )
        ps["qo_indptr"] = qo_indptr
        ps["kv_indptr"] = qo_indptr
        ps["kv_indices"] = self._get_kv_indices_buf(device, total_q)
        self._new_tokens_ps = ps

        # 2. Prep buffers for each context chunk (non-causal).
        self._context_ps = []
        cc = prefill_metadata.chunked_context
        if cc is not None:
            for chunk_idx in range(len(cc.seq_tot)):
                kv_indptr = cc.cu_seq_lens[chunk_idx]
                kv_indptr_cpu = cc.cu_seq_lens_cpu[chunk_idx].to(torch.int32)
                k_seq_lens_cpu = (kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).to(
                    torch.int32
                )
                total_k = int(kv_indptr_cpu[-1].item())
                chunk_ps = self._build_ps_metadata_for_chunk(
                    qo_indptr_cpu=qo_indptr_cpu,
                    kv_indptr_cpu=kv_indptr_cpu,
                    seq_lens_cpu=k_seq_lens_cpu,
                    is_causal=False,
                    device=device,
                    max_qlen=max_query_len,
                    max_kvlen=cc.max_seq_lens[chunk_idx],
                )
                chunk_ps["qo_indptr"] = qo_indptr
                chunk_ps["kv_indptr"] = kv_indptr
                chunk_ps["kv_indices"] = self._get_kv_indices_buf(device, total_k)
                self._context_ps.append(chunk_ps)

    def _run_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ps: dict,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the PS ASM kernel + reduce, returning `(out, lse)`.

        Output dtype matches `prefill_metadata.output_dtype` so the caller
        can feed it directly to `merge_attn_states` or copy it into the
        final `output` buffer.
        """
        from vllm.v1.worker.workspace import current_workspace_manager

        # mla_prefill_ps_asm_fwd requires contiguous V in (seq, head, v_head_dim).
        # cp_gather_cache produces V as a slice of the wider nope+rope buffer
        # so we need to copy.
        v = v.contiguous()

        num_partial_tiles = ps["num_partial_tiles"]
        total_q = q.shape[0]
        out_dtype = self._prefill_metadata.output_dtype
        assert out_dtype is not None

        # Partial/scratch buffers for the PS kernels.
        partial_q = num_partial_tiles * _FP8_PREFILL_TILE_Q
        logits, attn_lse, final_lse = current_workspace_manager().get_simultaneous(
            ((partial_q, self.num_heads, self.v_head_dim), torch.float32),
            ((partial_q, self.num_heads), torch.float32),
            ((total_q, self.num_heads), torch.float32),
        )
        out = torch.empty(
            (total_q, self.num_heads, self.v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )

        # Q/K/V are cast to FP8 with no additional rescaling for now, which relies on
        # activations staying within the e4m3 range. Since gfx950 uses e4m3fn, larger
        # values clamp to 488 instead of producing inf. Accuracy has been validated on
        # DeepSeek-V3 on GSM8k with no regressions.
        one_scale = self._get_one_scale(q.device)
        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            ps["qo_indptr"],
            ps["kv_indptr"],
            ps["kv_indices"],
            ps["work_indptr"],
            ps["work_info"],
            ps["max_q_len"],
            self.scale,
            is_causal,
            logits,
            attn_lse,
            out,
            one_scale,
            one_scale,
            one_scale,
        )
        self._mla_reduce_v1(
            logits,
            attn_lse,
            ps["reduce_indptr"],
            ps["reduce_final_map"],
            ps["reduce_partial_map"],
            _FP8_PREFILL_TILE_Q,
            0,
            out,
            final_lse,
        )

        # mla_reduce_v1 writes final_lse as (total_q, num_heads), but
        # triton_merge_attn_states wants it as (num_heads, total_q)
        return out, final_lse.transpose(0, 1).contiguous()

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._new_tokens_ps is not None, (
            "prepare_metadata must be called before run_prefill_new_tokens"
        )
        assert out is None and output_scale is None, (
            "fused/in-place FP8 output not supported by the AITER ASM "
            "MLA prefill backend"
        )
        out, lse = self._run_kernel(q, k, v, self._new_tokens_ps, is_causal=True)
        if return_softmax_lse:
            return out, lse
        return out

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= chunk_idx < len(self._context_ps), (
            f"context chunk {chunk_idx} requested but prepare_metadata built "
            f"{len(self._context_ps)} chunk(s). Call prepare_metadata first."
        )
        ps = self._context_ps[chunk_idx]
        return self._run_kernel(q, k, v, ps, is_causal=False)
