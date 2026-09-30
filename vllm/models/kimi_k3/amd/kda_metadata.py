# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
``GDNAttentionMetadataBuilder``. Only the FLA chunk metadata is built
differently on device rather than on the host.
"""

from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

_BLOCK_T = 256
_MIN_BLOCK_N = 128


@triton.jit(do_not_specialize=["N"])
def _chunk_metadata_kernel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    N,
    BT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i_n = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    is_seq = offs_n < N
    bos = tl.load(cu_seqlens + offs_n, mask=is_seq, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offs_n + 1, mask=is_seq, other=0).to(tl.int32)
    nt = tl.where(is_seq, tl.cdiv(eos - bos, BT), 0)

    base = tl.sum(tl.where(offs_n < i_n, nt, 0))
    num_chunks = tl.sum(tl.where(offs_n == i_n, nt, 0))

    tl.store(chunk_offsets + i_n, base)
    if i_n == 0:
        tl.store(chunk_offsets + N, tl.sum(nt))

    for t0 in range(0, num_chunks, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_chunks
        row = (base + offs_t) * 2
        tl.store(chunk_indices + row, tl.full([BLOCK_T], i_n, tl.int32), mask=mask_t)
        tl.store(chunk_indices + row + 1, offs_t.to(tl.int32), mask=mask_t)


def prepare_chunk_metadata_device(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FLA chunk metadata on device, with no host<->device transfer."""
    num_seqs = cu_seqlens_cpu.numel() - 1
    seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    num_chunks = int(((seq_lens + chunk_size - 1) // chunk_size).sum())

    chunk_indices = torch.empty(
        num_chunks, 2, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    chunk_offsets = torch.empty(
        num_seqs + 1, dtype=torch.int64, device=cu_seqlens.device
    )
    _chunk_metadata_kernel[(num_seqs,)](
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_seqs,
        BT=chunk_size,
        BLOCK_N=max(_MIN_BLOCK_N, next_power_of_2(num_seqs + 1)),
        BLOCK_T=_BLOCK_T,
        num_warps=4,
    )
    return chunk_indices, chunk_offsets


def _adaptive_verification_enabled(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    return speculative_config is not None and bool(
        getattr(speculative_config, "enable_adaptive_verification", False)
    )


class KimiK3ROCmKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["KimiK3ROCmKDAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # The spec path reads per-request offsets and accepted counts off device
        # within a fixed num_spec + 1 window (spec_state_indices_tensor, conv
        # max_query_len), so a graph captured at that width replays any 1..k+1
        # mix. Keep UNIFORM_BATCH rather than claiming ALWAYS: FULL capture here
        # is decode-only (build_for_cudagraph_capture asserts it).
        if not _adaptive_verification_enabled(vllm_config):
            return None
        if (
            cls.get_cudagraph_support(vllm_config, kv_cache_spec)
            != AttentionCGSupport.UNIFORM_BATCH
        ):
            return None
        return max_decode_query_len(vllm_config)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        if (
            metadata.uniform_spec_sequence_length is not None
            and _adaptive_verification_enabled(self.vllm_config)
        ):
            # Adaptive verification trims drafts on device after the CPU split
            # was made, so equal CPU lengths do not mean equal device lengths.
            # The fixed-length recurrent kernel places sequence i at i * L and
            # never reads cu_seqlens; force the cu_seqlens-driven kernel.
            metadata = replace(metadata, uniform_spec_sequence_length=None)
        return metadata

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_chunk_metadata_device(
            prefill_query_start_loc,
            prefill_query_start_loc_cpu,
            FLA_CHUNK_SIZE,
        )


class KimiK3ROCmKDABackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA_ROCM"

    @staticmethod
    def get_builder_cls() -> type[KimiK3ROCmKDAMetadataBuilder]:
        return KimiK3ROCmKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Pure spec-decode batches take their plan from DEVICE offsets
        # (spec_query_start_loc, num_accepted_tokens) once the builder drops the
        # CPU-derived uniform length under adaptive verification. Mixed batches
        # only need the CPU totals, which adaptive verification preserves.
        return True
