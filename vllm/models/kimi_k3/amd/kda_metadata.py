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
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.models.kimi_k3.amd import akr_kda_prefill
from vllm.models.kimi_k3.amd.specialized_decode.levers import EAGERPREP, lever_on
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

logger = init_logger(__name__)

_BLOCK_T = 256
_MIN_BLOCK_N = 128

# block_m of the fused prefill conv1d in kda.py; shared GDN builder prebuilds only 8
PREFILL_CONV_BLOCK_M = 128


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


@triton.jit(do_not_specialize=["num_spec_decodes", "batch_size", "spec_token_size"])
def _stage_spec_decode_fused_kernel(
    block_table_ptr,
    bt_stride_0: tl.int64,
    bt_stride_1,
    seq_lens_ptr,
    query_start_loc_ptr,
    num_accepted_ptr,
    state_out_ptr,
    state_out_stride_0,
    masks_out_ptr,
    token_indx_out_ptr,
    qsl_out_ptr,
    num_accepted_out_ptr,
    num_spec_decodes,
    batch_size,
    spec_token_size,
    cache_block_size,
    ALIGN: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    NULL_ID: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
):
    """One launch for GDN's FULL-graph spec-decode staging (pure verify batch).

    Writes exactly what ``GDNAttentionMetadataBuilder.build`` stages through
    ~9 launches (mamba align gather, mask H2D, arange, 5 copies + fills):
      state_out[r, :]  = aligned block_table[r] (r < ns), NULL_ID (ns <= r < bs)
      masks_out[r]     = r < ns                              (r < bs)
      token_indx[t]    = t                                   (t < spec_token_size)
      qsl_out[r]       = qsl[min(r, ns)]                     (r <= bs)
      num_accepted[r]  = num_accepted_in[r] (r < ns), 1      (ns <= r < bs)

    The state / query_start_loc / num_accepted staging mirrors
    ``nvidia/kda_metadata.py:_stage_spec_decode_metadata_kernel``; keep the
    two in sync.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    real = rows < num_spec_decodes
    in_batch = rows < batch_size

    slots = tl.arange(0, BLOCK_SLOTS)
    slot_ok = slots < NUM_SLOTS
    if ALIGN:
        seq_len = tl.load(seq_lens_ptr + rows, mask=real, other=1)
        start = tl.maximum((seq_len - 1) // cache_block_size, 0)
    else:
        start = tl.zeros([BLOCK_ROWS], dtype=tl.int32)
    cols = start[:, None] + slots[None, :]
    state = tl.load(
        block_table_ptr + rows[:, None].to(tl.int64) * bt_stride_0 + cols * bt_stride_1,
        mask=real[:, None] & slot_ok[None, :],
        other=NULL_ID,
    )
    tl.store(
        state_out_ptr + rows[:, None] * state_out_stride_0 + slots[None, :],
        state,
        mask=in_batch[:, None] & slot_ok[None, :],
    )
    tl.store(masks_out_ptr + rows, real, mask=in_batch)

    qsl = tl.load(query_start_loc_ptr + tl.minimum(rows, num_spec_decodes))
    tl.store(qsl_out_ptr + rows, qsl, mask=rows <= batch_size)

    nacc = tl.load(num_accepted_ptr + rows, mask=real, other=1)
    tl.store(num_accepted_out_ptr + rows, nacc, mask=in_batch)

    toks = pid * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tl.store(token_indx_out_ptr + toks, toks, mask=toks < spec_token_size)


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if PIN_MEMORY:
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def build_conv1d_chunk_schedule(
    query_start_loc_cpu: torch.Tensor, block_m: int, device: torch.device
) -> dict[str, int | torch.Tensor]:
    """Host-built AITER conv1d chunk schedule: no D2H sync, own buffers."""
    assert query_start_loc_cpu.device.type == "cpu"
    nums = (-(-query_start_loc_cpu.diff() // block_m)).to(torch.int64)
    tot = int(nums.sum())
    batch_ptr = torch.repeat_interleave(
        torch.arange(nums.numel(), dtype=torch.int32), nums
    )
    starts = torch.repeat_interleave(nums.cumsum(0) - nums, nums)
    token_chunk_offset_ptr = (torch.arange(tot, dtype=torch.int64) - starts).to(
        torch.int32
    )
    return {
        "tot": tot,
        "batch_ptr": _to_device(batch_ptr, device),
        "token_chunk_offset_ptr": _to_device(token_chunk_offset_ptr, device),
    }


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
        if not _adaptive_verification_enabled(vllm_config):
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
        if lever_on(EAGERPREP):
            fused = self._build_spec_decode_fused(
                common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu
            )
            if fused is not None:
                return fused
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        if attn_metadata.nums_dict is not None:
            # Same non-spec query_start_loc the base builder schedules over.
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            if attn_metadata.spec_sequence_masks_cpu is not None:
                query_lens_cpu = query_start_loc_cpu.diff()[
                    ~attn_metadata.spec_sequence_masks_cpu
                ]
                query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.numel() + 1, dtype=torch.int32
                )
                torch.cumsum(query_lens_cpu, dim=0, out=query_start_loc_cpu[1:])
            attn_metadata.nums_dict[PREFILL_CONV_BLOCK_M] = build_conv1d_chunk_schedule(
                query_start_loc_cpu,
                PREFILL_CONV_BLOCK_M,
                common_attn_metadata.query_start_loc.device,
            )
        if (
            attn_metadata.uniform_spec_sequence_length is not None
            and _adaptive_verification_enabled(self.vllm_config)
        ):
            attn_metadata = replace(attn_metadata, uniform_spec_sequence_length=None)
        if attn_metadata.checkpoint is not None:
            # The shared builder keys checkpoint rows on every request; the ROCm
            # prefill path indexes them by prefill_query_start_loc, so rebuild
            # with those rows.
            attn_metadata = replace(
                attn_metadata,
                checkpoint=self._build_checkpoint_metadata(
                    common_attn_metadata, attn_metadata, num_decode_draft_tokens_cpu
                ),
            )
        # Host chunk counts of the prefill sequences for amd_kernels' KDA prefill.
        akr_kda_prefill.attach_chunk_counts(
            attn_metadata, common_attn_metadata.query_start_loc_cpu
        )
        return attn_metadata

    def _build_spec_decode_fused(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata | None:
        """``GDNAttentionMetadataBuilder.build`` for a pure spec-decode batch
        staged into the FULL-graph buffers, in one launch instead of ~9.

        Returns None (caller falls back to the reference build) unless every
        real row is a spec decode, padding rows are zero-length and trail the
        spec rows, and the batch is staged. All decisions are host-side, from
        the same CPU tensors the reference path uses; no sync is added.
        """
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_accepted_tokens is None
            or getattr(self, "mamba_aligned_state_indices", None) is not None
        ):
            return None
        if not m.block_table_tensor.is_cuda:
            return None
        mode = self.vllm_config.cache_config.mamba_cache_mode
        if mode not in ("align", "none"):
            return None
        spec_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = int(spec_masks_cpu.sum())
        if num_spec_decodes == 0:
            return None
        batch_size = m.num_reqs
        if batch_size > self.decode_cudagraph_max_bs:
            return None
        query_start_loc_cpu = m.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        if spec_masks_cpu.shape[0] != batch_size or query_lens_cpu.shape[0] != (
            batch_size
        ):
            return None
        # Spec rows first, then only zero-length padding: then the reference
        # path takes its "num_prefills == 0 and num_decodes == 0" branch and
        # slices [:num_spec_decodes] exactly as staged here.
        if not bool(spec_masks_cpu[:num_spec_decodes].all()):
            return None
        if num_spec_decodes < batch_size and bool(
            (query_lens_cpu[num_spec_decodes:] != 0).any()
        ):
            return None
        num_spec_decode_tokens = int(query_lens_cpu.sum())
        if not self._stage_spec_decode(0, 0, num_spec_decodes, num_spec_decode_tokens):
            return None
        num_slots = self.num_spec + 1
        if mode == "align":
            assert isinstance(self.kv_cache_spec, MambaSpec)
            if 1 + self.kv_cache_spec.num_speculative_blocks < num_slots:
                return None
        elif m.block_table_tensor.shape[1] < num_slots:
            return None
        if num_accepted_tokens.shape[0] < num_spec_decodes:
            return None

        uniform_spec_sequence_length = None
        if not self._adaptive_verification:
            first = int(query_lens_cpu[0])
            if first > 0 and bool(
                torch.all(query_lens_cpu[:num_spec_decodes] == first)
            ):
                uniform_spec_sequence_length = first
        spec_token_size = min(
            num_spec_decodes * num_slots, int(query_start_loc_cpu[-1])
        )

        block_table = m.block_table_tensor
        state_out = self.spec_state_indices_tensor
        block_rows = 32
        # tl.arange needs a power of 2; 32 * num_slots is not for nst 2/4/5/6.
        block_tok = triton.next_power_of_2(block_rows * num_slots)
        grid = (
            max(
                triton.cdiv(batch_size + 1, block_rows),
                triton.cdiv(spec_token_size, block_tok),
            ),
        )
        _stage_spec_decode_fused_kernel[grid](
            block_table,
            block_table.stride(0),
            block_table.stride(1),
            m.seq_lens,
            m.query_start_loc,
            num_accepted_tokens,
            state_out,
            state_out.stride(0),
            self.spec_sequence_masks,
            self.spec_token_indx,
            self.spec_query_start_loc,
            self.num_accepted_tokens,
            num_spec_decodes,
            batch_size,
            spec_token_size,
            self.kv_cache_spec.block_size,
            ALIGN=mode == "align",
            NUM_SLOTS=num_slots,
            BLOCK_SLOTS=triton.next_power_of_2(num_slots),
            NULL_ID=NULL_BLOCK_ID,
            BLOCK_ROWS=block_rows,
            BLOCK_TOK=block_tok,
            num_warps=1,
        )
        attn_metadata = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            spec_query_start_loc=self.spec_query_start_loc[: batch_size + 1],
            spec_state_indices_tensor=state_out[:batch_size],
            spec_sequence_masks=self.spec_sequence_masks[:batch_size],
            spec_sequence_masks_cpu=spec_masks_cpu,
            spec_token_indx=self.spec_token_indx[:spec_token_size],
            non_spec_token_indx=self.non_spec_token_indx[:0],
            num_accepted_tokens=self.num_accepted_tokens[:batch_size],
            uniform_spec_sequence_length=uniform_spec_sequence_length,
        )
        if (
            attn_metadata.uniform_spec_sequence_length is not None
            and _adaptive_verification_enabled(self.vllm_config)
        ):
            attn_metadata = replace(attn_metadata, uniform_spec_sequence_length=None)
        return attn_metadata

    def _build_checkpoint_metadata(
        self,
        m: CommonAttentionMetadata,
        attn_metadata: GDNAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> MambaPrefillCheckpointMetadata | None:
        if attn_metadata.num_prefills == 0:
            return None
        # request_rows must line up with prefill_query_start_loc, either the
        # prefill tail of a decode-first batch, or every non-spec row
        # including padding
        num_decodes = attn_metadata.num_decodes
        request_rows = list(
            range(num_decodes, num_decodes + attn_metadata.num_prefills)
        )
        if attn_metadata.spec_sequence_masks is not None:
            assert num_decode_draft_tokens_cpu is not None
            request_rows = (
                (num_decode_draft_tokens_cpu < 0).nonzero().flatten().tolist()
            )
        return self.checkpoint_builder.build(m, request_rows)

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
        return True
