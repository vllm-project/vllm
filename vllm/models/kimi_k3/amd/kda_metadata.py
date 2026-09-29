# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
``GDNAttentionMetadataBuilder``. Only the FLA chunk metadata is built
differently on device rather than on the host.

When ``--use-replayssm`` is enabled on ROCm, KDA speculative decode uses the
ReplaySSM path (one checkpoint + ring record buffers) instead of
materializing one recurrent state per draft token.
"""

from dataclasses import dataclass, fields

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

logger = init_logger(__name__)

_BLOCK_T = 256
_MIN_BLOCK_N = 128
PENDING_SPEC_WINDOW = -1


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


@dataclass
class KimiK3ROCmKDAMetadata(GDNAttentionMetadata):
    """GDN metadata plus ROCm KDA ReplaySSM fields.

    Lives here so CUDA Qwen GDN can keep ``GDNAttentionMetadata`` unchanged.
    """

    replayssm: bool = False
    slot_idx: torch.Tensor | None = None
    write_pos: torch.Tensor | None = None
    replayssm_cache_len: int = 0
    replayssm_max_query_len: int = 1
    replayssm_fold_slots: torch.Tensor | None = None
    replayssm_fold_len: torch.Tensor | None = None
    replayssm_spec_slot_idx: torch.Tensor | None = None
    replayssm_decode_slot_idx: torch.Tensor | None = None


def _promote_gdn_metadata(md: GDNAttentionMetadata) -> KimiK3ROCmKDAMetadata:
    if isinstance(md, KimiK3ROCmKDAMetadata):
        return md
    return KimiK3ROCmKDAMetadata(
        **{f.name: getattr(md, f.name) for f in fields(GDNAttentionMetadata)}
    )


class KimiK3ROCmKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cache_config = vllm_config.cache_config
        self.use_kda_replayssm = (
            current_platform.is_rocm()
            and cache_config.use_replayssm
            and self.use_spec_decode
        )
        self.replayssm_max_query_len = self.num_spec + 1
        self.replayssm_cache_len = 0
        # Per slot: records the slot's last forward appended to its ring and
        # has not yet committed to write_pos. PENDING_SPEC_WINDOW means a
        # verify window, whose accepted count arrives with the next step.
        self.replayssm_pending_append: torch.Tensor | None = None
        self._replayssm_capturing = False
        self._replayssm_committed_this_step = False
        # Strong reference, not an id(): a freed metadata object's id can be
        # reused by the next step's object, which would suppress that commit.
        self._replayssm_step_marker: object | None = None
        self._step_fold_slots: torch.Tensor | None = None
        self._step_fold_len: torch.Tensor | None = None
        if self.use_kda_replayssm:
            requested = cache_config.replayssm_buffer_len
            min_cache_len = 2 * self.replayssm_max_query_len
            self.replayssm_cache_len = max(requested, min_cache_len)
            if self.replayssm_cache_len != requested:
                logger.warning(
                    "replayssm_buffer_len=%d is below 2*(mtp_k+1)=%d; "
                    "raising to %d for KDA ReplaySSM.",
                    requested,
                    min_cache_len,
                    self.replayssm_cache_len,
                )
            # All of this is per builder, i.e. per KV cache group: block ids are
            # only unique within a group, so state shared across groups would
            # let unrelated blocks collide on the same cursor entry.
            # Nightly constructs builders during CUDA-graph profiling *before*
            # cache_config.num_gpu_blocks is published; defer until it is.
            self.replayssm_write_pos: torch.Tensor | None = None
            self.replayssm_slot_buf: torch.Tensor | None = None
            self._ensure_replayssm_slots()
        else:
            self.replayssm_write_pos = None

    def _replayssm_num_slots(self) -> int | None:
        cfg = self.vllm_config.cache_config
        num_slots = cfg.num_gpu_blocks
        if num_slots is None:
            num_slots = cfg.num_gpu_blocks_override
        return num_slots

    def _ensure_replayssm_slots(self) -> bool:
        if not self.use_kda_replayssm:
            return False
        if self.replayssm_write_pos is not None:
            return True
        num_slots = self._replayssm_num_slots()
        if num_slots is None:
            return False
        device = self.device
        write_pos = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self.replayssm_write_pos = write_pos
        self.replayssm_pending_append = torch.zeros(
            num_slots, dtype=torch.int32, device=device
        )
        self.replayssm_slot_buf = torch.zeros(
            max(
                self.decode_cudagraph_max_bs,
                self.vllm_config.scheduler_config.max_num_seqs,
            ),
            dtype=torch.int32,
            device=device,
        )
        logger.info_once(
            "KDA ReplaySSM enabled on ROCm: cache_len=%d, verify window=%d, "
            "write_pos_slots=%d.",
            self.replayssm_cache_len,
            self.replayssm_max_query_len,
            write_pos.numel(),
        )
        return True

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

    def build(self, *args, **kwargs) -> KimiK3ROCmKDAMetadata:
        common_attn_metadata = (
            args[1] if len(args) > 1 else kwargs["common_attn_metadata"]
        )
        num_accepted_tokens = (
            args[2] if len(args) > 2 else kwargs.get("num_accepted_tokens")
        )
        if self._replayssm_step_marker is not common_attn_metadata:
            self._replayssm_step_marker = common_attn_metadata
            self._replayssm_committed_this_step = False
        metadata = _promote_gdn_metadata(super().build(*args, **kwargs))
        if not self.use_kda_replayssm:
            return metadata
        if not self._ensure_replayssm_slots():
            logger.warning(
                "KDA ReplaySSM: num_gpu_blocks still unset; skipping ReplaySSM "
                "metadata for this step (typical during CUDA-graph profiling)."
            )
            return metadata
        self._attach_kda_replayssm(metadata, common_attn_metadata, num_accepted_tokens)
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # Capture and DP dummy batches run on placeholder rows. Their cursor
        # bookkeeping would advance real slots by work that never happened.
        self._replayssm_capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._replayssm_capturing = False

    def _attach_kda_replayssm(
        self,
        md: KimiK3ROCmKDAMetadata,
        common_attn_metadata,
        num_accepted_tokens: torch.Tensor | None,
    ) -> None:
        write_pos = self.replayssm_write_pos
        pending = self.replayssm_pending_append
        slot_buf = self.replayssm_slot_buf
        assert write_pos is not None and pending is not None and slot_buf is not None

        md.replayssm = True
        md.write_pos = write_pos
        md.replayssm_cache_len = self.replayssm_cache_len
        md.replayssm_max_query_len = self.replayssm_max_query_len

        # spec_state_indices_tensor is [rows, num_spec + 1], so column 0 is
        # strided, while the kernels index slot_idx contiguously. Stage the
        # column in a persistent packed buffer: calling .contiguous() per step
        # would hand each captured cudagraph a pointer that dies after capture.
        spec_buf: torch.Tensor | None = None
        if md.spec_state_indices_tensor is not None:
            spec_buf = slot_buf[: md.spec_state_indices_tensor.shape[0]]
            spec_buf.copy_(md.spec_state_indices_tensor[:, 0])
            md.replayssm_spec_slot_idx = spec_buf
        if md.non_spec_state_indices_tensor is not None:
            md.replayssm_decode_slot_idx = md.non_spec_state_indices_tensor

        num_reqs = md.num_decodes + md.num_spec_decodes
        slot_idx: torch.Tensor | None = None
        if num_reqs > 0:
            if md.num_spec_decodes > 0:
                assert spec_buf is not None
                spec_slots = spec_buf[: md.num_spec_decodes]
                if md.num_decodes > 0:
                    assert md.non_spec_state_indices_tensor is not None
                    decode_slots = md.non_spec_state_indices_tensor[: md.num_decodes]
                    slot_idx = torch.cat([decode_slots, spec_slots])
                else:
                    slot_idx = spec_slots
            else:
                assert md.non_spec_state_indices_tensor is not None
                slot_idx = md.non_spec_state_indices_tensor[:num_reqs]
            # Trimmed to the live rows: the cursor bookkeeping must only touch
            # those, while the kernel keeps the padded rows so its grid stays
            # fixed across cudagraph replays.
            md.slot_idx = slot_idx

        if self._replayssm_capturing:
            md.replayssm_fold_slots = None
            md.replayssm_fold_len = None
            return

        # This group's cursors advance once per step, no matter how many layers
        # in the group later read the metadata.
        if not self._replayssm_committed_this_step:
            self._replayssm_committed_this_step = True
            # Commit before choosing each row's path: a row that appended on the
            # decode path last step may be a prefill row now (the GDN builder
            # reclassifies plain decodes whenever spec decodes are present), and
            # the fold below can only absorb records the cursor already counts.
            self._commit_pending_records(
                common_attn_metadata, num_accepted_tokens, write_pos, pending
            )
            self._mark_step_appends(md, pending)
            self._stage_replayssm_fold(md, write_pos)

        md.replayssm_fold_slots = self._step_fold_slots
        md.replayssm_fold_len = self._step_fold_len

    def _commit_pending_records(
        self,
        common_attn_metadata,
        num_accepted_tokens: torch.Tensor | None,
        write_pos: torch.Tensor,
        pending: torch.Tensor,
    ) -> None:
        """Advance every live row's cursor by what its last forward appended."""
        from vllm.models.kimi_k3.amd.ops.third_party.replayssm import (
            PAD_SLOT_ID,
            replayssm_commit,
        )

        m = common_attn_metadata
        num_reqs = m.num_reqs
        if num_reqs == 0:
            return
        block_table = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        slots = block_table[:num_reqs, 0].to(torch.int64)
        live = m.query_start_loc[1 : num_reqs + 1] > m.query_start_loc[:num_reqs]
        pending_rows = pending[slots]
        if num_accepted_tokens is None:
            accepted = torch.ones_like(pending_rows)
        else:
            accepted = num_accepted_tokens[:num_reqs].to(pending_rows)
        amount = torch.where(
            pending_rows == PENDING_SPEC_WINDOW, accepted, pending_rows
        )
        commit_slots = torch.where(
            live & (pending_rows != 0), slots, torch.full_like(slots, PAD_SLOT_ID)
        ).to(torch.int32)
        replayssm_commit(
            write_pos,
            commit_slots,
            amount,
            self.replayssm_max_query_len,
            self.replayssm_cache_len,
        )
        pending.index_fill_(0, slots, 0)

    def _mark_step_appends(
        self, md: KimiK3ROCmKDAMetadata, pending: torch.Tensor
    ) -> None:
        """Record what this step's decode-path rows will append."""
        if md.num_decodes > 0:
            assert md.non_spec_state_indices_tensor is not None
            decode_slots = md.non_spec_state_indices_tensor[: md.num_decodes]
            pending.index_fill_(0, decode_slots.to(torch.int64), 1)
        if md.num_spec_decodes > 0:
            assert md.replayssm_spec_slot_idx is not None
            spec_slots = md.replayssm_spec_slot_idx[: md.num_spec_decodes]
            pending.index_fill_(0, spec_slots.to(torch.int64), PENDING_SPEC_WINDOW)

    def _stage_replayssm_fold(
        self,
        md: KimiK3ROCmKDAMetadata,
        write_pos: torch.Tensor,
    ) -> None:
        """Snapshot the records that the chunk/prefill path has to absorb.

        The chunk kernel reads the checkpoint directly and cannot see the ring,
        so any row it consumes must have its records folded in first. Clearing
        the cursor here keeps every layer folding the same records, and the
        checkpoint is exact again once the chunk kernel writes the final state.
        """
        if md.num_prefills == 0 or md.prefill_state_indices is None:
            self._step_fold_slots = None
            self._step_fold_len = None
            return

        slots = md.prefill_state_indices.to(torch.int64)
        fold_len = write_pos[slots].clone()
        if md.prefill_has_initial_state is not None:
            # A sequence starting in this slot inherits whatever the block's
            # previous occupant left behind; those records are not its own.
            fold_len = torch.where(
                md.prefill_has_initial_state, fold_len, torch.zeros_like(fold_len)
            )

        self._step_fold_slots = md.prefill_state_indices
        self._step_fold_len = fold_len
        write_pos.index_fill_(0, slots, 0)


class KimiK3ROCmKDABackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA_ROCM"

    @staticmethod
    def get_builder_cls() -> type[KimiK3ROCmKDAMetadataBuilder]:
        return KimiK3ROCmKDAMetadataBuilder
