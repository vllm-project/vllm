# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
``GDNAttentionMetadataBuilder``. Only the FLA chunk metadata is built
differently on device rather than on the host.

When ``--use-replayssm`` is enabled on ROCm, KDA speculative decode uses the
ATOM-style ReplaySSM path (one checkpoint + ring record buffers) instead of
materializing one recurrent state per draft token.
"""

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
        self.replayssm_pending_reset: torch.Tensor | None = None
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
            num_slots = self._replayssm_num_slots(vllm_config)
            self.replayssm_write_pos = torch.zeros(
                num_slots, dtype=torch.int32, device=device
            )
            self.replayssm_pending_reset = torch.zeros(
                num_slots, dtype=torch.int32, device=device
            )
            self.replayssm_slot_buf = torch.zeros(
                max(
                    self.decode_cudagraph_max_bs,
                    vllm_config.scheduler_config.max_num_seqs,
                ),
                dtype=torch.int32,
                device=device,
            )
            logger.info_once(
                "KDA ReplaySSM enabled on ROCm: cache_len=%d, verify window=%d, "
                "write_pos_slots=%d.",
                self.replayssm_cache_len,
                self.replayssm_max_query_len,
                self.replayssm_write_pos.numel(),
            )
        else:
            self.replayssm_write_pos = None

    @staticmethod
    def _replayssm_num_slots(vllm_config) -> int:
        num_slots = vllm_config.cache_config.num_gpu_blocks
        if num_slots is None:
            raise RuntimeError(
                "KDA ReplaySSM metadata builder requires num_gpu_blocks to be set"
            )
        return num_slots

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

    def build(self, *args, **kwargs) -> GDNAttentionMetadata:
        common_attn_metadata = (
            args[1] if len(args) > 1 else kwargs["common_attn_metadata"]
        )
        if self._replayssm_step_marker is not common_attn_metadata:
            self._replayssm_step_marker = common_attn_metadata
            self._replayssm_committed_this_step = False
        metadata = super().build(*args, **kwargs)
        if not self.use_kda_replayssm:
            return metadata
        self._attach_kda_replayssm(metadata)
        return metadata

    def _attach_kda_replayssm(self, md: GDNAttentionMetadata) -> None:
        from vllm.models.kimi_k3.amd.ops.third_party.replayssm import replayssm_commit

        write_pos = self.replayssm_write_pos
        pending_reset = self.replayssm_pending_reset
        assert write_pos is not None and pending_reset is not None

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
            spec_buf = self.replayssm_slot_buf[: md.spec_state_indices_tensor.shape[0]]
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

        # This group's cursors advance once per step, no matter how many layers
        # in the group later read the metadata.
        if not self._replayssm_committed_this_step:
            self._replayssm_committed_this_step = True
            self._advance_replayssm_cursors(
                md, write_pos, pending_reset, slot_idx, num_reqs, replayssm_commit
            )
            self._stage_replayssm_fold(md, write_pos, pending_reset)

        md.replayssm_fold_slots = self._step_fold_slots
        md.replayssm_fold_len = self._step_fold_len

    def _advance_replayssm_cursors(
        self,
        md: GDNAttentionMetadata,
        write_pos: torch.Tensor,
        pending_reset: torch.Tensor,
        slot_idx: torch.Tensor | None,
        num_reqs: int,
        replayssm_commit,
    ) -> None:
        """Advance each row's cursor by the records the previous step wrote."""
        if slot_idx is None:
            return

        if md.num_spec_decodes == 0:
            # A plain decode still appends one record per row, so its cursor
            # must advance too, otherwise the next step overwrites the token.
            accepted = torch.ones(
                num_reqs, dtype=torch.int32, device=write_pos.device
            )
        elif md.num_accepted_tokens is not None:
            accepted = md.num_accepted_tokens[:num_reqs].to(torch.int32)
        else:
            return

        # Rows whose ring was emptied by a prefill have nothing to commit; the
        # accepted count from that step refers to tokens already folded into
        # the checkpoint.
        slots = slot_idx.to(torch.int64)
        accepted = torch.where(
            pending_reset[slots] != 0, torch.zeros_like(accepted), accepted
        )
        replayssm_commit(
            write_pos,
            slot_idx,
            accepted,
            self.replayssm_max_query_len,
            self.replayssm_cache_len,
        )
        pending_reset.index_fill_(0, slots, 0)

    def _stage_replayssm_fold(
        self,
        md: GDNAttentionMetadata,
        write_pos: torch.Tensor,
        pending_reset: torch.Tensor,
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
        pending_reset.index_fill_(0, slots, 1)


class KimiK3ROCmKDABackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA_ROCM"

    @staticmethod
    def get_builder_cls() -> type[KimiK3ROCmKDAMetadataBuilder]:
        return KimiK3ROCmKDAMetadataBuilder
