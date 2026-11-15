# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's model runner state on ROCm: the hybrid (KDA + MLA) state, plus the
lifecycle of the specialized decode when ``decode_backend=latency_specialized``
selects it, and two of its levers (``levers``): the target step skips the
draft-only attention groups (``VLLM_K31_BEAT_DRAFTMETA``) and gathers
num_accepted_tokens in one launch (``VLLM_K31_BEAT_EAGERPREP``)."""

from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.kimi_k3.amd.specialized_decode.levers import (
    DRAFTMETA,
    EAGERPREP,
    lever_on,
)
from vllm.models.kimi_k3.amd.specialized_decode.levers.runner import in_target_step
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup


class KimiK3ModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded, and prepares it
    for every bound KV cache (the CUDA-graph memory-profiling cache, then the real
    one)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()
        # The DFlash-family drafter's layers (``note_draft_attn_layers``).
        self._draft_layers: set[str] = set()
        # (attention groups, those minus the draft-only ones), last filtered.
        self._target_groups: tuple[list, list] | None = None

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()

    def note_draft_attn_layers(self, draft: set[str]) -> None:
        """From a DFlash-family drafter's ``set_attn`` (``levers.dspark``): the
        drafter's layers. Those drafters rebuild their own attention metadata
        from their own builders and never read the target step's."""
        self._draft_layers = set(draft)
        self._target_groups = None

    def target_attn_groups(
        self, attn_groups: list[list[AttentionGroup]]
    ) -> list[list[AttentionGroup]]:
        """The groups the step builds metadata for: under
        ``VLLM_K31_BEAT_DRAFTMETA``, in the runner's target step
        (``levers.runner.in_target_step``; not CUDA-graph capture or the
        drafter's calls), minus the draft-only ones."""
        if not (self._draft_layers and in_target_step() and lever_on(DRAFTMETA)):
            return attn_groups
        # The runner's groups are fixed once the KV cache is set up; filter once.
        target = self._target_groups
        if target is None or target[0] is not attn_groups:
            target = (
                attn_groups,
                drop_draft_only_attn_groups(attn_groups, self._draft_layers),
            )
            self._target_groups = target
        return target[1]

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        attn_groups = self.target_attn_groups(attn_groups)
        if not lever_on(EAGERPREP):
            return super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=ubatch_idx,
                model_specific_attn_metadata=model_specific_attn_metadata,
            )
        return self._prepare_attn_eagerprep(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )

    def _prepare_attn_eagerprep(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool,
        ubatch_idx: int,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None,
    ) -> dict[str, Any]:
        """``MambaHybridModelState.prepare_attn`` with num_accepted_tokens from
        ``gather_num_accepted`` (one launch, was three)."""
        assert ubatch_idx == 0, "DBO is not supported"
        assert model_specific_attn_metadata is None
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        # Prefer the promised bound: a capture dummy's measured max is its even
        # split, not the length the graph must replay.
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any
            # replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling_np = input_batch.is_prefilling_np
        if input_batch.prefill_runs_as_decode_np is not None:
            # A prompt tail the scheduler padded with placeholder drafts must run
            # as a spec-decode row: the prefill kernels can't roll them back.
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)
        # During CUDAGraph capture the builders create num_accepted_tokens and
        # num_decode_draft_tokens_cpu (build_for_cudagraph_capture).
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = gather_num_accepted(
                self.num_accepted_tokens_gpu, input_batch.idx_mapping, num_reqs
            )

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel. Decode rows without drafts stay spec rows:
            # only that path applies the previous step's accepted offset.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is None:
                num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)
            # Test request state, not num_scheduled_tokens == draft_count+1:
            # adaptive rewrites num_scheduled_tokens to an even split, so that
            # equality rarely holds and would demote every verify row to decode.
            is_decode = (
                ~is_prefilling_np
                & (input_batch.num_scheduled_tokens > 0)
                & (
                    input_batch.num_scheduled_tokens
                    <= self.vllm_config.num_speculative_tokens + 1
                )
            )
            num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                is_decode, num_draft_tokens_per_req, -1
            )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                assert self._mamba_ctx is not None
                all_group_indices = self._mamba_ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata, attn_groups, for_capture=for_capture
            )
        return attn_metadata


def drop_draft_only_attn_groups(
    attn_groups: list[list[AttentionGroup]], draft: set[str]
) -> list[list[AttentionGroup]]:
    """The groups minus those holding only draft layers. Group positions are
    kept (an emptied group is skipped by build_attn_metadata)."""
    return [
        [
            g
            for g in groups
            if isinstance(g.kv_cache_spec, MambaSpec) or not set(g.layer_names) <= draft
        ]
        for groups in attn_groups
    ]


@triton.jit(do_not_specialize=["num_real", "num_padded"])
def _gather_num_accepted_kernel(
    src_ptr,  # [max_num_reqs]
    idx_mapping_ptr,  # [num_real]
    out_ptr,  # [num_padded]
    num_real,
    num_padded,
    src_len,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = offs < num_real
    idx = tl.load(idx_mapping_ptr + offs, mask=real, other=0)
    # torch advanced indexing wraps negative indices.
    idx = tl.where(idx < 0, idx + src_len, idx)
    val = tl.load(src_ptr + idx, mask=real, other=1)
    tl.store(out_ptr + offs, val, mask=offs < num_padded)


def gather_num_accepted(
    src: torch.Tensor, idx_mapping: torch.Tensor, num_padded: int
) -> torch.Tensor:
    """``out = ones(num_padded); out[:n] = src[idx_mapping]`` in one launch."""
    out = torch.empty(num_padded, dtype=src.dtype, device=src.device)
    num_real = idx_mapping.shape[0]
    assert num_real <= num_padded
    if num_padded == 0:
        return out
    block = 256
    _gather_num_accepted_kernel[(triton.cdiv(num_padded, block),)](
        src, idx_mapping, out, num_real, num_padded, src.shape[0], BLOCK=block
    )
    return out
