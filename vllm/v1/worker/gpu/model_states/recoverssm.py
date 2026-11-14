# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_recoverssm import (
    GDNRecoverSSMMetadata,
    commit_gdn_recoverssm_groups,
    gdn_recoverssm_groupable,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.utils import AttentionGroup, groups_with_metadata_builders

logger = init_logger(__name__)


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing."""

    def __init__(self) -> None:
        self._step: tuple[RecoverSSMMetadata, ...] | None = None
        # The grouped GDN commit over the KV cache groups' commit contexts, keyed by
        # the contexts (they live as long as their metadata builders).
        self._grouped: tuple[tuple[Any, ...], Any] | None = None

    def reset(self) -> None:
        """Drop the recorded step and the grouped commit. Both hold commit contexts,
        which reference the KV cache tensors they were built for."""
        self._step = None
        self._grouped = None

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
    ) -> None:
        if for_capture:
            self._step = None
            return

        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in groups_with_metadata_builders(group_list):
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        self._step = tuple(step)

    def detach_step(self) -> tuple[RecoverSSMMetadata, ...] | None:
        """Hand the recorded step to a deferred commit (non-last PP ranks).

        A non-last pipeline-parallel rank commits in the PP postprocess
        ``pp_size`` steps later, after the other micro-batches have recorded
        their own steps and rebuilt the persistent metadata buffers. The commit
        index tensors (the replay-record rows among them) are therefore cloned
        here. The replay records themselves live in rows owned by the request
        (its request slot, or its state block for in-page records), and the
        request is not verified again before this commit, so they are not
        overwritten in between.
        """
        step, self._step = self._step, None
        if not step:
            return None
        detached = []
        for metadata in step:
            commit = getattr(metadata, "recoverssm_commit", None)
            if commit is not None:
                commit = dataclasses.replace(
                    commit,
                    **{
                        f.name: getattr(commit, f.name).clone()
                        for f in dataclasses.fields(commit)
                        if isinstance(getattr(commit, f.name), torch.Tensor)
                    },
                )
                metadata = dataclasses.replace(metadata, recoverssm_commit=commit)
            detached.append(metadata)
        return tuple(detached)

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        step: tuple[RecoverSSMMetadata, ...] | None = None,
    ) -> None:
        if step is None:
            step = self._step
            self._step = None
        elif not isinstance(num_sampled, int):
            # Deferred PP commit: rows whose request was freed while the commit
            # was pending carry idx -1 and may already have a new owner for
            # their state block; an accepted count of 0 makes the commit kernel
            # skip them. num_sampled from the PP broadcast is padded past
            # num_reqs, so only the first len(idx_mapping) rows are real.
            n = idx_mapping.shape[0]
            assert idx_mapping.ndim == 1 and num_sampled.ndim == 1
            assert num_sampled.shape[0] >= n, (num_sampled.shape, n)
            num_sampled = num_sampled.clone()
            num_sampled[:n] = torch.where(idx_mapping >= 0, num_sampled[:n], 0)
        if isinstance(num_sampled, int) or step is None:
            return

        postprocess: list[RecoverSSMPostprocessMetadata | None] = []
        # GDN layers split over several KV cache groups commit with one set of
        # launches instead of one set per group.
        gdn = [m for m in step if isinstance(m, GDNRecoverSSMMetadata)]
        grouped = self._grouped_commit(gdn) if gdn_recoverssm_groupable(gdn) else None
        if grouped is not None:
            postprocess.append(commit_gdn_recoverssm_groups(gdn, num_sampled, grouped))
            step = tuple(m for m in step if not isinstance(m, GDNRecoverSSMMetadata))
        postprocess.extend(m.commit_recoverssm_state(num_sampled) for m in step)

        for postprocess_meta in postprocess:
            if postprocess_meta is None:
                continue
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
            _postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                postprocess_meta.request_indices,
                postprocess_meta.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
                BLOCK_TABLE_WIDTH=postprocess_meta.block_table.shape[1],
            )

    def _grouped_commit(self, metas: list[GDNRecoverSSMMetadata]) -> Any:
        """The grouped commit over these groups' contexts, or None when their layers
        cannot share one context (then every group commits on its own)."""
        contexts = tuple(m.recoverssm_context for m in metas)
        # Compare by identity: the contexts are dataclasses over tensors.
        if (
            self._grouped is None
            or len(self._grouped[0]) != len(contexts)
            or any(a is not b for a, b in zip(self._grouped[0], contexts))
        ):
            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
                GDNRecoverSSMGroupedCommit,
            )

            try:
                grouped = GDNRecoverSSMGroupedCommit(
                    contexts, max_num_reqs=contexts[0].commit_lens.shape[0]
                )
            except ValueError as e:
                logger.info("GDN RecoverSSM commit stays per group: %s", e)
                grouped = None
            else:
                logger.info(
                    "GDN RecoverSSM grouped commit: %d KV cache groups, %d layers",
                    grouped.num_groups,
                    grouped.layer_groups.numel(),
                )
            self._grouped = (contexts, grouped)
        return self._grouped[1]


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    # Match the commit plan: the running state lives in the block holding the
    # last computed token, so the next step migrates it across a boundary.
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(
            tl.maximum(num_computed + num_sampled - 1, 0) // MAMBA_BLOCK_SIZE,
            BLOCK_TABLE_WIDTH - 1,
        ),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)
