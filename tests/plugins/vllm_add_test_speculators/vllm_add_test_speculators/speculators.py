# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only speculators, imported by qualified name in the worker."""

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    PlanInputs,
    TargetTokens,
    VerifyPlan,
)

from . import OFF_CHAIN
from .reference import walk_tree


class NgramTreeSpeculator(NgramGPUSpeculator):
    """Verifies the n-gram drafts as a tree: the first draft gets a sibling
    holding another token, and the chain continues from the first draft. Prints
    ``OFF_CHAIN`` the first time a step committed the first and second drafts
    (the path skips the sibling's row)."""

    _printed = False

    def tree_accept(
        self, plan: VerifyPlan, target: TargetTokens, width: int
    ) -> AcceptResult:
        return walk_tree(plan, target, width)

    def plan(self, inputs: PlanInputs) -> VerifyPlan | None:
        if inputs.chain_only:
            return None
        assert self.resources is not None
        num_reqs = len(inputs.req_ids)
        num_cols = self.resources.max_rows
        if not hasattr(self, "_parent"):
            shape = (self.req_states.max_num_reqs, num_cols)
            self._parent = torch.empty(shape, dtype=torch.int32, device=self.device)
            self._tokens = torch.empty(shape, dtype=torch.int32, device=self.device)
            # Per request slot: where the last tree's first draft would commit,
            # then its first and second drafts (-1: no tree).
            self._path = torch.full(
                (3, self.req_states.max_num_reqs), -1, device=self.device
            )
        self._check_skip(inputs.idx_mapping)
        rows = inputs.scheduled_rows.astype(np.int32)
        parent = np.tile(np.arange(-1, num_cols - 1, dtype=np.int32), (num_reqs, 1))
        # Row i holds draft i - 1 (row 0, the root, is ignored).
        src = np.tile(np.arange(-1, num_cols - 1), (num_reqs, 1)).clip(min=0)
        alt = np.zeros((num_reqs, num_cols), dtype=np.int64)
        tree = rows >= 4
        # Rows: root, d1, sibling of d1, d2 (child of d1), d3, ...
        parent[tree, 2] = 0
        parent[tree, 3] = 1
        src[tree, 2] = 0
        src[tree, 3:] -= 1
        alt[tree, 2] = 1
        drafts = self.req_states.draft_tokens[inputs.idx_mapping]
        src_gpu = torch.from_numpy(src).to(self.device, non_blocking=True)
        src_gpu = src_gpu.clamp(max=drafts.shape[1] - 1)
        tokens = drafts.gather(1, src_gpu)
        tokens = tokens + torch.from_numpy(alt).to(self.device, non_blocking=True)
        # Unproposed drafts are -1: any real token verifies the same.
        tokens = tokens.clamp(min=0) % self.vllm_config.model_config.get_vocab_size()
        self._parent[:num_reqs].copy_(torch.from_numpy(parent), non_blocking=True)
        self._tokens[:num_reqs].copy_(tokens.to(torch.int32))
        # The first committed token lands after the root; the second draft
        # follows the first only through its child row, past the sibling.
        root_pos = self.req_states.total_len.gpu[inputs.idx_mapping].long()
        is_tree = torch.from_numpy(tree).to(self.device, non_blocking=True)
        is_tree &= tokens[:, 2] != tokens[:, 1]
        self._path[0, inputs.idx_mapping] = torch.where(is_tree, root_pos, -1)
        self._path[1, inputs.idx_mapping] = tokens[:, 1]
        self._path[2, inputs.idx_mapping] = tokens[:, 3]
        return VerifyPlan(
            parent=self._parent[:num_reqs], tokens=self._tokens[:num_reqs]
        )

    def _check_skip(self, slots: torch.Tensor) -> None:
        if self._printed:
            return
        pos, first, second = self._path[:, slots]
        ids = self.req_states.all_token_ids.gpu
        at = pos.clamp(min=0)
        after = (at + 1).clamp(max=ids.shape[1] - 1)
        hit = (pos >= 0) & (pos + 1 < self.req_states.total_len.gpu[slots])
        hit &= (ids[slots, at] == first) & (ids[slots, after] == second)
        if bool(hit.any()):
            print(OFF_CHAIN, flush=True)
            self._printed = True
