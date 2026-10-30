# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the configured drafter with fixed, fully accepted target results."""

import numpy as np
import torch

from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states
from vllm.v1.worker.workspace import use_workspace_lane


class DraftBench:
    def __init__(self, runner):
        self.runner = runner
        self.speculator = runner.speculator
        self.method = runner.vllm_config.speculative_config.method
        if self.method not in ("mtp", "eagle", "eagle3", "dspark", "dflash"):
            raise ValueError(
                f"Drafter timing does not support {self.method!r}; "
                "use --exclude-drafter for target-only timing."
            )

    def prepare(self, input_batch, context):
        runner = self.runner
        self.batch = (
            runner.pcp_manager.global_batch if runner.pcp_manager else input_batch
        )
        self.context = context
        drafts = self.batch.num_draft_tokens_per_req
        accepted = (
            np.ones(self.batch.num_reqs, dtype=np.int32)
            if drafts is None
            else drafts.astype(np.int32) + 1
        )
        self.num_sampled = torch.tensor(accepted, device=runner.device)
        self.num_rejected = torch.zeros_like(self.num_sampled)
        # Fixed valid bonus token: target logits and rejection sampling are not timed.
        self.last_sampled = torch.zeros_like(runner.req_states.last_sampled_tokens)
        self.dp_sync_state = (
            DPSyncState(
                num_tokens_across_dp=torch.full(
                    (runner.dp_size,),
                    input_batch.num_tokens_after_padding,
                    dtype=torch.int32,
                ),
                uniform_token_count=None,
                eager=True,
                num_reqs=self.batch.num_reqs,
            )
            if runner.dp_size > 1
            else None
        )

    def propose(self, output):
        runner = self.runner
        if runner.use_aux_hidden_state_outputs:
            hidden, aux = output
        else:
            hidden, aux = output, None
        draft_hidden = get_drafter_hidden_states(runner.model, hidden)
        if runner.pcp_manager is not None and aux is not None:
            aux = [runner.pcp_manager.restore_hidden_states(states) for states in aux]
        with use_workspace_lane(runner._draft_workspace_lane):
            self.tokens = self.speculator.propose(
                self.batch,
                self.context.attn_metadata,
                self.context.slot_mapping,
                draft_hidden,
                aux,
                self.num_sampled,
                self.num_rejected,
                self.last_sampled,
                runner.req_states.next_prefill_tokens,
                runner.sampler.sampling_states.temperature.gpu,
                runner.sampler.sampling_states.seeds.gpu,
                dp_sync_state=self.dp_sync_state,
                num_speculative_tokens=runner.num_speculative_steps,
            )

    def check(self, reference):
        return {
            "method": self.method,
            "acceptance": "all",
            "num_draft_tokens": self.runner.num_speculative_steps,
            "output_shape": list(self.tokens.shape),
            "graph_matches_eager": bool(torch.equal(self.tokens, reference)),
            "valid_token_ids": bool(
                (
                    (self.tokens >= 0)
                    & (self.tokens < self.runner.model_config.get_vocab_size())
                ).all()
            ),
        }
