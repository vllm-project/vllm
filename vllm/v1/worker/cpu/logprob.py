# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    selected = logits.gather(-1, token_ids).to(torch.float32)
    # Accumulate the normalizer in FP32. Compilation fuses this cast into the
    # reduction instead of materializing a full FP32 logits tensor.
    normalizer = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True)
    return selected - normalizer


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def _compute_token_ranks(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(-1, token_ids.unsqueeze(-1))
    return (logits >= selected).sum(-1)


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: LogprobTokenIdsState | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    sampled_token_ids = sampled_token_ids.to(torch.int64)

    if max_per_req_token_ids == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_token_ids), dim=1)
        valid_mask = None
    else:
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None
        batch_size = logits.shape[0]
        num_cols = max(num_logprobs, max_per_req_token_ids)
        selected_token_ids = sampled_token_ids.new_zeros((batch_size, num_cols))
        valid_mask = torch.zeros_like(selected_token_ids, dtype=torch.bool)

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            selected_token_ids[:, :num_logprobs] = topk_token_ids
            valid_mask[:, :num_logprobs] = True

        req_indices = expanded_idx_mapping.to(torch.int64)
        num_custom = logprob_token_ids_state.num_token_ids.gpu[req_indices].to(
            torch.int64
        )
        custom_token_ids = torch.zeros_like(selected_token_ids)
        custom_token_ids[:, :max_per_req_token_ids] = (
            logprob_token_ids_state.token_ids.gpu[
                req_indices, :max_per_req_token_ids
            ].to(torch.int64)
        )
        custom_valid_mask = torch.arange(num_cols, device=logits.device).unsqueeze(
            0
        ) < num_custom.unsqueeze(1)
        use_custom = num_custom.unsqueeze(1) > 0
        selected_token_ids = torch.where(
            use_custom, custom_token_ids, selected_token_ids
        )
        valid_mask = torch.where(use_custom, custom_valid_mask, valid_mask)

        logprob_token_ids = torch.cat(
            (sampled_token_ids.unsqueeze(-1), selected_token_ids), dim=1
        )
        valid_mask = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=logits.device),
                valid_mask,
            ),
            dim=1,
        )

    if logits_mode:
        logprobs = logits.gather(-1, logprob_token_ids).to(torch.float32)
    else:
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    if valid_mask is not None:
        logprobs.masked_fill_(~valid_mask, float("-inf"))

    sampled_ranks = _compute_token_ranks(logits, sampled_token_ids)

    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids.to(torch.int32),
        logprobs=logprobs,
        selected_token_ranks=sampled_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


class CPUPromptLogprobsWorker(PromptLogprobsWorker):
    def _compute_token_logprobs(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return compute_token_logprobs(logits, token_ids)

    def _compute_topk_scores(
        self,
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        logits_mode: bool,
    ) -> LogprobsTensors:
        return compute_topk_scores(
            logits,
            num_logprobs,
            sampled_token_ids,
            logits_mode=logits_mode,
        )

    def get_prompt_logprobs_token_ids(
        self,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        all_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return ops.get_prompt_logprobs_token_ids_cpu(
            num_tokens,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
