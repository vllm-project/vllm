# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.cpu.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def _compute_token_logprobs(
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


class CPUSamplingStates(SamplingStates):
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            return

        temperature = self.temperature.gpu[expanded_idx_mapping].unsqueeze(1)
        logits.div_(torch.where(temperature == 0.0, 1.0, temperature))

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            return

        min_p = self.min_p.gpu[expanded_idx_mapping].unsqueeze(1)
        threshold = logits.amax(dim=1, keepdim=True) + min_p.log()
        logits.masked_fill_((min_p != 0.0) & (logits < threshold), float("-inf"))


class CPUSampler(Sampler):
    """CPU extension point for Model Runner V2 sampling."""

    sampling_states_cls = CPUSamplingStates

    def _compute_topk_scores(
        self,
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        cu_num_logits: list[int] | torch.Tensor | None,
        logprob_token_ids_state: LogprobTokenIdsState,
        expanded_idx_mapping: torch.Tensor,
        max_per_req_token_ids: int,
        logits_mode: bool,
    ) -> LogprobsTensors:
        sampled_token_ids = sampled_token_ids.to(torch.int64)

        if max_per_req_token_ids == 0:
            logprob_token_ids = sampled_token_ids.unsqueeze(-1)
            if num_logprobs > 0:
                topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
                logprob_token_ids = torch.cat(
                    (logprob_token_ids, topk_token_ids), dim=1
                )
            valid_mask = None
        else:
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
            logprobs = _compute_token_logprobs(logits, logprob_token_ids)
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

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not use_fused_sampler

        # TODO (fadara01): Add C++ fused ops for top-p / top-k sampling.
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)

        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=self.use_fp64_gumbel,
        )
        return sampled, processed_logits
