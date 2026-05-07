# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.probabilistic_rejection_sampler_utils import (
    probabilistic_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.synthetic_rejection_sampler_utils import (
    synthetic_rejection_sample,
)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.rejection_sample_method = spec_config.rejection_sample_method
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if self.rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        if self.rejection_sample_method == "standard":
            pos = input_batch.positions[input_batch.logits_indices]
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )
            logprobs_tensors = self._get_logprobs_tensors(
                input_batch,
                sampled,
                num_sampled,
                processed_logits
                if self.sampler.logprobs_mode == "processed_logprobs"
                else logits,
            )
        elif self.rejection_sample_method == "synthetic":
            sampler_output = self.sampler(logits, input_batch)
            logprobs_tensors = sampler_output.logprobs_tensors
            sampled, num_sampled = synthetic_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                input_batch.positions[input_batch.logits_indices],
                input_batch.idx_mapping,
                self.sampler.sampling_states.seeds.gpu,
                self.synthetic_conditional_rates,
                self.num_speculative_steps,
            )
        else:
            raise ValueError(
                f"Unknown rejection sample method: {self.rejection_sample_method}"
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
