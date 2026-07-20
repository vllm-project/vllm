# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState


class SimulatedSampler:
    def __init__(self, req_states: RequestState) -> None:
        self.req_states = req_states
        self.output_token_ids: dict[int, tuple[int, ...]] = {}

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        output_token_ids = (sampling_params.extra_args or {}).get(
            "simulated_output_token_ids"
        )
        if not isinstance(output_token_ids, list) or any(
            type(token_id) is not int for token_id in output_token_ids
        ):
            raise ValueError("simulated_output_token_ids must be a list of integers.")

        output_token_ids = tuple(output_token_ids)
        if sampling_params.eos_token_id is not None:
            output_token_ids += (sampling_params.eos_token_id,)

        self.output_token_ids[req_idx] = output_token_ids

    def sample(self, input_batch: InputBatch) -> SamplerOutput:
        seq_lens = input_batch.seq_lens[: input_batch.num_reqs]
        prefill_lens = torch.from_numpy(input_batch.prefill_len_np)
        num_sampled = (seq_lens >= prefill_lens).to(torch.int32)
        num_rejected = torch.zeros_like(num_sampled)
        sampled_token_ids = torch.zeros(
            (input_batch.num_reqs, 1),
            dtype=torch.int64,
            device=input_batch.seq_lens.device,
        )

        for batch_idx, req_idx in enumerate(input_batch.idx_mapping_np):
            output_token_ids = self.output_token_ids[req_idx]
            num_generated = int(
                self.req_states.total_len.gpu[req_idx]
                - self.req_states.prompt_len.np[req_idx]
            )
            if num_generated < len(output_token_ids):
                sampled_token_ids[batch_idx, 0] = output_token_ids[num_generated]

        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
