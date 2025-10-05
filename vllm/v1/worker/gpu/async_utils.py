# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.outputs import (AsyncModelRunnerOutput, LogprobsTensors,
                             ModelRunnerOutput, SamplerOutput)


class AsyncOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.copy_stream = copy_stream
        self.copy_event = torch.cuda.Event()

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(default_stream)

            self.sampled_token_ids = sampler_output.sampled_token_ids.to(
                "cpu", non_blocking=True)
            x = sampler_output.logprobs_tensors
            if x is not None:
                self.logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=x.logprob_token_ids.to(
                        "cpu", non_blocking=True),
                    logprobs=x.logprobs.to("cpu", non_blocking=True),
                    selected_token_ranks=x.selected_token_ranks.to(
                        "cpu", non_blocking=True),
                )
            else:
                self.logprobs_tensors = None
            self.copy_event.record()

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()
        self.model_runner_output.sampled_token_ids = (
            self.sampled_token_ids.numpy())
        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = (
                self.logprobs_tensors.tolists())
        return self.model_runner_output
