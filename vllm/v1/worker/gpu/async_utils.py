# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager

import numpy as np
import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    SamplerOutput,
)


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: np.ndarray,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_stream = copy_stream
        self.copy_event = torch.cuda.Event()

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(default_stream)

            # NOTE(woosuk): We should keep the CPU tensors unfreed, until the copy completes.
            self.sampled_token_ids = sampler_output.sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            else:
                self.logprobs_tensors = None
            self.prompt_logprobs_dict = {}
            if self.model_runner_output.prompt_logprobs_dict:
                for k, v in self.model_runner_output.prompt_logprobs_dict.items():
                    self.prompt_logprobs_dict[k] = v.to_cpu_nonblocking()
            self.copy_event.record(self.copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code ensures compatibility with OSS vLLM.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids_np = self.sampled_token_ids.numpy()
        sampled_token_ids = sampled_token_ids_np.tolist()
        for i, tokens in enumerate(sampled_token_ids):
            del tokens[self.num_sampled_tokens[i] :]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


@contextmanager
def async_barrier(event: torch.cuda.Event | None):
    if event is not None:
        event.synchronize()
    try:
        yield
    finally:
        if event is not None:
            event.record()
