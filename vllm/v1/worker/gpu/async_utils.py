# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import numpy as np
import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplingMaskLists,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.sampling_mask_token_ids: np.ndarray | None = None
            self.sampling_mask_counts: np.ndarray | None = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_token_ids = async_copy_to_np(
                    sampler_output.sampling_mask_tensors.token_ids
                )
                self.sampling_mask_counts = async_copy_to_np(
                    sampler_output.sampling_mask_tensors.counts
                )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampling_mask_token_ids is not None:
            assert self.sampling_mask_counts is not None
            self.model_runner_output.sampling_masks = _build_sampling_mask_lists(
                self.sampling_mask_token_ids,
                self.sampling_mask_counts,
                num_sampled_tokens,
                sampled_token_ids,
                self.model_runner_output.req_ids,
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


def _build_sampling_mask_lists(
    token_ids: np.ndarray,
    counts: np.ndarray,
    num_sampled_tokens: list[int],
    sampled_token_ids: list[list[int]],
    req_ids: list[str],
) -> SamplingMaskLists:
    num_reqs = len(req_ids)
    if (
        token_ids.ndim != 2
        or counts.shape != (num_reqs,)
        or token_ids.shape[0] != num_reqs
        or len(num_sampled_tokens) != num_reqs
        or len(sampled_token_ids) != num_reqs
    ):
        raise RuntimeError("sampling mask tensors are not aligned with requests")

    generated_indices: list[int] = []
    cu_num_generated_tokens = [0]
    row_width = token_ids.shape[1]
    for req_idx, (req_id, num_tokens, sampled_ids) in enumerate(
        zip(req_ids, num_sampled_tokens, sampled_token_ids)
    ):
        if num_tokens not in (0, 1):
            raise RuntimeError(
                "sampling distribution replay does not support multi-token "
                f"output for request {req_id}: {num_tokens} tokens"
            )
        if len(sampled_ids) != num_tokens:
            raise RuntimeError(
                f"sampled tokens are misaligned for request {req_id}: "
                f"expected {num_tokens}, got {len(sampled_ids)}"
            )
        if num_tokens:
            count = int(counts[req_idx])
            if count > row_width:
                raise RuntimeError(
                    f"sampling mask for request {req_id} has invalid count "
                    f"{count} for row width {row_width}"
                )
            if count <= 0:
                raise RuntimeError(
                    f"sampling mask for request {req_id} has an empty kept set"
                )
            kept_ids = token_ids[req_idx, :count]
            if np.any(kept_ids < 0):
                raise RuntimeError(
                    f"sampling mask for request {req_id} contains invalid padding"
                )
            if sampled_ids[0] not in kept_ids:
                raise RuntimeError(
                    f"sampled token {sampled_ids[0]} is absent from the sampling "
                    f"mask for request {req_id}"
                )
            generated_indices.append(req_idx)
        cu_num_generated_tokens.append(len(generated_indices))

    return SamplingMaskLists(
        token_ids[generated_indices],
        counts[generated_indices],
        cu_num_generated_tokens,
    )


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
