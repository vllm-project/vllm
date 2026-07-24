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
            self.sampling_mask_sparse_token_ids: np.ndarray | None = None
            self.sampling_mask_sparse_row_indices: list[int] | None = None
            self.sampling_mask_packed: np.ndarray | None = None
            self.sampling_mask_packed_row_indices: list[int] | None = None
            self.sampling_mask_counts: np.ndarray | None = None
            self.sampling_mask_vocab_size: int | None = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_sparse_token_ids = async_copy_to_np(
                    sampler_output.sampling_mask_tensors.sparse_token_ids
                )
                self.sampling_mask_sparse_row_indices = (
                    sampler_output.sampling_mask_tensors.sparse_row_indices
                )
                self.sampling_mask_packed = async_copy_to_np(
                    sampler_output.sampling_mask_tensors.packed_mask
                )
                self.sampling_mask_packed_row_indices = (
                    sampler_output.sampling_mask_tensors.packed_row_indices
                )
                self.sampling_mask_counts = async_copy_to_np(
                    sampler_output.sampling_mask_tensors.counts
                )
                self.sampling_mask_vocab_size = (
                    sampler_output.sampling_mask_tensors.vocab_size
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

        if self.sampling_mask_packed is not None:
            assert self.sampling_mask_sparse_token_ids is not None
            assert self.sampling_mask_sparse_row_indices is not None
            assert self.sampling_mask_packed_row_indices is not None
            assert self.sampling_mask_counts is not None
            assert self.sampling_mask_vocab_size is not None
            self.model_runner_output.sampling_masks = _build_sampling_mask_lists(
                sparse_token_ids=self.sampling_mask_sparse_token_ids,
                sparse_row_indices=self.sampling_mask_sparse_row_indices,
                packed_mask=self.sampling_mask_packed,
                packed_row_indices=self.sampling_mask_packed_row_indices,
                counts=self.sampling_mask_counts,
                vocab_size=self.sampling_mask_vocab_size,
                num_sampled_tokens=num_sampled_tokens,
                sampled_token_ids=sampled_token_ids,
                req_ids=self.model_runner_output.req_ids,
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
    sparse_token_ids: np.ndarray,
    sparse_row_indices: list[int],
    packed_mask: np.ndarray,
    packed_row_indices: list[int],
    counts: np.ndarray,
    vocab_size: int,
    num_sampled_tokens: list[int],
    sampled_token_ids: list[list[int]],
    req_ids: list[str],
) -> SamplingMaskLists:
    num_reqs = len(req_ids)
    packed_width = (vocab_size + 7) // 8
    if (
        sparse_token_ids.ndim != 2
        or sparse_token_ids.shape[0] != len(sparse_row_indices)
        or sparse_token_ids.dtype != np.int32
        or packed_mask.shape != (len(packed_row_indices), packed_width)
        or packed_mask.dtype != np.uint8
        or counts.shape != (num_reqs,)
        or counts.dtype != np.int32
        or len(num_sampled_tokens) != num_reqs
        or len(sampled_token_ids) != num_reqs
    ):
        raise RuntimeError("sampling mask tensors are not aligned with requests")

    row_sources: list[tuple[bool, int] | None] = [None] * num_reqs
    for is_sparse, row_indices in (
        (True, sparse_row_indices),
        (False, packed_row_indices),
    ):
        for source_idx, req_idx in enumerate(row_indices):
            if req_idx < 0 or req_idx >= num_reqs or row_sources[req_idx] is not None:
                raise RuntimeError("sampling mask row indices are invalid")
            row_sources[req_idx] = (is_sparse, source_idx)
    if any(source is None for source in row_sources):
        raise RuntimeError("sampling mask row indices do not cover every request")

    generated_indices: list[int] = []
    generated_rows: list[np.ndarray] = []
    cu_num_generated_tokens = [0]
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
            if count > vocab_size:
                raise RuntimeError(
                    f"sampling mask for request {req_id} has invalid count "
                    f"{count} for vocab size {vocab_size}"
                )
            if count <= 0:
                raise RuntimeError(
                    f"sampling mask for request {req_id} has an empty kept set"
                )

            source = row_sources[req_idx]
            assert source is not None
            is_sparse, source_idx = source
            if is_sparse:
                if count > sparse_token_ids.shape[1]:
                    raise RuntimeError(
                        f"sampling mask for request {req_id} has count {count}, "
                        f"but the sparse row width is {sparse_token_ids.shape[1]}"
                    )
                kept_ids = sparse_token_ids[source_idx, :count]
                if np.any((kept_ids < 0) | (kept_ids >= vocab_size)):
                    raise RuntimeError(
                        f"sampling mask for request {req_id} contains an invalid "
                        "token ID"
                    )
            else:
                unpacked = np.unpackbits(
                    packed_mask[source_idx], count=vocab_size, bitorder="little"
                )
                kept_ids = np.flatnonzero(unpacked).astype(np.int32, copy=False)
                if len(kept_ids) != count:
                    raise RuntimeError(
                        f"sampling mask for request {req_id} has count {count}, "
                        f"but the packed mask contains {len(kept_ids)} token IDs"
                    )

            if not np.any(kept_ids == sampled_ids[0]):
                raise RuntimeError(
                    f"sampled token {sampled_ids[0]} is absent from the sampling "
                    f"mask for request {req_id}"
                )
            generated_indices.append(req_idx)
            generated_rows.append(kept_ids)
        cu_num_generated_tokens.append(len(generated_indices))

    row_width = max((len(row) for row in generated_rows), default=0)
    token_ids = np.full((len(generated_rows), row_width), -1, dtype=np.int32)
    for row_idx, kept_ids in enumerate(generated_rows):
        token_ids[row_idx, : len(kept_ids)] = kept_ids

    return SamplingMaskLists(
        token_ids,
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
