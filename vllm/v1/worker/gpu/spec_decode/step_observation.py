# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The record a speculator receives after every sampled step."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode


@dataclass(frozen=True, slots=True)
class StepObservation:
    """What the model runner knows about a step once it has sampled.

    Passed to ``BaseSpeculator.observe_step`` once per step that ran the
    target forward and sampled, on the last pipeline-parallel rank: speculative
    steps, steps with no drafts (candidate budget 1 included) and prefill
    steps. Capture and profiling runs are not observed. Warm-up steps are,
    because they run as scheduled steps.

    Every field is something the runner already holds. Host values are ints
    and numpy views of the step's input arrays; they stay valid until the
    runner prepares the next step, so copy one to keep it. ``num_sampled`` and
    ``num_rejected`` stay on the device: reading them on the host needs a
    sync, so copy them asynchronously (``non_blocking`` into pinned memory,
    then wait on an event) instead.
    """

    # Observed steps before this one on this runner, from 0.
    step: int
    # Host clock (time.perf_counter_ns()) when the runner began the step.
    # GPU times are not recorded by the runner while serving: a consumer that
    # wants them records its own CUDA events (the drafter of this step has not
    # run yet when the step is observed).
    start_ns: int
    # Requests in the step, in batch row order.
    req_ids: list[str]
    num_reqs: int
    # Request rows the forward ran, CUDA graph padding included.
    num_reqs_padded: int
    # Query tokens scheduled (drafts to verify included).
    num_tokens: int
    # Query tokens the forward ran, CUDA graph padding included.
    num_tokens_padded: int
    cudagraph_mode: CUDAGraphMode
    # Query tokens per request when the batch was dispatched as uniform
    # decode, else None.
    uniform_token_count: int | None
    num_ubatches: int
    # Query tokens of requests that are still prefilling their prompt.
    num_prefill_tokens: int
    # Drafts the drafter proposes per request after this step: the candidate
    # budget minus one. 0 when the budget is 1 (``skip_drafts``) or no DP rank
    # can use drafts.
    num_spec_tokens: int
    # Host, per request: query tokens scheduled.
    num_scheduled_tokens: np.ndarray
    # Host, per request: tokens computed before this step.
    num_computed_tokens: np.ndarray
    # Host, per request: still prefilling its prompt.
    is_prefilling: np.ndarray
    # Host, per request: drafts verified this step (after the plan bound).
    # None when no request carries drafts.
    num_draft_tokens: np.ndarray | None
    # Device, per request: tokens this step committed (accepted drafts and
    # the bonus token) and scheduled query tokens it rejected.
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
