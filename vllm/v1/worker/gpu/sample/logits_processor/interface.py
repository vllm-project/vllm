# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 logits processor interface.

Import-light by design: custom processor classes are also loaded in the
frontend process (to validate per-request params), so this module must not
pull in model-runner side modules (torch, triton, worker state) at import
time. Heavy imports live under TYPE_CHECKING; annotations are deferred.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
    from vllm.v1.worker.gpu.states import RequestState


@dataclass(frozen=True)
class LogitsProcRequestState:
    """State associated with active requests, shared with logits processors.

    Wraps the model runner's per-slot buffers, which are mutated in place,
    so reads always see current values. Processors must treat every field
    as read-only.
    """

    device: torch.device
    max_num_reqs: int
    vocab_size: int

    # [max_num_reqs, max_model_len] committed token ids per request slot.
    all_token_ids: StagedWriteTensor
    # [max_num_reqs] tokens in the user-provided prompt.
    prompt_len: UvaBackedTensor
    # [max_num_reqs] tokens fed at the latest (re)fill: the prompt plus any
    # partial output on resumption after preemption.
    prefill_len: UvaBackedTensor
    # [max_num_reqs] prompt_len + output_len; grows as the request progresses.
    total_len: StagedWriteTensor

    @classmethod
    def from_request_state(cls, req_states: RequestState) -> LogitsProcRequestState:
        return cls(
            device=req_states.device,
            max_num_reqs=req_states.max_num_reqs,
            vocab_size=req_states.vocab_size,
            all_token_ids=req_states.all_token_ids,
            prompt_len=req_states.prompt_len,
            prefill_len=req_states.prefill_len,
            total_len=req_states.total_len,
        )


@dataclass(frozen=True)
class LogitsContext:
    """The current step's batch layout, passed to every ``apply()`` call.

    A row is a logits row, not a request: rows are reordered every step, and
    under speculative decoding a request owns one row per draft token.
    Committed tokens live in ``req_states.all_token_ids`` (valid up to
    ``total_len``); this step's draft tokens are only in ``input_ids``.
    """

    # [num_logits_rows] row -> persistent request slot.
    expanded_idx_mapping: torch.Tensor
    # [num_reqs] batch position -> persistent request slot.
    idx_mapping: torch.Tensor
    # [num_reqs] batch position -> persistent request slot, on the host, for
    # skipping work without a device sync.
    idx_mapping_np: np.ndarray
    # [num_logits_rows] row -> its offset among the rows of its own request.
    expanded_local_pos: torch.Tensor
    # [num_logits_rows] token fed to the model at each row's input position.
    input_ids: torch.Tensor
    # [num_logits_rows] position of each row within its sequence.
    pos: torch.Tensor
    # [num_reqs] batch position -> upper bound of tokens visible to the token
    # being sampled this step, on the host. Exact when spec decoding isn't in use.
    # Exact per-row lengths on device are `pos + 1`.
    seq_lens_upper_bound_np: np.ndarray


class LogitsProcessor(ABC):
    """Custom logits processor for Model Runner V2.

    Per-request state is keyed by the request slot index; slots are recycled
    through a free list, so per-slot state must be fully (re)initialized in
    ``add_request()``.

    ``apply()`` runs after the built-in bias, penalty, bad-words and grammar
    stages and before temperature, min_p and top-k/top-p, so it sees unscaled
    logits and must not re-inflate grammar-masked tokens. Thinking-budget
    forcing runs after ``apply()`` and wins over its edits for requests whose
    budget is exhausted.

    State that is constant for a request belongs in ``__init__()`` or
    ``add_request()``; ``apply()`` receives only what changes per step.
    """

    def __init__(  # noqa: B027
        self, vllm_config: VllmConfig, req_states: LogitsProcRequestState
    ):
        """Capture what stays constant for the processor's lifetime.

        ``req_states`` exposes the on-device token history and batch constants a
        processor may read. Treat it as read-only.
        """

    @classmethod  # noqa: B027
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Raise ``ValueError`` for invalid per-request arguments.

        Runs at request admission, so invalid arguments fail the request
        with an error instead of reaching the sampler.
        """

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        """Initialize per-slot state for a request entering the batch.

        The slot may hold a previous occupant's state; overwrite or
        neutralize all of it here.

        Returns whether this processor modifies logits for the request.
        """
        return True

    def apply_staged_writes(self) -> None:  # noqa: B027
        """Flush any host-side writes staged by ``add_request()`` to the device.

        Called once per step before the forward pass, after the model runner
        has flushed ``req_states``, so a processor that stages writes here can
        read the request's tokens back on device.
        """

    @abstractmethod
    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        """Modify logits in place or return a new tensor. In-place modification
        is preferred for efficiency.

        ``apply()`` is called once for the whole batch, including rows of
        requests this processor declined in ``add_request()``, so filter rows
        via ``ctx.expanded_idx_mapping``.

        Args:
            logits: [num_logits_rows, vocab_size] float32 tensor.
            ctx: this step's batch layout.

        """
        raise NotImplementedError
