# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class LogitsBatchState:
    """The persistent batch state visible to logits processors.

    Wraps the model runner's per-slot buffers, which are mutated in place,
    so reads always see current values. Processors must treat every field
    as read-only.
    """

    device: torch.device
    max_num_reqs: int
    vocab_size: int
    # [max_num_reqs, max_model_len] committed token ids per request slot.
    all_token_ids: StagedWriteTensor
    # [max_num_reqs] per-slot lengths; see RequestState for their meanings.
    prompt_len: UvaBackedTensor
    prefill_len: UvaBackedTensor
    total_len: StagedWriteTensor

    @classmethod
    def from_request_state(cls, req_states: RequestState) -> "LogitsBatchState":
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
    # [num_reqs] batch position -> persistent request slot, on the host, for
    # skipping work without a device sync.
    idx_mapping_np: np.ndarray
    # [num_logits_rows] row -> its offset among the rows of its own request.
    expanded_local_pos: torch.Tensor
    # [num_logits_rows] token fed to the model at each row's input position.
    input_ids: torch.Tensor
    # [num_logits_rows] position of each row within its sequence.
    pos: torch.Tensor


class LogitsProcessor(ABC):
    """Custom logits processor for Model Runner V2.

    Per-request state is keyed by the request slot index; slots are recycled
    through a free list, so per-slot state must be fully (re)initialized in
    ``add_request()``.

    ``apply()`` runs after the built-in bias, penalty, bad-words and grammar
    stages and before temperature, min_p and top-k/top-p, so it sees unscaled
    logits and must not re-inflate grammar-masked tokens.

    State that is constant for a request belongs in ``__init__()`` or
    ``add_request()``; ``apply()`` receives only what changes per step.
    """

    def __init__(  # noqa: B027
        self, vllm_config: "VllmConfig", state: LogitsBatchState
    ):
        """Capture what stays constant for the processor's lifetime.

        ``state`` exposes the on-device token history and batch constants a
        processor may read. Treat it as read-only.
        """

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        """Initialize per-slot state for a request entering the batch.

        The slot may hold a previous occupant's state; overwrite or
        neutralize all of it here.

        Returns whether this processor modifies logits for the request; the
        sampler ORs the returns into a per-request flag and skips the whole
        pipeline when no request needs it. Returning False gates admission
        only: when the pipeline runs, ``apply()`` still sees every row.
        """
        return True

    @abstractmethod
    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        """Modify logits in place or return a new tensor.

        ``apply()`` is called once for the whole batch, including rows of
        requests this processor declined in ``add_request()``, so filter rows
        via ``ctx.expanded_idx_mapping``.

        Args:
            logits: [num_logits_rows, vocab_size] float32 tensor.
            ctx: this step's batch layout.

        """
        raise NotImplementedError
