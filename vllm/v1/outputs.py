# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    KVConnectorStats = object


class LogprobsLists(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


class LogprobsTensors(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
        )

    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"
        )
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(
            num_positions, dtype=torch.int32, device="cpu"
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
PoolerOutput = torch.Tensor | list[torch.Tensor]


@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.invalid_block_ids
        )


@dataclass
class ECConnectorOutput:
    # [mm_hash]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]]

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None]

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None


# ModelRunnerOutput wrapper for async scheduling.
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass


@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """
    Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {rid: idx for idx, rid in enumerate(req_ids)}

    # No tokens generated yet ⇒ one empty list per request
    sampled_token_ids: list[list[int]] = [[0] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=pooler_output,
        kv_connector_output=None,
        ec_connector_output=None,
        num_nans_in_logits=None,
    )


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
    req_ids=[],
    req_id_to_index={},
    sampled_token_ids=[],
    logprobs=None,
    prompt_logprobs_dict={},
    pooler_output=[],
    num_nans_in_logits=None,
)
