# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

from vllm.v1.core.sched.output import SchedulerOutput


class LogprobsLists(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: list[list[int]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: list[list[float]]
    # [num_reqs]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty_cpu(num_positions: int,
                  num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions,
                                           dtype=torch.int32,
                                           device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass
class SamplerOutput:

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Optional[LogprobsTensors]


@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None


@dataclass
class ECConnectorOutput:
    # [mm_hash]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None


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
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

    # [num_reqs, hidden_size]
    pooler_output: list[Optional[torch.Tensor]]

    kv_connector_output: Optional[KVConnectorOutput] = None

    ec_connector_output: Optional[ECConnectorOutput] = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: Optional[dict[str, int]] = None


@dataclass
class DraftTokenIds:

    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput", ) -> ModelRunnerOutput:
    """
    Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {
        rid: idx
        for idx, rid in enumerate(req_ids)
    }

    # No tokens generated yet ⇒ one empty list per request
    sampled_token_ids: list[list[int]] = [[0] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[Optional[torch.Tensor]] = [None for _ in req_ids]

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


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[],
                                              req_id_to_index={},
                                              sampled_token_ids=[],
                                              logprobs=None,
                                              prompt_logprobs_dict={},
                                              pooler_output=[],
                                              num_nans_in_logits=None)
