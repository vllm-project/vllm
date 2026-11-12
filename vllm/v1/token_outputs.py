# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token outputs: per-request tensors with one row per generated token.

One channel carries them from the model runner to the caller:
`ModelRunnerOutput.token_outputs` (entry -> req_id -> rows),
`EngineCoreOutput.new_token_outputs` (entry -> rows) and `RequestState`, which
joins the chunks into the `CompletionOutput` field of the same name.

An entry is allowed by an engine flag that is off by default, asked for per
request on `SamplingParams`, and computed from the rows of the hidden states
that `compute_logits` receives. With every flag off the runner holds no state
and does no work; with one on, only the requests that ask pay. The last hidden
states are the first entry.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch

LAST_HIDDEN_STATES = "last_hidden_states"


def allowed(model_config: "ModelConfig") -> list[str]:
    """The entries the engine's flags allow."""
    return [LAST_HIDDEN_STATES] if model_config.enable_return_last_hidden_states else []


def requested(sampling_params: "SamplingParams") -> list[str]:
    """The entries a request asks for."""
    return [LAST_HIDDEN_STATES] if sampling_params.return_last_hidden_states else []


class TokenOutputs:
    """Worker side: which requests asked for which entry, and their rows."""

    def __init__(self, names: Iterable[str]) -> None:
        self._req_ids: dict[str, set[str]] = {name: set() for name in names}

    def add_request(self, req_id: str, sampling_params: "SamplingParams") -> None:
        for name in requested(sampling_params):
            self._req_ids[name].add(req_id)

    def remove_request(self, req_id: str) -> None:
        for req_ids in self._req_ids.values():
            req_ids.discard(req_id)

    def gather(
        self,
        hidden_states: torch.Tensor,
        input_batch: "InputBatch",
        compute: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]] | None:
        """Entry -> req_id -> [num_logits, width]: the rows of `hidden_states`
        that `sample` passed to `compute_logits`, for the scheduled requests
        that asked for the entry, mapped by `compute[entry]` when there is one
        (the identity otherwise). None when no scheduled request asked. Read
        before the next step can overwrite `hidden_states`."""
        if not any(self._req_ids.values()):
            return None
        cu_num_logits = input_batch.cu_num_logits_np
        out: dict[str, dict[str, torch.Tensor]] = {}
        for name, req_ids in self._req_ids.items():
            asking = [
                (req_id, int(cu_num_logits[i]), int(cu_num_logits[i + 1]))
                for i, req_id in enumerate(input_batch.req_ids)
                if req_id in req_ids and cu_num_logits[i + 1] > cu_num_logits[i]
            ]
            if not asking:
                continue
            rows = torch.cat([input_batch.logits_indices[a:b] for _, a, b in asking])
            values = hidden_states[rows]
            if compute is not None and name in compute:
                values = compute[name](values)
            out[name] = dict(
                zip(
                    (req_id for req_id, _, _ in asking),
                    torch.split(values, [b - a for _, a, b in asking]),
                )
            )
        return out or None
