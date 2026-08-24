# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped state for generative recommendation models."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.outputs import LogprobsTensors


@dataclass(frozen=True)
class GRRequestState:
    uids: tuple[Any, ...]
    candidate_num: int
    prompt_token_ids: tuple[int, ...]
    request_stage: int


class GRState:
    def __init__(self) -> None:
        self.requests: dict[str, GRRequestState] = {}

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: Any | None,
    ) -> None:
        extra_args = (
            sampling_params.extra_args if sampling_params is not None else None
        )
        gr_params = (extra_args or {}).get("gr_params")
        if gr_params is None:
            self.remove_request(req_id)
            return
        if not isinstance(gr_params, dict):
            raise ValueError(
                "Generative recommendation requests require "
                "sampling_params.extra_args['gr_params']."
            )
        if sampling_params is None or sampling_params.prompt_logprobs is None:
            raise ValueError(
                "Generative recommendation requests must set prompt_logprobs "
                "to receive candidate scores."
            )
        if len(prompt_token_ids) < 2:
            raise ValueError(
                "Generative recommendation prompts must contain at least two "
                "tokens (one context/candidate pair)."
            )

        uids = gr_params.get("uid")
        candidate_num = gr_params.get("candidate_num")
        request_stage = gr_params.get("request_stage", 2)
        if request_stage not in (2):
            raise NotImplementedError(
                "Generative recommendation currently supports request_stage "
                "2 only; PD-separated stage 0 or 1 is not supported."
            )
        if not isinstance(uids, list) or not uids:
            raise ValueError("gr_params['uid'] must be a non-empty list.")
        if not (
            isinstance(candidate_num, list)
            and len(candidate_num) == 1
            and isinstance(candidate_num[0], int)
            and candidate_num[0] > 0
        ):
            raise ValueError(
                "gr_params['candidate_num'] must be a one-element list "
                "containing a positive integer."
            )

        self.requests[req_id] = GRRequestState(
            uids=tuple(uids),
            candidate_num=candidate_num[0],
            prompt_token_ids=tuple(prompt_token_ids),
            request_stage=request_stage,
        )

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def _get_candidate_scores_dict(
        self,
        model_output: torch.Tensor,
        input_batch: Any,
    ) -> dict[str, LogprobsTensors]:
        """Convert packed HSTU scores into prompt-logprobs tensors."""
        scores_dict: dict[str, LogprobsTensors] = {}
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            request = self.requests.get(req_id)
            if request is None:
                continue

            if int(input_batch.num_computed_tokens_np[batch_idx]) != 0:
                raise ValueError(
                    "Generative recommendation currently requires unchunked "
                    "prefill. Chunked prefill needs cross-step candidate score "
                    "accumulation."
                )

            request_tokens = int(input_batch.num_scheduled_tokens[batch_idx])
            if request_tokens % len(request.uids) != 0:
                raise ValueError(
                    f"GR request {req_id!r} has {request_tokens} scheduled tokens "
                    f"for {len(request.uids)} uids."
                )
            tokens_per_uid = request_tokens // len(request.uids)
            if request.candidate_num > tokens_per_uid:
                raise ValueError(
                    f"GR request {req_id!r} candidate_num="
                    f"{request.candidate_num}, which exceeds tokens per uid "
                    f"({tokens_per_uid})."
                )

            request_start = int(input_batch.query_start_loc_np[batch_idx])
            token_ids: list[torch.Tensor] = []
            score_chunks: list[torch.Tensor] = []
            for uid_idx in range(len(request.uids)):
                uid_end = (uid_idx + 1) * tokens_per_uid
                candidate_start = uid_end - request.candidate_num
                token_ids.append(
                    torch.as_tensor(
                        request.prompt_token_ids[candidate_start:uid_end],
                        dtype=torch.int32,
                        device=model_output.device,
                    )
                )
                score_chunks.append(
                    model_output[
                        request_start + candidate_start : request_start + uid_end
                    ]
                )

            scores = torch.cat(score_chunks)
            scores_dict[req_id] = LogprobsTensors(
                logprob_token_ids=torch.cat(token_ids).unsqueeze(-1),
                logprobs=scores,
                selected_token_ranks=torch.zeros(
                    scores.shape[0], dtype=torch.int32, device=model_output.device
                ),
            )
        return scores_dict

    def collect_metadata(
        self, req_ids: list[str]
    ) -> list[dict[str, Any]] | None:
        metadata = [
            {
                "uid": list(request.uids),
                "candidate_num": [request.candidate_num],
                "request_stage": request.request_stage,
            }
            if (request := self.requests.get(req_id)) is not None
            else {}
            for req_id in req_ids
        ]
        return metadata if any(metadata) else None
