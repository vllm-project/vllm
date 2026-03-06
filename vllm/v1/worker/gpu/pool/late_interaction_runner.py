# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_KEY,
    LATE_INTERACTION_MODE_SCORE_DOC,
    LATE_INTERACTION_QUERY_KEY,
    LATE_INTERACTION_QUERY_USES_KEY,
    compute_maxsim_score,
)


class LateInteractionRunner:
    """Worker-side state and postprocessing for late-interaction scoring."""

    def __init__(self) -> None:
        # query_key -> token embeddings for late-interaction scoring.
        self._query_cache: dict[str, torch.Tensor] = {}
        # query_key -> remaining number of docs that should use this query.
        self._query_uses: dict[str, int] = {}
        # doc request id -> query key.
        self._doc_query_keys: dict[str, str] = {}

    def clear(self) -> None:
        self._query_cache.clear()
        self._query_uses.clear()
        self._doc_query_keys.clear()

    def register_request(
        self, req_id: str, pooling_params: PoolingParams | None
    ) -> None:
        mode, query_key, _ = self._parse_late_interaction_meta(pooling_params)
        if mode == LATE_INTERACTION_MODE_SCORE_DOC and query_key is not None:
            self._doc_query_keys[req_id] = query_key
        else:
            self._doc_query_keys.pop(req_id, None)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        for req_id in finished_req_ids:
            query_key = self._doc_query_keys.pop(req_id, None)
            if query_key is not None:
                self._release_query_use(query_key)

    def postprocess_pooler_output(
        self,
        raw_pooler_output: PoolerOutput,
        pooling_params: list[PoolingParams],
        req_ids: list[str],
        finished_mask: list[bool],
    ) -> PoolerOutput:
        if not isinstance(raw_pooler_output, list):
            return raw_pooler_output

        num_reqs = len(pooling_params)
        if len(raw_pooler_output) != num_reqs:
            raise ValueError(
                "raw_pooler_output and pooling_params must have the same length."
            )
        if len(req_ids) != num_reqs:
            raise ValueError("req_ids and pooling_params must have the same length.")
        if len(finished_mask) != num_reqs:
            raise ValueError(
                "finished_mask and pooling_params must have the same length."
            )

        if not any(finished_mask):
            return raw_pooler_output
        if not any(
            p.extra_kwargs and LATE_INTERACTION_MODE_KEY in p.extra_kwargs
            for p in pooling_params
        ):
            return raw_pooler_output

        outputs: list[torch.Tensor | None] = list(raw_pooler_output)
        for i, (req_id, output, params, finished) in enumerate(
            zip(req_ids, outputs, pooling_params, finished_mask)
        ):
            if not finished or output is None:
                continue

            mode, query_key, query_uses = self._parse_late_interaction_meta(params)
            if mode is None:
                continue

            assert query_key is not None
            if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
                assert query_uses is not None
                # `output` can be a view into the current step's hidden-states
                # buffer, so clone it before storing across scheduling steps.
                self._query_cache[query_key] = output.clone()
                self._query_uses[query_key] = query_uses
                outputs[i] = torch.zeros((), device=output.device, dtype=torch.float32)
                continue

            if mode == LATE_INTERACTION_MODE_SCORE_DOC:
                query_output = self._query_cache.get(query_key)
                if query_output is None:
                    raise ValueError(
                        "late-interaction query cache miss for key "
                        f"{query_key!r}. Ensure query requests are executed "
                        "before their paired document requests."
                    )

                outputs[i] = compute_maxsim_score(query_output, output)
                self._doc_query_keys.pop(req_id, None)
                self._release_query_use(query_key)
                continue

            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        return outputs

    def _release_query_use(self, query_key: str) -> None:
        remaining = self._query_uses.get(query_key, 1) - 1
        if remaining <= 0:
            self._query_uses.pop(query_key, None)
            self._query_cache.pop(query_key, None)
        else:
            self._query_uses[query_key] = remaining

    @staticmethod
    def _parse_late_interaction_meta(
        pooling_params: PoolingParams | None,
    ) -> tuple[str | None, str | None, int | None]:
        if pooling_params is None:
            return None, None, None

        extra_kwargs = pooling_params.extra_kwargs or {}
        mode = extra_kwargs.get(LATE_INTERACTION_MODE_KEY)
        if mode is None:
            return None, None, None

        query_key = extra_kwargs.get(LATE_INTERACTION_QUERY_KEY)
        if not isinstance(query_key, str) or not query_key:
            raise ValueError(
                "late-interaction request is missing a valid query key in "
                "pooling_params.extra_kwargs."
            )

        if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
            query_uses_raw = extra_kwargs.get(LATE_INTERACTION_QUERY_USES_KEY, 1)
            try:
                query_uses = max(1, int(query_uses_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "late-interaction query uses must be an integer value."
                ) from exc
            return mode, query_key, query_uses

        return mode, query_key, None
