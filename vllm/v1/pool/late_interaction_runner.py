# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC,
    compute_maxsim_score_batched,
)

logger = init_logger(__name__)


class LateInteractionRunner:
    """Worker-side state and postprocessing for late-interaction scoring."""

    def __init__(self, enable_flash: bool = True) -> None:
        # query_key -> token embeddings for late-interaction scoring.
        self._query_cache: dict[str, torch.Tensor] = {}
        # query_key -> remaining number of docs that should use this query.
        self._query_uses: dict[str, int] = {}
        # doc request id -> query key.
        self._doc_query_keys: dict[str, str] = {}
        # Fused Triton scoring (PoolerConfig.enable_flash_late_interaction
        # gates it; a runtime kernel failure disables it for the process).
        if enable_flash:
            try:
                from vllm.v1.pool.flash_maxsim import (  # noqa: F401
                    flash_maxsim_rerank_direct,
                )
            except ImportError:
                enable_flash = False
        self._flash_enabled = enable_flash

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
        if not any(p.late_interaction_params is not None for p in pooling_params):
            return raw_pooler_output

        outputs: list[torch.Tensor | None] = list(raw_pooler_output)
        score_indices: list[int] = []
        score_req_ids: list[str] = []
        score_query_keys: list[str] = []
        score_queries: list[torch.Tensor] = []
        score_docs: list[torch.Tensor] = []
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

                score_indices.append(i)
                score_req_ids.append(req_id)
                score_query_keys.append(query_key)
                score_queries.append(query_output)
                score_docs.append(output)
                continue

            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        if score_indices:
            score_values = self._score(score_queries, score_docs)
            for i, req_id, query_key, score in zip(
                score_indices, score_req_ids, score_query_keys, score_values
            ):
                outputs[i] = score
                self._doc_query_keys.pop(req_id, None)
                self._release_query_use(query_key)

        return outputs

    def _score(
        self,
        queries: list[torch.Tensor],
        docs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Score (query_i, doc_i) pairs: the fused Triton kernel handles the
        common rerank pattern (one shared query, CUDA inputs); multiple
        distinct queries, CPU tensors, or a kernel failure fall back to the
        reference scorer."""
        if self._flash_enabled and docs and queries[0].is_cuda:
            first = queries[0]
            if all(q is first for q in queries):
                try:
                    return self._score_flash_shared_query(first, docs)
                except Exception as exc:
                    # A persistent compile/launch failure must not take down
                    # requests: serve through the reference scorer and stop
                    # trying the kernel for the rest of the process.
                    self._flash_enabled = False
                    logger.warning(
                        "flash-maxsim scoring failed (%s); falling back to "
                        "the reference MaxSim path for this process.",
                        exc,
                    )
        return compute_maxsim_score_batched(queries, docs)

    @staticmethod
    def _score_flash_shared_query(
        query: torch.Tensor,
        docs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """One kernel launch for all docs against a single shared query:
        docs pack into one [total_tokens, d] tensor (single cat, no padding)
        and the kernel reads each by (offset, length) — the [B, Lq, Ld]
        similarity tensor is never materialised."""
        from vllm.v1.pool.flash_maxsim import flash_maxsim_rerank_direct

        device = docs[0].device
        lengths = [int(d.shape[0]) for d in docs]
        offsets = [0] * len(lengths)
        for i in range(1, len(lengths)):
            offsets[i] = offsets[i - 1] + lengths[i - 1]
        packed = docs[0] if len(docs) == 1 else torch.cat(docs, dim=0)
        scores = flash_maxsim_rerank_direct(
            query,
            packed,
            torch.tensor(offsets, device=device, dtype=torch.int32),
            torch.tensor(lengths, device=device, dtype=torch.int32),
            max(lengths),
        )
        return list(scores.unbind(0))

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
        if pooling_params is None or pooling_params.late_interaction_params is None:
            return None, None, None

        late_interaction_params = pooling_params.late_interaction_params
        mode = late_interaction_params.mode

        query_key = late_interaction_params.query_key
        if not isinstance(query_key, str) or not query_key:
            raise ValueError(
                "late-interaction request is missing a valid query key in "
                "pooling_params.late_interaction_params."
            )

        if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
            query_uses_raw = late_interaction_params.query_uses
            if query_uses_raw is None:
                query_uses_raw = 1
            try:
                query_uses = max(1, int(query_uses_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "late-interaction query uses must be an integer value."
                ) from exc
            return mode, query_key, query_uses

        return mode, query_key, None
