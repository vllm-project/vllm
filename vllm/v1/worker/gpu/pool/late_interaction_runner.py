# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
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
from vllm.v1.pool.metadata import PoolingCursor

logger = init_logger(__name__)


class LateInteractionRunner:
    """Worker-side state and postprocessing for late-interaction scoring."""

    def __init__(self) -> None:
        # query_key -> token embeddings for late-interaction scoring.
        self._query_cache: dict[str, torch.Tensor] = {}
        # query_key -> remaining number of docs that should use this query.
        self._query_uses: dict[str, int] = {}
        # doc request id -> query key.
        self._doc_query_keys: dict[str, str] = {}
        # Check if flash_maxsim_rerank_direct is available
        try:
            from vllm.v1.pool.flash_maxsim import (  # noqa: F401
                flash_maxsim_rerank_direct,
            )
            self._has_flash_maxsim_rerank = True
        except ImportError:
            self._has_flash_maxsim_rerank = False

        # Warm up Triton kernels to avoid compilation on first request
        self._warmup_kernels()

    def _warmup_kernels(self) -> None:
        """Pre-compile Triton kernels for all expected autotune buckets.

        Kernels and their autotune keys:
          flash_maxsim         : key=[Lq, d_pad]  (Lq not bucketed!)
          flash_maxsim_packed  : key=[Lq_bucket, d_pad]
          flash_maxsim_rerank_direct: key=[Lq_bucket, max_Ld_bucket, d_pad]

        Doc lengths and query lengths both bucket into
        {32, 64, 128, 256, 512, 1024, 2048, 4096}.  We pre-compile every
        bucket a realistic workload will hit so runtime never triggers
        autotune+benchmark (which can add seconds to first-request
        latency in each new bucket).
        """
        if not torch.cuda.is_available():
            return
        try:
            from vllm.v1.pool.flash_maxsim import (
                flash_maxsim,
                flash_maxsim_rerank_direct,
            )
            from vllm.v1.pool.flash_maxsim.flash_maxsim_varlen import (
                flash_maxsim_packed,
                pack_docs,
            )
            device = torch.cuda.current_device()
            # d=128 covers ColBERT (128) and ColPali (128).  Extend via
            # env var if running models with other embedding dims.
            d = int(os.environ.get("VLLM_FLASH_MAXSIM_WARMUP_D", "128"))

            # Lq buckets: cover typical query lengths (short questions to
            # long multi-sentence queries).
            lq_buckets = [32, 64, 128, 256]
            # Ld buckets: cover text docs (32-512) and image tokens
            # (ColPali ~1030).
            ld_buckets = [32, 64, 128, 256, 512, 1024]

            # --- flash_maxsim (B, Lq, Ld, d) — Lq NOT bucketed ---
            for Lq in lq_buckets:
                Q = torch.randn(Lq, d, device=device, dtype=torch.float16)
                for Ld in ld_buckets:
                    D = torch.randn(64, Ld, d, device=device,
                                    dtype=torch.float16)
                    flash_maxsim(Q, D)
                    del D
                del Q

            # --- flash_maxsim_rerank_direct (Lq × Ld buckets) ---
            for Lq in lq_buckets:
                Q = torch.randn(Lq, d, device=device, dtype=torch.float16)
                for Ld in ld_buckets:
                    batch = torch.randn(64 * Ld, d, device=device,
                                        dtype=torch.float16)
                    offs = torch.arange(0, 64 * Ld, Ld, device=device,
                                        dtype=torch.int32)
                    lens = torch.full((64,), Ld, device=device,
                                      dtype=torch.int32)
                    flash_maxsim_rerank_direct(Q, batch, offs, lens, Ld)
                    del batch, offs, lens
                del Q

            # --- flash_maxsim_packed (non-zerocopy fallback) ---
            for Lq in lq_buckets:
                Q = torch.randn(Lq, d, device=device, dtype=torch.float16)
                for Ld in ld_buckets:
                    docs = [torch.randn(Ld, d, device=device,
                                        dtype=torch.float16)
                            for _ in range(8)]
                    D_packed, cu_seqlens, max_ld = pack_docs(docs)
                    flash_maxsim_packed(Q, D_packed, cu_seqlens, max_ld)
                    del docs, D_packed, cu_seqlens
                del Q
        except Exception as e:
            # Surface the problem so first-request latency slowdown is
            # not silently swallowed.  Zero-copy path will still work —
            # it'll just trigger autotune on first real call.
            logger.warning("flash-maxsim kernel warmup failed: %s", e)
        finally:
            # Free any partial allocations from an interrupted warmup.
            if torch.cuda.is_available():
                torch.accelerator.empty_cache()

    @property
    def has_pending_docs(self) -> bool:
        """True when there are doc-scoring requests in flight."""
        return bool(self._doc_query_keys)

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
        projected_batch: torch.Tensor | None = None,
        pooling_cursor: PoolingCursor | None = None,
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

        # Can we use zero-copy scoring?
        use_zerocopy = (projected_batch is not None
                        and pooling_cursor is not None
                        and self._has_flash_maxsim_rerank)

        # Single GPU->CPU sync for offsets/lengths (avoids per-request
        # .item() calls which each trigger a sync).
        firsts_cpu: list[int] = []
        lasts_cpu: list[int] = []
        if use_zerocopy:
            assert pooling_cursor is not None  # narrowed by use_zerocopy
            firsts_cpu = pooling_cursor.first_token_indices_gpu.tolist()
            lasts_cpu = pooling_cursor.last_token_indices_gpu.tolist()

        outputs: list[torch.Tensor | None] = list(raw_pooler_output)
        score_indices: list[int] = []
        score_req_ids: list[str] = []
        score_query_keys: list[str] = []
        score_queries: list[torch.Tensor] = []
        # For zero-copy: collect (offset, length) instead of tensor copies
        score_doc_offsets: list[int] = []
        score_doc_lengths: list[int] = []
        # Fallback: collect tensor copies
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

                if use_zerocopy:
                    # Collect offset/length into projected batch — no copy
                    first = firsts_cpu[i]
                    last = lasts_cpu[i]
                    score_doc_offsets.append(first)
                    score_doc_lengths.append(last - first + 1)
                else:
                    score_docs.append(output)
                continue

            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        if score_indices:
            if use_zerocopy:
                assert projected_batch is not None  # narrowed by use_zerocopy
                score_values = self._score_zerocopy(
                    score_queries, projected_batch,
                    score_doc_offsets, score_doc_lengths,
                )
            else:
                score_values = compute_maxsim_score_batched(
                    score_queries, score_docs,
                )

            for i, req_id, query_key, score in zip(
                score_indices, score_req_ids, score_query_keys, score_values
            ):
                outputs[i] = score
                self._doc_query_keys.pop(req_id, None)
                self._release_query_use(query_key)

        return outputs

    @staticmethod
    def _score_zerocopy(
        queries: list[torch.Tensor],
        projected_batch: torch.Tensor,
        doc_offsets: list[int],
        doc_lengths: list[int],
    ) -> list[torch.Tensor]:
        """Score queries against docs using flash_maxsim_rerank_direct.

        Reads doc embeddings directly from projected_batch — zero copy.
        """
        from vllm.v1.pool.flash_maxsim import flash_maxsim_rerank_direct

        device = projected_batch.device

        # Group by query identity (same cached tensor = same data_ptr)
        groups: dict[
            int, tuple[torch.Tensor, list[int], list[int], list[int]]
        ] = {}
        for i, q in enumerate(queries):
            key = q.data_ptr()
            if key not in groups:
                groups[key] = (q, [], [], [])
            groups[key][1].append(doc_offsets[i])
            groups[key][2].append(doc_lengths[i])
            groups[key][3].append(i)

        results: list[torch.Tensor | None] = [None] * len(queries)

        for _, (query, offsets, lengths, result_indices) in groups.items():
            off_t = torch.tensor(offsets, device=device, dtype=torch.int32)
            len_t = torch.tensor(lengths, device=device, dtype=torch.int32)
            max_ld = max(lengths)

            scores = flash_maxsim_rerank_direct(
                query, projected_batch, off_t, len_t, max_ld
            )

            for j, idx in enumerate(result_indices):
                results[idx] = scores[j]

        return results  # type: ignore[return-value]

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
