# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import bisect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.v1.spec_decode.verify_adaptive_config import VerifyAdaptiveConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Core algorithm — pure function, stateless, unit-testable independently.
# ---------------------------------------------------------------------------

def choose_query_lens_discrete(
    probs: "list[list[float]] | np.ndarray",
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
) -> dict[str, Any]:
    """Discrete marginal-gain scan over the *measured* sum_query_len levels.

    Since verifier cost depends only on ``(batch_size, sum_query_len)``, the
    candidate Q values are exactly the profiled sum_query_len levels for the
    fixed batch size (e.g. ``bs*2, bs*4, …``).  For each level Q we greedily
    fill the ``S = Q - base_batch_size`` highest marginal gains and score it as
    ``(base_batch_size + top_S_gain_sum) / cost_lookup(Q)``, keeping the best Q.

    Args:
        probs: per-active-sequence accept probs; ``probs[i][t]`` is the
            predicted accept prob of draft position ``t`` for sequence ``i``.
        base_batch_size: full verifier batch size B.  Every sequence always
            contributes one anchor token, so ``sum_query_len = B + S``.
        q_levels: candidate sum_query_len values; must be real cost-table keys.
        cost_lookup: ``Q -> verifier ITL cost`` (batch size already fixed).
        max_draft_len: max draft tokens per sequence (``max_query_len - 1``).
        collect_records: if True, also return per-level debug records.
    """
    A = len(probs)

    # Marginal gains m[i,t] = prod_{k<=t} p[i,k], vectorised over the batch.
    mat = np.asarray(probs, dtype=np.float64).reshape(A, -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)

    seq_ids = np.repeat(np.arange(A), gains.shape[1])
    flat_gains = gains.ravel()
    order = np.argsort(-flat_gains, kind="stable")
    sorted_seq = seq_ids[order]
    # prefix_gain[S] = sum of the top-S marginal gains.
    prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
    total_available = flat_gains.shape[0]

    best_score = -math.inf
    best_Q, best_S = base_batch_size, 0
    records: list[dict[str, Any]] | None = [] if collect_records else None

    for Q in q_levels:
        S = Q - base_batch_size
        if S < 0:
            continue
        S = min(S, total_available)
        cost = cost_lookup(Q)
        if cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[S]) / cost
        if records is not None:
            records.append({"Q": Q, "S": int(S), "score": score, "cost": cost})
        if score > best_score:
            best_score, best_Q, best_S = score, Q, S

    # Reconstruct per-sequence draft lengths from the top-best_S marginals.
    draft_lens = np.bincount(sorted_seq[:best_S], minlength=A).tolist()

    return {
        "draft_lens": draft_lens,
        "best_Q": best_Q,
        "best_S": int(best_S),
        "best_score": best_score,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class VerifyAdaptiveController:
    """Per-request draft-length selector for the verifier step.

    Call order: ``__init__`` → ``profile_cost_table`` (once, after CUDA
    graph capture / JIT warmup) → ``process_draft_output`` (each step) →
    ``get_adaptive_draft_len`` (inside ``_prepare_inputs``).
    Call ``invalidate`` on request completion.
    """

    def __init__(
        self,
        config: VerifyAdaptiveConfig,
        num_spec_tokens: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        config.validate(num_spec_tokens)

        self.config = config
        self.num_spec_tokens = num_spec_tokens
        self.max_batch_size = max_batch_size
        self.device = device
        self.max_query_len_per_req: int = (
            config.max_query_len_per_req
            if config.max_query_len_per_req is not None
            else num_spec_tokens + 1
        )

        self._batch_size_levels: list[int] = self._build_batch_size_levels()
        self._query_len_levels: list[int] = self._build_query_len_levels()

        # (batch_size, sum_query_len) → ITL in seconds
        self._cost_table: dict[tuple[int, int], float] = {}
        self._sorted_bs: list[int] = []
        self._sorted_sql_per_bs: dict[int, list[int]] = {}

        # req_id → recommended draft_len for the next verifier step
        self._adaptive_draft_lens: dict[str, int] = {}

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: bs_levels=%s  ql_levels=%s",
                self._batch_size_levels,
                self._query_len_levels,
            )

    def _build_batch_size_levels(self) -> list[int]:
        """Step-2 range from min_warmup_batch_size to cap."""
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = (
            self.config.max_warmup_batch_size
            if self.config.max_warmup_batch_size is not None
            else self.max_batch_size
        )
        start = self.config.min_warmup_batch_size
        levels = list(range(start, cap + 1, 2))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        """``{min_q, min_q+step, …, max_q}`` with max_q forced in."""
        min_q = self.config.min_query_len_per_req
        max_q = self.max_query_len_per_req
        step = self.config.query_len_step_per_req

        levels = list(range(min_q, max_q + 1, step))
        if not levels or levels[-1] < max_q:
            levels.append(max_q)
        return sorted(set(levels))

    def profile_cost_table(self, runner: Any) -> None:
        """Measure verifier ITL at each (batch_size, query_len_per_req) point.

        INTEGRATION NOTE: ``runner._dummy_run`` must accept the kwarg
        ``explicit_scheduled_tokens: list[int] | None``.  When set it
        bypasses the internal token-distribution logic (see model-runner
        integration step).
        """
        if not self.config.enabled:
            return

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: profiling %d ITL cost points "
                "(%d bs × %d ql).",
                len(self._batch_size_levels) * len(self._query_len_levels),
                len(self._batch_size_levels),
                len(self._query_len_levels),
            )

        max_tokens = getattr(runner, "max_num_tokens", None)

        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []

            for ql in self._query_len_levels:
                num_tokens = bs * ql
                if max_tokens is not None and num_tokens > max_tokens:
                    logger.info(
                        "profile skip: bs=%d ql=%d num_tokens=%d > %d",
                        bs, ql, num_tokens, max_tokens,
                    )
                    continue

                sched_tokens = [ql] * bs

                runtime_mode, avg_ms, padded_tokens = runner._adaptive_profile_run(
                    sched_tokens,
                    self.config.warmup_seq_lens,
                    self.config.n_warmup_iters,
                    self.config.n_measure_iters,
                )
                elapsed_s = avg_ms / 1e3

                self._cost_table[(bs, num_tokens)] = elapsed_s
                self._sorted_sql_per_bs[bs].append(num_tokens)
                if (
                    get_tp_group().rank_in_group == 0
                    and get_pp_group().is_first_rank
                ):
                    logger.info(
                        "profile  bs=%-4d  ql=%-4d  sql=%-6d  padded=%-6d  "
                        "seq_lens=%-6d  mode=%-6s  avg=%.3f ms",
                        bs, ql, num_tokens, padded_tokens,
                        self.config.warmup_seq_lens,
                        runtime_mode,
                        avg_ms,
                    )

        self._sorted_bs = sorted(self._sorted_sql_per_bs.keys())
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()

        # TP correctness: GPU timings differ slightly per rank, which can
        # flip the argmax and cause divergent draft_lens -> NCCL deadlock.
        # Broadcast rank-0's table so all ranks decide identically.
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            self._cost_table = tp_group.broadcast_object(self._cost_table, src=0)

        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: cost table ready (%d entries).",
                len(self._cost_table),
            )

    def process_draft_output(
        self,
        selected_probs: torch.Tensor,  # [B, T] on CPU (pinned), already transferred
        req_ids: list[str],
        active_draft_req_ids: set[str],
        batch_size: int,
    ) -> None:
        """Compute and cache adaptive draft_lens from this step's drafter probs."""
        if not self.config.enabled or not active_draft_req_ids or not self._sorted_bs:
            return

        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        all_probs_np: np.ndarray = selected_probs[:n_rows].numpy()

        active_indices: list[int] = [
            i for i in range(n_rows) if req_ids[i] in active_draft_req_ids
        ]
        if not active_indices:
            return
        active_probs: np.ndarray = all_probs_np[active_indices]
        active_req_ids: list[str] = [req_ids[i] for i in active_indices]

        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs[bs_key]

        result = choose_query_lens_discrete(
            probs=active_probs,
            base_batch_size=batch_size,
            q_levels=q_levels,
            cost_lookup=lambda q: self._cost_table[(bs_key, q)],
            max_draft_len=self.max_query_len_per_req - 1,
        )

        draft_lens = result["draft_lens"]
        for req_id, draft_len in zip(active_req_ids, draft_lens):
            self._adaptive_draft_lens[req_id] = draft_len

        logger.debug(
            "adaptive: bs_key=%d best_Q=%d best_S=%d score=%.4f draft_lens=%s",
            bs_key, result["best_Q"], result["best_S"],
            result["best_score"], draft_lens,
        )

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        """Cached draft_len for *req_id*, or None (→ use full spec tokens)."""
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        """Drop cached state for a completed or evicted request."""
        self._adaptive_draft_lens.pop(req_id, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    """Smallest key ≥ val; falls back to max key when val is out of range."""
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]
