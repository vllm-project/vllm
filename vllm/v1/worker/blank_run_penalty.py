# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blank-run penalty for realtime streaming: worker-side implementation.

Streaming speech-to-text models that re-ingest their own output can fall into
a self-sustained decoding rut: once the visible context fills with the
blank/silence token, greedy decoding keeps emitting it over real speech for
minutes (observed with Voxtral realtime; same family as Whisper repetition
loops). Temperature does not help: the distribution collapses to P(blank)~1.

Once a request has sampled the blank token more than K consecutive times, a
progressive penalty is subtracted from that token's logit before sampling:

    penalty = min(cap, alpha * (run_length - K))

Healthy blank runs (inter-sentence silences) stay below K and are never
touched; genuinely silent audio keeps decoding as silence because its blank
margin exceeds `cap`.

This lives in the model runner, NOT in a v1 LogitsProcessor: the realtime
streaming path recycles the engine request for every audio chunk and clears
its output-token list in place, so the `output_tok_ids` reference a logits
processor receives via BatchUpdate is empty at every step. The run length
must be accumulated worker-side, keyed by request id, which is stable for
the whole streaming session.

Per-request opt-in via SamplingParams.extra_args["blank_run_penalty"]
(a dict with token_id, k, alpha, cap), wired by the realtime connection
from the --realtime-blank-run-* engine flags; inert otherwise.
"""

from dataclasses import dataclass

import torch

EXTRA_ARGS_KEY = "blank_run_penalty"


@dataclass(frozen=True)
class BlankRunConfig:
    token_id: int
    k: int
    alpha: float
    cap: float


def parse_config(sampling_params) -> BlankRunConfig | None:
    if sampling_params is None:
        return None
    cfg = (sampling_params.extra_args or {}).get(EXTRA_ARGS_KEY)
    if not cfg:
        return None
    try:
        parsed = BlankRunConfig(
            token_id=int(cfg["token_id"]),
            k=int(cfg["k"]),
            alpha=float(cfg["alpha"]),
            cap=float(cfg["cap"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if parsed.token_id < 0 or parsed.k < 1 or parsed.alpha <= 0 or parsed.cap <= 0:
        return None
    return parsed


class BlankRunPenalizer:
    """Per-request consecutive-blank counters + logit penalty.

    Usage from the model runner, once per decode step:
      1. apply(logits, req_ids, get_params) BEFORE sampling
      2. update(req_ids, sampled_token_ids) AFTER sampling
    prune(live_req_ids) drops state of finished requests (called lazily).
    """

    _PRUNE_EVERY = 512

    def __init__(self):
        # req_id -> parsed config (None = request opted out; cached to avoid
        # re-parsing extra_args every step)
        self._cfgs: dict[str, BlankRunConfig | None] = {}
        # req_id -> consecutive sampled-blank count
        self._runs: dict[str, int] = {}
        self._steps = 0

    def _cfg(self, req_id: str, get_params) -> BlankRunConfig | None:
        try:
            return self._cfgs[req_id]
        except KeyError:
            cfg = parse_config(get_params(req_id))
            self._cfgs[req_id] = cfg
            return cfg

    def apply(self, logits: torch.Tensor | None, req_ids, get_params) -> bool:
        """Subtract the penalty in place for locked requests. Returns True if
        any request in the batch has the penalty configured (callers may use
        it to skip `update` entirely for non-realtime workloads)."""
        if logits is None:
            return False
        any_active = False
        for i, req_id in enumerate(req_ids):
            cfg = self._cfg(req_id, get_params)
            if cfg is None:
                continue
            any_active = True
            run = self._runs.get(req_id, 0)
            if run > cfg.k:
                logits[i, cfg.token_id] -= min(cfg.cap, cfg.alpha * (run - cfg.k))
        return any_active

    def update(self, req_ids, sampled_token_ids) -> None:
        """Advance per-request counters from this step's sampled tokens.

        sampled_token_ids: list[int] aligned with req_ids (one token per
        request; the realtime decode path emits exactly one).
        """
        for i, req_id in enumerate(req_ids):
            cfg = self._cfgs.get(req_id)
            if cfg is None:
                continue
            if i < len(sampled_token_ids) and sampled_token_ids[i] == cfg.token_id:
                self._runs[req_id] = self._runs.get(req_id, 0) + 1
            else:
                self._runs[req_id] = 0
        self._steps += 1
        if self._steps % self._PRUNE_EVERY == 0 and len(self._cfgs) > 2 * len(req_ids):
            self.prune(set(req_ids))

    def prune(self, live_req_ids: set) -> None:
        self._cfgs = {r: c for r, c in self._cfgs.items() if r in live_req_ids}
        self._runs = {r: n for r, n in self._runs.items() if r in live_req_ids}
