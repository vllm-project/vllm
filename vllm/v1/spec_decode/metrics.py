# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1


class NearZeroAcceptanceCanary:
    """One-shot early-run warning for sustained near-zero draft acceptance.

    A mismatched or misconfigured draft/target pairing can run silently at
    ~0% acceptance: the server looks healthy, every draft is rejected, and
    speculative decoding is pure overhead (#61072). This canary accumulates
    the accepted and drafted counters over the first window of speculative
    decoding and, once the window is filled, warns at most once when
    acceptance stayed below a near-zero threshold. The window spans log
    intervals, so the state must not be reset with the interval buffers.

    Near-zero acceptance is a symptom with several possible causes (a draft
    trained for a different base checkpoint or tokenizer, a draft/target
    quantization mismatch, or simply hard prompts); the warning lists them
    without asserting that the pairing is invalid.
    """

    def __init__(
        self,
        window_draft_tokens: int = 4096,
        min_acceptance_rate: float = 0.01,
        log_fn=None,
    ):
        self.window_draft_tokens = window_draft_tokens
        self.min_acceptance_rate = min_acceptance_rate
        self._log_fn = log_fn
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
        self._fired = False

    def observe(self, spec_decoding_stats: SpecDecodingStats) -> None:
        if self._fired:
            return
        self._num_draft_tokens += spec_decoding_stats.num_draft_tokens
        self._num_accepted_tokens += spec_decoding_stats.num_accepted_tokens
        if self._num_draft_tokens < self.window_draft_tokens:
            return
        # Evaluate once on the filled window; later degradation is visible in
        # the periodic metrics log.
        self._fired = True
        acceptance_rate = self._num_accepted_tokens / self._num_draft_tokens
        if acceptance_rate >= self.min_acceptance_rate:
            return
        log_fn = self._log_fn if self._log_fn is not None else logger.warning
        log_fn(
            "Speculative decoding accepted %d of the first %d draft tokens "
            "(%.2f%%). Near-zero draft acceptance makes speculative decoding "
            "slower than running without it. Common causes are a draft model "
            "trained for a different target checkpoint or tokenizer, or a "
            "draft/target quantization mismatch, but low acceptance alone does "
            "not prove the pairing is invalid -- hard prompts can also depress "
            "it. Verify the draft matches the loaded target, or disable "
            "speculative decoding.",
            self._num_accepted_tokens,
            self._num_draft_tokens,
            100.0 * acceptance_rate,
        )


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self, is_diffusion: bool = False):
        # Diffusion (dLLM) models reuse the spec-decode data path with
        # overloaded semantics, so the raw spec-decode framing (drafts, bonus
        # token, per-position vector) is logged with diffusion-native terms.
        self.is_diffusion = is_diffusion
        # The canary accumulates across log intervals, so it lives outside
        # reset(); diffusion reuses the counters with different meaning and
        # is exempted.
        self.acceptance_canary = None if is_diffusion else NearZeroAcceptanceCanary()
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        if self.acceptance_canary is not None:
            self.acceptance_canary.observe(spec_decoding_stats)

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        if self.is_diffusion:
            self._log_diffusion(
                log_fn,
                num_denoising_steps=num_drafts,
                num_canvas_tokens=num_draft_tokens,
                num_committed_tokens=num_accepted_tokens,
                committed_throughput=accepted_throughput,
            )
            self.reset()
            return

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()

    def _log_diffusion(
        self,
        log_fn,
        num_denoising_steps: int,
        num_canvas_tokens: int,
        num_committed_tokens: int,
        committed_throughput: float,
    ):
        # Each "draft" is one denoising step that re-evaluates the canvas block
        # and finalizes some of its positions.
        mean_committed_per_step = (
            num_committed_tokens / num_denoising_steps
            if num_denoising_steps > 0
            else float("nan")
        )
        mean_steps_per_canvas = (
            num_canvas_tokens / num_committed_tokens
            if num_committed_tokens > 0
            else float("nan")
        )

        log_fn(
            "DiffusionDecoding metrics: "
            "Committed token throughput: %.2f tokens/s, "
            "Mean denoising steps per canvas: %.2f, "
            "Mean tokens committed per denoising step: %.2f, "
            "Committed: %d tokens, "
            "Denoising steps: %d, "
            "Canvas positions evaluated: %d",
            committed_throughput,
            mean_steps_per_canvas,
            mean_committed_per_step,
            num_committed_tokens,
            num_denoising_steps,
            num_canvas_tokens,
        )


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts_total[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos_total[$interval] /
      vllm:spec_decode_num_drafts_total[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        is_diffusion: bool = False,
    ):
        # Diffusion (dLLM) models reuse the spec-decode counters but expose them
        # under diffusion-native names; the per-position acceptance vector does
        # not apply, so it is omitted.
        self.is_diffusion = is_diffusion
        self.spec_decoding_enabled = speculative_config is not None or is_diffusion
        if not self.spec_decoding_enabled:
            return

        if is_diffusion:
            counter_specs = [
                ("vllm:diffusion_num_denoising_steps", "Number of denoising steps."),
                (
                    "vllm:diffusion_num_canvas_positions",
                    "Number of canvas positions evaluated.",
                ),
                (
                    "vllm:diffusion_num_committed_tokens",
                    "Number of committed (finalized) tokens.",
                ),
            ]
        else:
            counter_specs = [
                ("vllm:spec_decode_num_drafts", "Number of spec decoding drafts."),
                ("vllm:spec_decode_num_draft_tokens", "Number of draft tokens."),
                ("vllm:spec_decode_num_accepted_tokens", "Number of accepted tokens."),
            ]

        counters = [
            create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for name, doc in counter_specs
        ]
        # num_drafts/num_draft_tokens/num_accepted_tokens map onto denoising
        # steps/canvas positions/committed tokens in the diffusion path.
        self.counter_spec_decode_num_drafts = counters[0]
        self.counter_spec_decode_num_draft_tokens = counters[1]
        self.counter_spec_decode_num_accepted_tokens = counters[2]

        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not is_diffusion:
            assert speculative_config is not None
            num_spec_tokens = speculative_config.num_speculative_tokens
            pos_labelnames = labelnames + ["position"]
            base_counter = self._counter_cls(
                name="vllm:spec_decode_num_accepted_tokens_per_pos",
                documentation="Accepted tokens per draft position.",
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [
                    base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
