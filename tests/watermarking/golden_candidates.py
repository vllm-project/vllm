# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic watermarking golden candidates; see README.md."""

import json
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict, TypeVar, cast, get_args

import torch

from vllm.config.watermarking import (
    WatermarkConfig,
    WatermarkContextScope,
    WatermarkingAlgorithm,
    WatermarkPRFName,
    derive_watermark_key,
)
from vllm.v1.watermarking import (
    DualKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
    WatermarkDetection,
    WatermarkDetector,
    create_prf,
    create_watermarker,
)
from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask

_T = TypeVar("_T")

REGENERATE_COMMAND = "python -m tests.watermarking.generate_goldens"

GOLDEN_SCHEMA_VERSION = 2

# Hex round-trips exactly, but libm differs by an ULP across builds.
GOLDEN_FLOAT_RTOL = 1e-9
GOLDEN_FLOAT_ATOL = 1e-300

P_VALUE_RATIO_MIN_DISTANCE = 1e-6
REPETITIVE_MIN_KEY_PERTURBATION_TOKENS = 8
REPETITIVE_MAX_P_VALUE_RATIO = 0.5
ROUTING_BOUNDARY_MIN_DISTANCE = 1e-6

# round(((i + 1) * (5**0.5 - 1) / 2) % 1.0, 4) for i in range(64)
ROUTING_UNIFORMS: tuple[float, ...] = (
    0.618, 0.2361, 0.8541, 0.4721, 0.0902, 0.7082, 0.3262, 0.9443,
    0.5623, 0.1803, 0.7984, 0.4164, 0.0344, 0.6525, 0.2705, 0.8885,
    0.5066, 0.1246, 0.7426, 0.3607, 0.9787, 0.5967, 0.2148, 0.8328,
    0.4508, 0.0689, 0.6869, 0.305, 0.923, 0.541, 0.1591, 0.7771,
    0.3951, 0.0132, 0.6312, 0.2492, 0.8673, 0.4853, 0.1033, 0.7214,
    0.3394, 0.9574, 0.5755, 0.1935, 0.8115, 0.4296, 0.0476, 0.6656,
    0.2837, 0.9017, 0.5197, 0.1378, 0.7558, 0.3738, 0.9919, 0.6099,
    0.2279, 0.846, 0.464, 0.082, 0.7001, 0.3181, 0.9361, 0.5542,
)  # fmt: skip


class GoldenFormatError(ValueError):
    """The stored goldens do not have the shape this module expects."""


class GoldenGuardError(ValueError):
    """A candidate is too weak or too unstable to be frozen as a golden."""


class GoldenDetectionPayload(TypedDict):
    score: str
    p_value: str
    p_value_ratio: float
    num_scored_tokens: int
    is_watermarked: bool


class GoldenTracePayload(TypedDict):
    routing_draws: int
    key_b_routed: int | None
    key_b_effective: int | None
    dedup_skips: int
    partial_context_skips: int


class GoldenCandidatePayload(TypedDict):
    configuration: dict[str, object]
    resolved: dict[str, object]
    generation: list[int]
    trace: GoldenTracePayload
    detection: GoldenDetectionPayload


class GoldenPayload(TypedDict):
    schema_version: int
    environment: dict[str, str]
    candidates: dict[str, GoldenCandidatePayload]


@dataclass(frozen=True)
class DeterministicGenerationFixture:
    vocabulary_size: int = 128
    num_tokens: int = 64
    logit_denominator: int = 16
    logit_modulus: int = 131
    dominant_token: int | None = None
    dominant_bias: int = 0
    prompt: tuple[int, ...] = ()
    routing_uniforms: tuple[float, ...] = ROUTING_UNIFORMS

    def logits(self, position: int) -> torch.Tensor:
        token_ids = torch.arange(self.vocabulary_size, dtype=torch.int64)
        values = (token_ids * 37 + position * 17) % self.logit_modulus - 56
        if self.dominant_token is not None:
            values[self.dominant_token] += self.dominant_bias
        return (values.to(torch.float32) / self.logit_denominator).unsqueeze(0)


@dataclass(frozen=True)
class WatermarkingSchemeConfig:
    context_width: int = 4
    generation_alpha: float = 0.1
    detection_alpha: float = 0.1
    generation_deduplicate_contexts: WatermarkContextScope = "single_turn"
    generation_deduplicate_contexts_max_history: int | None = 8192
    detection_deduplicate_contexts: bool = True
    p_value_threshold: float = 0.01


@dataclass(frozen=True)
class GenerationTrace:
    routing_draws: int
    key_b_routed: int | None
    key_b_effective: int | None
    dedup_skips: int
    partial_context_skips: int


class _DeterministicSampler:
    def __init__(self, routing_uniforms: tuple[float, ...]) -> None:
        self.routing_uniforms = routing_uniforms
        self.routing_position = 0
        self.key_b_routed = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != 2:
            return logits.argmax(dim=-1)

        probability_a = logits.softmax(dim=-1)[:, 0]
        uniform = self.routing_uniforms[
            self.routing_position % len(self.routing_uniforms)
        ]
        self.routing_position += 1
        routed = (probability_a <= uniform).to(torch.int64)
        self.key_b_routed += int(routed.sum())
        return routed


def routing_boundary(alpha: float) -> float:
    """Return the realised key-A probability the deterministic sampler sees."""
    logits = torch.tensor([1 - alpha, alpha], dtype=torch.float32).log()
    return float(logits.softmax(dim=-1)[0])


def _create_gumbel_detector(
    key: int,
    config: WatermarkingSchemeConfig,
    prf: WatermarkPRFName,
) -> WatermarkDetector:
    return GumbelWatermarkDetector(
        key=key,
        context_width=config.context_width,
        p_value_threshold=config.p_value_threshold,
        prf=prf,
        deduplicate_contexts=config.detection_deduplicate_contexts,
    )


def _create_dual_key_gumbel_detector(
    key: int,
    config: WatermarkingSchemeConfig,
    prf: WatermarkPRFName,
) -> WatermarkDetector:
    return DualKeyGumbelWatermarkDetector(
        key=key,
        context_width=config.context_width,
        p_value_threshold=config.p_value_threshold,
        prf=prf,
        deduplicate_contexts=config.detection_deduplicate_contexts,
        alpha=config.detection_alpha,
    )


DETECTOR_FACTORIES = {
    "gumbel": _create_gumbel_detector,
    "dual_key_gumbel": _create_dual_key_gumbel_detector,
}


@dataclass(frozen=True)
class WatermarkingCandidate:
    id: str
    scheme: WatermarkingAlgorithm
    scheme_config: WatermarkingSchemeConfig
    prf: WatermarkPRFName
    key: int
    detection_key: int
    fixture: DeterministicGenerationFixture = DeterministicGenerationFixture()

    def configuration(self) -> dict[str, object]:
        prf_version = getattr(create_prf(self.prf, self.key), "version", None)
        if not isinstance(prf_version, str):
            raise ValueError(f"PRF {self.prf} does not define a version")
        return {
            "scheme": self.scheme,
            "scheme_config": {
                "context_width": self.scheme_config.context_width,
                "generation_alpha": self.scheme_config.generation_alpha,
                "detection_alpha": self.scheme_config.detection_alpha,
                "generation_deduplicate_contexts": (
                    self.scheme_config.generation_deduplicate_contexts
                ),
                "generation_deduplicate_contexts_max_history": (
                    self.scheme_config.generation_deduplicate_contexts_max_history
                ),
                "detection_deduplicate_contexts": (
                    self.scheme_config.detection_deduplicate_contexts
                ),
                "p_value_threshold": self.scheme_config.p_value_threshold,
            },
            "prf": self.prf,
            "prf_version": prf_version,
            "key": str(self.key),
            "detection_key": str(self.detection_key),
            "fixture": {
                "vocabulary_size": self.fixture.vocabulary_size,
                "num_tokens": self.fixture.num_tokens,
                "logit_denominator": self.fixture.logit_denominator,
                "logit_modulus": self.fixture.logit_modulus,
                "dominant_token": self.fixture.dominant_token,
                "dominant_bias": self.fixture.dominant_bias,
                "prompt": list(self.fixture.prompt),
                "routing_uniforms": list(self.fixture.routing_uniforms),
            },
        }

    def resolved(self) -> dict[str, object]:
        """Return the vLLM-side state this candidate resolves to.

        An allowlist rather than the whole dataclass, so a new production field
        is a deliberate golden change: WATERMARK_CONFIG_SPECS names the fields
        recorded here and test_goldens.py checks it still covers
        WatermarkConfig. Keys exceed 2**53 and are decimal strings.
        """
        config = self._watermark_config()
        detector = self._detector()
        key_b_prf = getattr(detector, "key_b_prf", None)
        dual_key = self.scheme == "dual_key_gumbel"
        return {
            "watermark_config": {
                field: getattr(config, field) for field in WATERMARK_CONFIG_SPECS
            },
            "derived_keys": {
                "key_a": str(derive_watermark_key(self.key, b"key_a"))
                if dual_key
                else None,
                "key_b": str(derive_watermark_key(self.key, b"key_b"))
                if dual_key
                else None,
            },
            "detector": {
                "type": type(detector).__name__,
                "context_width": detector.context_width,
                "p_value_threshold": detector.p_value_threshold,
                "deduplicate_contexts": detector.deduplicate_contexts,
                "alpha": getattr(detector, "alpha", None),
                "prf_key": str(detector.prf.key),
                "key_b_prf_key": None if key_b_prf is None else str(key_b_prf.key),
            },
        }

    def generate(self) -> list[int]:
        return self.generate_with_trace()[0]

    def generate_with_trace(self) -> tuple[list[int], GenerationTrace]:
        watermarker = create_watermarker(self._watermark_config())
        sampler = _DeterministicSampler(self.fixture.routing_uniforms)
        generated: list[int] = []
        dedup_skips = 0
        partial_context_skips = 0
        key_b_effective = 0

        for position in range(self.fixture.num_tokens):
            context = self._context(generated)
            skip_mask = self._skip_mask(generated, context)
            skipped = skip_mask is not None and bool(skip_mask.item())
            if skipped:
                dedup_skips += 1
                if len(generated) < self.scheme_config.context_width:
                    partial_context_skips += 1
            routed_before = sampler.key_b_routed
            sample = watermarker.sample(
                self.fixture.logits(position),
                context,
                sampler,
                skip_mask,
            )
            if sampler.key_b_routed > routed_before and not skipped:
                key_b_effective += 1
            generated.append(int(sample.token_ids.item()))

        routes = self.scheme == "dual_key_gumbel" and (
            0 < self.scheme_config.generation_alpha < 1
        )
        trace = GenerationTrace(
            routing_draws=sampler.routing_position,
            key_b_routed=sampler.key_b_routed if routes else None,
            key_b_effective=key_b_effective if routes else None,
            dedup_skips=dedup_skips,
            partial_context_skips=partial_context_skips,
        )
        return generated, trace

    def detect(self, token_ids: list[int]) -> WatermarkDetection:
        return self._detector().detect(token_ids)

    def golden_entry(
        self,
        detection_token_ids: list[int] | None = None,
    ) -> GoldenCandidatePayload:
        """Generate, detect and trace this candidate into one golden entry.

        ``detection_token_ids`` runs detection over stored tokens instead of
        the freshly generated ones, so a token drift and a detection drift are
        reported independently.
        """
        token_ids, trace = self.generate_with_trace()
        scored = token_ids if detection_token_ids is None else detection_token_ids
        detection = self.detect(scored)
        threshold = self.scheme_config.p_value_threshold
        return {
            "configuration": self.configuration(),
            "resolved": self.resolved(),
            "generation": token_ids,
            "trace": _trace_payload(trace),
            "detection": {
                "score": detection.score.hex(),
                "p_value": detection.p_value.hex(),
                "p_value_ratio": detection.p_value / threshold,
                "num_scored_tokens": detection.num_scored_tokens,
                "is_watermarked": detection.is_watermarked,
            },
        }

    def _watermark_config(self) -> WatermarkConfig:
        return WatermarkConfig(
            algorithm=self.scheme,
            key=self.key,
            alpha=self.scheme_config.generation_alpha,
            context_width=self.scheme_config.context_width,
            deduplicate_contexts=(self.scheme_config.generation_deduplicate_contexts),
            deduplicate_contexts_max_history=(
                self.scheme_config.generation_deduplicate_contexts_max_history
            ),
            prf=self.prf,
        )

    def _detector(self) -> WatermarkDetector:
        return DETECTOR_FACTORIES[self.scheme](
            self.detection_key,
            self.scheme_config,
            self.prf,
        )

    def _context(self, generated: list[int]) -> torch.Tensor:
        width = self.scheme_config.context_width
        padded = [-1] * width + generated
        return torch.tensor([padded[-width:]], dtype=torch.int64)

    def _skip_mask(
        self, generated: list[int], context: torch.Tensor
    ) -> torch.Tensor | None:
        scope = self.scheme_config.generation_deduplicate_contexts
        if scope == "none":
            return None

        prompt_len = len(self.fixture.prompt)
        storage = torch.zeros(
            (1, prompt_len + self.fixture.num_tokens),
            dtype=torch.int64,
        )
        if prompt_len:
            storage[0, :prompt_len] = torch.tensor(self.fixture.prompt)
        if generated:
            storage[0, prompt_len : prompt_len + len(generated)] = torch.tensor(
                generated
            )
        return repeated_context_mask(
            storage,
            torch.tensor([0]),
            torch.tensor([prompt_len]),
            torch.tensor([prompt_len + len(generated)]),
            context,
            self.scheme_config.generation_deduplicate_contexts_max_history,
            include_prompt=scope == "all",
            skip_partial_context=scope == "all",
        )


def _trace_payload(trace: GenerationTrace) -> GoldenTracePayload:
    return {
        "routing_draws": trace.routing_draws,
        "key_b_routed": trace.key_b_routed,
        "key_b_effective": trace.key_b_effective,
        "dedup_skips": trace.dedup_skips,
        "partial_context_skips": trace.partial_context_skips,
    }


BALANCED_FIXTURE = DeterministicGenerationFixture()

# The first five tokens of BALANCED_FIXTURE under scope "all", so position 4 repeats.
PROMPT_FIXTURE = DeterministicGenerationFixture(prompt=(46, 42, 38, 34, 122))

REPETITIVE_FIXTURE = DeterministicGenerationFixture(
    vocabulary_size=16,
    num_tokens=48,
    dominant_token=3,
    dominant_bias=52,
)

MIDDLE_FIXTURE = DeterministicGenerationFixture(
    vocabulary_size=8,
    num_tokens=64,
    dominant_token=3,
    dominant_bias=16,
)

# Few enough scored tokens that the p_value lands just under its threshold.
NEAR_THRESHOLD_FIXTURE = DeterministicGenerationFixture(
    vocabulary_size=16,
    num_tokens=8,
)


def _candidate(
    id: str,
    scheme: WatermarkingAlgorithm,
    key: int,
    *,
    prf: WatermarkPRFName,
    detection_key: int | None = None,
    context_width: int = 4,
    generation_alpha: float = 0.1,
    detection_alpha: float | None = None,
    generation_deduplicate_contexts: WatermarkContextScope = "single_turn",
    generation_deduplicate_contexts_max_history: int | None = 8192,
    detection_deduplicate_contexts: bool = True,
    p_value_threshold: float = 0.01,
    fixture: DeterministicGenerationFixture = BALANCED_FIXTURE,
) -> WatermarkingCandidate:
    return WatermarkingCandidate(
        id=id,
        scheme=scheme,
        scheme_config=WatermarkingSchemeConfig(
            context_width=context_width,
            generation_alpha=generation_alpha,
            detection_alpha=(
                generation_alpha if detection_alpha is None else detection_alpha
            ),
            generation_deduplicate_contexts=generation_deduplicate_contexts,
            generation_deduplicate_contexts_max_history=(
                generation_deduplicate_contexts_max_history
            ),
            detection_deduplicate_contexts=detection_deduplicate_contexts,
            p_value_threshold=p_value_threshold,
        ),
        prf=prf,
        key=key,
        detection_key=key if detection_key is None else detection_key,
        fixture=fixture,
    )


WATERMARKING_CANDIDATES = (
    _candidate("gumbel-philox-key0-cw1", "gumbel", 0, prf="philox", context_width=1),
    _candidate("gumbel-philox-key42-cw4", "gumbel", 42, prf="philox"),
    _candidate(
        "gumbel-philox-key42-cw4-wrong-key",
        "gumbel",
        42,
        prf="philox",
        detection_key=43,
    ),
    _candidate(
        "gumbel-philox-key2p32-cw5-no-dedup",
        "gumbel",
        2**32,
        prf="philox",
        context_width=5,
        generation_deduplicate_contexts="none",
        detection_deduplicate_contexts=False,
    ),
    _candidate(
        "gumbel-philox-keymax-cw16-all",
        "gumbel",
        2**64 - 1,
        prf="philox",
        context_width=16,
        generation_deduplicate_contexts="all",
    ),
    _candidate(
        "gumbel-philox-key42-cw4-all",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="all",
    ),
    _candidate(
        "gumbel-philox-key42-cw4-all-prompt",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="all",
        fixture=PROMPT_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-key42-cw4-mid-history8",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts_max_history=8,
        fixture=MIDDLE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-key42-cw4-mid-history-none",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts_max_history=None,
        fixture=MIDDLE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-key42-cw4-mid-no-dedup",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="none",
        generation_deduplicate_contexts_max_history=None,
        detection_deduplicate_contexts=False,
        fixture=MIDDLE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-key42-cw4-mid-all",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="all",
        generation_deduplicate_contexts_max_history=None,
        fixture=MIDDLE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-key42-cw4-near-threshold",
        "gumbel",
        42,
        prf="philox",
        p_value_threshold=0.0015,
        fixture=NEAR_THRESHOLD_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-repeated-cw1",
        "gumbel",
        42,
        prf="philox",
        context_width=1,
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-repeated-cw4-all",
        "gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="all",
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "dual-key-gumbel-philox-key0-cw1-alpha0",
        "dual_key_gumbel",
        0,
        prf="philox",
        context_width=1,
        generation_alpha=0.0,
    ),
    _candidate(
        "dual-key-gumbel-philox-key0-cw4-alpha0.1",
        "dual_key_gumbel",
        0,
        prf="philox",
    ),
    _candidate(
        "dual-key-gumbel-philox-key42-cw4-alpha-mismatch",
        "dual_key_gumbel",
        42,
        prf="philox",
        generation_alpha=0.1,
        detection_alpha=0.2,
    ),
    _candidate(
        "dual-key-gumbel-philox-key2p32-cw5-alpha0.5",
        "dual_key_gumbel",
        2**32,
        prf="philox",
        context_width=5,
        generation_alpha=0.5,
    ),
    _candidate(
        "dual-key-gumbel-philox-keymax-cw16-alpha1",
        "dual_key_gumbel",
        2**64 - 1,
        prf="philox",
        context_width=16,
        generation_alpha=1.0,
    ),
    _candidate(
        "dual-key-gumbel-philox-keymax-cw4-alpha0.4",
        "dual_key_gumbel",
        2**64 - 1,
        prf="philox",
        generation_alpha=0.4,
    ),
    _candidate(
        "dual-key-gumbel-philox-key7-cw4-near-threshold",
        "dual_key_gumbel",
        7,
        prf="philox",
        p_value_threshold=0.003,
        fixture=NEAR_THRESHOLD_FIXTURE,
    ),
    _candidate(
        "dual-key-gumbel-philox-repeated-cw1",
        "dual_key_gumbel",
        42,
        prf="philox",
        context_width=1,
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "dual-key-gumbel-philox-repeated-cw4-all",
        "dual_key_gumbel",
        42,
        prf="philox",
        generation_deduplicate_contexts="all",
        fixture=REPETITIVE_FIXTURE,
    ),
)

MAX_HISTORY_TWINS = (
    (
        "gumbel-philox-key42-cw4-mid-history8",
        "gumbel-philox-key42-cw4-mid-history-none",
    ),
)
PROMPT_TWINS = (
    (
        "gumbel-philox-key42-cw4-all",
        "gumbel-philox-key42-cw4-all-prompt",
    ),
)
SKIP_PARTIAL_TWINS = (
    (
        "gumbel-philox-key42-cw4-mid-history-none",
        "gumbel-philox-key42-cw4-mid-all",
    ),
)
DEDUPLICATION_TWINS = (
    (
        "gumbel-philox-key42-cw4-mid-history-none",
        "gumbel-philox-key42-cw4-mid-no-dedup",
    ),
)


def environment_block() -> dict[str, str]:
    """Return informational build details; these are recorded, never compared."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }


def frozen_entry(candidate: WatermarkingCandidate) -> GoldenCandidatePayload:
    """Build the entry to freeze for one candidate."""
    return candidate.golden_entry()


def validate_golden_guards(
    entries: Mapping[str, GoldenCandidatePayload],
) -> None:
    candidates_by_id = {
        candidate.id: candidate for candidate in WATERMARKING_CANDIDATES
    }
    combinations = {
        (candidate.scheme, candidate.prf) for candidate in WATERMARKING_CANDIDATES
    }
    expected_combinations = configured_algorithm_prf_combinations()
    if combinations != expected_combinations:
        missing_combinations = sorted(expected_combinations - combinations)
        unexpected_combinations = sorted(combinations - expected_combinations)
        raise GoldenGuardError(
            "candidate scheme/PRF coverage mismatch: "
            f"missing {missing_combinations}, unexpected {unexpected_combinations}"
        )

    missing = sorted(set(candidates_by_id) - set(entries))
    if missing:
        raise GoldenGuardError(f"golden entries are missing candidates: {missing}")

    for candidate in WATERMARKING_CANDIDATES:
        entry = entries[candidate.id]
        ratio = entry["detection"]["p_value_ratio"]
        if abs(ratio - 1.0) < P_VALUE_RATIO_MIN_DISTANCE:
            raise GoldenGuardError(
                f"{candidate.id}: p_value is within "
                f"{P_VALUE_RATIO_MIN_DISTANCE:g} of p_value_threshold "
                f"(ratio {ratio!r}), so is_watermarked is not reproducible"
            )

        if candidate.fixture == REPETITIVE_FIXTURE:
            alternate_tokens = replace(candidate, key=candidate.key ^ 1).generate()
            changed_tokens = sum(
                actual != alternate
                for actual, alternate in zip(
                    entry["generation"], alternate_tokens, strict=True
                )
            )
            if changed_tokens < REPETITIVE_MIN_KEY_PERTURBATION_TOKENS:
                raise GoldenGuardError(
                    f"{candidate.id}: key ^ 1 changes only {changed_tokens} tokens; "
                    f"expected at least {REPETITIVE_MIN_KEY_PERTURBATION_TOKENS}"
                )
            full_context_skips = (
                entry["trace"]["dedup_skips"] - entry["trace"]["partial_context_skips"]
            )
            if full_context_skips < 1:
                raise GoldenGuardError(
                    f"{candidate.id}: repetitive fixture has no full-context "
                    "deduplication skip"
                )
            if ratio > REPETITIVE_MAX_P_VALUE_RATIO:
                raise GoldenGuardError(
                    f"{candidate.id}: p_value_ratio {ratio!r} exceeds "
                    f"{REPETITIVE_MAX_P_VALUE_RATIO}"
                )

        alpha = candidate.scheme_config.generation_alpha
        if candidate.scheme == "dual_key_gumbel" and 0 < alpha < 1:
            boundary = routing_boundary(alpha)
            distance = min(
                abs(uniform - boundary)
                for uniform in candidate.fixture.routing_uniforms
            )
            if distance < ROUTING_BOUNDARY_MIN_DISTANCE:
                raise GoldenGuardError(
                    f"{candidate.id}: routing uniform is only {distance!r} from "
                    f"the realised boundary; expected at least "
                    f"{ROUTING_BOUNDARY_MIN_DISTANCE:g}"
                )

    for enabled_id, disabled_id in DEDUPLICATION_TWINS:
        enabled = candidates_by_id[enabled_id]
        disabled = candidates_by_id[disabled_id]
        expected_disabled = replace(
            enabled,
            id=disabled_id,
            scheme_config=replace(
                enabled.scheme_config,
                generation_deduplicate_contexts="none",
                detection_deduplicate_contexts=False,
            ),
        )
        if disabled != expected_disabled:
            raise GoldenGuardError(
                f"{enabled_id} and {disabled_id} must differ only in generation "
                "and detection deduplication"
            )

        enabled_entry = entries[enabled_id]
        disabled_entry = entries[disabled_id]
        if enabled_entry["trace"]["dedup_skips"] < 1:
            raise GoldenGuardError(f"{enabled_id}: expected at least one dedup skip")
        if disabled_entry["trace"]["dedup_skips"] != 0:
            raise GoldenGuardError(f"{disabled_id}: expected zero dedup skips")
        if enabled_entry["generation"] == disabled_entry["generation"]:
            raise GoldenGuardError(
                f"{enabled_id} and {disabled_id}: generations must differ"
            )
        enabled_scored = enabled_entry["detection"]["num_scored_tokens"]
        disabled_scored = disabled_entry["detection"]["num_scored_tokens"]
        if enabled_scored == disabled_scored:
            raise GoldenGuardError(
                f"{enabled_id} and {disabled_id}: detector scored-token counts "
                "must differ"
            )


def golden_payload() -> GoldenPayload:
    candidates = {
        candidate.id: frozen_entry(candidate) for candidate in WATERMARKING_CANDIDATES
    }
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "environment": environment_block(),
        "candidates": candidates,
    }


def floats_match(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=GOLDEN_FLOAT_RTOL,
        abs_tol=GOLDEN_FLOAT_ATOL,
    )


def _relative_difference(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual - expected) / abs(expected)


def _compare_float(
    field: str,
    actual: float,
    expected: float,
    differences: list[str],
) -> None:
    if not floats_match(actual, expected):
        differences.append(
            f"{field}: expected {expected!r}, got {actual!r}, "
            f"relative difference {_relative_difference(actual, expected):.3e}"
        )


def _compare_exact(
    field: str,
    actual: object,
    expected: object,
    differences: list[str],
) -> None:
    if actual != expected:
        differences.append(f"{field}: expected {expected!r}, got {actual!r}")


def _compare_hex_float(
    field: str,
    actual_hex: str,
    expected_hex: str,
    differences: list[str],
) -> None:
    actual = float.fromhex(actual_hex)
    expected = float.fromhex(expected_hex)
    if not floats_match(actual, expected):
        differences.append(
            f"{field}: expected {expected_hex} ({expected!r}), "
            f"got {actual_hex} ({actual!r}), relative difference "
            f"{_relative_difference(actual, expected):.3e}"
        )


def _compare_mapping(
    field: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    differences: list[str],
) -> None:
    for key in sorted(set(actual) | set(expected)):
        if actual.get(key) != expected.get(key):
            differences.append(
                f"{field}.{key}: expected {expected.get(key)!r}, "
                f"got {actual.get(key)!r}"
            )


def compare_entries(
    produced: GoldenCandidatePayload,
    golden: GoldenCandidatePayload,
) -> list[str]:
    """Return one line per field that differs, empty when the two agree.

    Configuration, resolved state, tokens, trace,
    num_scored_tokens and is_watermarked are compared exactly; score, p_value
    and p_value_ratio are compared with GOLDEN_FLOAT_RTOL. The environment
    block is informational and is never looked at.
    """
    differences: list[str] = []
    for field in ("configuration", "resolved", "trace"):
        _compare_mapping(field, produced[field], golden[field], differences)

    if produced["generation"] != golden["generation"]:
        first_difference = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(produced["generation"], golden["generation"])
                )
                if actual != expected
            ),
            min(len(produced["generation"]), len(golden["generation"])),
        )
        differences.append(
            f"generation: first differs at index {first_difference} "
            f"({len(produced['generation'])} tokens generated, "
            f"{len(golden['generation'])} in the golden): "
            f"expected {golden['generation']}, got {produced['generation']}"
        )

    detection = produced["detection"]
    expected_detection = golden["detection"]
    _compare_hex_float(
        "detection.score",
        detection["score"],
        expected_detection["score"],
        differences,
    )
    _compare_hex_float(
        "detection.p_value",
        detection["p_value"],
        expected_detection["p_value"],
        differences,
    )
    _compare_float(
        "detection.p_value_ratio",
        detection["p_value_ratio"],
        expected_detection["p_value_ratio"],
        differences,
    )
    _compare_exact(
        "detection.num_scored_tokens",
        detection["num_scored_tokens"],
        expected_detection["num_scored_tokens"],
        differences,
    )
    _compare_exact(
        "detection.is_watermarked",
        detection["is_watermarked"],
        expected_detection["is_watermarked"],
        differences,
    )
    return differences


def compare_golden(
    candidate: WatermarkingCandidate,
    golden: GoldenCandidatePayload,
) -> list[str]:
    """Return one line per field of ``candidate`` that drifted from ``golden``."""
    return compare_entries(candidate.golden_entry(golden["generation"]), golden)


def _format_error(where: str, problem: str) -> str:
    return f"{where}: {problem}"


def _require_mapping(value: object, where: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GoldenFormatError(
            _format_error(where, f"expected a mapping, got {type(value).__name__}")
        )
    return value


def _require_keys(
    mapping: Mapping[str, object], expected: frozenset[str], where: str
) -> None:
    problems = []
    missing = sorted(expected - set(mapping))
    unexpected = sorted(set(mapping) - expected)
    if missing:
        problems.append(f"missing {missing}")
    if unexpected:
        problems.append(f"unexpected {unexpected}")
    if problems:
        raise GoldenFormatError(_format_error(where, " and ".join(problems)))


def _require_type(value: object, expected: type[_T], where: str) -> _T:
    if type(value) is not expected:
        raise GoldenFormatError(
            _format_error(
                where,
                f"expected {expected.__name__}, got {type(value).__name__}",
            )
        )
    return cast(_T, value)


_FieldSpec = tuple[type, bool, type | None]

_ENVIRONMENT_SPECS: dict[str, _FieldSpec] = {
    "python": (str, False, None),
    "torch": (str, False, None),
    "platform": (str, False, None),
    "cpu_capability": (str, False, None),
}
_CANDIDATE_KEYS = frozenset(
    {"configuration", "resolved", "generation", "trace", "detection"}
)
_CONFIGURATION_SPECS: dict[str, _FieldSpec] = {
    "scheme": (str, False, None),
    "scheme_config": (dict, False, None),
    "prf": (str, False, None),
    "prf_version": (str, False, None),
    "key": (str, False, None),
    "detection_key": (str, False, None),
    "fixture": (dict, False, None),
}
_SCHEME_CONFIG_SPECS: dict[str, _FieldSpec] = {
    "context_width": (int, False, None),
    "generation_alpha": (float, False, None),
    "detection_alpha": (float, False, None),
    "generation_deduplicate_contexts": (str, False, None),
    "generation_deduplicate_contexts_max_history": (int, True, None),
    "detection_deduplicate_contexts": (bool, False, None),
    "p_value_threshold": (float, False, None),
}
_FIXTURE_SPECS: dict[str, _FieldSpec] = {
    "vocabulary_size": (int, False, None),
    "num_tokens": (int, False, None),
    "logit_denominator": (int, False, None),
    "logit_modulus": (int, False, None),
    "dominant_token": (int, True, None),
    "dominant_bias": (int, False, None),
    "prompt": (list, False, int),
    "routing_uniforms": (list, False, float),
}
_RESOLVED_SPECS: dict[str, _FieldSpec] = {
    "watermark_config": (dict, False, None),
    "derived_keys": (dict, False, None),
    "detector": (dict, False, None),
}
WATERMARK_CONFIG_SPECS: dict[str, _FieldSpec] = {
    "algorithm": (str, False, None),
    "alpha": (float, False, None),
    "context_width": (int, False, None),
    "deduplicate_contexts": (str, False, None),
    "deduplicate_contexts_max_history": (int, True, None),
    "prf": (str, False, None),
    "allow_target_only_watermarking": (bool, False, None),
}
_DERIVED_KEYS_SPECS: dict[str, _FieldSpec] = {
    "key_a": (str, True, None),
    "key_b": (str, True, None),
}
_DETECTOR_SPECS: dict[str, _FieldSpec] = {
    "type": (str, False, None),
    "context_width": (int, False, None),
    "p_value_threshold": (float, False, None),
    "deduplicate_contexts": (bool, False, None),
    "alpha": (float, True, None),
    "prf_key": (str, False, None),
    "key_b_prf_key": (str, True, None),
}
_TRACE_SPECS: dict[str, _FieldSpec] = {
    "routing_draws": (int, False, None),
    "key_b_routed": (int, True, None),
    "key_b_effective": (int, True, None),
    "dedup_skips": (int, False, None),
    "partial_context_skips": (int, False, None),
}
_DETECTION_SPECS: dict[str, _FieldSpec] = {
    "score": (str, False, None),
    "p_value": (str, False, None),
    "p_value_ratio": (float, False, None),
    "num_scored_tokens": (int, False, None),
    "is_watermarked": (bool, False, None),
}


def _validate_fields(
    value: object,
    specs: dict[str, _FieldSpec],
    where: str,
) -> dict[str, object]:
    mapping = _require_mapping(value, where)
    _require_keys(mapping, frozenset(specs), where)
    for field, (expected, optional, element) in specs.items():
        location = f"{where}.{field}"
        item = mapping[field]
        if optional and item is None:
            continue
        _require_type(item, expected, location)
        if element is not None:
            for index, entry in enumerate(cast("list[object]", item)):
                _require_type(entry, element, f"{location}[{index}]")
    return mapping


def _validate_candidate(value: object, where: str) -> None:
    candidate = _require_mapping(value, where)
    _require_keys(candidate, _CANDIDATE_KEYS, where)

    configuration = _validate_fields(
        candidate["configuration"], _CONFIGURATION_SPECS, f"{where}.configuration"
    )
    _validate_fields(
        configuration["scheme_config"],
        _SCHEME_CONFIG_SPECS,
        f"{where}.configuration.scheme_config",
    )
    _validate_fields(
        configuration["fixture"],
        _FIXTURE_SPECS,
        f"{where}.configuration.fixture",
    )

    resolved = _validate_fields(
        candidate["resolved"], _RESOLVED_SPECS, f"{where}.resolved"
    )
    _validate_fields(
        resolved["watermark_config"],
        WATERMARK_CONFIG_SPECS,
        f"{where}.resolved.watermark_config",
    )
    _validate_fields(
        resolved["derived_keys"],
        _DERIVED_KEYS_SPECS,
        f"{where}.resolved.derived_keys",
    )
    _validate_fields(
        resolved["detector"], _DETECTOR_SPECS, f"{where}.resolved.detector"
    )

    generation = _require_type(candidate["generation"], list, f"{where}.generation")
    for index, token_id in enumerate(generation):
        _require_type(token_id, int, f"{where}.generation[{index}]")

    _validate_fields(candidate["trace"], _TRACE_SPECS, f"{where}.trace")
    detection = _validate_fields(
        candidate["detection"], _DETECTION_SPECS, f"{where}.detection"
    )
    for field in ("score", "p_value"):
        try:
            float.fromhex(cast(str, detection[field]))
        except ValueError:
            raise GoldenFormatError(
                _format_error(
                    f"{where}.detection.{field}",
                    f"{detection[field]!r} is not a hexadecimal float",
                )
            ) from None


def load_goldens(
    payload: object,
    *,
    source: str,
) -> dict[str, GoldenCandidatePayload]:
    """Validate a decoded goldens payload and return its candidate mapping."""
    top_level = _require_mapping(payload, source)
    schema_version = top_level.get("schema_version")
    if schema_version is not None:
        _require_type(schema_version, int, f"{source}: schema_version")
    if schema_version != GOLDEN_SCHEMA_VERSION:
        raise GoldenFormatError(
            _format_error(
                f"{source}: schema_version",
                f"expected {GOLDEN_SCHEMA_VERSION}, got {schema_version}",
            )
        )
    _require_keys(
        top_level, frozenset({"schema_version", "environment", "candidates"}), source
    )
    _validate_fields(
        top_level["environment"], _ENVIRONMENT_SPECS, f"{source}: environment"
    )

    candidates = _require_mapping(top_level["candidates"], f"{source}: candidates")
    for candidate_id, candidate in candidates.items():
        _validate_candidate(candidate, f"{source}: candidate {candidate_id}")
    return cast("dict[str, GoldenCandidatePayload]", candidates)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    # The path is not known here; read_goldens puts it in front of the message.
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise GoldenFormatError(f"key {key!r} appears more than once")
        mapping[key] = value
    return mapping


def read_goldens(path: Path) -> dict[str, GoldenCandidatePayload]:
    """Read and validate the goldens file, returning its candidate mapping."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise GoldenFormatError(_format_error(str(path), str(error))) from None
    except GoldenFormatError as error:
        raise GoldenFormatError(_format_error(str(path), str(error))) from None
    return load_goldens(payload, source=str(path))


def configured_algorithms() -> set[str]:
    return set(get_args(WatermarkingAlgorithm))


def configured_prfs() -> set[str]:
    return set(get_args(WatermarkPRFName))


def configured_algorithm_prf_combinations() -> set[tuple[str, str]]:
    return {
        (algorithm, prf)
        for algorithm in configured_algorithms()
        for prf in configured_prfs()
    }
