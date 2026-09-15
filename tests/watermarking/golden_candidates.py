# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import TypedDict, get_args

import torch

from vllm.config.watermarking import (
    WatermarkConfig,
    WatermarkContextScope,
    WatermarkingAlgorithm,
    WatermarkPRFName,
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

REGENERATE_COMMAND = "python -m tests.watermarking.generate_goldens"

# Relative tolerance for score and p_value. The hex payload round-trips
# exactly, but libm differs by an ULP across builds, so the reproduced value is
# compared with GOLDEN_FLOAT_RTOL; the absolute floor only matters when the
# golden value is exactly zero.
GOLDEN_FLOAT_RTOL = 1e-9
GOLDEN_FLOAT_ATOL = 1e-300


def floats_match(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=GOLDEN_FLOAT_RTOL,
        abs_tol=GOLDEN_FLOAT_ATOL,
    )


class GoldenDetectionPayload(TypedDict):
    score: str
    p_value: str
    num_scored_tokens: int
    is_watermarked: bool


class GoldenCandidatePayload(TypedDict):
    configuration: dict[str, object]
    generation: list[int]
    detection: GoldenDetectionPayload


class GoldenPayload(TypedDict):
    schema_version: int
    candidates: dict[str, GoldenCandidatePayload]


@dataclass(frozen=True)
class DeterministicGenerationFixture:
    vocabulary_size: int = 128
    num_tokens: int = 64
    logit_denominator: int = 16
    dominant_token: int | None = None
    dominant_bias: int = 0
    routing_uniforms: tuple[float, ...] = (0.05, 0.95, 0.35, 0.75)

    def logits(self, position: int) -> torch.Tensor:
        token_ids = torch.arange(self.vocabulary_size, dtype=torch.int64)
        values = (token_ids * 37 + position * 17) % 113 - 56
        if self.dominant_token is not None:
            values[self.dominant_token] += self.dominant_bias
        return (values.to(torch.float32) / self.logit_denominator).unsqueeze(0)


@dataclass(frozen=True)
class WatermarkingSchemeConfig:
    context_width: int = 4
    generation_alpha: float = 0.1
    detection_alpha: float = 0.2
    generation_deduplicate_contexts: WatermarkContextScope = "single_turn"
    generation_deduplicate_contexts_max_history: int | None = 8192
    detection_deduplicate_contexts: bool = True
    p_value_threshold: float = 0.01


class _DeterministicSampler:
    def __init__(self, routing_uniforms: tuple[float, ...]) -> None:
        self.routing_uniforms = routing_uniforms
        self.routing_position = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != 2:
            return logits.argmax(dim=-1)

        probability_a = logits.softmax(dim=-1)[:, 0]
        uniform = self.routing_uniforms[
            self.routing_position % len(self.routing_uniforms)
        ]
        self.routing_position += 1
        return (probability_a <= uniform).to(torch.int64)


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
            "fixture": {
                "vocabulary_size": self.fixture.vocabulary_size,
                "num_tokens": self.fixture.num_tokens,
                "logit_denominator": self.fixture.logit_denominator,
                "dominant_token": self.fixture.dominant_token,
                "dominant_bias": self.fixture.dominant_bias,
                "routing_uniforms": list(self.fixture.routing_uniforms),
            },
        }

    def generate(self) -> list[int]:
        config = WatermarkConfig(
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
        watermarker = create_watermarker(config)
        sampler = _DeterministicSampler(self.fixture.routing_uniforms)
        generated: list[int] = []

        for position in range(self.fixture.num_tokens):
            context = self._context(generated)
            skip_mask = self._skip_mask(generated, context)
            sample = watermarker.sample(
                self.fixture.logits(position),
                context,
                sampler,
                skip_mask,
            )
            generated.append(int(sample.token_ids.item()))

        return generated

    def detect(self, token_ids: list[int]) -> WatermarkDetection:
        detector = DETECTOR_FACTORIES[self.scheme](
            self.key,
            self.scheme_config,
            self.prf,
        )
        return detector.detect(token_ids)

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

        storage = torch.zeros(
            (1, self.fixture.num_tokens),
            dtype=torch.int64,
        )
        if generated:
            storage[0, : len(generated)] = torch.tensor(generated)
        return repeated_context_mask(
            storage,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([len(generated)]),
            context,
            self.scheme_config.generation_deduplicate_contexts_max_history,
            include_prompt=scope == "all",
            skip_partial_context=scope == "all",
        )


BALANCED_FIXTURE = DeterministicGenerationFixture()
REPETITIVE_FIXTURE = DeterministicGenerationFixture(
    vocabulary_size=16,
    num_tokens=32,
    logit_denominator=16,
    dominant_token=3,
    dominant_bias=160,
)


def _candidate(
    id: str,
    scheme: WatermarkingAlgorithm,
    key: int,
    *,
    context_width: int = 4,
    generation_alpha: float = 0.1,
    detection_alpha: float = 0.2,
    generation_deduplicate_contexts: WatermarkContextScope = "single_turn",
    generation_deduplicate_contexts_max_history: int | None = 8192,
    detection_deduplicate_contexts: bool = True,
    fixture: DeterministicGenerationFixture = BALANCED_FIXTURE,
) -> WatermarkingCandidate:
    return WatermarkingCandidate(
        id=id,
        scheme=scheme,
        scheme_config=WatermarkingSchemeConfig(
            context_width=context_width,
            generation_alpha=generation_alpha,
            detection_alpha=detection_alpha,
            generation_deduplicate_contexts=generation_deduplicate_contexts,
            generation_deduplicate_contexts_max_history=(
                generation_deduplicate_contexts_max_history
            ),
            detection_deduplicate_contexts=detection_deduplicate_contexts,
        ),
        prf="philox",
        key=key,
        fixture=fixture,
    )


WATERMARKING_CANDIDATES = (
    _candidate("gumbel-philox-key0-cw1", "gumbel", 0, context_width=1),
    _candidate("gumbel-philox-key42-cw4", "gumbel", 42),
    _candidate(
        "gumbel-philox-key2p32-cw5-no-dedup",
        "gumbel",
        2**32,
        context_width=5,
        generation_deduplicate_contexts="none",
        detection_deduplicate_contexts=False,
    ),
    _candidate(
        "gumbel-philox-keymax-cw16-all",
        "gumbel",
        2**64 - 1,
        context_width=16,
        generation_deduplicate_contexts="all",
    ),
    _candidate(
        "gumbel-philox-repeated-cw1",
        "gumbel",
        42,
        context_width=1,
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "gumbel-philox-repeated-cw4-all",
        "gumbel",
        42,
        generation_deduplicate_contexts="all",
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "dual-key-gumbel-philox-key0-cw1-alpha0",
        "dual_key_gumbel",
        0,
        context_width=1,
        generation_alpha=0.0,
        detection_alpha=0.0,
    ),
    _candidate(
        "dual-key-gumbel-philox-key42-cw4",
        "dual_key_gumbel",
        42,
    ),
    _candidate(
        "dual-key-gumbel-philox-key2p32-cw5-balanced",
        "dual_key_gumbel",
        2**32,
        context_width=5,
        generation_alpha=0.5,
        detection_alpha=0.5,
    ),
    _candidate(
        "dual-key-gumbel-philox-keymax-cw16-alpha1",
        "dual_key_gumbel",
        2**64 - 1,
        context_width=16,
        generation_alpha=1.0,
        detection_alpha=1.0,
    ),
    _candidate(
        "dual-key-gumbel-philox-repeated-cw1",
        "dual_key_gumbel",
        42,
        context_width=1,
        fixture=REPETITIVE_FIXTURE,
    ),
    _candidate(
        "dual-key-gumbel-philox-repeated-cw4-all",
        "dual_key_gumbel",
        42,
        generation_deduplicate_contexts="all",
        fixture=REPETITIVE_FIXTURE,
    ),
)


def _relative_difference(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual - expected) / abs(expected)


def _compare_hex_float(
    field: str,
    actual: float,
    expected_hex: object,
    differences: list[str],
) -> None:
    if not isinstance(expected_hex, str):
        differences.append(f"{field}: golden value {expected_hex!r} is not a string")
        return
    try:
        expected = float.fromhex(expected_hex)
    except ValueError:
        differences.append(
            f"{field}: golden value {expected_hex!r} is not a hexadecimal float"
        )
        return
    if not floats_match(actual, expected):
        differences.append(
            f"{field}: expected {expected_hex} ({expected!r}), "
            f"got {actual.hex()} ({actual!r}), relative difference "
            f"{_relative_difference(actual, expected):.3e}"
        )


def _compare_exact(
    field: str,
    actual: object,
    expected: object,
    differences: list[str],
) -> None:
    if actual != expected:
        differences.append(f"{field}: expected {expected!r}, got {actual!r}")


def compare_golden(
    candidate: WatermarkingCandidate,
    golden: GoldenCandidatePayload,
) -> list[str]:
    """Return one line per field that drifted, empty when the golden reproduces.

    Configuration, tokens, num_scored_tokens and is_watermarked are compared
    exactly; score and p_value are compared with GOLDEN_FLOAT_RTOL.
    """
    differences: list[str] = []

    configuration = candidate.configuration()
    expected_configuration = golden["configuration"]
    for key in sorted(set(configuration) | set(expected_configuration)):
        _compare_exact(
            f"configuration.{key}",
            configuration.get(key),
            expected_configuration.get(key),
            differences,
        )

    generated = candidate.generate()
    expected_generation = golden["generation"]
    if generated != expected_generation:
        first_difference = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(generated, expected_generation)
                )
                if actual != expected
            ),
            min(len(generated), len(expected_generation)),
        )
        differences.append(
            f"generation: first differs at index {first_difference} "
            f"({len(generated)} tokens generated, "
            f"{len(expected_generation)} in the golden): "
            f"expected {expected_generation}, got {generated}"
        )

    # Detection runs on the golden tokens so a token drift and a detection
    # drift are reported independently.
    detection = candidate.detect(expected_generation)
    expected_detection = golden["detection"]
    _compare_hex_float(
        "detection.score",
        detection.score,
        expected_detection["score"],
        differences,
    )
    _compare_hex_float(
        "detection.p_value",
        detection.p_value,
        expected_detection["p_value"],
        differences,
    )
    _compare_exact(
        "detection.num_scored_tokens",
        detection.num_scored_tokens,
        expected_detection["num_scored_tokens"],
        differences,
    )
    _compare_exact(
        "detection.is_watermarked",
        detection.is_watermarked,
        expected_detection["is_watermarked"],
        differences,
    )
    return differences


def golden_payload() -> GoldenPayload:
    candidates: dict[str, GoldenCandidatePayload] = {}
    for candidate in WATERMARKING_CANDIDATES:
        token_ids = candidate.generate()
        detection = candidate.detect(token_ids)
        candidates[candidate.id] = {
            "configuration": candidate.configuration(),
            "generation": token_ids,
            "detection": {
                "score": detection.score.hex(),
                "p_value": detection.p_value.hex(),
                "num_scored_tokens": detection.num_scored_tokens,
                "is_watermarked": detection.is_watermarked,
            },
        }
    return {"schema_version": 1, "candidates": candidates}


def load_goldens(payload: object) -> dict[str, GoldenCandidatePayload]:
    if not isinstance(payload, dict):
        raise ValueError("Watermarking golden payload must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported watermarking golden schema")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("Watermarking goldens must contain a candidate mapping")
    return candidates


def configured_algorithms() -> set[str]:
    return set(get_args(WatermarkingAlgorithm))


def configured_prfs() -> set[str]:
    return set(get_args(WatermarkPRFName))
