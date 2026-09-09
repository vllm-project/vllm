# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestStaticYarnConfig:
    """Validated immutable YaRN profiles selected per request."""

    factors: tuple[float, ...]
    original_max_position: int
    factor_offsets: tuple[tuple[float, int], ...]
    factor_profile_ids: tuple[tuple[float, str], ...]

    @classmethod
    def from_hf_config(
        cls,
        factors: list[float] | tuple[float, ...] | None,
        hf_text_config: Any,
    ) -> "RequestStaticYarnConfig | None":
        if factors is None:
            return None

        normalized = tuple(float(factor) for factor in factors)
        if not normalized or normalized != tuple(sorted(set(normalized))):
            raise ValueError(
                "Request-static YaRN factors must be non-empty, unique, and sorted"
            )
        if any(factor < 1.0 for factor in normalized):
            raise ValueError("Request-static YaRN factors must be at least 1")

        rope_parameters = getattr(hf_text_config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            raise ValueError("Request-static YaRN requires one RoPE parameter mapping")
        if rope_parameters.get("rope_type") != "yarn":
            raise ValueError("Request-static YaRN requires rope_type='yarn'")
        if "mrope_section" not in rope_parameters:
            raise ValueError("Request-static YaRN currently requires mRoPE")

        configured_factor = float(rope_parameters["factor"])
        if configured_factor != normalized[-1]:
            raise ValueError(
                "The configured YaRN factor must equal the largest request-static "
                f"factor ({normalized[-1]:g})"
            )

        original_max_position = int(rope_parameters["original_max_position_embeddings"])
        offsets: list[tuple[float, int]] = []
        profiles: list[tuple[float, str]] = []
        offset = 0
        for factor in normalized:
            cache_rows = 4 * original_max_position * factor
            if not cache_rows.is_integer():
                raise ValueError(
                    f"Request-static YaRN factor {factor:g} produces a fractional "
                    "mRoPE cache size"
                )
            offsets.append((factor, offset))
            offset += int(cache_rows)

            profile_parameters = dict(rope_parameters)
            profile_parameters["factor"] = factor
            profile_parameters.pop("request_static_factors", None)
            canonical = json.dumps(
                profile_parameters,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            profiles.append((factor, f"yarn:{digest}"))

        return cls(
            factors=normalized,
            original_max_position=original_max_position,
            factor_offsets=tuple(offsets),
            factor_profile_ids=tuple(profiles),
        )

    def apply_to_hf_config(self, hf_text_config: Any) -> None:
        hf_text_config.rope_parameters["request_static_factors"] = list(self.factors)

    def select_factor(self, required_tokens: int) -> float:
        for factor in self.factors:
            if required_tokens <= self.original_max_position * factor:
                return factor
        raise ValueError(
            f"Request needs {required_tokens} tokens, but the largest "
            "request-static YaRN profile covers "
            f"{self.original_max_position * self.factors[-1]:g}"
        )

    def offset_for_factor(self, factor: float) -> int:
        for candidate, offset in self.factor_offsets:
            if candidate == factor:
                return offset
        raise ValueError(f"Unknown request-static YaRN factor: {factor:g}")

    def profile_id_for_factor(self, factor: float) -> str:
        for candidate, profile_id in self.factor_profile_ids:
            if candidate == factor:
                return profile_id
        raise ValueError(f"Unknown request-static YaRN factor: {factor:g}")
