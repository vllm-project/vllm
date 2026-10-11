# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vllm.logger import init_logger

logger = init_logger(__name__)

QuantAxis = Literal["channel", "token", "tensor"]
SplitType = Literal["head", "layer"]
Transport = Literal["raw", "compression", "raw_aggregate"]

# Product-facing key for the lossless aggregate transport (alias ``raw_agg``).
RAW_AGGREGATE_KEY = "raw_aggregate"
RAW_AGGREGATE_ALIAS = "raw_agg"
RAW_AGGREGATE_SLOT_COUNT = 2
RAW_AGGREGATE_LOGICAL_BLOCKS_PER_CHUNK = 160

# ``from_extra_config`` runs in several components (memory plan, connector);
# warn once per process when a raw_aggregate key shadows compression settings.
_RAW_AGGREGATE_WARNING_EMITTED = False


def _parse_raw_aggregate(
    extra_config: dict[str, Any],
) -> tuple[int, int] | None:
    """Return ``(slot_count, logical_blocks_per_chunk)`` or ``None``.

    The key is considered absent when it is missing or explicitly disabled
    (``false``/``null``).  ``raw_aggregate`` wins over the ``raw_agg`` alias.
    """
    if RAW_AGGREGATE_KEY in extra_config:
        raw = extra_config[RAW_AGGREGATE_KEY]
    elif RAW_AGGREGATE_ALIAS in extra_config:
        raw = extra_config[RAW_AGGREGATE_ALIAS]
    else:
        return None
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{RAW_AGGREGATE_KEY} must be true or an object with "
            f"{RAW_AGGREGATE_SLOT_COUNT}/{RAW_AGGREGATE_LOGICAL_BLOCKS_PER_CHUNK}"
        )
    unknown = set(raw) - {"slot_count", "logical_blocks_per_chunk"}
    if unknown:
        raise ValueError(
            f"Unknown {RAW_AGGREGATE_KEY} parameters: {sorted(unknown)}; "
            "supported: slot_count, logical_blocks_per_chunk"
        )
    slot_count = int(raw.get("slot_count", RAW_AGGREGATE_SLOT_COUNT))
    logical_blocks_per_chunk = int(
        raw.get("logical_blocks_per_chunk", RAW_AGGREGATE_LOGICAL_BLOCKS_PER_CHUNK)
    )
    if slot_count < 1:
        raise ValueError(f"{RAW_AGGREGATE_KEY}.slot_count must be >= 1")
    if logical_blocks_per_chunk < 1:
        raise ValueError(f"{RAW_AGGREGATE_KEY}.logical_blocks_per_chunk must be >= 1")
    return slot_count, logical_blocks_per_chunk


@dataclass(frozen=True)
class MooncakeCompressionConfig:
    # ``enabled`` is the internal name; the public JSON key is
    # ``enable-compression-pipeline``.
    enabled: bool = False
    # Which product transport this configuration describes.  ``raw`` is only
    # used for disabled configurations.
    transport: Transport = "compression"
    transformer: Literal["hadamard"] = "hadamard"
    seed: int = 3237919422
    split_type: SplitType = "layer"
    score_path: str | None = None
    hybrid_ratio: float = 0.8
    axis_key: QuantAxis = "channel"
    axis_value: QuantAxis = "token"
    high_key_num_levels: int = 12
    high_value_num_levels: int = 8
    low_key_num_levels: int = 6
    low_value_num_levels: int = 4
    codec: Literal["ans"] = "ans"
    slot_count: int = 2
    logical_blocks_per_chunk: int = 1
    pipeline: tuple[str, ...] = ("transformer", "quantizer", "codec")

    @property
    def available(self) -> bool:
        return self.enabled

    @property
    def is_raw_aggregate(self) -> bool:
        return self.transport == "raw_aggregate"

    @property
    def uses_aggregate(self) -> bool:
        return "aggregate" in self.pipeline

    @property
    def uses_transformer(self) -> bool:
        return not self.uses_aggregate and "transformer" in self.pipeline

    @property
    def uses_quantizer(self) -> bool:
        return not self.uses_aggregate and "quantizer" in self.pipeline

    @property
    def uses_codec(self) -> bool:
        return not self.uses_aggregate and "codec" in self.pipeline

    @classmethod
    def from_extra_config(
        cls, extra_config: dict[str, Any]
    ) -> MooncakeCompressionConfig:
        global _RAW_AGGREGATE_WARNING_EMITTED
        extra_config = extra_config or {}
        raw_aggregate = _parse_raw_aggregate(extra_config)
        if raw_aggregate is not None:
            # raw_aggregate has the highest priority: any enabled compression
            # settings are ignored (warn once per process so operators notice).
            compression = extra_config.get("compression") or {}
            if (
                compression.get("enable-compression-pipeline", False)
                and not _RAW_AGGREGATE_WARNING_EMITTED
            ):
                _RAW_AGGREGATE_WARNING_EMITTED = True
                logger.warning(
                    "raw_aggregate transport selected; ignoring compression "
                    "parameters: %s",
                    ", ".join(sorted(str(key) for key in compression)),
                )
            slot_count, logical_blocks_per_chunk = raw_aggregate
            return cls(
                enabled=True,
                transport="raw_aggregate",
                pipeline=("aggregate",),
                slot_count=slot_count,
                logical_blocks_per_chunk=logical_blocks_per_chunk,
            )

        raw = extra_config.get("compression", {})
        if raw is None:
            raw = {}
        enabled = raw.get("enable-compression-pipeline", False)
        # Disabled deployments intentionally ignore the rest of the object.
        if not enabled:
            return cls(enabled=False, transport="raw")

        normalized = dict(raw)
        normalized.pop("enable-compression-pipeline")
        if "pipeline" in normalized:
            normalized["pipeline"] = tuple(normalized["pipeline"])
        config = cls(enabled=True, transport="compression", **normalized)
        if config.transformer != "hadamard":
            raise ValueError(
                "Mooncake compression only supports transformer='hadamard'."
            )
        if config.codec != "ans":
            raise ValueError("Mooncake compression only supports codec='ans'.")
        if type(config.slot_count) is not int or config.slot_count < 1:
            raise ValueError("Mooncake compression slot_count must be >= 1.")
        if (
            type(config.logical_blocks_per_chunk) is not int
            or config.logical_blocks_per_chunk < 1
        ):
            raise ValueError(
                "Mooncake compression logical_blocks_per_chunk must be >= 1."
            )
        supported_stages = {"aggregate", "transformer", "quantizer", "codec"}
        unknown = set(config.pipeline) - supported_stages
        if unknown:
            raise ValueError(
                f"Unknown Mooncake compression stages: {sorted(unknown)}; "
                f"supported stages: {sorted(supported_stages)}"
            )
        if len(set(config.pipeline)) != len(config.pipeline):
            raise ValueError("Mooncake compression stages must be unique.")
        if "aggregate" not in config.pipeline:
            ordered = tuple(
                stage
                for stage in ("transformer", "quantizer", "codec")
                if stage in config.pipeline
            )
            if config.pipeline != ordered:
                raise ValueError(
                    "Mooncake compression stages must follow "
                    "transformer, quantizer, codec order."
                )
        if "aggregate" in config.pipeline and len(config.pipeline) > 1:
            logger.warning(
                "The aggregate stage takes precedence over other compression "
                "stages; configure raw_aggregate for the preferred spelling."
            )
        if not 0.0 <= config.hybrid_ratio <= 1.0:
            raise ValueError("Mooncake compression hybrid_ratio must be in [0, 1].")
        for name in (
            "high_key_num_levels",
            "high_value_num_levels",
            "low_key_num_levels",
            "low_value_num_levels",
        ):
            levels = getattr(config, name)
            if type(levels) is not int or not 2 <= levels <= 256:
                raise ValueError(
                    f"Mooncake compression {name} must be an integer in [2, 256]."
                )
        if config.split_type not in ("head", "layer"):
            raise ValueError(
                f"Unsupported Mooncake compression split_type: {config.split_type}"
            )
        if config.axis_key not in ("channel", "token", "tensor"):
            raise ValueError(
                f"Unsupported Mooncake key quantization axis: {config.axis_key}"
            )
        if config.axis_value not in ("channel", "token", "tensor"):
            raise ValueError(
                f"Unsupported Mooncake value quantization axis: {config.axis_value}"
            )
        return config
