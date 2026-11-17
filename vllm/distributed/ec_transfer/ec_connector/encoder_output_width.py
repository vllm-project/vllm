# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder output widths for sizing EC transfers.

An encoder can emit features wider than the LM embedding (DeepStack packs
several layers on the last dim) by an amount only the model knows. Producers
size transfers before the first real item is encoded, so the engine measures
the width once at startup by encoding a dummy item per modality.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _measure_on_worker(worker: Any) -> dict[str, int]:
    """Measure encoder output widths on a worker."""
    from vllm.multimodal.encoder_budget import measure_encoder_output_layouts

    return {
        modality: width
        for modality, (width, _) in measure_encoder_output_layouts(worker).items()
    }


def measure_encoder_output_widths(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> None:
    """Record the encoder output widths on an EC producer's config.

    Call after the model is loaded and before the KV cache is allocated: the
    Scheduler's connector reads the widths, and once the KV cache holds the
    memory budget there is no room left for a dummy encode.

    Raises:
        ValueError: No worker could encode a dummy item, or workers disagree.

    """
    ec_config = vllm_config.ec_transfer_config
    if ec_config is None or not ec_config.is_ec_producer:
        return
    reports = [widths for widths in collective_rpc(_measure_on_worker) if widths]
    if not reports:
        raise ValueError(
            "EC producer could not measure the encoder output width: the model "
            "encoded no dummy multimodal input."
        )
    if any(widths != reports[0] for widths in reports):
        raise ValueError(f"Workers disagree on the encoder output widths: {reports}")
    logger.info("EC producer encoder output widths: %s", reports[0])
    ec_config.mm_encoder_output_widths = reports[0]


def get_encoder_output_width(vllm_config: "VllmConfig", modality: str) -> int:
    """Encoder output width for `modality`, as measured at producer startup.

    Raises:
        ValueError: No width was measured for `modality`, so a transfer
            cannot be sized.

    """
    assert vllm_config.ec_transfer_config is not None
    widths = vllm_config.ec_transfer_config.mm_encoder_output_widths
    if modality not in widths:
        raise ValueError(
            f"No encoder output width was measured for modality {modality!r}; "
            f"measured: {sorted(widths)}"
        )
    return widths[modality]
