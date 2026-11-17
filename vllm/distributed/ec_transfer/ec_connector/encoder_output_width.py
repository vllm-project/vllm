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

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def bytes_per_embedding_from_layouts(
    reports: list[dict[str, tuple[int, int]]],
) -> int:
    """Largest embedding payload in bytes across the measured layouts.

    Each report maps a modality to `(width, element_size)`. Different
    modalities and ranks share one token capacity, so the budget uses the
    maximum cost. A non-positive size is not a usable layout.
    """
    bytes_per_embedding = 0
    for report in reports:
        for modality, (width, element_size) in report.items():
            if width <= 0 or element_size <= 0:
                raise ValueError(
                    "Measured encoder output layout for modality "
                    f"{modality!r} is {width} x {element_size} bytes, which "
                    "cannot size an encoder cache."
                )
            bytes_per_embedding = max(bytes_per_embedding, width * element_size)
    if bytes_per_embedding <= 0:
        raise ValueError(
            "Encoder output measurement did not produce a usable embedding size."
        )
    return bytes_per_embedding


@torch.inference_mode()
def _measure_on_worker(worker: Any) -> dict[str, tuple[int, int]]:
    """Encode one dummy item per modality; runs on every worker.

    Returns `modality -> (width, element_size)`, or {} on ranks that hold no
    encoder. Both values come from the emitted tensor, not from model config.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.utils import group_and_batch_mm_kwargs
    from vllm.utils.torch_utils import PIN_MEMORY

    runner = worker.model_runner
    if not runner.supports_mm_inputs or not get_pp_group().is_first_rank:
        return {}
    budget = MultiModalBudget(
        worker.vllm_config, MULTIMODAL_REGISTRY, enable_cache=False
    )
    model = runner.get_model()
    layouts: dict[str, tuple[int, int]] = {}
    for modality in budget.mm_max_toks_per_item:
        items = budget.get_dummy_encoder_profile_inputs(modality, 1)
        if not items:
            continue
        _, _, mm_kwargs = next(
            group_and_batch_mm_kwargs(
                items, device=runner.device, pin_memory=PIN_MEMORY
            )
        )
        outputs = model.embed_multimodal(**mm_kwargs)
        output = outputs[0]
        if output.ndim != 2 or output.shape[-1] <= 0:
            raise ValueError(
                "Cannot size encoder outputs for modality "
                f"{modality!r}: expected a 2D (embeddings, width) tensor, "
                f"got shape {tuple(output.shape)}."
            )
        layouts[modality] = (int(output.shape[-1]), int(output.element_size()))
        del outputs
    return layouts


def measure_encoder_output_widths(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> int | None:
    """Measure encoder output layouts once, before the KV cache is allocated.

    EC producers still record per-modality widths for transfer sizing. When
    `--encoder-cache-size-gb` is set, the same dummy encode also supplies the
    bytes per embedding used to resolve the token capacity. Ranks without an
    encoder contribute nothing. Widths must agree across ranks that do measure;
    byte cost uses the maximum so one token budget covers every layout.

    Returns:
        Bytes per embedding when a byte budget is configured, otherwise None.

    Raises:
        ValueError: A required measurement is missing, widths disagree, or the
            output is not a 2D embedding tensor.

    """
    ec_config = vllm_config.ec_transfer_config
    need_widths = ec_config is not None and ec_config.is_ec_producer
    need_bytes = vllm_config.scheduler_config.encoder_cache_size_gb is not None
    if not need_widths and not need_bytes:
        return None
    reports = [layouts for layouts in collective_rpc(_measure_on_worker) if layouts]
    if not reports:
        if need_widths:
            raise ValueError(
                "EC producer could not measure the encoder output width: the "
                "model encoded no dummy multimodal input."
            )
        raise ValueError(
            "--encoder-cache-size-gb requires a local encoder tower so startup "
            "can measure the encoder output layout. This process encoded no "
            "dummy multimodal input. Instances that drop the encoder tower and "
            "have no local layout measurement are not supported."
        )
    width_maps = [
        {modality: width for modality, (width, _) in report.items()}
        for report in reports
    ]
    if any(widths != width_maps[0] for widths in width_maps):
        if need_widths:
            raise ValueError(
                f"Workers disagree on the encoder output widths: {width_maps}"
            )
        logger.warning(
            "Encoder output widths differ across ranks (%s). The byte budget "
            "uses the largest measured embedding.",
            width_maps,
        )
    if need_widths:
        assert ec_config is not None
        logger.info("EC producer encoder output widths: %s", width_maps[0])
        ec_config.mm_encoder_output_widths = width_maps[0]
    if not need_bytes:
        return None
    bytes_per_embedding = bytes_per_embedding_from_layouts(reports)
    if len({bytes_per_embedding_from_layouts([report]) for report in reports}) > 1:
        logger.warning(
            "Encoder embedding sizes differ across measurements (%s). "
            "Using %d bytes per embedding.",
            reports,
            bytes_per_embedding,
        )
    else:
        logger.info(
            "Encoder embedding size for the cache budget: %d bytes.",
            bytes_per_embedding,
        )
    return bytes_per_embedding


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
