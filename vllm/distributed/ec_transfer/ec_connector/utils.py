# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector helper utilities."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def placeholder_metadata_fields(
    modality: str, model_config: "ModelConfig", cache: dict[str, set[str]]
) -> set[str]:
    """Which processed keys this model needs published for `modality`.

    Read from `MultiModalDataParser.embedding_fields`, the same declaration
    the consumer's parser requires, so the two cannot drift. An empty set
    means the modality cannot be delivered out of band, and the consumer
    will process the media itself.

    Args:
        modality: the modality to report fields for.
        model_config: the producer's model config.
        cache: per-connector memo of the answer, keyed by modality.
    """
    if modality in cache:
        return cache[modality]

    fields: set[str] = set()
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY

        info = MULTIMODAL_REGISTRY.create_processor(model_config).info
        fields = info.data_parser.placeholder_metadata_fields(modality)
    except Exception:
        # Reporting nothing is a safe degradation: the consumer falls back to
        # processing the media itself.
        logger.warning(
            "Could not determine the placeholder metadata fields for "
            "modality %s; the consumer will preprocess the media itself.",
            modality,
            exc_info=True,
        )

    cache[modality] = fields
    return fields


def build_ec_items(
    request: "Request", model_config: "ModelConfig", cache: dict[str, set[str]]
) -> list[dict[str, Any]]:
    """Report each item's cache key and grid so a consumer can skip the
    image transform.

    A consumer only needs the grid to size the prompt's placeholder range;
    the embedding itself arrives through the connector. Reporting the grid
    the producer actually computed keeps the two sides in agreement without
    the caller re-deriving it from the raw media.
    """
    items: list[dict[str, Any]] = []
    for feature in request.mm_features:
        metadata = {}
        # `data` is None for items served from the processor cache, in which
        # case the metadata is unavailable here and the consumer has to fall
        # back to processing the media itself.
        if feature.data is not None:
            wanted = placeholder_metadata_fields(feature.modality, model_config, cache)
            metadata = {
                key: value.tolist()
                for key, value in feature.data.get_data().items()
                if key in wanted and isinstance(value, torch.Tensor)
            }
        items.append({"mm_hash": feature.identifier, **metadata})
    return items


class ECOutputAggregator:
    """Merge every worker's EC connector output onto the single
    ModelRunnerOutput that reaches the scheduler.

    Mirrors KVOutputAggregator: only `output_rank`'s output is returned to the
    scheduler, but the EC connector may have run on any rank.
    """

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        output = outputs[output_rank]
        if not output:
            return None

        finished_sending = set[str]()
        finished_recving = set[str]()
        worker_meta = None
        for model_runner_output in outputs:
            assert model_runner_output is not None
            ec_output = model_runner_output.ec_connector_output
            if not ec_output:
                continue

            finished_sending |= ec_output.finished_sending or set()
            finished_recving |= ec_output.finished_recving or set()

            if meta := ec_output.ec_connector_worker_meta:
                worker_meta = (
                    meta if worker_meta is None else worker_meta.aggregate(meta)
                )

        aggregated = ECConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            ec_connector_worker_meta=worker_meta,
        )
        if aggregated.is_empty():
            output.ec_connector_output = None
            return output

        # `output` is the shared empty output whenever `output_rank` had no work,
        # so attach through the copy-on-write helper.
        return ModelRunnerOutput.with_ec_conn_output(output, aggregated)
