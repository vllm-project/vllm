# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector helper utilities."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal.inputs import MultiModalFeatureSpec

logger = init_logger(__name__)


class PlaceholderMetadataResolver:
    """Resolves which processed keys a model needs published per modality.

    Reads `MultiModalDataParser.embedding_fields`, the same declaration the
    consumer's parser requires, so the two cannot drift. An empty set means
    the modality cannot be delivered out of band, and the consumer will
    process the media itself.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        self._model_config = model_config
        self._cache: dict[str, set[str]] = {}

    def fields_for(self, modality: str) -> set[str]:
        if modality in self._cache:
            return self._cache[modality]

        fields: set[str] = set()
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.create_processor(self._model_config).info
            fields = info.data_parser.placeholder_metadata_fields(modality)
        except Exception:
            logger.warning(
                "Could not determine the placeholder metadata fields for "
                "modality %s; the consumer will preprocess the media itself.",
                modality,
                exc_info=True,
            )

        self._cache[modality] = fields
        return fields


def collect_ec_item_metadata(
    mm_features: "list[MultiModalFeatureSpec]",
    resolver: PlaceholderMetadataResolver,
) -> list[dict[str, Any]]:
    """Build one `ec_items` entry per feature for `request_finished()`.

    Each entry carries the feature's mm_hash plus whatever placeholder
    metadata `resolver` says this model needs published for its modality, so
    a consumer can skip the image transform. `data` is None for items served
    from the processor cache, in which case the metadata is unavailable here
    and the consumer has to fall back to processing the media itself.
    """
    items: list[dict[str, Any]] = []
    for feature in mm_features:
        metadata: dict[str, Any] = {}
        if feature.data is not None:
            wanted = resolver.fields_for(feature.modality)
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
