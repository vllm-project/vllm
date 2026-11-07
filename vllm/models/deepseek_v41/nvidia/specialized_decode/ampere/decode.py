# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 Native Specialized Decode Executor for Ampere."""

from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING

from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
)

from .. import DecodeConfig, max_rows
from ..boundary.boundary import AmpereBoundaries
from ..runtime.decode import (
    LayerPlan,
    SpecializedDecode,
)
from .steps import AmpereAttentionStep, AmpereMoEStep

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4Model

    from .draft import AmpereDraftDecode

logger = init_logger(__name__)


class AmpereSpecializedDecode(SpecializedDecode):
    """The decode executor over one backbone's local layers on Ampere SM80."""

    def __init__(
        self,
        backbone: "DeepseekV4Model",
        config: DecodeConfig,
        lm_head: nn.Module,
    ) -> None:
        super().__init__(backbone, config, lm_head)
        self.max_rows = max_rows()
        layers = [
            layer
            for layer in islice(
                backbone.layers, backbone.start_layer, backbone.end_layer
            )
            if isinstance(layer, DeepseekV4DecoderLayer)
        ]
        self.mhc_stream = getattr(layers[0], "mhc_stream", None) if layers else None
        self.boundaries = AmpereBoundaries(self.mhc_stream)

        self.plans = [
            LayerPlan(
                layer_id=layer_id,
                layer=layer,
                attention=AmpereAttentionStep(layer.attn),
                moe=AmpereMoEStep(layer.ffn, self.max_rows),
                engram=layer.engram,
            )
            for layer_id, layer in zip(
                range(backbone.start_layer, backbone.end_layer), layers
            )
        ]
        logger.info_once(
            "Ampere SM80 Specialized Decode initialized with %d decoder layers.",
            len(self.plans),
        )

    def for_draft(self, backbone: "DSparkDeepseekV4Model") -> "AmpereDraftDecode":
        from .draft import AmpereDraftDecode

        return AmpereDraftDecode(self, backbone)

    def warm_up(
        self,
        vllm_config: VllmConfig,
        drafts: Sequence = (),
    ) -> None:
        """Warm up specialized decode kernels for configured graph capture sizes."""
        sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
        cap = self.max_rows
        capture_rows = sorted({n for n in sizes if 1 <= n <= cap})
        logger.info_once(
            "Ampere SM80 specialized decode warmed up for capture sizes: %s",
            capture_rows,
        )
