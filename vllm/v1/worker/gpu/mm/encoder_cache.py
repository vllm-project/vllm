# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec


class EncoderCache:
    def __init__(self):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.encoder_outputs)

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        # Own the list: the uniprocess executor may share the scheduler objects.
        self.mm_features[req_id] = list(mm_features)

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def free_encoder_inputs(self, req_id: str, input_ids: list[int]) -> None:
        features = self.mm_features.get(req_id)
        if features is None:
            return
        for input_id in input_ids:
            # Keep positions/identifiers for embedding gathering and M-RoPE.
            # Replacing rather than mutating preserves EngineCore's replay input
            # and other occurrences of the same feature/hash.
            features[input_id] = replace(features[input_id], data=None)

    def restore_encoder_inputs(
        self, req_id: str, features: dict[int, MultiModalFeatureSpec]
    ) -> None:
        for input_id, feature in features.items():
            self.mm_features[req_id][input_id] = replace(
                self.mm_features[req_id][input_id], data=feature.data
            )

    def reset_mm_cache(self) -> None:
        """Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # NOTE: v2 encoder cache profiling skips the multi-modal cache
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
