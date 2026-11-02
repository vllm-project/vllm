# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem


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
        # The uniprocess executor can share this list with EngineCore.
        self.mm_features[req_id] = list(mm_features)

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def update_encoder_inputs(
        self, req_id: str, updates: dict[int, MultiModalKwargsItem | None]
    ) -> None:
        features = self.mm_features[req_id]
        for input_id, data in updates.items():
            # Replace entries so EngineCore keeps its original payload for replay.
            features[input_id] = replace(features[input_id], data=data)

    def reset_mm_cache(self) -> None:
        """Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # NOTE: v2 encoder cache profiling skips the multi-modal cache
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
