# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec


class EncoderCache:
    def __init__(self):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        self.mm_features[req_id] = mm_features

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # TODO: Implement MM budget for encoder dummy run
        pass

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
