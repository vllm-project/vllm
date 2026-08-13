# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.utils.torch_utils import async_tensor_h2d


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
        self.mm_features[req_id] = mm_features

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
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

    def cache_passthrough_embeds(
        self,
        mm_hashes: list[str],
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
        device: torch.device,
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]:
        """Cache `prompt_embeds` items directly and drop them from the batch.

        `prompt_embeds` is a passthrough modality: the tensor is already in the
        model's embedding space, so no encoder runs. Caching it here lets
        `gather_mm_embeddings` splice it via the standard `is_mm_embed` path,
        and keeps it out of `mm_kwargs`, which the real encoder cannot consume.
        """
        if not any(modality == "prompt_embeds" for modality, _ in mm_kwargs):
            return mm_hashes, mm_kwargs

        kept_hashes: list[str] = []
        kept_kwargs: list[tuple[str, MultiModalKwargsItem]] = []
        for mm_hash, (modality, mm_item) in zip(mm_hashes, mm_kwargs):
            if modality != "prompt_embeds":
                kept_hashes.append(mm_hash)
                kept_kwargs.append((modality, mm_item))
                continue
            embeds = mm_item["embedding"].data
            assert isinstance(embeds, torch.Tensor)
            self.encoder_outputs[mm_hash] = async_tensor_h2d(embeds, device=device)
        return kept_hashes, kept_kwargs
