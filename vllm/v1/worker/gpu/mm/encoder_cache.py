# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem

logger = init_logger(__name__)


class EncoderCache:
    def __init__(self, *, allow_storage_relocation: bool = True):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs
        self.encoder_outputs: dict[str, torch.Tensor] = {}
        self._allow_storage_relocation = allow_storage_relocation

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
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_outputs.clear()

    def add_encoder_outputs(
        self, mm_hashes: list[str], outputs: list[torch.Tensor]
    ) -> None:
        new_outputs = dict(zip(mm_hashes, outputs))
        self.encoder_outputs.update(new_outputs)
        shared: dict[int, list[str]] = {}
        for mm_hash, output in new_outputs.items():
            storage = output.untyped_storage()
            if storage.nbytes() > output.nbytes:
                shared.setdefault(storage.data_ptr(), []).append(mm_hash)
        for mm_hashes in shared.values():
            # Cropped outputs or duplicate hashes can leave an allocation only
            # partly represented in the cache even before the first eviction.
            # Connectors receive the final addresses after insertion returns;
            # either compact or original storage must then stay in place.
            self._compact_encoder_outputs(mm_hashes)

    def _compact_encoder_outputs(self, mm_hashes: list[str]) -> None:
        if not mm_hashes:
            return
        if self._allow_storage_relocation:
            storage_bytes = (
                self.encoder_outputs[mm_hashes[0]].untyped_storage().nbytes()
            )
            live_bytes = sum(self.encoder_outputs[h].nbytes for h in mm_hashes)
            # Amortize copies against bytes reclaimed. Connector inputs are
            # isolated at insertion instead: their addresses cannot move later.
            if live_bytes > storage_bytes - live_bytes:
                return
        replacements: dict[str, torch.Tensor] = {}
        try:
            for mm_hash in mm_hashes:
                replacements[mm_hash] = self.encoder_outputs[mm_hash].clone()
        except torch.OutOfMemoryError:
            # Compaction temporarily needs both allocations. Keep the complete
            # original group if any optional copy cannot be allocated.
            replacements.clear()
            logger.warning_once(
                "Insufficient memory to compact encoder cache; retaining "
                "shared storage."
            )
            return
        self.encoder_outputs.update(replacements)

    def free_encoder_cache(self, mm_hashes: list[str]) -> None:
        shared_storage: dict[int, list[str]] = {}
        for mm_hash in mm_hashes:
            output = self.encoder_outputs.pop(mm_hash, None)
            if output is not None and self._allow_storage_relocation:
                storage = output.untyped_storage()
                if storage.nbytes() > output.nbytes:
                    shared_storage.setdefault(storage.data_ptr(), [])
        if not shared_storage:
            return
        del storage
        # Process the whole eviction batch first: a fully evicted allocation
        # needs no copy. Only survivors can retain the evicted items' storage.
        for mm_hash, output in self.encoder_outputs.items():
            if (ptr := output.untyped_storage().data_ptr()) in shared_storage:
                shared_storage[ptr].append(mm_hash)
        del output
        for survivors in shared_storage.values():
            self._compact_encoder_outputs(survivors)
