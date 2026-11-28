# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
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
        self.allow_storage_relocation = allow_storage_relocation
        # Backing storage data_ptr -> hashes inserted via add_encoder_outputs
        # that share that storage. External insertions (e.g. EC connector
        # receive-pool views) are intentionally untracked so they are never
        # relocated.
        self._storage_groups: dict[int, set[str]] = {}
        self._hash_to_storage: dict[str, int] = {}

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
        self._storage_groups.clear()
        self._hash_to_storage.clear()

    def _untrack(self, mm_hash: str) -> int | None:
        ptr = self._hash_to_storage.pop(mm_hash, None)
        if ptr is None:
            return None
        group = self._storage_groups.get(ptr)
        if group is not None:
            group.discard(mm_hash)
            if not group:
                del self._storage_groups[ptr]
                return None
        return ptr

    def add_encoder_outputs(
        self, mm_hashes: Sequence[str], outputs: Sequence[torch.Tensor]
    ) -> None:
        if not mm_hashes:
            return
        touched: set[int] = set()
        for mm_hash, output in zip(mm_hashes, outputs, strict=True):
            old_ptr = self._untrack(mm_hash)
            if old_ptr is not None:
                touched.add(old_ptr)
            self.encoder_outputs[mm_hash] = output
            storage = output.untyped_storage()
            if storage.nbytes() > output.nbytes:
                ptr = storage.data_ptr()
                self._storage_groups.setdefault(ptr, set()).add(mm_hash)
                self._hash_to_storage[mm_hash] = ptr
                touched.add(ptr)
        # Drop local tensor references before compacting so reclaimed storages
        # can be returned to the allocator between groups.
        del outputs, output, storage
        # Cropped outputs and duplicate hashes can retain unused batch storage.
        # Compact before connectors register the resulting addresses.
        for ptr in touched:
            self._compact_storage(ptr)

    def _compact_storage(self, ptr: int) -> None:
        group = self._storage_groups.get(ptr)
        if not group:
            return
        live_hashes: list[str] = []
        live_tensors: list[torch.Tensor] = []
        for mm_hash in group:
            tensor = self.encoder_outputs.get(mm_hash)
            if tensor is not None and tensor.untyped_storage().data_ptr() == ptr:
                live_hashes.append(mm_hash)
                live_tensors.append(tensor)
        if not live_hashes:
            self._storage_groups.pop(ptr, None)
            return
        if len(live_hashes) != len(group):
            self._storage_groups[ptr] = set(live_hashes)
        storage_bytes = live_tensors[0].untyped_storage().nbytes()
        live_bytes = sum(t.nbytes for t in live_tensors)
        # Amortize copies against bytes reclaimed.
        if live_bytes > storage_bytes - live_bytes:
            return
        try:
            first = live_tensors[0]
            if len(live_tensors) > 1 and all(
                t.device == first.device and t.dtype == first.dtype
                for t in live_tensors[1:]
            ):
                cloned = list(torch._foreach_clone(live_tensors))
            else:
                cloned = [t.clone() for t in live_tensors]
        except torch.OutOfMemoryError:
            # Publish only complete groups; failed compaction preserves originals.
            logger.warning_once(
                "Insufficient memory to compact encoder cache; retaining "
                "shared storage."
            )
            return
        for mm_hash, replacement in zip(live_hashes, cloned, strict=True):
            self.encoder_outputs[mm_hash] = replacement
            self._hash_to_storage.pop(mm_hash, None)
        self._storage_groups.pop(ptr, None)

    def free_encoder_cache(self, mm_hashes: Sequence[str]) -> None:
        touched: set[int] = set()
        for mm_hash in mm_hashes:
            output = self.encoder_outputs.pop(mm_hash, None)
            ptr = self._untrack(mm_hash)
            if output is not None and ptr is not None:
                touched.add(ptr)
        if not touched or not self.allow_storage_relocation:
            return
        del output
        # Process the whole eviction batch first: a fully evicted allocation
        # needs no copy. Only survivors can retain the evicted items' storage.
        for ptr in touched:
            self._compact_storage(ptr)
