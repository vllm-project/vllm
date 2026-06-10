# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec


class EncoderCache:
    """GPU-side cache of encoder outputs (e.g. vision embeddings) keyed by
    mm_hash, plus per-request multimodal feature metadata.

    Eviction is requested by the scheduler via `free_encoder_cache()`, but
    the scheduler's view of request progress can be stale by the time the
    request reaches the worker: under async scheduling `num_computed_tokens`
    is advanced speculatively and rolled back on draft-token rejection, and
    an entry can be shared by concurrent requests with identical content
    (same mm_hash). Evicting such an entry while an in-flight request may
    still read it crashes the engine with "Encoder cache miss" in
    `_gather_mm_embeddings` (https://github.com/vllm-project/vllm/issues/38551).

    To prevent this, eviction of an entry that is still referenced by any
    in-flight request is deferred until the last referencing request is
    removed. Memory overhead is bounded by the encoder outputs of in-flight
    requests; entries with no live references are evicted eagerly, exactly
    as before.

    For encoder-decoder models (e.g. Whisper), pass `eager_eviction=True`:
    the scheduler frees those entries only once the first decode token
    exists, at which point cross-attention KV is cached and the encoder
    output is provably dead, so deferral is unnecessary.
    """

    def __init__(
        self,
        eager_eviction: bool = False,
        encoder_outputs: dict[str, torch.Tensor] | None = None,
    ):
        # req_id -> MM features
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        # MM hash -> encoder outputs. May be a caller-owned dict so that
        # existing code holding a reference to it observes evictions.
        self.encoder_outputs: dict[str, torch.Tensor] = (
            encoder_outputs if encoder_outputs is not None else {}
        )
        # Evict on free_encoder_cache() even while referenced.
        self.eager_eviction = eager_eviction
        # MM hash -> ids of in-flight requests referencing it.
        self._mm_hash_refs: dict[str, set[str]] = {}
        # Hashes whose eviction was requested while still referenced.
        self._pending_free: set[str] = set()

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        self.mm_features[req_id] = mm_features
        if not mm_features:
            return
        for mm_hash in {f.identifier for f in mm_features}:
            self._mm_hash_refs.setdefault(mm_hash, set()).add(req_id)
            # The entry is referenced again; cancel any deferred eviction.
            self._pending_free.discard(mm_hash)

    def update_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        """Replace the request's MM features, carrying references over."""
        old_features = self.mm_features.get(req_id) or []
        old_hashes = {f.identifier for f in old_features}
        self.add_request(req_id, mm_features)
        new_hashes = {f.identifier for f in mm_features or []}
        for mm_hash in old_hashes - new_hashes:
            self._unref(mm_hash, req_id)

    def remove_request(self, req_id: str) -> None:
        mm_features = self.mm_features.pop(req_id, None)
        if not mm_features:
            return
        for mm_hash in {f.identifier for f in mm_features}:
            self._unref(mm_hash, req_id)

    def _unref(self, mm_hash: str, req_id: str) -> None:
        refs = self._mm_hash_refs.get(mm_hash)
        if refs is None:
            return
        refs.discard(req_id)
        if refs:
            return
        del self._mm_hash_refs[mm_hash]
        if mm_hash in self._pending_free:
            # The scheduler requested eviction while the entry was still
            # referenced; the last reference is gone, so evict now.
            self._pending_free.discard(mm_hash)
            self.encoder_outputs.pop(mm_hash, None)

    def free_encoder_cache(self, mm_hash: str) -> None:
        if not self.eager_eviction and self._mm_hash_refs.get(mm_hash):
            # An in-flight request may still read this entry: the scheduler
            # frees based on speculative progress that can be rolled back on
            # draft-token rejection, and concurrent requests can share the
            # same content hash. Defer eviction until the last referencing
            # request is removed.
            self._pending_free.add(mm_hash)
        else:
            self.encoder_outputs.pop(mm_hash, None)

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
        self._pending_free.clear()
