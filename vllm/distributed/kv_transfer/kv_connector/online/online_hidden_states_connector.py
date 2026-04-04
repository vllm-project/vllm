# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
    extract_from_kv_cache,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class OnlineHiddenStatesConnector(ExampleHiddenStatesConnector):
    """Async two-file hidden states capture for EAGLE drafter training.

    Scheduler and worker run as separate instances in the same process.
    Percentile filtering decisions are communicated via the class-level
    ``discard_req_ids`` set: the scheduler adds IDs in request_finished,
    and the worker checks them in get_finished.
    """

    # Shared between scheduler and worker instances (same process)
    discard_req_ids: set[str] = set()

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role,
                         kv_cache_config=kv_cache_config)
        kv = self._kv_transfer_config.get_from_extra_config
        self.use_compression = kv("use_compression", False)
        self.compression_level = kv("compression_level", 3)
        self.percentile_tracker = self.create_percentile_tracker(kv)
        self.decode_filenames: dict[str, str] = {}
        self.request_filenames: dict[str, tuple[str, str]] = {}
        self.prev_step_info: dict[str, tuple[int, int]] = {}
        self.acceptance_lengths: dict[str, list[float]] = {}
        self.total_captured = 0
        self.total_skipped = 0
        self.writer = None

    def create_percentile_tracker(self, kv):
        pct = kv("capture_percentile", 0.0)
        if pct <= 0:
            return None
        from vllm.distributed.kv_transfer.kv_connector.online.percentile_tracker import PercentileTracker  # noqa: E501
        return PercentileTracker(percentile=pct,
                                 window_size=kv("capture_window_size", 1000),
                                 min_samples=kv("capture_min_samples", 100))

    def get_writer(self):
        if self.writer is None:
            from vllm.distributed.kv_transfer.kv_connector.online.hidden_states_writer import HiddenStatesWriter  # noqa: E501
            self.writer = HiddenStatesWriter(self._storage_path,
                                             use_compression=self.use_compression,
                                             compression_level=self.compression_level)
        return self.writer

    # ==============================
    # Worker-side methods
    # ==============================

    def wait_for_save(self):
        if self.writer is not None:
            self.writer.flush(timeout=10.0)

    def register_kv_caches(self, kv_caches):
        """Override parent to handle both extract_hidden_states and live EAGLE.

        With extract_hidden_states: finds CacheOnlyAttentionLayer (parent behavior).
        With live EAGLE: no CacheOnlyAttentionLayer exists, so skip the assertion.
        In live mode, capture happens via capture_hidden_states() from the proposer.
        """
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
        from vllm.config import get_layers_from_vllm_config
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer,
            list(kv_caches.keys()),
        )
        self.cache_layers = list(layers.keys())

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if layer_name not in self.cache_layers:
            return
        from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionMetadata  # noqa: E501
        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleHiddenStatesConnectorMetadata)
        writer = self.get_writer()
        for request in metadata.requests:
            hs = extract_from_kv_cache(kv_layer, request.slot_mapping,
                                       request.token_ids.shape[0])
            if request.new_req:
                self.write_prefill(writer, request, hs)
            else:
                writer.accumulate_async(request.req_id, hs, request.token_ids)

    def write_prefill(self, writer, request, hidden_states):
        os.makedirs(os.path.dirname(request.filename), exist_ok=True)
        writer.write_async(hidden_states, request.token_ids, request.filename)
        self.decode_filenames[request.req_id] = request.filename.replace(
            ".safetensors", "_decode.safetensors")

    def get_finished(self, finished_req_ids):
        for req_id in finished_req_ids:
            fn = self.decode_filenames.pop(req_id, None)
            if self.writer is None:
                continue
            # Check if scheduler marked this request for discard
            if req_id in OnlineHiddenStatesConnector.discard_req_ids:
                OnlineHiddenStatesConnector.discard_req_ids.discard(req_id)
                self.writer.discard_request(req_id)
            elif fn is not None:
                self.writer.flush_request(req_id, fn)
            else:
                self.writer.discard_request(req_id)
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def build_connector_meta(self, scheduler_output):
        meta = ExampleHiddenStatesConnectorMetadata()
        cached = scheduler_output.scheduled_cached_reqs
        self.track_acceptance(cached)
        self.record_step_info(scheduler_output, cached)
        self.register_new_requests(meta, scheduler_output)
        return meta

    def track_acceptance(self, cached_reqs):
        if self.percentile_tracker is None:
            return
        for i, req_id in enumerate(cached_reqs.req_ids):
            prev = self.prev_step_info.get(req_id)
            if prev is None or prev[1] <= 0:
                continue
            actual = cached_reqs.num_computed_tokens[i]
            accepted = max(0, actual - prev[0] + prev[1])
            self.acceptance_lengths.setdefault(req_id, []).append(
                1.0 + accepted / prev[1])

    def record_step_info(self, sched_out, cached_reqs):
        self.prev_step_info.clear()
        spec = sched_out.scheduled_spec_decode_tokens
        computed = {cid: cached_reqs.num_computed_tokens[j]
                    for j, cid in enumerate(cached_reqs.req_ids)}
        for r in sched_out.scheduled_new_reqs:
            computed[r.req_id] = r.num_computed_tokens
        for rid, n in sched_out.num_scheduled_tokens.items():
            c = computed.get(rid)
            if c is not None:
                self.prev_step_info[rid] = (c + n, len(spec.get(rid, [])))

    def register_new_requests(self, meta, sched_out):
        for r in sched_out.scheduled_new_reqs:
            tids = r.prompt_token_ids or []
            sub = r.lora_request.lora_name if r.lora_request else "base"
            pfn = os.path.join(self._storage_path, sub, f"{r.req_id}.safetensors")
            dfn = os.path.join(self._storage_path, sub, f"{r.req_id}_decode.safetensors")
            meta.add_request(r.req_id, filename=pfn, token_ids=tids,
                             block_ids=r.block_ids[0], block_size=self._block_size)
            self.request_filenames[r.req_id] = (pfn, dfn)
            self._active_requests[r.req_id] = r

    # ==============================
    # Request completion
    # ==============================

    def request_finished(self, request, block_ids):
        rid = request.request_id
        fns = self.request_filenames.pop(rid, None)
        self._active_requests.pop(rid, None)
        self.prev_step_info.pop(rid, None)
        acc = self.acceptance_lengths.pop(rid, None)

        empty = False, {"hidden_states_prefill": None, "hidden_states_decode": None}
        if fns is None:
            return empty
        pfn, dfn = fns
        if self.should_discard(acc):
            # Mark for discard — worker's get_finished will skip writing.
            # Also delete any prefill file that was already written to disk.
            OnlineHiddenStatesConnector.discard_req_ids.add(rid)
            self.delete_captured_files(pfn, dfn)
            self.total_skipped += 1
            return empty
        self.total_captured += 1
        return False, {"hidden_states_prefill": pfn, "hidden_states_decode": dfn}

    def should_discard(self, acc):
        if self.percentile_tracker is None or not acc:
            return False
        return not self.percentile_tracker.observe_and_check(sum(acc) / len(acc))

    @staticmethod
    def delete_captured_files(*paths):
        for p in paths:
            for c in (p, p + ".zst"):
                try:
                    os.remove(c)
                except FileNotFoundError:
                    pass

    # ==============================
    # Lifecycle
    # ==============================

    def shutdown(self):
        if self.writer is not None:
            self.writer.flush(timeout=10.0)
            self.writer.shutdown(timeout=5.0)

    # ==============================
    # Live EAGLE capture
    # ==============================

    def capture_hidden_states(self, hidden_states, token_ids,
                              query_start_loc, req_ids,
                              lora_mapping, lora_lookup):
        """Accumulate per-request hidden states during live EAGLE serving.

        Data is buffered in pinned CPU memory and written to disk only when
        the request finishes (via get_finished), enabling percentile filtering.
        """
        writer = self.get_writer()
        for i, req_id in enumerate(req_ids):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            if start == end:
                continue

            lora_id = int(lora_mapping[i])
            if lora_id != 0 and lora_id in lora_lookup:
                subdir = lora_lookup[lora_id].lora_name
            else:
                subdir = "base"

            # Track filename for flush at request completion
            if req_id not in self.decode_filenames:
                self.decode_filenames[req_id] = os.path.join(
                    self._storage_path, subdir,
                    f"{req_id}.safetensors")

            writer.accumulate_async(
                req_id, hidden_states[start:end], token_ids[start:end])
