# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online (async) hidden states connector.

Drop-in replacement for ExampleHiddenStatesConnector that uses a
background thread for disk I/O instead of blocking the forward pass.

Same interface, same metadata, same extraction logic — the only
difference is that safetensors.torch.save_file runs in a background
thread after an async GPU→pinned-memory copy.

Supports percentile-based capture filtering: when ``capture_percentile``
is set in ``kv_connector_extra_config``, acceptance lengths are
accumulated per request during decode.  At request completion, the
request's average acceptance length is checked against the global
percentile threshold.  Only requests whose average falls in the worst
X% keep their captured file; the rest are deleted.  This focuses data
collection on the cases where the drafter performs poorly — exactly the
training signal needed to fine-tune LoRA drafter heads.

Write strategy:
  - Prefill: always write (hidden states for all prompt tokens).
  - Decode: skip writes (prompt hidden states don't change).
  - request_finished (filtering enabled): delete the file if the
    request's average acceptance is above the threshold.

Output layout:
  Files are organized by LoRA adapter under ``shared_storage_path``::

    shared_storage_path/
      base/                          # requests without LoRA
        <req_id>.safetensors[.zst]
      <lora_name>/                   # requests with LoRA adapter
        <req_id>.safetensors[.zst]
"""

import os
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.\
    example_hidden_states_connector import (
    ExampleHiddenStatesConnectorMetadata,
    extract_from_kv_cache,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class OnlineHiddenStatesConnector(KVConnectorBase_V1):
    """Async hidden states connector with non-blocking disk I/O.

    Identical to ExampleHiddenStatesConnector except that the
    safetensors write happens asynchronously via GPU→pinned copy
    on a CUDA transfer stream + a background writer thread.

    Write strategy:
      - Prefill (new requests): always extract and write hidden states.
        The KV cache holds all prompt tokens' hidden states, so a single
        write at prefill time captures everything needed.
      - Decode (cached requests): skip writes entirely.  The prompt
        hidden states in the KV cache don't change during decode, so
        re-extracting would just overwrite the same data.
      - When percentile filtering is enabled, acceptance lengths are
        accumulated per request.  At request_finished, if the request's
        average acceptance is above the threshold (drafter was good
        enough), the file is deleted.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        assert self._vllm_config.speculative_config is not None, (
            "OnlineHiddenStatesConnector only works when using "
            "'extract_hidden_states' speculative method"
        )
        spec_config = (
            self._vllm_config.speculative_config
            .draft_model_config.hf_config
        )
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self._request_filenames: dict[str, str] = {}
        self._active_requests: dict[str, NewRequestData] = {}
        self._req_blocks: dict[str, list[int]] = {}

        # Compression config (read from kv_connector_extra_config)
        self._use_compression = self._kv_transfer_config\
            .get_from_extra_config("use_compression", True)
        self._compression_level = self._kv_transfer_config\
            .get_from_extra_config("compression_level", 3)

        # Percentile-based capture filtering config
        self._capture_percentile: float = self._kv_transfer_config\
            .get_from_extra_config("capture_percentile", 0.0)
        self._capture_window_size: int = self._kv_transfer_config\
            .get_from_extra_config("capture_window_size", 1000)
        self._capture_min_samples: int = self._kv_transfer_config\
            .get_from_extra_config("capture_min_samples", 100)
        self._percentile_tracker = None
        if self._capture_percentile > 0:
            from vllm.distributed.kv_transfer.kv_connector.online\
                .percentile_tracker import PercentileTracker
            self._percentile_tracker = PercentileTracker(
                percentile=self._capture_percentile,
                window_size=self._capture_window_size,
                min_samples=self._capture_min_samples,
            )
            logger.info(
                "Percentile capture filtering enabled: "
                "percentile=%.1f, window=%d, min_samples=%d",
                self._capture_percentile,
                self._capture_window_size,
                self._capture_min_samples,
            )

        # Per-request tracking for acceptance rate computation.
        # Maps req_id -> (num_computed_tokens, num_spec_tokens) from
        # the previous scheduler step, used to derive acceptance length.
        self._prev_step_info: dict[str, tuple[int, int]] = {}
        # Per-request accumulated acceptance lengths for deferred
        # filtering.  At request_finished, the average is checked
        # against the global percentile threshold.
        self._req_acceptance_lengths: dict[str, list[float]] = {}
        # Stats
        self._total_captured = 0
        self._total_skipped = 0

        # Lazy-init async writer (needs CUDA to be ready)
        self._writer = None

    def _get_writer(self):
        if self._writer is None:
            from vllm.distributed.kv_transfer.kv_connector.online\
                .hidden_states_writer import HiddenStatesWriter
            self._writer = HiddenStatesWriter(
                self._storage_path,
                use_compression=self._use_compression,
                compression_level=self._compression_level,
            )
        return self._writer

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer,
            list(kv_caches.keys()),
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, "
            f"got {len(self.cache_layers)}"
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )
        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata), (
            "OnlineHiddenStatesConnector only supports "
            "CacheOnlyAttentionBackend"
        )

        connector_metadata = self._get_connector_metadata()
        assert isinstance(
            connector_metadata, ExampleHiddenStatesConnectorMetadata
        )

        os.makedirs(self._storage_path, exist_ok=True)
        writer = self._get_writer()
        for request in connector_metadata.requests:
            # Only write on prefill (new requests).  The KV cache holds
            # all prompt tokens' hidden states, so a single write at
            # prefill captures everything.  Decode steps would just
            # overwrite the same data.
            if not request.new_req:
                continue
            # Ensure the target subdirectory exists (e.g. base/ or
            # <lora_name>/) before handing off to the writer thread.
            os.makedirs(
                os.path.dirname(request.filename), exist_ok=True,
            )
            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping,
                request.token_ids.shape[0],
            )
            writer.write_async(
                hidden_states=hidden_states,
                token_ids=request.token_ids,
                filename=request.filename,
            )
            self._total_captured += 1

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = ExampleHiddenStatesConnectorMetadata()

        # --- Accumulate per-request acceptance lengths ---
        # For each cached request, compare actual num_computed_tokens
        # with what we expected from the previous step.  The difference
        # tells us how many draft tokens were accepted.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if self._percentile_tracker is not None:
            for i, req_id in enumerate(cached_reqs.req_ids):
                if req_id not in self._prev_step_info:
                    continue
                prev_computed, prev_spec = self._prev_step_info[req_id]
                if prev_spec <= 0:
                    continue
                actual_computed = cached_reqs.num_computed_tokens[i]
                num_accepted = max(
                    0, actual_computed - prev_computed - 1
                )
                acceptance_length = 1.0 + (num_accepted / prev_spec)
                # Accumulate per-request (decision deferred to
                # request_finished).
                if req_id not in self._req_acceptance_lengths:
                    self._req_acceptance_lengths[req_id] = []
                self._req_acceptance_lengths[req_id].append(
                    acceptance_length,
                )

        # --- Record current step info for next iteration ---
        self._prev_step_info.clear()
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Build O(1) lookup for num_computed_tokens by req_id
        computed_by_id: dict[str, int] = {}
        for j, cid in enumerate(cached_reqs.req_ids):
            computed_by_id[cid] = cached_reqs.num_computed_tokens[j]
        for new_req in scheduler_output.scheduled_new_reqs:
            computed_by_id[new_req.req_id] = new_req.num_computed_tokens

        for req_id, num_sched in scheduler_output.num_scheduled_tokens.items():
            num_spec = len(spec_tokens.get(req_id, []))
            computed = computed_by_id.get(req_id)
            if computed is not None:
                self._prev_step_info[req_id] = (
                    computed + num_sched, num_spec,
                )

        # --- Build metadata ---
        # Only new requests are included — decode requests don't need
        # re-extraction since prompt hidden states don't change.
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            # Organize output by LoRA adapter: files go into
            # <storage_path>/<lora_name>/ (or <storage_path>/base/
            # for requests without a LoRA adapter).
            subdir = "base"
            if new_req.lora_request is not None:
                subdir = new_req.lora_request.lora_name
            req_dir = os.path.join(self._storage_path, subdir)
            filename = os.path.join(
                req_dir,
                f"{new_req.req_id}.safetensors",
            )
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])

        # Still track block growth for cached requests (needed if we
        # ever want to re-extract, and for compatibility).
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue
            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is not None:
                self._req_blocks[req_id].extend(new_block_ids[0])

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        self._active_requests.pop(req_id, None)
        self._req_blocks.pop(req_id, None)
        self._prev_step_info.pop(req_id, None)
        acceptance_lengths = self._req_acceptance_lengths.pop(req_id, None)

        # --- Deferred percentile filtering ---
        # If filtering is enabled and we have acceptance data, compute
        # the request's average acceptance length and check against the
        # global threshold.  If the drafter performed well enough (above
        # threshold), delete the captured file — we only want training
        # data from requests where the drafter struggled.
        if (self._percentile_tracker is not None
                and acceptance_lengths
                and req_filename is not None):
            avg_acceptance = sum(acceptance_lengths) / len(acceptance_lengths)
            # Feed the average into the global tracker.
            should_keep = self._percentile_tracker.observe_and_check(
                avg_acceptance,
            )
            if not should_keep:
                # Drafter was good enough — discard the file.
                self._delete_captured_file(req_filename)
                self._total_skipped += 1
                req_filename = None  # signal: no file kept
            else:
                self._total_captured += 1

            # Log stats periodically
            total = self._total_captured + self._total_skipped
            if total > 0 and total % 500 == 0:
                stats = self._percentile_tracker.get_stats()
                logger.info(
                    "Capture filter stats: captured=%d, skipped=%d, "
                    "threshold=%.3f, mean_acceptance=%.3f, samples=%d",
                    self._total_captured, self._total_skipped,
                    stats.get("percentile_threshold") or 0.0,
                    stats.get("mean_acceptance") or 0.0,
                    stats.get("num_samples") or 0,
                )

        return False, {"hidden_states_path": req_filename}

    def _delete_captured_file(self, filename: str) -> None:
        """Delete a captured safetensors file (plain or compressed)."""
        for path in (filename, filename + ".zst"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", path, e)

    @classmethod
    def get_required_kvcache_layout(
        cls, vllm_config: "VllmConfig"
    ) -> str | None:
        return "NHD"
