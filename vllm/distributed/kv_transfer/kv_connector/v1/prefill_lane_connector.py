# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefillLaneConnector: prefill long prompts concurrently with decode.

The bulk of a long prompt is handed to a prefill lane
(vllm/v1/worker/gpu/prefill_lane.py) as an async KV load, so the decode
steps of the running requests keep going while it prefills instead of
stalling behind every chunk.

Usage:
    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "PrefillLaneConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"min_tokens": 4096, "compute_units": 36}
    }'

Options (kv_connector_extra_config):
    - min_tokens: prompts with fewer uncached tokens take the normal path
      (default: two chunks).
    - chunk_tokens: lane chunk size (default: the largest multiple of the
      block size within max_num_batched_tokens).
    - compute_units: compute units the lane's stream may use, counted from
      the top; 0 means all (default: 0).
    - always: use the lane even when no other request is running (default:
      false; then nothing would stall and the normal path is faster).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MambaSpec,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.prefill_lane import PrefillJob, PrefillLane

logger = init_logger(__name__)

# How long an idle step waits for the lane before returning, so an engine with
# only lane work does not spin.
_IDLE_WAIT_S = 0.005


@dataclass
class PrefillLaneMetadata(KVConnectorMetadata):
    jobs: list["PrefillJob"] = field(default_factory=list)
    # Nothing else is scheduled this step.
    idle: bool = False


class PrefillLaneConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        self.block_size = vllm_config.cache_config.block_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.chunk_tokens = int(
            extra.get(
                "chunk_tokens",
                max(self.block_size, max_tokens // self.block_size * self.block_size),
            )
        )
        assert self.chunk_tokens % self.block_size == 0
        self.min_tokens = int(extra.get("min_tokens", 2 * self.chunk_tokens))
        self.compute_units = int(extra.get("compute_units", 0))
        self.always = bool(extra.get("always", False))
        # Scheduler side.
        self._pending: list[PrefillJob] = []
        # Whether the last step ran any request, i.e. someone would stall.
        self._busy = False
        # Worker side.
        self.lane: PrefillLane | None = None

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_model_runner(self, runner: "GPUModelRunner") -> None:
        from vllm.v1.worker.gpu.prefill_lane import PrefillLane

        self.lane = PrefillLane(runner, self.chunk_tokens, self.compute_units)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, PrefillLaneMetadata)
        assert self.lane is not None
        for job in meta.jobs:
            self.lane.submit(job)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.lane is not None
        meta = self._get_connector_metadata()
        idle = isinstance(meta, PrefillLaneMetadata) and meta.idle
        done = self.lane.take_finished(_IDLE_WAIT_S if idle else 0.0)
        return None, done or None

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        if (
            not (self._busy or self.always)
            or request.num_computed_tokens
            or request.mm_features
            or request.prompt_embeds is not None
            or request.sampling_params is None
            or request.sampling_params.prompt_logprobs is not None
        ):
            return 0, False
        # The lane stops one token short; the normal path computes that token
        # and samples, as after a full prefix hit.
        end = request.num_tokens - 1
        junction = request.shared_prefix_boundary
        if num_computed_tokens < junction < end and junction % self.block_size:
            # A sub-block shared-prefix state the lane would not materialize.
            return 0, False
        num_tokens = end - num_computed_tokens
        if num_tokens < self.min_tokens:
            return 0, False
        return num_tokens, True

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return
        from vllm.v1.worker.gpu.prefill_lane import PrefillJob

        end = request.num_tokens - 1
        start = end - num_external_tokens
        new_req = NewRequestData.from_request(
            request, self._lane_block_ids(blocks, end), request._all_token_ids
        )
        new_req.num_computed_tokens = start
        self._pending.append(
            PrefillJob(new_req, end, self._zero_block_ids(blocks, start, end))
        )

    def _lane_block_ids(
        self, blocks: "KVCacheBlocks", end: int
    ) -> tuple[list[int], ...]:
        """In align mode an async load gets a real Mamba state block only where
        it ends; the slots before are null. The lane advances the state chunk by
        chunk, so it keeps the running state in that last block, as the normal
        path does, by pointing the null slots at it."""
        block_ids = blocks.get_block_ids()
        if self._vllm_config.cache_config.mamba_cache_mode != "align":
            return block_ids
        lane_ids = []
        for group, group_blocks, ids in zip(
            self._kv_cache_config.kv_cache_groups,
            blocks.blocks,
            block_ids,
            strict=True,
        ):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                lane_ids.append(ids)
                continue
            last = (end - 1) // group.kv_cache_spec.block_size
            running = group_blocks[last]
            assert not running.is_null
            lane_ids.append(
                [
                    running.block_id if i < last and block.is_null else block.block_id
                    for i, block in enumerate(group_blocks)
                ]
            )
        return tuple(lane_ids)

    def _zero_block_ids(
        self, blocks: "KVCacheBlocks", start: int, end: int
    ) -> list[int]:
        """Blocks the scheduler skipped zeroing because the load writes them;
        the lane zeroes them itself before its first chunk."""
        if not self._kv_cache_config.needs_kv_cache_zeroing:
            return []
        ids: list[int] = []
        for group, group_blocks in zip(
            self._kv_cache_config.kv_cache_groups, blocks.blocks, strict=True
        ):
            if any(
                isinstance(spec, CircularBufferSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            ):
                continue
            block_size = group.kv_cache_spec.block_size
            lo, hi = start // block_size, cdiv(end, block_size)
            ids.extend(block.block_id for block in group_blocks[lo:hi])
        return ids

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        self._busy = scheduler_output.total_num_scheduled_tokens > 0
        meta = PrefillLaneMetadata(jobs=self._pending, idle=not self._busy)
        self._pending = []
        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
