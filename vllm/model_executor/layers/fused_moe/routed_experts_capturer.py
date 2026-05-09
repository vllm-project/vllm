# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed

from vllm.config.model import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom op for routing capture -- traceable by torch.compile / Dynamo.
#
# Registered as a formal custom op so that torch.compile traces through it
# cleanly without graph breaks.  ALL TP ranks call this op with a real
# device buffer to ensure identical CUDA graph structure (symmetry).
# Non-rank-0 buffers are written but never read for D2H.
# ---------------------------------------------------------------------------


@torch.library.custom_op("vllm::capture_routing", mutates_args={"buffer"})
def capture_routing_op(
    buffer: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_id: int,
    batch_size: int,
) -> None:
    buffer[layer_id, :batch_size, :].copy_(
        topk_ids[:batch_size].to(buffer.dtype), non_blocking=True
    )


@capture_routing_op.register_fake
def _capture_routing_op_fake(
    buffer: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_id: int,
    batch_size: int,
) -> None:
    pass


_MB = 1024 * 1024


class _RoutedExpertsDeviceCache:
    """Per-device (GPU) cache for capturing routed expert IDs during forward
    pass.  Always writes at row 0 so that CUDA graph replay sees the same
    addresses that were recorded at capture time.
    """

    DTYPE = torch.int16

    def __init__(
        self,
        max_num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        device: str,
    ) -> None:
        # Layout: (L, N, K) so that buffer[layer_id] is a contiguous (N, K)
        # view — required by the FlashInfer routing-replay kernel which
        # writes expert IDs assuming contiguous row-major memory.
        self.num_hidden_layers = num_hidden_layers
        self.buffer = torch.zeros(
            (num_hidden_layers, max_num_batched_tokens, num_experts_per_tok),
            dtype=self.DTYPE,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        return self.buffer.nbytes

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[layer_id, :batch, :].copy_(topk_ids, non_blocking=True)

    def _finalize_allocation_log(self):
        buf_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            "Routing experts device buffer allocated. shape=%s, size=%.2f MB",
            tuple(self.buffer.shape),
            buf_mb,
        )


class _RoutedExpertsHostCache:
    """Host (CPU) cache using numpy arrays for per-request routing data.

    Numpy arrays avoid torch dispatcher overhead for scatter operations.
    Lazy per-request allocation avoids a massive up-front buffer.
    """

    DTYPE = np.int16

    def __init__(
        self,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        max_model_len: int,
    ) -> None:
        self.max_model_len = max_model_len
        self.num_hidden_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok

        self._req_buffers: dict[str, np.ndarray] = {}
        self._filled_len: dict[str, int] = {}
        self._total_allocated_bytes = 0

        self._finalize_allocation_log()

    def get_buffer_size_bytes(self) -> int:
        return self._total_allocated_bytes

    def get_or_grow_buffer(self, req_id: str, max_pos: int) -> np.ndarray:
        required_len = max_pos + 1

        if req_id not in self._req_buffers:
            buf = np.zeros(
                (required_len, self.num_hidden_layers, self.num_experts_per_tok),
                dtype=self.DTYPE,
            )
            self._req_buffers[req_id] = buf
            self._total_allocated_bytes += buf.nbytes
            return buf

        buf = self._req_buffers[req_id]
        if buf.shape[0] >= required_len:
            return buf

        new_len = min(max(required_len, buf.shape[0] * 2), self.max_model_len)
        new_buf = np.zeros(
            (new_len, self.num_hidden_layers, self.num_experts_per_tok),
            dtype=self.DTYPE,
        )
        new_buf[: buf.shape[0]] = buf
        self._total_allocated_bytes += new_buf.nbytes - buf.nbytes
        self._req_buffers[req_id] = new_buf
        return new_buf

    def get_buffer(self, req_id: str) -> np.ndarray | None:
        return self._req_buffers.get(req_id)

    def update_filled_len(self, req_id: str, max_pos: int) -> None:
        new_len = max_pos + 1
        self._filled_len[req_id] = max(self._filled_len.get(req_id, 0), new_len)

    def get_filled_len(self, req_id: str) -> int:
        return self._filled_len.get(req_id, 0)

    def free_request(self, req_id: str) -> None:
        if req_id in self._req_buffers:
            self._total_allocated_bytes -= self._req_buffers.pop(req_id).nbytes
        self._filled_len.pop(req_id, None)

    def _finalize_allocation_log(self):
        logger.info(
            "Routing experts host cache initialized (lazy allocation). "
            "max_model_len=%s, layers=%s, experts_per_tok=%s",
            self.max_model_len,
            self.num_hidden_layers,
            self.num_experts_per_tok,
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: str,
        shared_host_cache: _RoutedExpertsHostCache | None = None,
        skip_host_cache: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                max_num_batched_tokens=max_num_batched_tokens,
                num_fused_shared_experts=num_fused_shared_experts,
                max_model_len=max_model_len,
                device=device,
                shared_host_cache=shared_host_cache,
                skip_host_cache=skip_host_cache,
            )
        return _RoutedExpertsCapturerNoop()

    @abstractmethod
    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self, req_id: str, seqlen: int | None = None, free_slot: bool = True
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ):
        raise NotImplementedError

    def finalize_pending_copy(self):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


def _count_moe_layers(hf_config) -> int:
    """Count the number of MoE layers in a model.

    Resolves three known config shapes:
    - Nemotron-style: an explicit ``layers_block_type`` list with "moe" entries.
    - Qwen3MoE / DeepSeek-style sparse: ``decoder_sparse_step > 1`` with optional
      ``mlp_only_layers`` exclusions.
    - Default: every layer is MoE except those listed in ``mlp_only_layers``.
    """
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        return layers_block_type.count("moe")
    n = hf_config.num_hidden_layers
    mlp_only = getattr(hf_config, "mlp_only_layers", None) or []
    step = getattr(hf_config, "decoder_sparse_step", 1) or 1
    if step > 1:
        return sum(1 for i in range(n) if (i + 1) % step == 0 and i not in mlp_only)
    return n - sum(1 for i in mlp_only if 0 <= i < n)


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer with GPU device cache and CPU host cache.

    Performance strategy -- async D2H with optimized host-cache scatter:

    Every decode step we issue a non-blocking D2H copy on a dedicated
    CUDA stream.  The scatter into per-request host-cache buffers is
    deferred to the start of the NEXT step (by which time the copy has
    finished).  The scatter loop is optimized with direct scalar access
    to avoid numpy slice views, int() conversions, and .max() calls.

    At extraction time (when a request finishes), data is already in a
    contiguous host buffer -- just a numpy slice, no concatenation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        max_num_batched_tokens: int,
        num_fused_shared_experts: int,
        max_model_len: int,
        device: str,
        shared_host_cache: _RoutedExpertsHostCache | None = None,
        skip_host_cache: bool = False,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = _count_moe_layers(model_config.hf_text_config)
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self._skip_host_cache = skip_host_cache

        if skip_host_cache:
            self.host_cache = None
            logger.info("Skipping host cache for device %s (non-rank-0)", device)
        elif shared_host_cache is not None:
            self.host_cache = shared_host_cache
        else:
            self.host_cache = _RoutedExpertsHostCache(
                num_hidden_layers=self.num_hidden_layers,
                num_experts_per_tok=self.num_experts_per_tok,
                max_model_len=self.max_model_len,
            )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_num_batched_tokens=self.max_num_batched_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            device=device,
        )

        # ---- Async D2H pipeline (rank-0 only) ----
        # Non-rank-0 workers only need the device buffer for symmetric
        # CUDA graph capture; they skip the D2H pipeline entirely.
        self._has_pending_copy = False
        self._pending_positions: np.ndarray | None = None
        self._pending_num_scheduled: dict[str, int] | None = None
        self._pending_total_tokens: int = 0

        if not skip_host_cache:
            # Same (L, N, K) layout as device_cache.buffer.
            self._pinned_staging = torch.zeros(
                (
                    self.num_hidden_layers,
                    max_num_batched_tokens,
                    self.num_experts_per_tok,
                ),
                dtype=_RoutedExpertsDeviceCache.DTYPE,
                pin_memory=True,
            )
            # Private device snapshot: source for the async D2H. Decouples
            # the in-flight copy from device_cache.buffer, which the next
            # step's MoE writes overwrite in place on main_stream.
            self._device_staging = torch.empty_like(self.device_cache.buffer)
            self._copy_stream = torch.cuda.Stream(device=device)
            self._copy_event = torch.cuda.Event()

            pinned_mb = self._pinned_staging.nbytes / _MB
            logger.info(
                "Routing experts pinned staging buffer allocated. "
                "shape=%s, size=%.2f MB",
                tuple(self._pinned_staging.shape),
                pinned_mb,
            )
        else:
            self._pinned_staging = None
            self._device_staging = None
            self._copy_stream = None
            self._copy_event = None
            logger.info(
                "Routing experts device-only capturer (rank != 0). "
                "Device buffer shape=%s",
                tuple(self.device_cache.buffer.shape),
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    # ------------------------------------------------------------------
    # sync_fwd_experts_buffer_DtoH -- called AFTER the forward pass
    # ------------------------------------------------------------------

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ):
        if self.host_cache is None:
            return

        # 1. Finalize previous async copy -- the copy had an entire
        #    forward pass to complete so event.synchronize() is ~free.
        if self._has_pending_copy:
            self._copy_event.synchronize()
            self._scatter_to_host()
            self._has_pending_copy = False

        total_tokens = sum(num_scheduled_tokens.values())
        if total_tokens == 0:
            return

        # 2. Snapshot the device buffer on main_stream into a private
        #    staging buffer, then issue the D2H from the staging buffer
        #    on a dedicated copy stream. The snapshot serializes after
        #    the current step's MoE writes (same stream) and is private
        #    from the next step's MoE writes, so the in-flight D2H is
        #    not aliased by step N+1's forward under async scheduling.
        main_stream = torch.cuda.current_stream(self._copy_stream.device)
        self._device_staging[:, :total_tokens, :].copy_(
            self.device_cache.buffer[:, :total_tokens, :], non_blocking=True
        )
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(main_stream)
            self._pinned_staging[:, :total_tokens, :].copy_(
                self._device_staging[:, :total_tokens, :], non_blocking=True
            )
            self._copy_event.record()

        # 3. Save metadata for deferred scatter.
        self._pending_positions = positions.numpy().copy()
        self._pending_num_scheduled = num_scheduled_tokens
        self._pending_total_tokens = total_tokens
        self._has_pending_copy = True

    # ------------------------------------------------------------------
    # Optimized scatter into pre-allocated host-cache buffers
    # ------------------------------------------------------------------

    def _scatter_to_host(self):
        """Scatter D2H data into per-request host cache buffers.

        Staging layout is (L, N, K).  Host cache layout is (seq_len, L, K).
        We transpose the staging slice to (N, L, K) before scattering so
        that indexing by token position naturally yields (L, K) rows.
        """
        # Transpose (L, N, K) -> (N, L, K) for the active token range.
        host_values = (
            self._pinned_staging[:, : self._pending_total_tokens, :]
            .numpy()
            .transpose(1, 0, 2)
        )
        positions_np = self._pending_positions
        host_cache = self.host_cache
        assert self._pending_num_scheduled is not None
        assert positions_np is not None
        assert host_cache is not None

        offset = 0
        for req_id, n_tokens in self._pending_num_scheduled.items():
            if n_tokens == 0:
                continue

            if n_tokens == 1:
                pos_val = int(positions_np[offset])
                buf = host_cache.get_or_grow_buffer(req_id, pos_val)
                buf[pos_val] = host_values[offset]
                host_cache.update_filled_len(req_id, pos_val)
            else:
                pos = positions_np[offset : offset + n_tokens]
                max_pos = int(pos[-1]) if n_tokens > 0 else 0
                if n_tokens > 1:
                    max_pos = int(pos.max())
                buf = host_cache.get_or_grow_buffer(req_id, max_pos)
                buf[pos] = host_values[offset : offset + n_tokens]
                host_cache.update_filled_len(req_id, max_pos)

            offset += n_tokens

        self._pending_positions = None
        self._pending_num_scheduled = None
        self._pending_total_tokens = 0

    # ------------------------------------------------------------------
    # finalize_pending_copy -- call before reading host cache
    # ------------------------------------------------------------------

    def finalize_pending_copy(self):
        """Ensure the most recent async D2H copy has been scattered into
        host cache buffers.  Call before get_routed_experts."""
        if self._has_pending_copy:
            self._copy_event.synchronize()
            self._scatter_to_host()
            self._has_pending_copy = False

    # ------------------------------------------------------------------
    # Extraction -- O(1), just a numpy slice
    # ------------------------------------------------------------------

    def get_routed_experts(
        self,
        req_id: str,
        seqlen: int | None = None,
        free_slot: bool = True,
    ):
        if self.host_cache is None:
            return None
        buf = self.host_cache.get_buffer(req_id)
        if buf is None:
            return None
        filled = self.host_cache.get_filled_len(req_id)
        if filled <= 0:
            return None
        effective_len = min(filled, seqlen) if seqlen is not None else filled
        result = buf[:effective_len].copy()
        if free_slot:
            self.host_cache.free_request(req_id)
        return result

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(self, req_id: str, seqlen=None, free_slot=True):
        return None

    def sync_fwd_experts_buffer_DtoH(self, positions, num_scheduled_tokens):
        pass

    def finalize_pending_copy(self):
        pass

    def get_host_cache(self):
        return None

    def get_device_cache(self):
        pass


# Global capturer instance (per-process)
_global_expert_capturer: RoutedExpertsCapturer | None = _RoutedExpertsCapturerNoop()
_shared_host_cache: _RoutedExpertsHostCache | None = None


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_for_current_batch(
    req_ids: list[str],
    requests: dict,
    req_id_to_index: dict[str, int],
    num_tokens_no_spec: np.ndarray,
    max_model_len: int,
) -> dict[str, np.ndarray] | None:
    """Extract routed experts for requests predicted to finish this step.

    Checks all stop conditions the scheduler will check (max_tokens,
    EOS token, stop tokens, max_model_len) so that every finished
    request gets its routing data attached to the ModelRunnerOutput.

    Args:
        req_ids: Ordered request IDs for the current batch.
        requests: Map of req_id to CachedRequestState (read-only).
        req_id_to_index: Map of req_id to input batch index.
        num_tokens_no_spec: Array of total token counts per request index.
        max_model_len: Maximum model sequence length.
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return None
    host_cache = capturer.get_host_cache()
    if host_cache is None:
        return None

    finishing_req_ids: list[str] = []
    for req_id in req_ids:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        sp = req_state.sampling_params
        if sp is None:
            continue
        output_ids = req_state.output_token_ids
        if not output_ids:
            continue
        if len(output_ids) < sp.min_tokens:
            continue

        finishing = False
        last_token = output_ids[-1]

        # EOS token (mirrors check_stop: eos_token_id is None
        # when ignore_eos=True, so this naturally respects that)
        if last_token == sp.eos_token_id:
            finishing = True

        # Explicit stop token IDs
        if not finishing and sp.stop_token_ids and last_token in sp.stop_token_ids:
            finishing = True

        # max_tokens / max_model_len length cap
        if not finishing:
            if sp.max_tokens is not None and len(output_ids) >= sp.max_tokens:
                finishing = True
            else:
                req_idx = req_id_to_index.get(req_id)
                if req_idx is not None:
                    total = num_tokens_no_spec[req_idx]
                    if total >= max_model_len:
                        finishing = True

        if finishing:
            finishing_req_ids.append(req_id)

    if not finishing_req_ids:
        return None

    # At least one request is finishing: ensure the latest async D2H
    # copy has been scattered into the host cache.
    capturer.finalize_pending_copy()

    result: dict[str, np.ndarray] = {}
    for req_id in finishing_req_ids:
        seqlen = host_cache.get_filled_len(req_id)
        if seqlen <= 0:
            continue
        experts = capturer.get_routed_experts(req_id, seqlen=seqlen, free_slot=False)
        if experts is not None:
            result[req_id] = experts

    return result if result else None


def free_routing_buffers(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None = None,
) -> None:
    """Free host cache buffers for finished and preempted requests.

    Finished requests had their routing data extracted in the previous
    step.

    Preempted requests are re-prefilled from scratch when they resume,
    so their host-cache buffer is freed here. This means any routing
    already accumulated in the host cache for the preempted request is
    dropped without being emitted on a ``ModelRunnerOutput`` --
    consumers see ``routed_experts=None`` for those requests with no
    other signal. Partial-rollout / async-RL pipelines that depend on
    receiving routing for preempted requests should treat preemption
    as a routing-data loss event and either keep preemption disabled
    or reconstruct routing on the resumed prefill.
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return
    host_cache = capturer.get_host_cache()
    if host_cache is None:
        return

    for req_id in finished_req_ids:
        host_cache.free_request(req_id)
    if preempted_req_ids:
        for req_id in preempted_req_ids:
            host_cache.free_request(req_id)


def issue_routing_d2h_copy(
    input_batch_req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    positions: torch.Tensor,
    positions_cpu: torch.Tensor,
) -> None:
    """Issue async D2H copy of routed experts after the forward pass.

    Called EARLY in the execute_model epilogue so the copy overlaps with
    eplb, kv_connector finalization, and draft work.
    finalize_pending_copy() + get_routed_experts() happen later in
    extract_routed_experts_for_current_batch().
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return

    ordered = {
        req_id: num_scheduled_tokens[req_id]
        for req_id in input_batch_req_ids
        if req_id in num_scheduled_tokens
    }
    n = sum(ordered.values())
    positions_cpu[:n].copy_(positions[:n])
    capturer.sync_fwd_experts_buffer_DtoH(
        positions=positions_cpu[:n],
        num_scheduled_tokens=ordered,
    )


def split_routed_experts(
    routed_experts: np.ndarray,
    prompt_len: int,
    num_output_tokens: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split routing data into prompt and generation portions.

    Args:
        routed_experts: Full routing array of shape (seq_len, L, K).
        prompt_len: Number of prompt tokens for the request.
        num_output_tokens: Actual number of generated tokens (from
            detokenizer).  When provided, the generation portion is
            clipped to this length — necessary with MTP where the model
            runner may capture routing for more tokens than the final
            output contains.

    Returns:
        (prompt_routed_experts, gen_routed_experts) numpy arrays, either
        of which may be None if the corresponding portion is empty.
    """
    prompt_routed_experts = routed_experts[:prompt_len]
    gen_routed_experts = routed_experts[prompt_len:]

    # Clip generation routing to match actual output tokens.
    if (
        num_output_tokens is not None
        and gen_routed_experts.shape[0] > num_output_tokens
        and num_output_tokens > 0
    ):
        gen_routed_experts = gen_routed_experts[:num_output_tokens]

    if prompt_routed_experts.size == 0:
        prompt_routed_experts = None
    if gen_routed_experts.size == 0:
        gen_routed_experts = None

    return prompt_routed_experts, gen_routed_experts


def get_shared_host_cache() -> _RoutedExpertsHostCache | None:
    return _shared_host_cache


def create_shared_host_cache(
    model_config: ModelConfig,
    max_model_len: int,
) -> _RoutedExpertsHostCache:
    global _shared_host_cache
    num_hidden_layers = _count_moe_layers(model_config.hf_text_config)
    num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
    _shared_host_cache = _RoutedExpertsHostCache(
        num_hidden_layers=num_hidden_layers,
        num_experts_per_tok=num_experts_per_tok,
        max_model_len=max_model_len,
    )
    return _shared_host_cache


def init_routed_experts_capturer_with_shared_cache(
    enable: bool,
    model_config: ModelConfig,
    num_fused_shared_experts: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> RoutedExpertsCapturer:
    """Initialize capturer with rank-aware handling (only rank 0 captures)."""
    if not enable:
        capturer = _RoutedExpertsCapturerNoop()
        set_global_experts_capturer(capturer)
        return capturer

    if world_size > 1 and rank != 0:
        # Non-rank-0 workers get a device-only capturer (no host cache,
        # no D2H pipeline) so that ALL ranks have a real device buffer.
        # This ensures the custom op call in every MoE layer produces
        # identical CUDA graph structure across TP ranks.
        logger.info("Creating device-only routed experts capturer for rank %s", rank)
        capturer = RoutedExpertsCapturer.create(
            enable=True,
            model_config=model_config,
            num_fused_shared_experts=num_fused_shared_experts,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            device=device,
            skip_host_cache=True,
        )
        set_global_experts_capturer(capturer)
        return capturer

    capturer = RoutedExpertsCapturer.create(
        enable=True,
        model_config=model_config,
        num_fused_shared_experts=num_fused_shared_experts,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
        skip_host_cache=False,
    )
    set_global_experts_capturer(capturer)
    return capturer


def bind_routing_capture_to_model(model) -> None:
    """Bind routing capture buffers to all FusedMoE layers in the model.

    Must be called AFTER init_routed_experts_capturer_with_shared_cache()
    and BEFORE CUDA graph capture.  All TP ranks get a real buffer so
    that the custom op call produces identical graph structure.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    capturer = get_global_experts_capturer()
    device_cache = capturer.get_device_cache()
    if device_cache is None:
        return  # routing capture not enabled

    buffer = device_cache.buffer

    # Mark the buffer so CUDA graphs do NOT snapshot/restore its contents.
    if hasattr(torch.compiler, "cudagraph_mark_tensor_static"):
        torch.compiler.cudagraph_mark_tensor_static(buffer)
    elif hasattr(torch._C, "_set_static_address_tag"):
        torch._C._set_static_address_tag(buffer, True)
    with contextlib.suppress(Exception):
        torch._dynamo.mark_static_address(buffer)

    bound = 0
    for module in model.modules():
        if isinstance(module, FusedMoE) and hasattr(module, "moe_layer_id"):
            # Per-FusedMoE configurations not yet validated for routing
            # capture. These signals are only set after model init, so a
            # config-level guard cannot see them.
            if module.moe_config.is_sequence_parallel:
                raise NotImplementedError(
                    "routed-experts capture is not yet validated with "
                    "sequence parallelism on the FusedMoE layer "
                    "(moe_config.is_sequence_parallel=True)."
                )
            if (
                module.moe_config.dp_size > 1
                and not module.quant_method.supports_internal_mk
            ):
                raise NotImplementedError(
                    "routed-experts capture is not yet validated with "
                    "naive DP dispatch (non-modular quant method "
                    f"{type(module.quant_method).__name__}, "
                    f"dp_size={module.moe_config.dp_size})."
                )

            layer_id = module.moe_layer_id
            layer_buf = buffer[layer_id]  # (N_max, K)
            module._routing_replay_out = layer_buf
            # Mark each per-layer view as static so CUDA graphs don't
            # snapshot/restore or relocate the buffer during replay.
            if hasattr(torch.compiler, "cudagraph_mark_tensor_static"):
                torch.compiler.cudagraph_mark_tensor_static(layer_buf)
            with contextlib.suppress(Exception):
                torch._dynamo.mark_static_address(layer_buf)
            bound += 1

    logger.info(
        "Bound routing capture buffer to %s FusedMoE layers. Buffer shape=%s",
        bound,
        tuple(buffer.shape),
    )
