import logging
from abc import ABC
from typing import Optional

import numpy as np
import torch
import torch.distributed
from vllm.config.model import ModelConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom op for routing capture — traceable by torch.compile / Dynamo.
#
# Using a registered custom op instead of a Python lambda callback avoids
# CUDA graph breaks.  On non-rank-0 workers the buffer attribute is None,
# so the call site's `if buffer is not None` guard compiles to False and
# the op is elided entirely — giving identical graph structure across all
# TP ranks and preventing NCCL collective deadlocks on multi-node setups.
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

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    """Per-device (GPU) cache for capturing routed expert IDs during forward
    pass.  Always writes at row 0 so that CUDA graph replay sees the same
    addresses that were recorded at capture time.
    """

    DTYPE = torch.int16

    def __init__(
        self,
        num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        # Layout: (L, N, K) so that buffer[layer_id] is a contiguous (N, K)
        # view — required by the FlashInfer routing-replay kernel which
        # writes expert IDs assuming contiguous row-major memory.
        self.num_hidden_layers = num_hidden_layers
        self.buffer = torch.zeros(
            (num_hidden_layers, num_batched_tokens, num_experts_per_tok),
            dtype=self.DTYPE,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, (
            "capturing routing experts but get layer_id None"
        )
        batch, _ = topk_ids.shape
        self.buffer[layer_id, :batch, :].copy_(topk_ids, non_blocking=True)

    def _finalize_allocation_log(self):
        buf_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. "
            f"shape={tuple(self.buffer.shape)}, size={buf_mb:.2f} MB"
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
        max_running_requests: int,
        max_model_len: int,
        use_shared_memory: bool = True,
    ) -> None:
        self.max_model_len = max_model_len
        self.max_running_requests = max_running_requests
        self.num_hidden_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        self._use_shared_memory = use_shared_memory

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
            f"Routing experts host cache initialized (lazy allocation). "
            f"max_model_len={self.max_model_len}, "
            f"layers={self.num_hidden_layers}, "
            f"experts_per_tok={self.num_experts_per_tok}"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_batched_tokens: int,
        max_running_requests: int,
        max_model_len: int,
        device: str,
        shared_host_cache: Optional[_RoutedExpertsHostCache] = None,
        skip_host_cache: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_batched_tokens=num_batched_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                max_model_len=max_model_len,
                device=device,
                shared_host_cache=shared_host_cache,
                skip_host_cache=skip_host_cache,
            )
        return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self, req_id: str, seqlen: Optional[int] = None, free_slot: bool = True
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self, positions: torch.Tensor, num_scheduled_tokes: dict[str, int],
    ):
        raise NotImplementedError

    def finalize_pending_copy(self):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


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
        num_batched_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        max_model_len: int,
        device: str,
        shared_host_cache: Optional[_RoutedExpertsHostCache] = None,
        skip_host_cache: bool = False,
    ):
        self.forward_batch = None
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_batched_tokens = num_batched_tokens
        self.max_model_len = max_model_len
        self._skip_host_cache = skip_host_cache

        if skip_host_cache:
            self.host_cache = None
            logger.info(f"Skipping host cache for device {device} (non-rank-0)")
        elif shared_host_cache is not None:
            self.host_cache = shared_host_cache
        else:
            self.host_cache = _RoutedExpertsHostCache(
                max_running_requests=max_running_requests,
                num_hidden_layers=self.num_hidden_layers,
                num_experts_per_tok=self.num_experts_per_tok,
                max_model_len=self.max_model_len,
                use_shared_memory=False,
            )

        self.device_cache = _RoutedExpertsDeviceCache(
            num_batched_tokens=self.num_batched_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

        # ---- Async D2H pipeline ----
        # Same (L, N, K) layout as device_cache.buffer.
        self._pinned_staging = torch.zeros(
            (self.num_hidden_layers, num_batched_tokens, self.num_experts_per_tok),
            dtype=_RoutedExpertsDeviceCache.DTYPE,
            pin_memory=True,
        )
        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_event = torch.cuda.Event()
        self._has_pending_copy = False

        self._pending_positions: np.ndarray | None = None
        self._pending_num_scheduled: dict[str, int] | None = None
        self._pending_total_tokens: int = 0

        pinned_mb = get_tensor_size_bytes(self._pinned_staging) / _MB
        logger.info(
            f"Routing experts pinned staging buffer allocated. "
            f"shape={tuple(self._pinned_staging.shape)}, size={pinned_mb:.2f} MB"
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    # ------------------------------------------------------------------
    # sync_fwd_experts_buffer_DtoH -- called AFTER the forward pass
    # ------------------------------------------------------------------

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokes: dict[str, int],
    ):
        if self.host_cache is None:
            return

        # 1. Finalize previous async copy -- the copy had an entire
        #    forward pass to complete so event.synchronize() is ~free.
        if self._has_pending_copy:
            self._copy_event.synchronize()
            self._scatter_to_host()
            self._has_pending_copy = False

        total_tokens = sum(num_scheduled_tokes.values())
        if total_tokens == 0:
            return

        # 2. Issue new async D2H copy on a dedicated stream.
        #    Device buffer layout is (L, N, K); copy the first total_tokens
        #    along the N dimension for every layer.
        main_stream = torch.cuda.current_stream(self._copy_stream.device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(main_stream)
            self._pinned_staging[:, :total_tokens, :].copy_(
                self.device_cache.buffer[:, :total_tokens, :], non_blocking=True
            )
            self._copy_event.record()

        # 3. Save metadata for deferred scatter.
        self._pending_positions = positions.numpy().copy()
        self._pending_num_scheduled = num_scheduled_tokes
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
        host_values = self._pinned_staging[
            :, :self._pending_total_tokens, :
        ].numpy().transpose(1, 0, 2)
        positions_np = self._pending_positions
        host_cache = self.host_cache

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
                pos = positions_np[offset:offset + n_tokens]
                max_pos = int(pos[-1]) if n_tokens > 0 else 0
                if n_tokens > 1:
                    max_pos = int(pos.max())
                buf = host_cache.get_or_grow_buffer(req_id, max_pos)
                buf[pos] = host_values[offset:offset + n_tokens]
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
        self, req_id: str, seqlen: int | None = None, free_slot: bool = True,
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

    def sync_fwd_experts_buffer_DtoH(self, positions, num_scheduled_tokes):
        pass

    def finalize_pending_copy(self):
        pass

    def get_host_cache(self):
        return None

    def get_device_cache(self):
        pass


# Global capturer instance (per-process)
_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()
_shared_host_cache: Optional[_RoutedExpertsHostCache] = None


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def get_shared_host_cache() -> Optional[_RoutedExpertsHostCache]:
    return _shared_host_cache


def create_shared_host_cache(
    model_config: ModelConfig,
    max_running_requests: int,
    max_model_len: int,
) -> _RoutedExpertsHostCache:
    global _shared_host_cache
    num_hidden_layers = model_config.hf_text_config.layers_block_type.count("moe")
    num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
    _shared_host_cache = _RoutedExpertsHostCache(
        max_running_requests=max_running_requests,
        num_hidden_layers=num_hidden_layers,
        num_experts_per_tok=num_experts_per_tok,
        max_model_len=max_model_len,
        use_shared_memory=False,
    )
    return _shared_host_cache


def init_routed_experts_capturer_with_shared_cache(
    enable: bool,
    model_config: ModelConfig,
    num_fused_shared_experts: int,
    num_batched_tokens: int,
    max_running_requests: int,
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
        logger.info(f"Skipping routed experts capturer for rank {rank}")
        capturer = _RoutedExpertsCapturerNoop()
        set_global_experts_capturer(capturer)
        return capturer

    capturer = RoutedExpertsCapturer.create(
        enable=True,
        model_config=model_config,
        num_fused_shared_experts=num_fused_shared_experts,
        num_batched_tokens=num_batched_tokens,
        max_running_requests=max_running_requests,
        max_model_len=max_model_len,
        device=device,
        skip_host_cache=False,
    )
    set_global_experts_capturer(capturer)
    return capturer
