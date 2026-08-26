# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange PLE data between a GPU worker and the CPU-offload process."""

import os
import queue
import threading
from multiprocessing.reduction import ForkingPickler
from typing import Any

import msgspec
import torch
import torch.nn as nn
import zmq
from cuda.bindings import driver as cuda_driver

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
)
from vllm.v1.ple_offload.protocol import (
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class PleOffloadConnector:
    """Connect a GPU runner to the shared PLE CPU worker.

    MRV1 and MRV2 share the same CPU-input and CUDA-output IPC protocol.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        ipc_addr: str,
        *,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor | None,
    ) -> None:
        self.device = device
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self._layers = self._setup_layers(vllm_config, model)

        # Both runner paths stage into the same shared buffers. TP0 registers
        # them with CUDA so MRV2 can use asynchronous D2H copies.
        scheduler_config = vllm_config.scheduler_config
        self._input_ids_buf = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._ngram_context_buf = None
        config = vllm_config.model_config.hf_text_config
        ngram_context_len = int(config.ngram_size) - 1
        if ngram_context_len > 0:
            self._ngram_context_buf = torch.empty(
                scheduler_config.max_num_seqs,
                ngram_context_len,
                dtype=torch.int32,
                device="cpu",
            ).share_memory_()

        # Runner input allocations are address-stable, so bind them once and
        # pass only batch sizes through the per-forward request queue.
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._uses_cuda_inputs = self._input_ids_source.is_cuda
        self._validate_input_sources()

        self._pinned_input_buffers: list[torch.Tensor] = []
        # PLE rejects DBO, and each forward consumes its output before the
        # next launch, so one pending request is sufficient.
        self._request_queue: queue.Queue[PleOffloadRequest | None] = queue.Queue(
            maxsize=1
        )
        self._request_thread: threading.Thread | None = None
        self._request_thread_ready = threading.Event()
        self._zmq_ctx: zmq.Context | None = None
        self._registration_socket: zmq.Socket | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        self._input_ready_event: torch.cuda.Event | None = None
        self._d2h_done_event: torch.cuda.Event | None = None

        try:
            self._zmq_ctx = zmq.Context()
            self._registration_socket = self._zmq_ctx.socket(zmq.PUSH)
            self._registration_socket.connect(ipc_addr)
            self._register_with_offload_worker(vllm_config, ipc_addr)

            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses.
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                self._start_request_thread(ipc_addr)
        except Exception:
            self.close()
            raise

    def _setup_layers(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
    ) -> dict[str, PleOffloadLayer]:
        """Attach output buffers and semaphores to GPU PLE placeholders."""
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )

        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for layer in layers.values():
            # The CPU worker writes results here through CUDA IPC. The GPU
            # placeholder waits on the paired cross-process semaphore.
            output_buffer = torch.empty(
                max_num_tokens,
                int(config.ple_embed_dim),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=self.device,
            )
            layer.setup_cross_process_offload(
                output_buffer,
                CpuGpuSemaphore(self.device),
            )
        return layers

    def _pin_input_buffers(self) -> None:
        """Page-lock shared input allocations without replacing their storage."""
        buffers = [self._input_ids_buf, self._query_start_loc_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            if buffer.device.type != "cpu" or not buffer.is_shared():
                raise RuntimeError("PLE input buffers must be shared CPU tensors")
            if not buffer.is_contiguous():
                raise RuntimeError("PLE input buffers must be contiguous")
            _cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    cuda_driver.CU_MEMHOSTREGISTER_PORTABLE,
                ),
                "cuMemHostRegister(PLE input buffer)",
            )
            self._pinned_input_buffers.append(buffer)
            if not buffer.is_pinned():
                raise RuntimeError("CUDA did not page-lock a PLE input buffer")

    def _unpin_input_buffers(self) -> None:
        """Release CUDA registrations after the request thread has stopped."""
        for buffer in reversed(self._pinned_input_buffers):
            try:
                _cuda_check(
                    cuda_driver.cuMemHostUnregister(buffer.data_ptr()),
                    "cuMemHostUnregister(PLE input buffer)",
                )
            except RuntimeError:
                logger.exception("Failed to unregister a PLE input buffer")
        self._pinned_input_buffers.clear()

    def _register_with_offload_worker(
        self, vllm_config: VllmConfig, ipc_addr: str
    ) -> None:
        """Register CUDA IPC outputs and shared CPU inputs with the worker."""
        # Each GPU worker owns distinct output buffers, while TP0's shared
        # inputs become the request source for its DP rank.
        registration = PleOffloadRegistration(
            worker_id=(
                self.dp_rank * vllm_config.parallel_config.world_size
                + vllm_config.parallel_config.rank
            ),
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            gpu_output_buffers={
                name: layer._gpu_output_buffer for name, layer in self._layers.items()
            },
            sem_flag_tensors={
                name: layer._sem.flag_tensor for name, layer in self._layers.items()
            },
            input_ids_buf=self._input_ids_buf,
            query_start_loc_buf=self._query_start_loc_buf,
            ngram_context_buf=self._ngram_context_buf,
        )

        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        assert self._registration_socket is not None
        self._registration_socket.send(payload)

        logger.info(
            "PleOffload: registered %d PleOffloadLayer(s) "
            "(dp_rank=%d, tp_rank=%d, ipc_addr=%s): %s",
            len(self._layers),
            self.dp_rank,
            self.tp_rank,
            ipc_addr,
            sorted(self._layers),
        )

    def _start_request_thread(self, ipc_addr: str) -> None:
        """Start the thread that publishes batches after inputs are ready."""
        self._request_thread = threading.Thread(
            target=self._request_loop,
            args=(ipc_addr,),
            name=f"ple-offload-dp{self.dp_rank}",
            daemon=True,
        )
        self._request_thread.start()
        if not self._request_thread_ready.wait(timeout=10):
            raise RuntimeError("Timed out starting the PLE request thread")

    def _request_loop(self, ipc_addr: str) -> None:
        """Stage fixed runner inputs, then notify the CPU worker."""
        socket: zmq.Socket | None = None
        try:
            if self._zmq_ctx is None:
                raise RuntimeError("PLE ZMQ context closed before thread startup")
            socket = self._zmq_ctx.socket(zmq.PUSH)
            socket.connect(ipc_addr)
            self._request_thread_ready.set()
            while True:
                request = self._request_queue.get()
                if request is None:
                    return
                self._process_request(request, socket)
        except Exception:
            logger.exception("PLE request thread failed")
            os._exit(1)
        finally:
            self._request_thread_ready.set()
            if socket is not None:
                socket.close(linger=0)

    def _process_request(self, request: PleOffloadRequest, socket: zmq.Socket) -> None:
        """Stage one batch from fixed sources and publish its request."""
        if self._uses_cuda_inputs:
            self._copy_cuda_inputs(request)
        else:
            self._copy_cpu_inputs(request)

        with torch.cuda.nvtx.range("ple_offload.send_request"):
            socket.send(msgspec.msgpack.encode(request))

    def _copy_cpu_inputs(self, request: PleOffloadRequest) -> None:
        """Stage MRV1's existing CPU mirrors in the notifier thread."""
        num_tokens = request.num_tokens
        num_reqs = request.num_reqs
        with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
            self._input_ids_buf[:num_tokens].copy_(self._input_ids_source[:num_tokens])
        with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
            self._query_start_loc_buf[: num_reqs + 1].copy_(
                self._query_start_loc_source[: num_reqs + 1]
            )
        if self._ngram_context_buf is not None:
            assert self._ngram_context_source is not None
            with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                self._ngram_context_buf[:num_reqs].copy_(
                    self._ngram_context_source[:num_reqs]
                )

    def _validate_input_sources(self) -> None:
        """Validate fixed runner sources against shared input buffers."""
        sources = [
            ("input_ids", self._input_ids_source, self._input_ids_buf),
            (
                "query_start_loc",
                self._query_start_loc_source,
                self._query_start_loc_buf,
            ),
        ]
        if (self._ngram_context_source is None) != (self._ngram_context_buf is None):
            raise ValueError("PLE ngram_context source and buffer must match")
        if self._ngram_context_source is not None:
            assert self._ngram_context_buf is not None
            sources.append(
                (
                    "ngram_context",
                    self._ngram_context_source,
                    self._ngram_context_buf,
                )
            )

        expected_device = self.device if self._uses_cuda_inputs else torch.device("cpu")
        for name, source, buffer in sources:
            if (
                source.device != expected_device
                or source.dtype != buffer.dtype
                or source.ndim != buffer.ndim
                or source.shape[0] < buffer.shape[0]
                or source.shape[1:] != buffer.shape[1:]
            ):
                raise ValueError(f"PLE {name} source is incompatible")

    def _copy_cuda_inputs(self, request: PleOffloadRequest) -> None:
        """Stage MRV2 inputs on the background D2H stream."""
        if (
            self._d2h_stream is None
            or self._input_ready_event is None
            or self._d2h_done_event is None
        ):
            raise RuntimeError("PLE D2H resources are not initialized")

        with torch.accelerator.device_index(self.device.index):
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(self._input_ready_event)
                with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
                    self._input_ids_buf[: request.num_tokens].copy_(
                        self._input_ids_source[: request.num_tokens],
                        non_blocking=True,
                    )
                with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
                    self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                        self._query_start_loc_source[: request.num_reqs + 1],
                        non_blocking=True,
                    )
                if self._ngram_context_buf is not None:
                    assert self._ngram_context_source is not None
                    with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                        self._ngram_context_buf[: request.num_reqs].copy_(
                            self._ngram_context_source[: request.num_reqs],
                            non_blocking=True,
                        )
                self._d2h_done_event.record(self._d2h_stream)
            with torch.cuda.nvtx.range("ple_offload.wait_d2h"):
                self._d2h_done_event.synchronize()

    def _launch(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        """Queue one batch while keeping staging off the model thread."""
        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return

        if self._uses_cuda_inputs:
            assert self._input_ready_event is not None
            # The background copy stream waits for runner input production
            # without making the model stream wait for D2H completion.
            self._input_ready_event.record(torch.cuda.current_stream(self.device))
        request = PleOffloadRequest(
            dp_rank=self.dp_rank,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
        self._request_queue.put_nowait(request)

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
    ) -> None:
        """Submit real inputs or satisfy the PLE wait for a dummy forward."""
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        self._launch(num_reqs, num_tokens)

    def signal_dummy_outputs(self, num_tokens: int) -> None:
        """Locally satisfy PLE waits for dummy and capture forwards."""
        # Dummy and capture forwards do not send CPU requests, but every PLE
        # placeholder still waits for a completed output semaphore.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer._gpu_output_buffer[:num_tokens].zero_()
            layer._sem.signal(stream)

    def release_outputs(self) -> None:
        """Mark GPU output buffers reusable after the model consumes them."""
        # Reset only after the consumer forward so the CPU worker cannot
        # overwrite an output that a GPU PLE placeholder may still read.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)

    def close(self) -> None:
        """Stop request transport and release host registrations."""
        request_thread = self._request_thread
        if request_thread is not None and request_thread.is_alive():
            try:
                self._request_queue.put(None, timeout=5)
            except queue.Full:
                logger.error("Timed out stopping the PLE request thread")
            request_thread.join(timeout=5)
        if request_thread is not None and request_thread.is_alive():
            # The thread may still access the registered buffers or ZMQ context.
            logger.error("PLE request thread did not stop; deferring resource cleanup")
            return
        self._request_thread = None

        if self._pinned_input_buffers:
            with torch.accelerator.device_index(self.device.index):
                self._unpin_input_buffers()
        self._d2h_done_event = None
        self._input_ready_event = None
        self._d2h_stream = None
        if self._registration_socket is not None:
            self._registration_socket.close(linger=0)
            self._registration_socket = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
