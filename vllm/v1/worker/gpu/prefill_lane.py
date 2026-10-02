# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill lane: long prompts prefill on a side thread while decode keeps going.

A step that carries a prefill chunk takes as long as the chunk, so every
running request stalls behind a long prompt. Decode is memory-bound and
prefill compute-bound, so both fit on the GPU at once: the lane runs the bulk
of a long prompt through a second model runner on its own thread, CU-masked
stream and TP communicator, while the main runner keeps stepping.

The scheduler sees the lane as an async KV load (PrefillLaneConnector): the
request waits in WAITING_FOR_REMOTE_KVS with its blocks reserved until the
lane reports it done, and its last token then goes through the normal path,
which samples it.
"""

import copy
import ctypes
import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import init_tp_lane_group, use_tp_lane
from vllm.forward_context import use_thread_local_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu import model_runner as model_runner_mod
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)

# Chunks at or below this many tokens take the decode attention kernels, whose
# scratch buffers live on the layers and are shared with the main runner.
_MIN_CHUNK_TOKENS = 256


@dataclass
class PrefillJob:
    """Prefill tokens [new_req.num_computed_tokens, end) of one request."""

    new_req: NewRequestData
    end: int
    zero_block_ids: list[int]


class PrefillLaneRunner(GPUModelRunner):
    """Model runner for the lane.

    Shares the main runner's model, drafter weights and KV cache, keeps its
    own request state, input buffers and attention metadata builders, and
    never runs CUDA graphs.
    """

    def __init__(self, main: GPUModelRunner, chunk_tokens: int):
        self.main = main
        # Merging short first and last chunks takes a chunk past chunk_tokens,
        # and past the scheduler's budget that sizes the runner's buffers.
        vllm_config = copy.copy(main.vllm_config)
        vllm_config.scheduler_config = copy.copy(main.vllm_config.scheduler_config)
        vllm_config.scheduler_config.max_num_batched_tokens = max(
            main.max_num_tokens, chunk_tokens + 2 * _MIN_CHUNK_TOKENS
        )
        super().__init__(vllm_config, main.device)
        self.load_model()
        self._init_kv_cache()

    def load_model(self, *args, **kwargs) -> None:
        main = self.main
        loader = SimpleNamespace(load_model=lambda **_: main.model)
        if self.speculator is not None:
            assert isinstance(main.speculator, DraftModelSpeculator)
            draft_model = main.speculator.model
            self.speculator.load_draft_model = lambda *_: draft_model  # type: ignore[method-assign]
        with patch.object(model_runner_mod, "get_model_loader", lambda _: loader):
            super().load_model()
        if self.speculator is not None:
            # Derived at load by diffing the layers before and after loading the
            # drafter, which the main runner already did.
            assert isinstance(main.speculator, DraftModelSpeculator)
            assert isinstance(self.speculator, DraftModelSpeculator)
            self.speculator.draft_attn_layer_names = (
                main.speculator.draft_attn_layer_names
            )

    def _init_kv_cache(self) -> None:
        main = self.main
        self.kv_cache_config = main.kv_cache_config
        self.cp_interleave = main.cp_interleave
        draft_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            draft_layer_names = self.speculator.draft_attn_layer_names
        self.attn_groups, _, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
            draft_layer_names=draft_layer_names,
        )
        self.block_tables = BlockTables(
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            **{
                **main.block_table_kwargs,
                "max_num_batched_tokens": self.max_num_tokens,
            },
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.NONE,
            decode_query_len=self.decode_query_len,
        )
        if isinstance(self.speculator, DraftModelSpeculator):
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            self.speculator.init_cudagraph_manager(CUDAGraphMode.NONE)
        self.kv_caches = main.kv_caches
        self.kv_connector = NO_OP_KV_CONNECTOR
        if self.kv_cache_config.needs_kv_cache_zeroing:
            self._init_kv_zero_meta()

    def prefill(self, job: PrefillJob, chunk_tokens: int) -> None:
        req_id = job.new_req.req_id
        start = job.new_req.num_computed_tokens
        for i, (lo, hi) in enumerate(split_chunks(start, job.end, chunk_tokens)):
            out = SchedulerOutput.make_empty()
            if i == 0:
                out.scheduled_new_reqs = [job.new_req]
                out.new_block_ids_to_zero = job.zero_block_ids or None
            else:
                out.scheduled_cached_reqs = CachedRequestData(
                    req_ids=[req_id],
                    resumed_req_ids=set(),
                    new_token_ids=[[]],
                    all_token_ids={},
                    new_block_ids=[None],
                    num_computed_tokens=[lo],
                    num_output_tokens=[0],
                )
            out.num_scheduled_tokens = {req_id: hi - lo}
            out.total_num_scheduled_tokens = hi - lo
            self.execute_model(out)
            self.sample_tokens(None)
        self._remove_request(req_id)


def split_chunks(start: int, end: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """Chunk boundaries on multiples of chunk_tokens, like the scheduler's
    Mamba-aligned split; a short first or last chunk is merged into its
    neighbour."""
    bounds = [start]
    nxt = (start // chunk_tokens + 1) * chunk_tokens
    while nxt < end:
        bounds.append(nxt)
        nxt += chunk_tokens
    bounds.append(end)
    if len(bounds) > 2 and bounds[-1] - bounds[-2] <= _MIN_CHUNK_TOKENS:
        del bounds[-2]
    if len(bounds) > 2 and bounds[1] - bounds[0] <= _MIN_CHUNK_TOKENS:
        del bounds[1]
    return list(zip(bounds[:-1], bounds[1:]))


def _cu_masked_stream(device: torch.device, num_units: int) -> torch.cuda.Stream:
    """A stream restricted to the last num_units compute units (ROCm)."""
    total = torch.cuda.get_device_properties(device).multi_processor_count
    if not current_platform.is_rocm() or not 0 < num_units < total:
        return torch.cuda.Stream(device)
    hip = ctypes.CDLL("libamdhip64.so")
    mask = (ctypes.c_uint32 * ((total + 31) // 32))()
    for unit in range(total - num_units, total):
        mask[unit // 32] |= 1 << (unit % 32)
    handle = ctypes.c_void_p()
    err = hip.hipExtStreamCreateWithCUMask(ctypes.byref(handle), len(mask), mask)
    if err != 0:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")
    logger.info("Prefill lane stream on %d of %d compute units", num_units, total)
    return torch.cuda.ExternalStream(handle.value, device=device)


class PrefillLane:
    """Runs prefill jobs in submission order on a side thread.

    Every TP rank gets the same jobs in the same order, so the lane's
    collectives line up across ranks without extra coordination.
    """

    def __init__(self, main: GPUModelRunner, chunk_tokens: int, num_units: int):
        self.chunk_tokens = chunk_tokens
        self.device = main.device
        init_tp_lane_group()
        self.stream = _cu_masked_stream(self.device, num_units)
        with torch.cuda.stream(self.stream):
            self.runner = PrefillLaneRunner(main, chunk_tokens)
        self.jobs: queue.Queue[PrefillJob] = queue.Queue()
        self.num_pending = 0
        self.done: list[str] = []
        self.cond = threading.Condition()
        self.thread = threading.Thread(
            target=self._loop, name="prefill-lane", daemon=True
        )
        self.thread.start()

    def submit(self, job: PrefillJob) -> None:
        logger.debug(
            "Prefill lane: %s tokens [%d, %d)",
            job.new_req.req_id,
            job.new_req.num_computed_tokens,
            job.end,
        )
        with self.cond:
            self.num_pending += 1
        self.jobs.put(job)

    def take_finished(self, wait_s: float = 0.0) -> set[str]:
        """Return the jobs finished so far; with wait_s, wait that long for one
        if none has finished and some are still running."""
        with self.cond:
            if wait_s and not self.done and self.num_pending:
                self.cond.wait(wait_s)
            done, self.done = self.done, []
        return set(done)

    def _loop(self) -> None:
        torch.accelerator.set_device_index(self.device)
        use_thread_local_forward_context()
        with use_tp_lane(), torch.cuda.stream(self.stream), torch.inference_mode():
            while True:
                job = self.jobs.get()
                self.runner.prefill(job, self.chunk_tokens)
                self.stream.synchronize()
                with self.cond:
                    self.num_pending -= 1
                    self.done.append(job.new_req.req_id)
                    self.cond.notify_all()
