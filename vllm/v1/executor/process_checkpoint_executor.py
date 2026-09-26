# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-node sleep executor preserving complete CUDA worker state."""

import json
import os
import threading
import time
from concurrent.futures import Future
from functools import wraps
from multiprocessing.process import BaseProcess
from typing import Any, cast

from vllm.logger import init_logger
from vllm.utils.cuda_process_checkpoint import CudaProcessCheckpoint
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._checkpoint_lock:
            return method(self, *args, **kwargs)

    return call


def _validate_nccl_checkpoint(worker, suspended):
    from vllm.distributed import get_tp_group

    comm = cast(Any, get_tp_group().device_communicator).pynccl_comm
    if (
        comm is None
        or not comm.available
        or comm.disabled
        or not comm.nccl.has_symbol("ncclCommSuspend")
        or not comm.nccl.has_symbol("ncclCommResume")
        or comm._suspended != suspended
    ):
        raise RuntimeError(
            "NCCL checkpoint requires an active suspend/resume implementation"
        )


class ProcessCheckpointExecutor(MultiprocExecutor):
    """Use dedicated workers even at TP=1 so failed restoration is terminable."""

    def __init__(self, vllm_config, monitor_workers=True):
        self._checkpoint_lock = threading.RLock()
        self._checkpoint = None
        self._checkpoint_active = False
        parallel = vllm_config.parallel_config
        compilation = getattr(vllm_config, "compilation_config", None)
        passes = getattr(compilation, "pass_config", None)
        fused_allreduce = bool(getattr(compilation, "mode", 0)) and bool(
            getattr(passes, "fuse_allreduce_rms", False)
        )
        standalone_flashinfer = os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER") == "1"
        # The compiler fusion allocates FlashInfer workspaces independently
        # of the standalone all-reduce environment switch.
        self._checkpoint_flashinfer = parallel.tensor_parallel_size > 1 and (
            standalone_flashinfer or fused_allreduce
        )
        self._use_nccl_suspend = (
            parallel.tensor_parallel_size > 1
            and vllm_config.model_config.enable_nccl_comm_suspend
        )
        if (
            parallel.tensor_parallel_size not in (1, 2, 4)
            or parallel.nnodes != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.world_size != parallel.tensor_parallel_size
        ):
            raise ValueError("Process checkpoint requires single-node TP=1, 2 or 4")
        if vllm_config.kv_transfer_config or vllm_config.ec_transfer_config:
            raise ValueError(
                "Process checkpoint does not support external cache transfer"
            )
        if not vllm_config.model_config.enable_sleep_mode:
            raise ValueError("Process checkpoint requires enable_sleep_mode=True")
        if parallel.tensor_parallel_size > 1:
            required = {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "NCCL_CUMEM_HOST_ENABLE": "0",
                "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
                "VLLM_USE_NCCL_SYMM_MEM": "0",
            }
            if vllm_config.model_config.enable_nccl_comm_suspend:
                # NCCL suspend releases P2P mappings while retaining addresses.
                # NVLS keeps persistent shared allocations that CUDA driver
                # checkpoint cannot offload on the tested driver 580.
                if os.environ.get("NCCL_DISABLE_MEM_MANAGER", "0") != "0":
                    raise ValueError("NCCL checkpoint requires the memory manager")
                required.update(
                    {
                        "NCCL_P2P_DISABLE": "0",
                        "NCCL_CUMEM_ENABLE": "1",
                        "NCCL_NVLS_ENABLE": "0",
                        "VLLM_ALLREDUCE_USE_FLASHINFER": "1"
                        if standalone_flashinfer
                        else "0",
                    }
                )
            if self._checkpoint_flashinfer:
                required["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] = "trtllm"
            if not parallel.disable_custom_all_reduce or any(
                os.environ.get(k) != v for k, v in required.items()
            ):
                raise ValueError(
                    "Process checkpoint currently requires disabled custom "
                    f"all-reduce and NCCL checkpoint configuration: {required}"
                )
        super().__init__(vllm_config, monitor_workers)

    def _record(self, **event):
        event["monotonic_s"] = time.monotonic()
        logger.info("CUDA_PROCESS_CHECKPOINT %s", json.dumps(event))

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        kv_output_aggregator=None,
        ec_output_aggregator=None,
    ):
        if self.is_failed:
            raise RuntimeError("Process checkpoint executor failed")
        if self._checkpoint_active:
            # EngineCore's first pause synchronized and cleared these caches.
            # No new worker work can run while checkpointed, so a repeated
            # pause can acknowledge that state without calling locked CUDA APIs.
            if (
                method
                in ("synchronize_device", "reset_mm_cache", "reset_encoder_cache")
                and not args
                and not kwargs
                and unique_reply_rank is None
                and kv_output_aggregator is None
                and ec_output_aggregator is None
            ):
                result = [None] * len(self.workers)
                if non_block:
                    future: Future[Any] = Future()
                    future.set_result(result)
                    return future
                return result
            raise RuntimeError(
                "Workers are checkpointed; wake them before issuing RPCs"
            )
        if method in ("sleep", "wake_up") and timeout is None:
            timeout = 120
        return super().collective_rpc(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            non_block=non_block,
            unique_reply_rank=unique_reply_rank,
            kv_output_aggregator=kv_output_aggregator,
            ec_output_aggregator=ec_output_aggregator,
        )

    def check_health(self):
        if self.is_failed:
            raise RuntimeError("Process checkpoint executor failed")
        if not self._checkpoint_active:
            super().check_health()

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        MultiprocExecutor._ensure_worker_termination(worker_procs)
        # A checkpointed worker cannot finish CUDA-dependent cleanup. The base
        # executor may SIGKILL it and return before waitpid reaps the child.
        # Reap before notifying EngineCore of failure and losing the parent.
        deadline = time.monotonic() + 5
        for proc in worker_procs:
            proc.join(timeout=max(0, deadline - time.monotonic()))
            if proc.exitcode is None:
                logger.error(
                    "Worker pid=%s did not exit within cleanup deadline", proc.pid
                )

    def _close_checkpoint(self):
        checkpoint, self._checkpoint = self._checkpoint, None
        if checkpoint is not None:
            try:
                checkpoint.close()
            except Exception:
                logger.exception("Failed to close CUDA checkpoint helper")

    def _stop_failed(self):
        self.is_failed = True
        # Terminate worker processes even if a CUDA API is blocked. Never try
        # to continue serving a partially checkpointed rank set.
        super().shutdown()
        self._close_checkpoint()
        callback = self.failure_callback
        if callback is not None:
            self.failure_callback = None
            callback()

    @_serialized
    def sleep(self, level=1):
        if self.is_failed:
            raise RuntimeError("Process checkpoint executor failed")
        if level != 1:
            raise ValueError("Process checkpoint currently supports sleep level 1 only")
        if self._checkpoint_active:
            return
        # Match the parent's no-op after KV-first partial wake: weights are
        # still asleep, but CUDA/NCCL have resumed and must not be assumed
        # suspended. Fully wake all tags before starting another checkpoint.
        if "weights" in self.sleeping_tags:
            return
        if self._checkpoint is None:
            self._checkpoint = CudaProcessCheckpoint(self._record)
        try:
            if self._use_nccl_suspend:
                self.collective_rpc(
                    _validate_nccl_checkpoint, args=(False,), timeout=30
                )
            super().sleep(level)
            if self._use_nccl_suspend:
                self.collective_rpc(_validate_nccl_checkpoint, args=(True,), timeout=30)
            if self._checkpoint_flashinfer:
                self.collective_rpc("checkpoint_prepare", timeout=120)
            self._checkpoint.suspend([w.proc.pid for w in self.workers])
            self._checkpoint_active = True
        except Exception:
            self._stop_failed()
            raise

    @_serialized
    def wake_up(self, tags=None):
        if self.is_failed:
            raise RuntimeError("Process checkpoint executor failed")
        if tags and any(tag not in self.sleeping_tags for tag in tags):
            raise ValueError("Invalid or already awake sleep tag")
        try:
            if self._checkpoint_active:
                cast(CudaProcessCheckpoint, self._checkpoint).resume()
                self._checkpoint_active = False
                if self._checkpoint_flashinfer:
                    self.collective_rpc("checkpoint_restore", timeout=120)
            super().wake_up(tags)
            if self._use_nccl_suspend:
                self.collective_rpc(
                    _validate_nccl_checkpoint, args=(False,), timeout=30
                )
        except Exception:
            self._stop_failed()
            raise

    @_serialized
    def shutdown(self):
        try:
            if self._checkpoint_active and not getattr(self, "is_failed", False):
                try:
                    self.wake_up()
                except Exception:
                    logger.exception(
                        "Failed to restore during shutdown; workers terminated"
                    )
            super().shutdown()
        finally:
            self._close_checkpoint()
