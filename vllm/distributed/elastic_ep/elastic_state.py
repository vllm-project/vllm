# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
import weakref
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import torch.distributed
from torch.distributed import ReduceOp

from vllm.config import ParallelConfig
from vllm.distributed import (
    sched_yield,
    stateless_destroy_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.engine.core import DPEngineCoreProc

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WorkerType = Literal["existing", "new", "shutdown"]


class ScaleUpExistingWorkerState(enum.IntEnum):
    WAIT_NEW_WORKERS_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_WORKERS_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY = 5
    SWITCH_AND_PREPARE = 6
    EPLB_RESHUFFLE = 7
    COMPLETE = 8


class ScaleUpNewWorkerState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2


class ScaleDownRemainingWorkerState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    SWITCH_AND_PREPARE = 2
    COMPLETE = 3


class ScaleDownShutdownWorkerState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2


class ElasticScalingState:
    def __init__(
        self,
        model_executor: "Executor",
        engine_core: "DPEngineCoreProc",
        vllm_config: "VllmConfig",
        new_parallel_config: ParallelConfig,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ):
        self.model_executor_ref = weakref.ref(model_executor)
        self.engine_core_ref = weakref.ref(engine_core)
        self.vllm_config = vllm_config
        self.old_dp_group = self.engine_core.dp_group if worker_type != "new" else None
        self.old_dp_store = self.engine_core.dp_store if worker_type != "new" else None
        self.new_dp_group = (
            self.engine_core.dp_group if worker_type == "new" else new_parallel_config
        )
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request
        self.waiting_for_notification = False
        self.last_barrier_timeout = False

        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewWorkerState.PREPARE
                if worker_type == "new"
                else ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_INIT
            )
        else:
            self.state = (
                ScaleDownShutdownWorkerState.PREPARE
                if worker_type == "shutdown"
                else ScaleDownRemainingWorkerState.PREPARE
            )

    @property
    def model_executor(self) -> "Executor":
        model_executor = self.model_executor_ref()
        if model_executor is None:
            raise RuntimeError("Model executor has been garbage collected")
        return model_executor

    @property
    def engine_core(self) -> "DPEngineCoreProc":
        engine_core = self.engine_core_ref()
        if engine_core is None:
            raise RuntimeError("Engine core has been garbage collected")
        return engine_core

    def progress(self) -> bool:
        if self.waiting_for_notification:
            return False

        if self.scale_type == "scale_up":
            return (
                self._progress_new_worker()
                if self.worker_type == "new"
                else self._progress_existing_worker()
            )
        return (
            self._progress_shutdown_worker()
            if self.worker_type == "shutdown"
            else self._progress_remaining_worker()
        )

    def _do_store_barrier(
        self, dp_store, group_rank, group_size, barrier_id, timeout=None
    ):
        arrival_key = f"arrival_{barrier_id}_{group_rank}"
        dp_store.set(arrival_key, b"1")

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < group_size:
            if (
                timeout is not None
                and time.time() - start_time > timeout.total_seconds()
            ):
                raise RuntimeError(
                    f"Barrier timed out after {timeout.total_seconds()} seconds"
                )

            for i in range(group_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                present = dp_store.check([key])
                if present:
                    processes_arrived.add(i)

            if len(processes_arrived) < group_size:
                sched_yield()

    def _staged_barrier(self, use_new_group: bool) -> bool:
        dp_store = self.new_dp_store if use_new_group else self.old_dp_store
        dp_group = self.new_dp_group if use_new_group else self.old_dp_group

        group_rank = dp_group.rank()
        group_size = dp_group.size()
        barrier_id = "elastic_ep_barrier"

        if self.last_barrier_timeout:
            self._do_store_barrier(
                dp_store, group_rank, group_size, barrier_id, timeout=None
            )
            torch.distributed.barrier(dp_group)
            self.last_barrier_timeout = False
            if group_rank == 0:
                dp_store.delete_key("elastic_ep_sync")
                for i in range(group_size):
                    dp_store.delete_key(f"arrival_{barrier_id}_{i}")
            return True

        # NOTE(yongji): figure out appropriate timeout
        timeout = None if dp_store.check(["elastic_ep_sync"]) else timedelta(seconds=5)

        try:
            self._do_store_barrier(
                dp_store, group_rank, group_size, barrier_id, timeout=timeout
            )
            torch.distributed.barrier(dp_group)
            self.last_barrier_timeout = False
            if group_rank == 0:
                dp_store.delete_key("elastic_ep_sync")
                for i in range(group_size):
                    dp_store.delete_key(f"arrival_{barrier_id}_{i}")
            return True
        except RuntimeError:
            self.last_barrier_timeout = True
            dp_store.compare_set("elastic_ep_sync", "", b"1")
            return False

    def _progress_existing_worker(self) -> bool:
        state = self.state

        if state == ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_INIT:
            self.waiting_for_notification = True
            return False

        elif state == ScaleUpExistingWorkerState.CREATE_STANDBY_GROUPS:
            # NOTE(yongji): wait for all exisiting workers to receive the request
            if (
                int(self.old_dp_store.get("elastic_ep_ticket"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=False):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("elastic_ep_ticket")
            self._create_standby_groups()
            self.state = ScaleUpExistingWorkerState.TRANSFER_EXPERT_MAPPING
            return True

        elif state == ScaleUpExistingWorkerState.TRANSFER_EXPERT_MAPPING:
            self._transfer_expert_mapping()
            self.state = ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_WEIGHTS_INIT
            return True

        elif state == ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_WEIGHTS_INIT:
            self.waiting_for_notification = True
            return False

        elif state == ScaleUpExistingWorkerState.TRANSFER_WEIGHTS:
            if (
                int(self.old_dp_store.get("elastic_ep_ticket"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=False):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("elastic_ep_ticket")
            self._transfer_weights()
            self.state = ScaleUpExistingWorkerState.SYNC_KV_CACHE_MEMORY
            return True

        elif state == ScaleUpExistingWorkerState.SYNC_KV_CACHE_MEMORY:
            self._sync_kv_cache_memory()
            self.state = ScaleUpExistingWorkerState.SWITCH_AND_PREPARE
            return True

        elif state == ScaleUpExistingWorkerState.SWITCH_AND_PREPARE:
            self._switch_and_prepare()
            self.state = ScaleUpExistingWorkerState.EPLB_RESHUFFLE
            self.new_dp_store.add("elastic_ep_ticket", 1)
            return True

        elif state == ScaleUpExistingWorkerState.EPLB_RESHUFFLE:
            if (
                int(self.new_dp_store.get("elastic_ep_ticket"))
                < self.new_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=True):
                return False
            if self.new_dp_group.rank() == 0:
                self.new_dp_store.delete_key("elastic_ep_ticket")
            self._eplb_reshuffle()
            self.state = ScaleUpExistingWorkerState.COMPLETE
            self._update_parallel_config()
            return True

        return False

    def _progress_new_worker(self) -> bool:
        state = self.state

        if state == ScaleUpNewWorkerState.PREPARE:
            tensor = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(
                tensor, op=ReduceOp.MAX, group=self.new_dp_group
            )
            data = tensor.tolist()
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self.state = ScaleUpNewWorkerState.EPLB_RESHUFFLE
            self.new_dp_store.add("elastic_ep_ticket", 1)
            return True

        elif state == ScaleUpNewWorkerState.EPLB_RESHUFFLE:
            if (
                int(self.new_dp_store.get("elastic_ep_ticket"))
                < self.new_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=True):
                return False
            assert self.new_dp_group.rank() > 0
            self._eplb_reshuffle()
            self.state = ScaleUpNewWorkerState.COMPLETE
            return True

        return False

    def _progress_remaining_worker(self) -> bool:
        state = self.state

        if state == ScaleDownRemainingWorkerState.PREPARE:
            self.state = ScaleDownRemainingWorkerState.EPLB_RESHUFFLE
            self.old_dp_store.add("elastic_ep_ticket", 1)
            return True

        elif state == ScaleDownRemainingWorkerState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("elastic_ep_ticket"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=False):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("elastic_ep_ticket")
            self._eplb_reshuffle_before_scale_down()
            self.state = ScaleDownRemainingWorkerState.SWITCH_AND_PREPARE
            return True
        elif state == ScaleDownRemainingWorkerState.SWITCH_AND_PREPARE:
            self._create_standby_groups()
            self._switch_and_prepare()
            self.state = ScaleDownRemainingWorkerState.COMPLETE
            return True

        return False

    def _progress_shutdown_worker(self) -> bool:
        state = self.state

        if state == ScaleDownShutdownWorkerState.PREPARE:
            self.state = ScaleDownShutdownWorkerState.EPLB_RESHUFFLE
            self.old_dp_store.add("elastic_ep_ticket", 1)
            return True

        if state == ScaleDownShutdownWorkerState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("elastic_ep_ticket"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(use_new_group=False):
                return False
            assert self.old_dp_group.rank() > 0
            self._eplb_reshuffle_before_scale_down()
            self.state = ScaleDownShutdownWorkerState.COMPLETE
            self.engine_core._send_worker_notification("SHUTDOWN_COMPLETE")
            self.engine_core.shutdown()
            return True

        return False

    def handle_notification(self, notification_type: str):
        assert self.worker_type != "new"
        if (
            notification_type == "NEW_WORKERS_INIT_READY"
            and self.state == ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_INIT
        ):
            self.old_dp_store.add("elastic_ep_ticket", 1)
            self.waiting_for_notification = False
            self.state = ScaleUpExistingWorkerState.CREATE_STANDBY_GROUPS
        elif (
            notification_type == "NEW_WORKERS_WEIGHTS_INIT_READY"
            and self.state == ScaleUpExistingWorkerState.WAIT_NEW_WORKERS_WEIGHTS_INIT
        ):
            self.old_dp_store.add("elastic_ep_ticket", 1)
            self.waiting_for_notification = False
            self.state = ScaleUpExistingWorkerState.TRANSFER_WEIGHTS

    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewWorkerState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingWorkerState.COMPLETE
            )
        return (
            self.state == ScaleDownShutdownWorkerState.COMPLETE
            if self.worker_type == "shutdown"
            else self.state == ScaleDownRemainingWorkerState.COMPLETE
        )

    def _create_standby_groups(self):
        assert isinstance(self.new_dp_group, ParallelConfig)
        self.new_dp_group, self.new_dp_store = (
            self.new_dp_group.stateless_init_dp_group(return_store=True)
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("create_standby_groups", self.reconfig_request)
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Created standby communication groups")

    def _transfer_weights(self):
        assert self.reconfig_request is not None
        old_dp_size = self.old_dp_group.size()
        new_dp_size = self.reconfig_request.new_data_parallel_size

        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("transfer_weights", old_dp_size, new_dp_size)
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Transferred weights to new workers")

    def _transfer_expert_mapping(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("broadcast_expert_mapping",)
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Broadcasted expert mapping to new workers")

    def _sync_kv_cache_memory(self):
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        ParallelConfig.sync_kv_cache_memory_size(
            self.new_dp_group, self.engine_core.available_gpu_memory_for_kv_cache
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Synced KV cache memory size to new workers")

    def _switch_and_prepare(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("switch_and_prepare",)
        )
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        tensor = torch.tensor(
            [engines_running, current_wave, step_counter],
            dtype=torch.int32,
            device="cpu",
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=self.new_dp_group)
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _eplb_reshuffle(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("perform_eplb_reshuffle",)
        )
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _eplb_reshuffle_before_scale_down(self):
        assert self.reconfig_request is not None
        self.model_executor.collective_rpc(
            "elastic_ep_execute",
            args=(
                "perform_eplb_reshuffle",
                self.reconfig_request.new_data_parallel_size,
            ),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _update_parallel_config(self):
        assert self.reconfig_request is not None
        reconfig_request = self.reconfig_request
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        parallel_config._stateless_world_group_port_list = (
            reconfig_request.new_stateless_world_group_port_list
        )
        parallel_config._stateless_dp_group_port_list = (
            reconfig_request.new_stateless_dp_group_port_list
        )
        parallel_config._stateless_ep_group_port_list = (
            reconfig_request.new_stateless_ep_group_port_list
        )
