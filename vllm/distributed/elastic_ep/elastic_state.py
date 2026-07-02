# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import weakref
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch.distributed

from vllm.config import ParallelConfig
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group,
)
from vllm.distributed.elastic_ep.async_utils import SingleMethodAsyncRunner
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.v1.engine import (
    EEPNotificationType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WorkerType = Literal["existing", "new", "removing"]


class ScaleUpExistingEngineState(enum.IntEnum):
    CREATE_STANDBY_GROUPS = 0
    STAGE_QUANT_METHODS = 1
    TRANSFER_WEIGHTS = 2
    SYNC_KV_CACHE_MEMORY_SIZE = 3
    COMMIT_SCALE_UP = 4
    COMPLETE = 5


class ScaleUpNewEngineState(enum.IntEnum):
    PRE_KV_INIT = 0
    PREPARE = 1
    COMPLETE = 2


class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    COMMIT_SCALE_DOWN = 1
    COMPLETE = 2


class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    COMPLETE = 1


EngineState: TypeAlias = (
    ScaleUpExistingEngineState
    | ScaleUpNewEngineState
    | ScaleDownRemainingEngineState
    | ScaleDownRemovingEngineState
)


class ElasticEPScalingState:
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
        self.new_parallel_config: ParallelConfig = new_parallel_config
        self.new_dp_group = self.engine_core.dp_group if worker_type == "new" else None
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request
        self.commit_requested = False
        self._prepare_runner = SingleMethodAsyncRunner()
        self._prepare_future: Future[Any] | None = None
        self._new_dp_sync: tuple[object, Any] | None = None
        self.state: EngineState
        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewEngineState.PRE_KV_INIT
                if worker_type == "new"
                else ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS
            )
        else:
            self.state = (
                ScaleDownRemovingEngineState.PREPARE
                if worker_type == "removing"
                else ScaleDownRemainingEngineState.PREPARE
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

    def _collective_rpc(self, *args, **kwargs):
        return self.model_executor.collective_rpc(*args, **kwargs)

    def _execute_async(self, execute_method: str, *args) -> bool:
        if self._prepare_future is None:
            done_keys = self._collective_rpc(
                "elastic_ep_execute",
                args=("start_async", execute_method, *args),
            )
            assert self.reconfig_request is not None
            coord_store = get_cached_tcp_store_client(
                self.reconfig_request.new_data_parallel_master_ip,
                self.reconfig_request.coord_store_port,
            )
            self._prepare_future = self._prepare_runner.start(
                coord_store.wait, done_keys
            )
        if not self._prepare_future.done():
            return False

        self._prepare_runner.clear()
        self._collective_rpc("elastic_ep_execute", args=("clear_async",))
        self._prepare_future = None
        return True

    def progress(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self._progress_new_engine()
                if self.worker_type == "new"
                else self._progress_existing_engine()
            )
        return (
            self._progress_removing_engine()
            if self.worker_type == "removing"
            else self._progress_remaining_engine()
        )

    def run_pre_kv_init_states(self) -> None:
        assert self.scale_type == "scale_up" and self.worker_type == "new"
        assert self.state == ScaleUpNewEngineState.PRE_KV_INIT
        assert self.progress()
        assert self.state == ScaleUpNewEngineState.PREPARE

    def _progress_existing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS:
            if not self._create_standby_groups():
                return False
            self.state = ScaleUpExistingEngineState.STAGE_QUANT_METHODS
            return True

        elif state == ScaleUpExistingEngineState.STAGE_QUANT_METHODS:
            if not self._execute_async("stage_standby_moe_quant_methods"):
                return False
            self.state = ScaleUpExistingEngineState.TRANSFER_WEIGHTS
            return True

        elif state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS:
            if not self._transfer_weights():
                return False
            self.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
            return True

        elif state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            if not self._sync_kv_cache_memory_size():
                return False
            self.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
            self._mark_ready_for_switch()
            return True

        elif state == ScaleUpExistingEngineState.COMMIT_SCALE_UP:
            if not self.commit_requested:
                return False
            self._commit_new_dp_group()
            self._collective_rpc("elastic_ep_execute", args=("commit_scale_up", True))
            self.state = ScaleUpExistingEngineState.COMPLETE
            self._update_parallel_config()
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleUpExistingEngineState.COMPLETE
            return True

    def _progress_new_engine(self) -> bool:
        state = self.state
        assert self.new_dp_group is not None and self.new_dp_store is not None

        if state == ScaleUpNewEngineState.PRE_KV_INIT:
            self._collective_rpc("elastic_ep_execute", args=("receive_weights",))
            self.engine_core.available_gpu_memory_for_kv_cache = (
                ParallelConfig.sync_kv_cache_memory_size(self.new_dp_group, -1)
            )
            self._collective_rpc("elastic_ep_execute", args=("prepare_new_worker",))
            self.state = ScaleUpNewEngineState.PREPARE
            return True

        elif state == ScaleUpNewEngineState.PREPARE:
            self._mark_ready_for_switch()
            tensor = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.new_dp_group,
            )
            data = tensor.tolist()
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self._collective_rpc("elastic_ep_execute", args=("commit_scale_up", False))
            self.state = ScaleUpNewEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleUpNewEngineState.COMPLETE
            return True

    def _progress_remaining_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleDownRemainingEngineState.PREPARE:
            if self._create_standby_groups():
                self.state = ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN
                self._mark_ready_for_switch()
                return True
            return False

        elif state == ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN:
            if not self.commit_requested:
                return False
            self._commit_scale_down(removing=False)
            self._commit_new_dp_group()
            self._update_parallel_config()
            self.state = ScaleDownRemainingEngineState.COMPLETE
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleDownRemainingEngineState.COMPLETE
            return True

    def _progress_removing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleDownRemovingEngineState.PREPARE:
            assert self.old_dp_group.rank() > 0
            self._commit_scale_down(removing=True)
            self.state = ScaleDownRemovingEngineState.COMPLETE
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.SHUTDOWN_COMPLETE
            )
            return True

        else:
            assert self.state == ScaleDownRemovingEngineState.COMPLETE
            return True

    def is_ready_for_switch(self) -> bool:
        return self.worker_type == "existing" and (
            self.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
            or self.state is ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN
        )

    @property
    def ready_key(self) -> str:
        return f"eep_ready/{self.engine_core.dp_rank}"

    def _mark_ready_for_switch(self) -> None:
        parallel_config = self.new_parallel_config
        get_cached_tcp_store_client(
            parallel_config.data_parallel_master_ip,
            parallel_config._coord_store_port,
        ).set(self.ready_key, b"1")

    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        return (
            self.state == ScaleDownRemovingEngineState.COMPLETE
            if self.worker_type == "removing"
            else self.state == ScaleDownRemainingEngineState.COMPLETE
        )

    def _init_new_dp_group(self) -> tuple[Any, Any]:
        return self.new_parallel_config.stateless_init_dp_group(return_store=True)

    def _ensure_new_dp_group(self) -> bool:
        if self.new_dp_group is not None:
            return True

        if self._prepare_future is None:
            self._prepare_future = self._prepare_runner.start(self._init_new_dp_group)
        if not self._prepare_future.done():
            return False

        self.new_dp_group, self.new_dp_store = self._prepare_runner.clear()
        self._prepare_future = None
        return True

    def _create_standby_groups(self) -> bool:
        assert self.old_dp_group is not None
        if not self._ensure_new_dp_group():
            return False
        if not self._execute_async("create_standby_groups", self.reconfig_request):
            return False
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Created standby communication groups")
        return True

    def _transfer_weights(self) -> bool:
        assert self.reconfig_request is not None and self.old_dp_group is not None
        old_dp_size = self.old_dp_group.size()
        new_dp_size = self.reconfig_request.new_data_parallel_size

        if not self._execute_async("transfer_weights", old_dp_size, new_dp_size):
            return False
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Transferred weights to new workers")
        return True

    def _sync_kv_cache_memory_size(self) -> bool:
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        assert self.new_dp_group is not None and self.old_dp_group is not None

        if self._new_dp_sync is None:
            tensor = torch.tensor(
                [self.engine_core.available_gpu_memory_for_kv_cache],
                dtype=torch.int64,
                device="cpu",
            )
            work = torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.new_dp_group,
                async_op=True,
            )
            self._new_dp_sync = (tensor, work)
            return False

        _, work = self._new_dp_sync
        if not work.is_completed():
            return False
        work.wait()
        self._new_dp_sync = None
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Synced KV cache memory size to new workers")
        return True

    def _commit_new_dp_group(self):
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        assert self.new_dp_group is not None
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
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=new_dp_group
        )
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _send_reconfigure_finished(self):
        assert self.new_dp_group is not None
        if self.new_dp_group.rank() == 0:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )

    def _commit_scale_down(self, removing: bool):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        self._collective_rpc(
            "elastic_ep_execute",
            args=(
                "commit_scale_down",
                self.reconfig_request.new_data_parallel_size,
                removing,
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
        parallel_config._coord_store_port = reconfig_request.coord_store_port
