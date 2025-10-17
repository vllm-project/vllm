# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.tt_model_runner import TTModelRunner
from vllm.v1.worker.worker_base import WorkerBase
from vllm.worker.tt_model_runner import TTModelInput
from vllm.worker.tt_worker import (close_mesh_device, get_mesh_grid,
                                   get_num_available_blocks_tt,
                                   open_mesh_device)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class TTWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker)

        # Initialized by init_device
        self.mesh_device = None

        # Whether to use ttnn tracing for model execution
        override_tt_config = self.model_config.override_tt_config
        trace_key = "trace_mode"
        self.trace_mode = True
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in [True, False], \
                f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            self.trace_mode = override_tt_config[trace_key]

    def init_device(self) -> None:
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        if dp_rank == 0:
            self.mesh_device = open_mesh_device(
                self.model_config.override_tt_config, self.trace_mode, dp_rank)
            self.device_config.device = self.mesh_device
            assert self.mesh_device is not None
            self.device_config.num_devices = self.mesh_device.get_num_devices()
        else:
            mesh_grid = get_mesh_grid(dp_rank)
            self.mesh_device = None
            # Num devices is required for determining num blocks in KV cache.
            self.device_config.num_devices = mesh_grid[0] * mesh_grid[1]
        # Init ModelRunner here, so that we have access to self.mesh_device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            mesh_device=self.mesh_device,
            trace_mode=self.trace_mode,
        )

    def load_model(self):
        # Only DP rank 0 loads the model
        if self.vllm_config.parallel_config.data_parallel_rank == 0:
            self.model_runner.load_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        For the GPU/TPU backends, this method generates the KVCacheSpec by
        parsing the kv cache format from each Attention module in the static
        forward context (compilation_config.static_forward_context).
        core/kv_cache_utils.py uses the KVCacheSpec along with available
        memory info from a profiling run to determine num blocks.
        
        For the TT backend, the static forward context is not populated since
        the modelling code is independent so we currently skip creating a
        kv cache spec for each layer, similar to the Spyre/Neuron backends.
        Currently we also don't run profiling to determine available memory.
        
        Return a dummy single layer KVCacheSpec and in the
        determine_available_memory function override num blocks using
        self.cache_config.num_gpu_blocks_override.
        """

        # TODO: Once we're able to populate a static forward context,
        # generate separate specs per layer (e.g. also sliding window, local
        # attention).

        model_config = self.model_config
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        # Excludes TP factor since that is handled on the model side for TT.
        total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        dtype = (model_config.dtype if cache_config.cache_dtype == "auto" else
                 STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype])

        attn_spec = FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=total_num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            use_mla=model_config.use_mla,
            sliding_window=model_config.get_sliding_window())
        kv_cache_spec: dict[str, KVCacheSpec] = {"foo": attn_spec}
        return kv_cache_spec

    def determine_available_memory(self) -> int:
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).
        
        Currenly we just return a large dummy number of bytes similar to the
        Spyre/Neuron backends and override the number of kv cache blocks.
        """

        # TODO: Once we can run profiling, return real available memory
        # instead of overriding the number of blocks.
        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 1 << 64

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache (only DP rank 0) and initialize persistent
        input batch (all DP ranks) with the specified kv_cache_config.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        # Cache is already initialized in initialize_from_config.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        # Currently skip and compile/capture-trace during the first execution.
        pass

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        assert self.is_driver_worker, "There should only be one Worker for TT"
        output = self.model_runner.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    # ---- DP gather hooks called by DPEngineCoreProc in core.py ----

    def build_dp_model_input(
            self,
            scheduler_output: "SchedulerOutput") -> Optional[TTModelInput]:
        """Called by each DP rank to build TTModelInput from scheduler output.
        Returns None if there is no scheduled work in this step.
        """
        return self.model_runner.build_model_input(scheduler_output)

    def concat_and_execute_dp(
            self,
            inputs: list[Optional[TTModelInput]]) -> list[list[list[int]]]:
        """Called only by DP rank 0 to concatenate DP-sized inputs and execute.
        Returns per-DP sampled ids."""
        assert self.vllm_config.parallel_config.data_parallel_rank == 0, \
            "concat_and_execute_dp must run on DP rank 0"
        assert self.is_driver_worker, "concat_and_execute_dp must run on driver"
        merged = self.model_runner.concat_model_inputs(inputs)
        return self.model_runner.execute_with_model_input(merged)

    def apply_dp_execution_result(
            self, sampled_token_ids: list[list[int]]) -> ModelRunnerOutput:
        """Called by each DP rank to apply sampled tokens to internal caches.
        """
        return self.model_runner.generate_runner_output(sampled_token_ids)

    # ---- Destructor (used to close devices) ----

    def __del__(self):
        # Delete model runner first in case there are model artifacts
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

            if self.mesh_device:
                close_mesh_device(self.mesh_device,
                                  self.model_config.override_tt_config)
                del self.mesh_device

        if hasattr(super(), '__del__'):
            super().__del__()  # type: ignore
