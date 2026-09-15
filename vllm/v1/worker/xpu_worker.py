# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_model_runner import XPUModelRunner, XPUModelRunnerV2

from .utils import request_memory

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

    def init_device(self):
        # In DP mode, XPU workers see all visible devices.
        # Offset local_rank by the local DP shard.
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and (
                parallel_config.data_parallel_external_lb
                or parallel_config.nnodes_within_dp == 1
            )
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            replica_world_size = parallel_config.world_size
            visible_device_count = torch.accelerator.device_count()

            if parallel_config.data_parallel_external_lb:
                if parallel_config.nnodes_within_dp > 1:
                    # The replica spans nodes, so use its node-local shard.
                    replica_world_size = parallel_config.local_world_size
                    if replica_world_size > visible_device_count:
                        raise ValueError(
                            f"Local TP/PP/PCP replica size ({replica_world_size}) "
                            "exceeds the number of visible XPU devices "
                            f"({visible_device_count})."
                        )
                    local_dp_capacity = visible_device_count // replica_world_size
                elif replica_world_size < visible_device_count:
                    # A node can host multiple complete TP/PP/PCP replicas.
                    local_dp_capacity = visible_device_count // replica_world_size
                    if visible_device_count % replica_world_size != 0:
                        logger.warning_once(
                            "XPU external LB cannot evenly divide "
                            "%d visible devices into TP/PP/PCP replicas of "
                            "size %d. This node can host %d complete DP "
                            "replicas, leaving %d visible devices unused.",
                            visible_device_count,
                            replica_world_size,
                            local_dp_capacity,
                            visible_device_count % replica_world_size,
                        )
                elif replica_world_size == visible_device_count:
                    # A node hosts exactly one complete TP/PP/PCP replica.
                    local_dp_capacity = 1
                    logger.warning_once(
                        "XPU external LB sees exactly enough devices for one "
                        "TP/PP/PCP replica. This may be the intended "
                        "configuration, but it may also indicate that device "
                        "visibility is misconfigured. Every DP rank must see "
                        "all XPU devices on its node; consider removing "
                        "ZE_AFFINITY_MASK or setting it to expose the complete "
                        "device set."
                    )
                else:
                    # The topology says single-node, but the replica does not fit.
                    raise ValueError(
                        f"TP/PP/PCP replica size ({replica_world_size}) exceeds "
                        f"the number of visible XPU devices ({visible_device_count}), "
                        "but nnodes_within_dp is 1. Configure the multi-node "
                        "topology, or ensure every DP rank can see all devices "
                        "on its node."
                    )
                # Strip the node component off the global DP index to get this
                # engine's slot on its own node. Assumes the launcher assigns
                # DP ranks to nodes in contiguous blocks (node 0 gets ranks
                # 0..capacity-1, and so on), which is what the usual sequential
                # and one-pod-per-rank deployments do. A round-robin or
                # unbalanced assignment would silently map two engines onto the
                # same device.
                dp_local_rank = parallel_config.data_parallel_index % local_dp_capacity

            self.local_rank += dp_local_rank * replica_world_size

            assert self.local_rank < visible_device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds. "
            )
            assert parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # oneCCL warm-up; only meaningful for multi-device runs. Requiring it
        # with a single worker breaks platforms where oneCCL cannot enumerate
        # device topology (e.g. paravirtualized GPUs).
        if (
            self.parallel_config.world_size > 1
            and torch.distributed.is_xccl_available()
        ):
            torch.distributed.all_reduce(torch.zeros(1).xpu())

        if self.use_v2_model_runner:
            logger.info_once("Using V2 Model Runner")

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        model_runner = XPUModelRunnerV2 if self.use_v2_model_runner else XPUModelRunner
        self.model_runner = model_runner(  # type: ignore
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start and self.profiler is None:
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            trace_name = (
                f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
            )

            self.profiler = TorchProfilerWrapper(
                self.profiler_config,
                worker_name=trace_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )
            logger.debug("Starting torch profiler with trace name: %s", trace_name)

        super().profile(is_start=is_start, profile_prefix=profile_prefix)

    def shutdown(self) -> None:
        logger.info(
            "XPUWorker shutdown: cleaning up (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )
        super().shutdown()
        from vllm.device_allocator.xpumem import XpuMemAllocator

        if XpuMemAllocator.instance is not None:
            XpuMemAllocator.instance.release_pools()
        logger.info(
            "XPUWorker shutdown: done (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )
