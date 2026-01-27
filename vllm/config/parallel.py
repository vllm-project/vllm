# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless, get_open_ports_list

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.executor.executor_base import ExecutorBase
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    ExecutorBase = Any

logger = init_logger(__name__)

DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]


@config
@dataclass
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP)."""

    window_size: int = 1000
    """Window size for expert load recording."""
    step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    num_redundant_experts: int = 0
    """Number of redundant experts to use for expert parallelism."""

    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """


@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: Optional[int] = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: str = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    is provided explicitly to vllm serve."""
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank."""
    enable_prefix_aware_routing: bool = False
    """Enable prefix-aware routing for data parallel load balancing.
    When enabled, requests with the same token prefix will be routed
    to the same engine to improve prefix cache hit rates."""
    prefix_routing_length: int = 16
    """Number of tokens to use as the prefix for routing decisions.
    Only used when enable_prefix_aware_routing is True."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    eplb_config: EPLBConfig = field(default_factory=EPLBConfig)
    """Expert parallelism configuration."""
    num_redundant_experts: Optional[int] = None
    """`num_redundant_experts` is deprecated and has been replaced with
    `eplb_config.num_redundant_experts`. This will be removed in v0.12.0.
    Please use `eplb_config.num_redundant_experts` instead."""
    eplb_window_size: Optional[int] = None
    """`eplb_window_size` is deprecated and has been replaced with
    `eplb_config.window_size`. This will be removed in v0.12.0.
    Please use `eplb_config.window_size` instead."""
    eplb_step_interval: Optional[int] = None
    """`eplb_step_interval` is deprecated and has been replaced with
    `eplb_config.step_interval`. This will be removed in v0.12.0.
    Please use `eplb_config.step_interval` instead."""
    eplb_log_balancedness: Optional[bool] = None
    """`eplb_log_balancedness` is deprecated and has been replaced with
    `eplb_config.log_balancedness`. This will be removed in v0.12.0.
    Please use `eplb_config.log_balancedness` instead."""

    max_parallel_loading_workers: Optional[int] = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    ray_runtime_env: Optional[RuntimeEnv] = None
    """Ray runtime environment to pass to distributed workers."""

    placement_group: Optional[PlacementGroup] = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: Optional[Union[str,
                                                 DistributedExecutorBackend,
                                                 type[ExecutorBase]]] = None
    """Backend to use for distributed model
    workers, either "ray" or "mp" (multiprocessing). If the product
    of pipeline_parallel_size and tensor_parallel_size is less than
    or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, this will default
    to "ray" if Ray is installed and fail otherwise. Note that tpu
    only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""

    world_size: int = field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    _data_parallel_master_port_list: list[int] = field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    decode_context_parallel_size: int = 1
    """Number of decode context parallel groups, because the world size does
    not change by dcp, it simply reuse the GPUs of TP group, and tp_size
    needs to be divisible by dcp_size."""

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def stateless_init_dp_group(self) -> ProcessGroup:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group)

        max_retries = 5
        last_exc: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo")
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning(
                        "Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup,
                          has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished],
                              dtype=torch.int32,
                              device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup,
                                  kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory],
                              dtype=torch.int64,
                              device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        factors.append(self.enable_expert_parallel)
        factors.append(self.data_parallel_size)
        factors.append(envs.VLLM_ALL2ALL_BACKEND)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        # Forward deprecated fields to their new location
        if self.num_redundant_experts is not None:
            self.eplb_config.num_redundant_experts = (
                self.num_redundant_experts)
            logger.warning_once(
                "num_redundant_experts is deprecated and has been replaced "
                "with eplb_config.num_redundant_experts. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect.")
        if self.eplb_window_size is not None:
            self.eplb_config.window_size = self.eplb_window_size
            logger.warning_once(
                "eplb_window_size is deprecated and has been replaced "
                "with eplb_config.window_size. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect.")
        if self.eplb_step_interval is not None:
            self.eplb_config.step_interval = self.eplb_step_interval
            logger.warning_once(
                "eplb_step_interval is deprecated and has been replaced "
                "with eplb_config.step_interval. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect.")
        if self.eplb_log_balancedness is not None:
            self.eplb_config.log_balancedness = self.eplb_log_balancedness
            logger.warning_once(
                "eplb_log_balancedness is deprecated and has been replaced "
                "with eplb_config.log_balancedness. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect.")

        # Continue with the rest of the initialization
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})")

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if not self._data_parallel_master_port_list:
                self._data_parallel_master_port_list = get_open_ports_list(5)
            self.data_parallel_master_port = \
                self._data_parallel_master_port_list.pop()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})")
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_external_lb:
                raise ValueError("data_parallel_external_lb can only "
                                 "be set when data_parallel_size > 1")

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.enable_eplb:
            if not current_platform.is_cuda():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices now.")
            if self.eplb_config.num_redundant_experts < 0:
                raise ValueError(
                    "num_redundant_experts must be non-negative, but got "
                    f"{self.eplb_config.num_redundant_experts}.")
            if not self.enable_expert_parallel:
                raise ValueError(
                    "enable_expert_parallel must be True to use EPLB.")
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts should be used with EPLB."
                    f"{self.eplb_config.num_redundant_experts}.")
        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray: "
                                     f"{ray_utils.ray_import_err}. Ray is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.")
                backend = "ray"
            elif self.data_parallel_backend == "ray":
                logger.info("Using ray distributed inference because "
                            "data_parallel_backend is ray")
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference",
                         backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False))

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase
        from vllm.platforms import current_platform
        if self.distributed_executor_backend is not None and not isinstance(
                self.distributed_executor_backend, str) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom ExecutorBase subclass or its import path.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")

        return self
