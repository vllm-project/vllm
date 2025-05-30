# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
from pydantic.dataclasses import dataclass
from torch.distributed import ProcessGroup, ReduceOp

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils import cuda_device_count_stateless, get_open_port

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from vllm.config.tokenizerpool_config import TokenizerPoolConfig
    from vllm.executor.executor_base import ExecutorBase
else:
    TokenizerPoolConfig = type
    PlacementGroup = Any
    ExecutorBase = Any

logger = init_logger(__name__)

DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]


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
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    max_parallel_loading_workers: Optional[int] = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None
    """This parameter is deprecated and will be removed in a future release.
    Please remove it from your configs"""

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    placement_group: Optional[PlacementGroup] = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: Optional[Union[DistributedExecutorBackend,
                                                 type[ExecutorBase]]] = None
    """Backend to use for distributed model
    workers, either "ray" or "mp" (multiprocessing). If the product
    of pipeline_parallel_size and tensor_parallel_size is less than
    or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, this will default
    to "ray" if Ray is installed and fail otherwise. Note that tpu
    and hpu only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decofing.
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
        increment the port number each time we need to initialize a
        new process group related to data parallelism.
        """
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += 1
        return answer

    def stateless_init_dp_group(self) -> ProcessGroup:
        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group)

        # use gloo since the engine process might not have cuda device
        dp_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            self.get_next_dp_init_port(),
            self.data_parallel_rank,
            self.data_parallel_size,
            backend="gloo")

        return dp_group

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
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})")

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            self.data_parallel_master_port = get_open_port()
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        ray_only_devices: list[str] = []
        from vllm.platforms import current_platform
        if (current_platform.device_type in ray_only_devices
                and self.world_size > 1):
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    f"{current_platform.device_type.upper()} backend only "
                    "supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
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
            logger.info("Defaulting to use %s for distributed inference",
                        backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        self._verify_args()

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and self.distributed_executor_backend.uses_ray)

    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase
        from vllm.platforms import current_platform
        if self.distributed_executor_backend not in (
                "ray", "mp", "uni",
                "external_launcher", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher' or"
                " custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.info(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")

        assert isinstance(self.worker_extension_cls, str), (
            "worker_extension_cls must be a string (qualified class name).")
