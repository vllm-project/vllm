# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import regex as re
import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp, Store
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.fault_tolerance import FaultToleranceConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.config.fault_tolerance import FaultToleranceConfig
    from vllm.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)
_NUMACTL_CPUSET_PATTERN = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")

# [CN] 本文件定义 vLLM 分布式执行的全部并行配置。
# [CN] 三条链路交汇处：EngineArgs 收集 -> ParallelConfig 校验 ->
# [CN] __post_init__ 推导 -> 运行时按 world_size 拉起 worker。
# [CN] 核心概念区分：
# [CN]   TP/PP/PCP 决定新增进程数，MoE 层按 TP x PCP x DP 之积切片；
# [CN]   DCP 只切 KV cache 而不新增进程，因此可以复用已有 rank。

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
EPLBPolicyOption = Literal["default"]
DCPCommBackend = Literal["ag_rs", "a2a"]
EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_v2",
    "mori_high_throughput",
    "mori_low_latency",
    "nixl_ep",
    "allgather_reducescatter",
    "flashinfer_all2allv",  # temporary alias for flashinfer_nvlink_two_sided
    "flashinfer_nvlink_two_sided",
    "flashinfer_nvlink_one_sided",
]


# [CN] 专家并行负载均衡（EPLB）配置：周期性统计专家负载并重排专家到 rank 的映射。
@config
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP)."""

    # [CN] 负载统计滑动窗口：只保留最近这么多步的专家命中计数。

    window_size: int = Field(default=1000, gt=0)
    """Window size for expert load recording."""
    # [CN] 重排周期间隔。大于 window_size 时不报错，只是实际只用到最近
    # [CN] window_size 步的统计数据来决定新排布。

    step_interval: int = Field(default=3000, gt=0)
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    # [CN] 冗余专家数：多余的槽位用来复制热点专家，是热负载均衡的主要手段。

    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism."""

    # [CN] 是否每步记录均衡度指标。默认关：统计本身需要一次跨 rank 通信。

    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """
    log_balancedness_interval: int = Field(default=1, gt=0)
    """
    Interval for logging the balancedness.
    """
    # [CN] 非阻塞 EPLB：权重搬运与正常前向重叠，代价是只能用 default 策略，
    # [CN] 且不能用 NCCL 类通信器（多流冲突）。

    use_async: bool = True
    """
    Whether to use non-blocking EPLB.
    """

    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB)."""

    # [CN] 权重搬运通道。None 走自动选择：优先 nixl，退回 gloo。

    communicator: EPLBCommunicatorBackend | None = None
    """
    Backend for EPLB expert weight communication:
    - "torch_nccl": Use torch.distributed on the device process group
    - "torch_gloo": Use torch.distributed gloo with CPU staging
    - "nixl": Use NIXL with staged send/recv buffers
    - "pynccl": Use PyNccl send/recv
    - None: Auto-select backend (prefers "nixl", falls back to "torch_gloo")
    """

    # [CN] 异步模式的两条硬约束：策略必须 default、通信器不能是 NCCL 系。
    # [CN] 这里拒绝而非静默降级，因为 NCCL 多流冲突会直接 hang。
    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        if self.use_async and self.communicator in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"{self.communicator} communicator is incompatible with "
                "async EPLB due to NCCL multi-stream conflicts. Use "
                "'torch_gloo' or 'nixl' instead, or leave communicator "
                "unset for automatic selection."
            )
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


# [CN] 分布式执行的唯一权威配置。几乎所有字段都参与 compute_hash()。
@config
class ParallelConfig:
    """Configuration for the distributed execution."""

    # [CN] PP：按层切开，stage 之间只传 hidden states，通信量最小但引入气泡。

    pipeline_parallel_size: int = Field(default=1, ge=1)
    """Number of pipeline parallel groups."""
    # [CN] TP：按权重矩阵切开，每层要两次 all-reduce，通信频繁但无气泡。

    tensor_parallel_size: int = Field(default=1, ge=1)
    """Number of tensor parallel groups."""
    # [CN] PCP：把 prefill 的序列维度切开。注意它扩大 world_size（要新进程），
    # [CN] 但 KV cache 分片数不随之增加 —— 与 DCP 的关键区别。

    prefill_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that split prefill sequence computation. PCP expands
    the process world size but does not increase the KV-cache shard count."""
    # [CN] DP：完整副本各自跑不同请求。MoE 的专家按 TP x PCP x DP 展开，
    # [CN] 所以增大 DP 等价于给专家池更多承载槽位。

    data_parallel_size: int = Field(default=1, ge=1)
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor, prefill-context, and data parallel sizes."""
    # [CN] 单节点内的 DP 组数。0 是哨兵：表示 DP 由外部指定，
    # [CN] __post_init__ 见到它就跳过自动推导，转去读环境变量。

    data_parallel_size_local: int = Field(default=1, ge=0)
    """Number of local data parallel groups. A value of 0 is a sentinel used by
    the engine-args layer to signal that data parallelism was specified
    externally (see `ParallelConfig.__post_init__`)."""
    # [CN] 本进程在 DP 组内的全局编号，参与 torch process group 的建立。

    data_parallel_rank: int = Field(default=0, ge=0)
    """Rank of the data parallel group. The runtime check at
    ``__post_init__`` further bounds this by ``data_parallel_size``."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group, set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    # [CN] 固定 RPC 端口且所有节点共用，不像 master_port 那样按进程递增。

    data_parallel_rpc_port: int = Field(default=29550, ge=1, le=65535)
    """Fixed port for data parallel messaging, shared by all nodes."""
    # [CN] 每 N 步做一次 DP 收尾同步。所有 DP rank 必须取相同值，
    # [CN] 否则 collective 次数不一致会 hang（compute_hash 因此把它算进去）。

    dp_sync_interval: int = Field(default=16, ge=1)
    """Steps between DP finish-sync all-reduces; must match across DP ranks."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    # [CN] 外部 LB 模式（k8s one-pod-per-rank）。每 rank 是独立服务实例，
    # [CN] 由外层负载均衡转发，vLLM 内部不做调度。

    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Supported only for MoE deployments; non-MoE
    models should use independent vLLM instances without --data-parallel-*
    arguments. Set implicitly when --data-parallel-rank is provided explicitly
    to vllm serve."""
    # [CN] 混合 LB：节点内部本地 DP rank 之间由 vLLM 均衡，
    # [CN] 节点之间交给外部 LB。要配合 --data-parallel-start-rank 使用。

    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank."""
    # [CN] 三态字段：None=未知（跳过相关校验），True/False 用于拒绝无意义配置。

    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    # [CN] MoE 层改用 EP（专家分散到各 rank + all2all 派发）而不是 TP。
    # [CN] EP 下每张卡只持有部分专家，注意力部分仍是 TP。

    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    # [CN] 采样按 rank 切批：每 rank 只采自己那一段，省掉重复的 logits 计算。
    # [CN] 需要模型实现 compute_logits_local，否则退化为全量采样。

    enable_batch_sharded_sampling: bool | None = None
    """Use sharded sampling across tensor parallel ranks. Each rank samples
    a slice of the batch instead of every rank sampling all of it. Currently
    defaults to False if not set. Enabling it explicitly raises when the config
    cannot support it (`tensor_parallel_size` must be > 1, `max_num_seqs` at
    least `tensor_parallel_size`, and `max_logprobs` non-negative). Models opt in
    by implementing `compute_logits_local`."""
    # [CN] 加载权重时按 rank 过滤：每张卡只读自己那几个专家的分片。
    # [CN] 对按专家分文件的 checkpoint（DeepSeek/Mixtral）省 I/O 极为显著。
    # [CN] 对 3D 融合的专家权重（GPT-OSS）无效 —— 张量本身就是一整块。

    enable_ep_weight_filter: bool = False
    """Skip non-local expert weights during model loading when expert
    parallelism is active.  Each rank only reads its own expert shard from
    disk, which can drastically reduce storage I/O for MoE models with
    per-expert weight tensors (e.g. DeepSeek, Mixtral, Kimi-K2.5).  Has no
    effect on 3D fused-expert checkpoints (e.g. GPT-OSS) or non-MoE
    models."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """Expert parallelism configuration."""
    # [CN] linear 让相邻专家在同一 rank；round_robin 交错分布。
    # [CN] 分组专家模型（如共享中间层）在无冗余时更适合交错，
    # [CN] 因为同组专家不会全挤在同一张卡上。

    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers:

    - "linear": Experts are placed in a contiguous manner. For example, with 4
      experts and 2 ranks, rank 0 will have experts [0, 1] and rank 1 will have
      experts [2, 3].
    - "round_robin": Experts are placed in a round-robin manner. For example,
      with 4 experts and 2 ranks, rank 0 will have experts [0, 2] and rank 1
      will have experts [1, 3]. This strategy can help improve load balancing
      for grouped expert models with no redundant experts."""
    # [CN] EP 的 token 派发/回收实现。不同后端对 batch、延迟、
    # [CN] 是否需要 NVLink 的假设差别很大，直接影响 MoE 吞吐。

    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:

    - "allgather_reducescatter": All2all based on allgather and reducescatter
    - "deepep_high_throughput": Use deepep high-throughput kernels
    - "deepep_low_latency": Use deepep low-latency kernels
    - "mori_high_throughput": MoRI EP with InterNodeV1 for multi-node
    - "mori_low_latency": MoRI EP with InterNodeV1LL for multi-node
    - "nixl_ep": Use nixl-ep kernels
    - "flashinfer_nvlink_one_sided": Use flashinfer high-throughput a2a kernels
    - "flashinfer_nvlink_two_sided": Use flashinfer two-sided kernels for mnnvl"""

    # [CN] 分批加载时的并发 worker 数上限，防 TP 大模型时主机内存 OOM。
    # [CN] 当前实现已不支持，__post_init__ 会对非空值告警并忽略。

    max_parallel_loading_workers: int | None = Field(default=None, ge=1)
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    # [CN] 关掉自研 all-reduce kernel 退回 NCCL。平台不支持时会被动置 True。

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    # [CN] 弹性 EP：用无状态 NCCL 组支持运行期扩缩容 DP/EP  rank。

    enable_elastic_ep: bool = False
    """Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""

    # [CN] 双批重叠（DBO）：把一个 batch 拆成两个微批流水跑，
    # [CN] 让通信与计算互相掩盖，等价于 ubatch_size=2 的动态版本。

    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    # [CN] 显式指定微批数。0 或 1 表示不拆分。

    ubatch_size: int = Field(default=0, ge=0)
    """Number of ubatch size."""

    # [CN] 纯 decode 批的拆批阈值：token 数超过才微批化。
    # [CN] 小 decode 批拆开反而增加 kernel 启动开销，不如单批快。

    dbo_decode_token_threshold: int = Field(default=32, ge=0)
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch."""
    # [CN] 含 prefill 的批用更大阈值（默认 512）：
    # [CN] prefill 已经是计算密集，要足够大才值得拆。

    dbo_prefill_token_threshold: int = Field(default=512, ge=0)  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch."""

    # [CN] DP 同步改用 Gloo。异步调度下 NCCL 的同步语义会打断流水线，
    # [CN] 故此时默认 True；注意用可调式 None 而非硬编码 False。

    disable_nccl_for_dp_synchronization: bool | None = None
    """Forces the dp synchronization logic in vllm/v1/worker/dp_utils.py 
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    """

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""

    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """
    Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    Note:
        [TPU](https://docs.vllm.ai/projects/tpu/en/latest/) platform only supports Ray
        for distributed inference.
    """

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
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    master_port: int = 29501
    """distributed master port for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    node_rank: int = Field(default=0, ge=0)
    """distributed node rank for multi-node distributed
    inference when distributed_executor_backend is mp."""
    nnodes: int = Field(default=1, ge=1)
    """num of nodes for multi-node distributed
    inference when distributed_executor_backend is mp."""
    # [CN] NUMA 绑定：worker 子进程绑到 GPU 所属 NUMA 节点的 CPU 与内存。
    # [CN] 跨 socket 访问显存/内存的延迟差异在多卡上非常可观。

    numa_bind: bool = False
    """Enable NUMA binding for GPU worker subprocesses.

    By default, workers are pinned to their GPU's NUMA-local CPUs and
    memory; on PCT-capable Xeons they also auto-bind to the SKU's
    PCT priority cores.
    """
    # [CN] 每张可见卡绑哪个 NUMA 节点，长度需等于可见 GPU 数。
    # [CN] 不设且 numa_bind=True 时自动探测 GPU-NUMA 拓扑。

    numa_bind_nodes: list[int] | None = None
    """NUMA node to bind each GPU worker to.

    Specify one NUMA node per visible GPU, for example `[0, 0, 1, 1]`
    for a 4-GPU system with GPUs 0-1 on NUMA node 0 and GPUs 2-3 on
    NUMA node 1. If unset and `numa_bind=True`, vLLM auto-detects the
    GPU-to-NUMA topology. The values are passed to `numactl --membind`
    and `--cpunodebind`, so they must be valid `numactl` NUMA node indices.
    """
    # [CN] 每张卡绑的 CPU 列表，语法同 numactl --physcpubind。
    # [CN] 设了它就用 --physcpubind 覆盖 --cpunodebind，
    # [CN] 便于精确绑到 PCT 等高主频核上。

    numa_bind_cpus: list[str] | None = None
    """Optional CPU lists to bind each GPU worker to.

    Specify one CPU list per visible GPU, for example
    `["0-3", "4-7", "8-11", "12-15"]`. When set, vLLM uses
    `numactl --physcpubind` instead of `--cpunodebind`. This is useful
    for custom policies such as binding to PCT or other high-frequency cores.
    Each entry must use `numactl --physcpubind` CPU-list syntax, for example
    `"0-3"` or `"0,2,4-7"`.
    """
    # [CN] vLLM 逻辑 GPU id -> 物理 GPU id 的映射（如 [2,3] 表示逻辑 0 是物理 2）。
    # [CN] 只在平台边界处生效：NVML 查询、网卡亲和性、P2P 检查与最终设备选择。

    assigned_physical_gpu_ids: list[int] | None = None
    """Mapping from vLLM-local logical GPU IDs to physical GPU IDs.

    For example, ``[2, 3]`` means logical GPU 0 maps to physical GPU 2,
    and logical GPU 1 maps to physical GPU 3. Physical IDs are used only
    at platform/topology boundaries such as NVML, NIC affinity, P2P
    checks, and final CUDA device selection when needed. When None,
    logical IDs map to visible device IDs in order."""

    # [CN] init_process_group 的超时。多节点下载模型很慢时要把这个值调大，
    # [CN] 否则会出现看起来像是"某节点掉线"的超时错误。

    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group).
    If set, this value is passed to torch.distributed.init_process_group as the
    timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL).
    Increase this for multi-node setups where model downloads may be slow."""

    cpu_distributed_timeout_seconds: int | None = None
    """Timeout (in seconds) for cpu communication groups. If None, PyTorch's
    default timeout is used (1800s for gloo)."""

    # [CN] init=False 的派生量：PP x TP x PCP，决定要起多少个 worker 进程。
    # [CN] 注意不含 DP —— DP 副本由另外的进程树承载。

    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    # [CN] 私有字段：预先申请的空闲端口池，避免多处建组时抢同一端口。

    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    # [CN] 协调用 TCPStore 端口：worker 作为 client 连上来交换自选端口号，
    # [CN] 这样就不必事先把所有端口算好。

    _coord_store_port: int = 0
    """Port of the coordination TCPStore. Can be set by the API server; workers
    connect as clients to exchange self-picked group ports at runtime."""

    # [CN] DCP：decode 阶段把 KV cache 切到多个 rank 上，但不增加进程数。
    # [CN] 无 PCP 时直接复用 TP 的 rank；有 PCP 时可横跨 PCP 轴或整个 TPxPCP。

    decode_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that shard the decode KV cache. DCP does not expand
    the process world size. Without PCP, DCP reuses TP ranks. With PCP, DCP
    either spans the PCP axis or the full TP x PCP block."""

    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.

    """
    # [CN] None 表示交给模型默认值，模型通过 set_dcp_defaults 覆盖。

    dcp_comm_backend: DCPCommBackend | None = None
    """Communication backend for Decode Context Parallel (DCP).
    - "ag_rs": AllGather + ReduceScatter (existing behavior)
    - "a2a": All-to-All exchange of partial outputs + LSE, then
      combine with Triton kernel. Reduces NCCL calls from 3 to 2
      per layer for MLA models.

    `None` selects the model default, which is "ag_rs" unless the model
    overrides it via [`set_dcp_defaults`][vllm.config.ParallelConfig.set_dcp_defaults].
    """

    # [CN] MLA 场景下在各 DCP rank 上复制 query 投影：
    # [CN] 用小矩阵的冗余计算换掉每步都要做的 query all-gather。

    dcp_q_replicate: bool | None = None
    """Replicate the MLA query projection within each DCP group so decode can skip the
    query all-gather.

    With DCP the KV cache is sharded across the group, so the standard MLA decode path
    all-gathers the query every step. Replicating the (small) query projection at load
    time lets each rank materialize the full group-local head set and skip that 
    collective, at the cost of computing the projection redundantly on every rank 
    in the group.
    """

    # [CN] KV cache 在各 CP rank 上的交错粒度。1 = token 级（token i 在
    # [CN] rank i % dcp_world_size）；等于 block_size = block 级。
    # [CN] block_size 必须是它的整数倍，否则 block 无法对齐切分。

    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP.
    Store interleave_size tokens on dcp_rank i, then store next
    interleave_size tokens on dcp_rank i+1.
    Interleave_size=1: token-level alignment, where token `i` is stored on
        dcp_rank `i % dcp_world_size`.
    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks. Tokens are then stored
        in (rank i+1, block j) only after (rank i, block j) is fully occupied.
    Block_size should be greater than or equal to cp_kv_cache_interleave_size.
    Block_size should be divisible by cp_kv_cache_interleave_size.
    """

    # [CN] 与 data_parallel_rank 初值相同，但 Dense 模型不会被覆盖。
    # [CN] 这是"逻辑编号"与"进程组编号"分离的地方。

    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""

    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    enable_fault_tolerance: bool = False
    """Enable fault tolerance for detailed error recovery,
    such as scaling down fault DPEngineCore.
    """

    fault_tolerance_config: FaultToleranceConfig = Field(
        default_factory=FaultToleranceConfig
    )
    """The configurations for fault tolerance."""

    # [CN] mode="wrap" 的校验器：值为 None 时直接返回，不交给下游校验。
    # [CN] 这样这个字段才能保持"未设置"的三态语义直到 __post_init__ 推导。
    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    # [CN] 允许 None（未设置），但一旦给了列表就必须是非空非负 —— 空列表
    # [CN] 会让下游按索引取 NUMA 节点时越界。
    @field_validator("numa_bind_nodes")
    @classmethod
    def _validate_numa_bind_nodes(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_nodes must not be empty.")
        if any(node < 0 for node in value):
            raise ValueError("numa_bind_nodes must contain non-negative integers.")
        return value

    # [CN] 逐项校验 numactl CPU 列表语法，并额外检查区间必须升序。
    # [CN] "3-1" 能被宽泛的正则放过，但传给 numactl 会失败。
    @field_validator("numa_bind_cpus")
    @classmethod
    def _validate_numa_bind_cpus(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_cpus must not be empty.")

        for cpuset in value:
            if not cpuset:
                raise ValueError("numa_bind_cpus entries must not be empty.")
            if not _NUMACTL_CPUSET_PATTERN.fullmatch(cpuset):
                raise ValueError(
                    "numa_bind_cpus entries must use numactl CPU list syntax, "
                    "for example '0-3' or '0,2,4-7'."
                )
            for part in cpuset.split(","):
                if "-" not in part:
                    continue
                start_str, end_str = part.split("-", 1)
                if int(start_str) > int(end_str):
                    raise ValueError(
                        f"numa_bind_cpus ranges must be ascending, but got '{cpuset}'."
                    )
        return value

    # [CN] 结构性约束集中在这里：不合理的组合直接 ValueError，
    # [CN] 而不是等到运行时出现 collective 次数不匹配导致 hang。
    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        # [CN] -1 是引擎核心进程的保留值，其余必须落在 [0, count) 内。

        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        # [CN] 容错机制假设只有一个 AsyncMPClient 统管所有引擎，
        # [CN] 多 API 进程下故障降级的路径无法收敛。

        if self.enable_fault_tolerance and self._api_process_count > 1:
            raise ValueError(
                "Fault tolerance requires a single API server process "
                f"(--api-server-count=1), but got {self._api_process_count}. "
                "The FT system assumes one AsyncMPClient manages all engines."
            )

        # [CN] 已下线的后端不是报错而是静默回退到默认，避免老脚本直接挂掉。

        if self.all2all_backend in ["pplx", "naive"]:
            logger.warning(
                "The '%s' all2all backend has been removed. "
                "Falling back to 'allgather_reducescatter'.",
                self.all2all_backend,
            )
            self.all2all_backend = "allgather_reducescatter"

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        # [CN] 外部 LB 只在真正有多副本时才有意义，单 rank 下开了会造成请求无人处理。

        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if not self.numa_bind and (
            self.numa_bind_nodes is not None or self.numa_bind_cpus is not None
        ):
            raise ValueError(
                "numa_bind_nodes and numa_bind_cpus require numa_bind=True."
            )

        # [CN] EPLB 的前提是专家分布在多个 rank 上：TP/PCP/DP 至少要有一个 > 1。

        if self.enable_eplb:
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            # The EP group spans the TP x PCP x DP ranks. EPLB therefore needs
            # TP, PCP, or DP > 1.
            if (
                self.tensor_parallel_size
                * self.prefill_context_parallel_size
                * self.data_parallel_size
                <= 1
            ):
                raise ValueError(
                    "EPLB requires tensor, prefill-context, or data parallelism, "
                    f"but got TP={self.tensor_parallel_size}, "
                    f"PCP={self.prefill_context_parallel_size}, "
                    f"DP={self.data_parallel_size}."
                )
        # [CN] 未开 EPLB 却设了冗余专家是典型笔误 —— 冗余槽位无处安放。

        else:
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        # [CN] TP/PCP/DCP 三者的整除约束：DCP 要能整除它依附的那组 rank。

        tp = self.tensor_parallel_size
        pcp = self.prefill_context_parallel_size
        dcp = self.decode_context_parallel_size
        # [CN] PCP 与 DP 组合会让序列维与数据维的通信域交叉，暂未支持。

        if pcp > 1 and self.data_parallel_size > 1:
            raise ValueError("PCP does not support data parallelism yet.")
        if pcp == 1:
            # DCP reuses the TP ranks when PCP is disabled.
            if tp % dcp != 0:
                raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")
        # [CN] 开 PCP 后 DCP 只能取 1 / 跨 PCP / 跨整个 TPxPCP 三种形态，
        # [CN] 中间值无法构造出规整的通信域。

        elif dcp not in (1, pcp, tp * pcp):
            raise ValueError(
                "When PCP is enabled, DCP must be disabled, span the PCP "
                "axis, or span the full TP x PCP axis. "
                f"Got TP={tp}, PCP={pcp}, DCP={dcp}; valid DCP sizes are "
                f"{sorted({1, pcp, tp * pcp})}."
            )

        return self

    # [CN] 供模型在 verify_and_update_config 里声明偏好。只填 None 的字段，
    # [CN] 显式用户配置优先 —— 这是"模型建议 vs 用户指定"的边界。

    def set_dcp_defaults(
        self,
        comm_backend: DCPCommBackend = "ag_rs",
        q_replicate: bool = False,
    ) -> None:
        """Fill in the DCP options the user left unset.

        Models can set their preferred DCP settings by calling this from their
        `verify_and_update_config` hook.
        """
        if self.dcp_comm_backend is None:
            self.dcp_comm_backend = comm_backend
        if self.dcp_q_replicate is None:
            self.dcp_q_replicate = q_replicate

    # [CN] 含 DP 副本的总进程数，决定是否真的需要分布式执行器。
    @property
    def world_size_across_dp(self) -> int:
        """Process world size across TP, PCP, PP, and DP."""
        return self.world_size * self.data_parallel_size

    # [CN] DBO 打开或显式 ubatch_size>1 都算启用微批。
    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    # [CN] DBO 固定是两个微批（双重叠），其余情况按配置值。
    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else self.ubatch_size

    # [CN] 决定客户端只管本地 EngineCore 还是连同远端一起管：
    # [CN] 外部/混合 LB 下每个 vLLM 实例只负责本机副本。
    @property
    def local_engines_only(self) -> bool:
        """
        Client manages local+remote EngineCores in pure internal LB case.
        Client manages local EngineCores in hybrid and external LB case.
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    # [CN] worker 进程与 engine 进程可能各自要建 DP 组，端口不能复用。
    # [CN] 有预申请池就弹一个，否则从基准端口递增。

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
        # [CN] 离线 SPMD：没有 EngineArgs 层，全部从环境变量回读。

        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    # [CN] 有协调 store 时由 rank0 绑端口并发布，其余 rank 去读；
    # [CN] 没有协调者时退回预分配端口池。

    def _pick_stateless_dp_port(self) -> tuple[int, socket.socket | None]:
        """Return ``(port, listen_socket)`` for DP group init.

        With a coord store, rank 0 binds a socket and publishes the port;
        others read it.  Without one, pops a pre-allocated port and
        returns ``listen_socket=None``.
        """
        if not self._coord_store_port:
            return self.get_next_dp_init_port(), None

        from vllm.distributed.utils import get_cached_tcp_store_client

        store = get_cached_tcp_store_client(
            self.data_parallel_master_ip, self._coord_store_port
        )

        key = "dp_master_port"
        if self.data_parallel_rank == 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.data_parallel_master_ip, 0))
            s.listen()
            port = s.getsockname()[1]
            store.set(key, str(port).encode())
            return port, s
        else:
            return int(store.get(key).decode()), None

    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[False] = ...
    ) -> ProcessGroup: ...
    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[True] = ...
    ) -> tuple[ProcessGroup, Store]: ...
    # [CN] 无状态地建一个临时的 DP 进程组（用完即弃，不占用全局默认组）。
    # [CN] 高并发下多个进程可能同时选到同一个"当前空闲"端口，
    # [CN] 首个进程 bind 成功后其余会 EADDRINUSE，因此这里带重试。

    def stateless_init_dp_group(
        self, return_store: bool = False
    ) -> ProcessGroup | tuple[ProcessGroup, Store]:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                port, listen_socket = self._pick_stateless_dp_port()
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    port,
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                    listen_socket=listen_socket,
                )
            # [CN] 只针对 EADDRINUSE 重试：其他网络错误重试也不会好转。

            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism, replicated tokens results in useless
    # duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #

    # [CN] 上面的注释是理解这个属性的关键：注意力末尾（o_proj）的 all-reduce
    # [CN] 让每个 TP rank 都持有全份 token。EP 下这份复制会导致专家层收到
    # [CN] 重复 token，白白多做一遍计算和通信，因此改成序列并行输入。
    @property
    def use_sequence_parallel_moe(self) -> bool:
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "deepep_high_throughput",
                "deepep_low_latency",
                "flashinfer_nvlink_one_sided",
                "mori_high_throughput",
                "mori_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    # [CN] 需要 all2all 的三种情形：DP、序列并行 MoE、EP 下的 PCP。
    @property
    def use_all2all(self) -> bool:
        return (
            self.data_parallel_size > 1
            or self.use_sequence_parallel_moe
            or (self.enable_expert_parallel and self.prefill_context_parallel_size > 1)
        )

    # [CN] 只有低延迟类后端支持把多个请求的 all2all 合并成一批发送。
    @property
    def use_batched_dp_moe(self) -> bool:
        return (
            self.all2all_backend
            in (
                "deepep_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.data_parallel_size > 1
        )

    @property
    def node_rank_within_dp(self) -> int:
        return self.node_rank % self.nnodes_within_dp

    # [CN] 一个 DP 组跨越的节点数：总节点数除以 DP 组数。
    # [CN] 它是 local_world_size 与 NUMA 拓扑推导的分母。
    @property
    def nnodes_within_dp(self) -> int:
        if self.nnodes == 1:
            return 1
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    @property
    def local_world_size(self) -> int:
        return self.world_size // self.nnodes_within_dp

    # [CN] 用 MAX 归约实现逻辑或：只要任一 rank 还有活要干，整体就不能停。
    # [CN] 必须所有人都同意才能停机，否则会有人提前退出导致 collective 失败。
    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    # [CN] 把"是否还有未完成工作"和"是否有暂停请求"塞进同一个 2 元素张量，
    # [CN] 一次 SUM 归约同时拿到两个结论 —— DP 同步是每步都走的开销。
    # [CN] pause 要的是"全体一致"（SUM==size），unfinished 要的是"任一个"（SUM>0）。
    @staticmethod
    def sync_dp_state(
        dp_group: ProcessGroup, has_unfinished: bool, pending_pause: bool
    ) -> tuple[bool, bool]:
        """Combined all-reduce for DP state synchronization.

        Uses a single SUM all-reduce on a 2-element tensor:
          [0] = 1 if this rank has unfinished work, else 0.
                SUM > 0 ≡ logical OR across ranks → any rank has work.
          [1] = 1 if this rank has a pending pause request, else 0.
                SUM == dp_size ≡ all ranks reached pause consensus.

        has_unfinished_global is true if any rank has unfinished work,
        or if some ranks are waiting for a pause consensus.

        Returns:
            (has_unfinished_global, pause_consensus)
        """
        tensor = torch.tensor(
            [int(has_unfinished), int(pending_pause)], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=dp_group)
        dp_size = dp_group.size()
        pause_count = tensor[1].item()
        # [CN] 取模而非判等：部分 rank 已投票时结果既不是 0 也不是 size，
        # [CN] 此时不能算达成共识，必须继续跑以便让投票收敛。

        has_unfinished_global = tensor[0].item() > 0 or pause_count % dp_size != 0
        return has_unfinished_global, pause_count == dp_size

    # [CN] 各 DP 副本可用显存不同，取 MIN 保证所有人都能放下同样的 KV cache。
    # [CN] -1 先换成 int64 最大值（等价于"我不限制"），否则 MIN 会恒为 -1。
    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    # [CN] 只哈希影响计算图结构的字段。这份哈希用于校验各 DP rank 配置一致：
    # [CN] 一旦不一致，collective 调用序列会错开，表现为整个集群 hang 而非报错。

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        # [CN] 被排除的都是"怎么起进程"层面的信息（拓扑、网络、端口、Ray/NUMA 细节），
        # [CN] 它们不改变张量计算图，因此不会造成 collective 模式差异。

        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "_coord_store_port",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
            # NUMA binding is per-rank host-side memory locality; it does
            # not affect collective-communication semantics. When numa_bind
            # is enabled with auto-detection, each DP rank stores its own
            # NUMA node in numa_bind_nodes (see vllm/utils/numa_utils.py
            # `_get_numa_node`), which would otherwise diverge the DP hash.
            "numa_bind",
            "numa_bind_nodes",
            "numa_bind_cpus",
            "assigned_physical_gpu_ids",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    # [CN] 推导阶段：把用户给的粗粒度参数算成运行时真正要用的量。
    # [CN] 顺序很重要 —— world_size 必须先算出来，后面的后端选择依赖它。

    def __post_init__(self) -> None:
        # Continue with the rest of the initialization
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        # [CN] 外部启动器由外层（如 torchrun）负责起进程，这里只记录信息。
        # [CN] 注意这次判断的 world_size 会把 DP 也算进去。

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        # [CN] 弹性 EP 依赖单一控制面协调扩缩，因此禁掉 PP 和外部/混合 LB。

        if self.enable_elastic_ep:
            if not self.enable_eplb:
                raise ValueError("Elastic EP is only supported with enable_eplb=True.")
            if self.pipeline_parallel_size > 1:
                raise ValueError(
                    "Elastic EP is not supported with pipeline parallelism "
                    f"(pipeline_parallel_size={self.pipeline_parallel_size})."
                )
            if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
                raise NotImplementedError(
                    "Elastic EP is not compatible with data_parallel_external_lb "
                    "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                    "server and core client to coordinate scale up/down."
                )
            if self.eplb_config.use_async:
                from vllm.distributed.nixl_utils import is_nixl_available

                if not is_nixl_available():
                    raise ValueError(
                        "Elastic EP with async EPLB requires the NIXL "
                        "package. Either install NIXL or set "
                        "--eplb-config.use_async=false."
                    )

        # [CN] 两条进入 DP 的入口：显式给了 DP size，或 local=0 这个"外部指定"哨兵。

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            # [CN] 弹性 EP 用无状态组，端口走协调 store 协商，不需要预分配。

            if not self.enable_elastic_ep:
                if not self._data_parallel_master_port_list:
                    self._data_parallel_master_port_list = get_open_ports_list(5)
                self.data_parallel_master_port = (
                    self._data_parallel_master_port_list.pop()
                )

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            # [CN] 稠密模型开离线 DP 只会让每份都跑全量却各自算一半请求，毫无收益。
            # [CN] 这里特意判 is False 而不是 not —— True/None 都不拦。

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        # [CN] 后端自动选择的优先级：TPU+SPMD -> uni；多节点 CUDA -> mp；
        # [CN] 单节点放不下 -> 直接报错而不是偷偷转 ray；最后才看 ray 是否可用。

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and current_platform.device_count() < self.world_size
            ):
                gpu_count = current_platform.device_count()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
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
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        # [CN] 曾经用于限制加载并发，当前实现已不支持，保留告警避免静默失效。

        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        # [CN] 自动选择通信器：优先 nixl（零拷贝 RDMA，兼容异步与弹性）；
        # [CN] 弹性 EP 用 pynccl（无状态组需要）；其余退回 gloo。
        # [CN] torch_nccl 被刻意跳过：异步 EPLB 下多流冲突、高负载下批量收发会 hang。

        if self.enable_eplb and self.eplb_config.communicator is None:
            # Prefer NIXL when available: zero-copy RDMA reads, compatible
            # with both async EPLB and elastic EP.
            # Fallbacks: pynccl for elastic EP (stateless groups need it),
            # torch_gloo for static EP.  torch_nccl is avoided because NCCL
            # is incompatible with async EPLB (multi-stream conflicts) and
            # batched isend/irecv hangs under high load.
            # See https://github.com/pytorch/pytorch/issues/174288
            from vllm.distributed.nixl_utils import is_nixl_available

            if is_nixl_available():
                self.eplb_config.communicator = "nixl"
            elif self.enable_elastic_ep:
                self.eplb_config.communicator = "pynccl"
            else:
                self.eplb_config.communicator = "torch_gloo"

    # [CN] backend 既可能是字符串也可能是 Executor 子类，
    # [CN] 后者通过类属性 uses_ray 声明自己是否依赖 ray。
    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    # [CN] 最后一轮跨模块的校验，需要 import 其它模块，故延迟到此为止。
    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from vllm.v1.executor import Executor

        # Enable batch invariance settings if requested
        # [CN] 批不变性要求所有归约走确定性路径，自研 kernel 有 split 差异故禁用。

        if envs.VLLM_BATCH_INVARIANT:
            self.disable_custom_all_reduce = True

        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            from vllm.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        # A batch below one token per microbatch cannot be split, so the
        # thresholds have to keep it out rather than the split having to cope.
        # [CN] 每个微批至少要摊到 1 个 token：阈值低于微批数就意味着分不出批，
        # [CN] 与其让下游在 0 长度张量上崩掉，不如在这里拦住。

        if self.use_ubatching and (
            min(self.dbo_decode_token_threshold, self.dbo_prefill_token_threshold)
            < self.num_ubatches
        ):
            raise ValueError(
                "dbo_decode_token_threshold and dbo_prefill_token_threshold must "
                f"be at least the number of microbatches ({self.num_ubatches})."
            )

        return self

    # [CN] 把一个 DP rank 降级为独立实例：nnodes/node_rank 要在改 DP 字段
    # [CN] 之前算好并保存，因为这两个 property 依赖 data_parallel_size_local。

    def reconfigure_for_independent_dp_rank(self) -> None:
        """Reconfigure for a single independent non-MoE DP rank."""
        # Capture these before changing DP fields.
        nnodes = self.nnodes_within_dp
        node_rank = self.node_rank_within_dp
        self.data_parallel_size = 1
        self.data_parallel_size_local = 1
        self.data_parallel_rank = 0
        self.nnodes = nnodes
        self.node_rank = node_rank
