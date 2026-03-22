# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU 微批次包装器模块。

本模块实现了微批次相关的包装器和元数据类，负责：
- 管理微批次元数据（UbatchMetadata）
- 管理 CUDA Graph 元数据（CUDAGraphMetaData）
- 控制 SM（流式多处理器）分配
- 包装微批次执行逻辑

主要类：
- UbatchMetadata: 微批次元数据
- CUDAGraphMetaData: CUDA Graph 元数据
- SMControlContextManager: SM 控制上下文管理器
- UBatchWrapper: 微批次包装器
"""
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts

logger = init_logger(__name__)


@dataclass
class UbatchMetadata:
    """微批次元数据类。

    存储微批次执行所需的元数据信息。

    Attributes:
        context: 微批次上下文
        input_ids: 输入 ID 张量
        positions: 位置张量
        inputs_embeds: 输入嵌入张量（可选）
        intermediate_tensors: 中间张量（用于流水线并行）
        num_tokens: token 数量
    """
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


@dataclass
class CUDAGraphMetaData:
    """CUDA Graph 元数据类。

    存储 CUDA Graph 执行所需的元数据信息。

    Attributes:
        cudagraph: CUDA Graph
        ubatch_metadata: 微批次元数据
        outputs: 输出（可选）
    """
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


class SMControlContextManager:
    """SM（流式多处理器）控制上下文管理器。

    控制通信和计算的 SM 分配。进入上下文时，将通信和计算的 SM 数量
    分别设置为 comm_sms 和 total_sms - comm_sms。退出时，恢复为使用
    所有可用的 SM。

    Attributes:
        total_sms: 总 SM 数量
        compute_sms: 计算用 SM 数量
        comm_sms: 通信用 SM 数量
        set_comm_sms: 设置通信 SM 数量的函数
        set_compute_sms: 设置计算 SM 数量的函数
    """
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """初始化 SM 控制上下文管理器。

        Args:
            comm_sms: 通信 SM 数量（剩余用于计算）
            set_comm_sms: 设置通信 SM 数量的函数
            set_compute_sms: 设置计算 SM 数量的函数
        """

        assert current_platform.is_cuda(), (
            "SM control is currently only supported on CUDA"
        )
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


class UBatchWrapper:
    """微批次包装器类。

    包装可执行对象以支持微批次执行和 CUDA Graph 捕获。

    Attributes:
        runnable: 可执行对象
        vllm_config: vLLM 配置
        compilation_config: 编译配置
        comm_stream: 通信 CUDA 流
        ready_barrier: 就绪屏障
        cudagraphs: CUDA Graph 元数据字典
        cudagraph_wrapper: CUDA Graph 包装器
        sm_control: SM 控制上下文管理器
        device: 设备
        is_debugging_mode: 是否为调试模式
    """
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        """初始化微批次包装器。

        Args:
            runnable: 可执行对象
            vllm_config: vLLM 配置
            runtime_mode: CUDA Graph 模式
            device: CUDA 设备
        """
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        # Ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode
            )

        self.sm_control = self._create_sm_control_context(vllm_config)
        self.device = device
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

    @property
    def graph_pool(self):
        """获取图形池。

        Returns:
            图形池（如果有）
        """
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        """清除所有图形。"""
        self.cudagraphs.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.clear_graphs()

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        """创建 SM 控制上下文管理器。

        Args:
            vllm_config: vLLM 配置

        Returns:
            SM 控制上下文管理器
        """
        comm_sms: int = envs.VLLM_DBO_COMM_SMS

        set_comm_sms = lambda sms: None
        if vllm_config.parallel_config.enable_expert_parallel:
            # Currently only DeepEP highthroughput supports SM control so this
            # only affects that case.
            ep_group = get_ep_group()
            device_communicator = ep_group.device_communicator
            all2all_manager = None
            if device_communicator is not None:
                all2all_manager = device_communicator.all2all_manager

            if all2all_manager is not None:
                max_sms_used = all2all_manager.max_sms_used()
                if max_sms_used is not None:
                    comm_sms = min(comm_sms, max_sms_used)

            if comm_sms > 0 and all2all_manager is not None:
                set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            import deep_gemm as dg

            set_compute_sms = lambda sms: dg.set_num_sms(sms)

        return SMControlContextManager(
            comm_sms=comm_sms,
            set_comm_sms=set_comm_sms,
            set_compute_sms=set_compute_sms,
        )

    def __getattr__(self, key: str):
        """允许访问可执行对象的属性。

        Args:
            key: 属性名

        Returns:
            属性值
        """
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable:
        """unwrap 返回原始可执行对象。

        Returns:
            原始可执行对象
        """
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """捕获微批次运行的 CUDA Graph。

        逻辑比较复杂，因为需要确保每个微批次线程在开始图形捕获之前
        初始化其 CUDA 上下文。

        流程如下：
        1. 主线程启动每个微批次线程。每个线程在进入 ubatch_context 之前
           初始化其 CUDA 上下文（torch.cuda.current_blas_handle()）。
        2. 主线程开始图形捕获并唤醒第一个微批次线程。
        3. 每个微批次线程运行模型到完成并将输出张量返回给主线程。
        4. 主线程存储捕获的 CUDA Graph 及其元数据并返回。

        Args:
            ubatch_metadata: 微批次元数据
            model: 模型

        Returns:
            输出张量
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.accelerator.set_device_index(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = CUDAGraphMetaData(
                cudagraph=torch.cuda.CUDAGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())

            # Sync offloader's copy stream before capture.
            # Ensure any pre-capture prefetches from offloader are complete.
            get_offloader().sync_prev_onload()

            with torch.cuda.graph(
                cudagraph_metadata.cudagraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = torch.cat(sorted_results, dim=0)
                cudagraph_metadata.outputs = result
                # Join offloader's copy stream after forward to avoid unjoined
                # stream error. The last layer's start_prefetch forks copy_stream,
                # but wait_prefetch only happens in the next forward pass.
                get_offloader().join_after_forward()
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """运行微批次。

        Args:
            ubatch_metadata: 微批次元数据
            model: 模型

        Returns:
            输出张量
        """
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        model,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        """生成微批次元数据列表。

        Args:
            ubatch_slices: 微批次切片列表
            attn_metadata: 注意力元数据
            slot_mapping: slot 映射
            input_ids: 输入 ID
            positions: 位置
            inputs_embeds: 输入嵌入
            intermediate_tensors: 中间张量
            compute_stream: 计算流
            dp_metadata: DP 元数据
            batch_descriptor: 批次描述符
            cudagraph_runtime_mode: CUDA Graph 运行时模式

        Returns:
            微批次元数据列表
        """
        # Create one forward context per ubatch
        forward_contexts = []
        # slot_mapping can be None, an empty dict (from create_forward_context
        # converting None to {}), or a list of dicts (one per ubatch)
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        """切片模型输入。

        Args:
            tokens_slice: token 切片
            input_ids: 输入 ID
            positions: 位置
            inputs_embeds: 输入嵌入
            intermediate_tensors: 中间张量

        Returns:
            切片后的 (input_ids, positions, inputs_embeds, intermediate_tensors)
        """
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds else None
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice] if intermediate_tensors else None
        )

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def __call__(self, *args, **kwargs):
        """执行可执行对象。

        如果启用了微批次，则使用微批次执行；否则直接运行可执行对象。

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            执行结果
        """
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.model)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            # Sync offloader before replay - ensures any external dependencies
            # from pre-capture prefetches are satisfied.
            get_offloader().sync_prev_onload()
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.model)
