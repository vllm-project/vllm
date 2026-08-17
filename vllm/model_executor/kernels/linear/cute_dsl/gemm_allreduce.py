# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlashInfer GEMM with a fused tensor-parallel all-reduce.

FlashInfer's two-shot all-reduce requires the tokens dimension to be
divisible by 128. Dispatches are padded to the next 128-token
specialization.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.utils.flashinfer import get_flashinfer_cutedsl_gemm_allreduce

logger = init_logger(__name__)

_BUCKET_STEP = 128
_2CTA_MMA_TILER_M = 256
_2CTA_CLUSTER_SHAPE_MN = (2, 1)
_1CTA_MMA_TILER_M = 128
_1CTA_CLUSTER_SHAPE_MN = (1, 1)
_DTYPE_CONFIGS: dict[
    torch.dtype,
    tuple[
        type[cutlass.Numeric],
        type[cutlass.Numeric],
        type[cutlass.Numeric],
        torch.dtype,
        int,
        int,
    ],
] = {
    torch.bfloat16: (
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.BFloat16,
        torch.bfloat16,
        8,
        256,
    ),
}


def _as_cute_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    leading_dim: int | None = None,
) -> cute.Tensor:
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = dtype
    if leading_dim is None:
        return result.mark_layout_dynamic()
    return result.mark_layout_dynamic(leading_dim=leading_dim)


@dataclass
class CuteDSLFusedGemmAR:
    workspace: _CuteDSLFusedGemmARWorkspace
    weight: torch.Tensor
    weight_cute: cute.Tensor

    def should_run(self, x: torch.Tensor) -> bool:
        return self.workspace.should_run(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.workspace(x, self.weight, self.weight_cute)


@dataclass
class _CuteDSLFusedGemmARBucket:
    M: int
    use_2cta_instrs: bool
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    input_cute: cute.Tensor
    output_cute: cute.Tensor
    output_mc_cute: cute.Tensor
    flags: torch.Tensor
    flags_cute: cute.Tensor
    flags_mc_cute: cute.Tensor
    compiled: Any = None


def gemm_ar_bucket_sizes(max_M: int) -> tuple[int, ...]:
    if max_M < 1:
        raise ValueError(f"Fused GEMM-AR requires max_M >= 1, got {max_M}")

    aligned_max = (max_M + _BUCKET_STEP - 1) // _BUCKET_STEP * _BUCKET_STEP
    return tuple(range(_BUCKET_STEP, aligned_max + 1, _BUCKET_STEP))


def _bucket_kernel_config(
    M: int, mma_tiler_n: int
) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    if M % _2CTA_MMA_TILER_M == 0:
        return (
            True,
            (_2CTA_MMA_TILER_M, mma_tiler_n),
            _2CTA_CLUSTER_SHAPE_MN,
        )
    return False, (_1CTA_MMA_TILER_M, mma_tiler_n), _1CTA_CLUSTER_SHAPE_MN


class _CuteDSLFusedGemmARWorkspace:
    def __init__(self, *, max_M: int, N: int, K: int, dtype: torch.dtype) -> None:
        tp_group = get_tp_group()
        self.group = tp_group.device_group
        self.world_size = tp_group.world_size
        self.device = torch.device("cuda", torch.accelerator.current_device_index())

        assert self.world_size == 8
        assert dist.get_world_size() == self.world_size
        assert dtype in _DTYPE_CONFIGS
        (
            self.ab_dtype,
            self.acc_dtype,
            self.output_dtype,
            self.torch_output_dtype,
            k_alignment,
            self.mma_tiler_n,
        ) = _DTYPE_CONFIGS[dtype]
        assert N % self.mma_tiler_n == 0
        assert K % k_alignment == 0

        self.bucket_sizes = gemm_ar_bucket_sizes(max_M)
        self.max_M = self.bucket_sizes[-1]
        self.N = N
        self.K = K
        self.dtype = dtype

        self.input = torch.empty(
            (self.max_M, K),
            dtype=dtype,
            device=self.device,
        )
        self.output = symm_mem.empty(
            (self.max_M, N, 1),
            dtype=self.torch_output_dtype,
            device=self.device,
        )
        self.output_handle = symm_mem.rendezvous(
            self.output, group=self.group.group_name
        )
        if self.output_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-allreduce requires NVLink multicast memory")
        self.output_mc_alias = cutlass_torch.as_tensor(
            self.output_handle.multicast_ptr,
            self.output.shape,
            self.output.dtype,
        )
        self.generation_barrier = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.device,
        )
        major, minor = torch.cuda.get_device_capability(self.device)
        self.kernel_cls = get_flashinfer_cutedsl_gemm_allreduce()
        self.sm_version = f"sm_{major}{minor}"

        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        bucket_configs = {
            M: _bucket_kernel_config(M, self.mma_tiler_n) for M in self.bucket_sizes
        }
        bucket_flag_sizes = {}
        for M, (use_2cta, mma_tiler, cluster_shape) in bucket_configs.items():
            cta_tile_m = mma_tiler[0] // (2 if use_2cta else 1)
            num_clusters_m = (M + cta_tile_m * cluster_shape[0] - 1) // (
                cta_tile_m * cluster_shape[0]
            )
            num_clusters_n = (N + mma_tiler[1] * cluster_shape[1] - 1) // (
                mma_tiler[1] * cluster_shape[1]
            )
            bucket_flag_sizes[M] = (
                num_clusters_m * num_clusters_n * cluster_shape[0] * cluster_shape[1]
                + num_sms
            )
        self.flags = symm_mem.empty(
            sum(bucket_flag_sizes.values()),
            dtype=torch.int32,
            device=self.device,
        )
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, group=self.group.group_name)
        if self.flags_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-allreduce requires NVLink multicast memory")
        self.flags_mc_alias = cutlass_torch.as_tensor(
            self.flags_handle.multicast_ptr,
            self.flags.shape,
            self.flags.dtype,
        )

        flag_offset = 0
        self.buckets = {}
        for M, flag_size in bucket_flag_sizes.items():
            self.buckets[M] = self._make_bucket(
                M, *bucket_configs[M], flag_offset, flag_size
            )
            flag_offset += flag_size
        _fused_gemm_ar_workspaces.add(self)

    def _make_bucket(
        self,
        M: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        flag_offset: int,
        flag_size: int,
    ) -> _CuteDSLFusedGemmARBucket:
        flags = self.flags[flag_offset : flag_offset + flag_size]
        flags_mc_alias = self.flags_mc_alias[flag_offset : flag_offset + flag_size]
        return _CuteDSLFusedGemmARBucket(
            M=M,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            input_cute=_as_cute_tensor(
                self.input[:M].unsqueeze(-1),
                self.ab_dtype,
                leading_dim=1,
            ),
            output_cute=_as_cute_tensor(
                self.output[:M],
                self.output_dtype,
                leading_dim=1,
            ),
            output_mc_cute=_as_cute_tensor(
                self.output_mc_alias[:M],
                self.output_dtype,
                leading_dim=1,
            ),
            flags=flags,
            flags_cute=_as_cute_tensor(flags, cutlass.Int32),
            flags_mc_cute=_as_cute_tensor(
                flags_mc_alias,
                cutlass.Int32,
            ),
        )

    def _compile_buckets(
        self,
        kernel_cls: Any,
        sm_version: str,
        compile_weight_cute: cute.Tensor,
        stream: Any,
    ) -> None:
        for bucket in self.buckets.values():
            if bucket.compiled is not None:
                continue
            max_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
                bucket.cluster_shape_mn[0] * bucket.cluster_shape_mn[1]
            )
            kernel = kernel_cls(
                self.acc_dtype,
                bucket.use_2cta_instrs,
                bucket.mma_tiler_mn,
                bucket.cluster_shape_mn,
                True,
                all_reduce="two_shot",
                sm_version=sm_version,
            )
            bucket.compiled = cute.compile(
                kernel,
                bucket.input_cute,
                compile_weight_cute,
                bucket.output_cute,
                max_clusters,
                stream,
                c_mc=bucket.output_mc_cute,
                barrier_flag=bucket.flags_cute,
                barrier_flag_mc=bucket.flags_mc_cute,
            )

    def compile(self) -> int:
        uncompiled = sum(bucket.compiled is None for bucket in self.buckets.values())
        if uncompiled == 0:
            return 0

        compile_weight = torch.empty(
            (self.N, self.K), dtype=self.dtype, device=self.device
        )
        compile_weight_cute = _as_cute_tensor(
            compile_weight.unsqueeze(-1), self.ab_dtype, leading_dim=1
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compile_buckets(
            self.kernel_cls,
            self.sm_version,
            compile_weight_cute,
            stream,
        )
        torch.accelerator.synchronize(self.device)
        return uncompiled

    def make_weight_projection(self, weight: torch.Tensor) -> CuteDSLFusedGemmAR | None:
        if (
            weight.shape != (self.N, self.K)
            or weight.dtype != self.dtype
            or weight.device != self.device
            or not weight.is_contiguous()
        ):
            return None
        weight_cute = _as_cute_tensor(
            weight.unsqueeze(-1), self.ab_dtype, leading_dim=1
        )
        return CuteDSLFusedGemmAR(self, weight, weight_cute)

    def should_run(self, x: torch.Tensor) -> bool:
        if not (
            0 < x.shape[0] <= self.max_M
            and x.shape[1] == self.K
            and x.dtype == self.dtype
            and x.device == self.device
            and x.is_contiguous()
        ):
            return False
        return self._bucket_for_M(x.shape[0]).compiled is not None

    def _bucket_for_M(self, M: int) -> _CuteDSLFusedGemmARBucket:
        return next(bucket for size, bucket in self.buckets.items() if size >= M)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_cute: cute.Tensor,
    ) -> torch.Tensor:
        M = x.shape[0]
        assert self.should_run(x)
        assert weight.shape == (self.N, self.K)
        bucket = self._bucket_for_M(M)
        assert bucket.compiled is not None

        self.input[:M].copy_(x)
        if bucket.M > M:
            self.input[M : bucket.M].zero_()

        bucket.flags.zero_()
        dist.all_reduce(self.generation_barrier, group=self.group)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        bucket.compiled(
            bucket.input_cute,
            weight_cute,
            bucket.output_cute,
            stream,
            c_mc=bucket.output_mc_cute,
            barrier_flag=bucket.flags_cute,
            barrier_flag_mc=bucket.flags_mc_cute,
        )
        return self.output[:M, :, 0].clone()


_fused_gemm_ar_workspaces: weakref.WeakSet[_CuteDSLFusedGemmARWorkspace] = (
    weakref.WeakSet()
)
_fused_gemm_ar_workspace_cache: weakref.WeakValueDictionary[
    tuple[int, int, int, int, torch.dtype], _CuteDSLFusedGemmARWorkspace
] = weakref.WeakValueDictionary()


def _get_compatible_weight(linear: LinearBase) -> torch.Tensor | None:
    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return None
    if getattr(linear, "bias", None) is not None or not getattr(
        linear, "reduce_results", False
    ):
        return None
    weight = linear.weight
    incompatible_weight = (
        weight.ndim != 2
        or weight.dtype not in _DTYPE_CONFIGS
        or not weight.is_contiguous()
    )
    if incompatible_weight:
        return None
    return weight


def get_or_create_cutedsl_fused_gemm_ar_workspace(
    *, max_M: int, N: int, K: int, dtype: torch.dtype
) -> _CuteDSLFusedGemmARWorkspace:
    aligned_max_M = gemm_ar_bucket_sizes(max_M)[-1]
    device_index = torch.accelerator.current_device_index()
    key = (device_index, aligned_max_M, N, K, dtype)
    workspace = _fused_gemm_ar_workspace_cache.get(key)
    if workspace is None:
        workspace = _CuteDSLFusedGemmARWorkspace(max_M=max_M, N=N, K=K, dtype=dtype)
        _fused_gemm_ar_workspace_cache[key] = workspace
    return workspace


def make_cutedsl_fused_gemm_ar(
    linear: LinearBase, *, max_M: int
) -> CuteDSLFusedGemmAR | None:
    weight = _get_compatible_weight(linear)
    if weight is None:
        return None
    N, K = map(int, weight.shape)
    _, _, _, _, k_alignment, mma_tiler_n = _DTYPE_CONFIGS[weight.dtype]
    if N % mma_tiler_n != 0 or K % k_alignment != 0:
        return None
    try:
        workspace = get_or_create_cutedsl_fused_gemm_ar_workspace(
            max_M=max_M, N=N, K=K, dtype=weight.dtype
        )
    except (
        AttributeError,
        ImportError,
        AssertionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        logger.warning_once("CuTe DSL fused GEMM-AR initialization failed: %s", error)
        return None
    return workspace.make_weight_projection(weight)


def warmup_cutedsl_fused_gemm_ar() -> int:
    return sum(workspace.compile() for workspace in list(_fused_gemm_ar_workspaces))
