import copy
from dataclasses import dataclass
from itertools import product
from typing import Tuple


@dataclass
class Cutlass3xArgs:
    dtype_str: str
    arch: int
    tile_shape: Tuple[int, int, int]
    cluster_shape: Tuple[int, int, int]
    kernel_schedule: str
    epilogue_schedule: str
    tile_schedule: str
    gemm_mode: str
    acc_type: str

    def with_tile_shape(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_shape = ts
        return clone

    def with_cluster_shape(self, cs):
        clone = copy.deepcopy(self)
        clone.cluster_shape = cs
        return clone

    def with_tile_schedule(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_schedule = ts
        return clone

    def with_kernel_schedule(self, ks):
        clone = copy.deepcopy(self)
        clone.kernel_schedule = ks
        return clone

    def with_epilogue_schedule(self, es):
        clone = copy.deepcopy(self)
        clone.epilogue_schedule = es
        return clone

    def with_gemm_mode(self, gm):
        clone = copy.deepcopy(self)
        clone.gemm_mode = gm
        return clone

    def with_acc_type(self, acc):
        clone = copy.deepcopy(self)
        clone.acc_type = acc
        return clone

    def with_dtype_str(self, dtype_str):
        clone = copy.deepcopy(self)
        clone.dtype_str = dtype_str
        return clone


DefaultCutlass3xArgsFP8 = Cutlass3xArgs(
    "fp8", 90, (128, 128, 128), (1, 2, 1),
    "cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative",
    "cutlass::epilogue::TmaWarpSpecializedCooperative",
    "cutlass::gemm::PersistentScheduler",
    "cutlass::gemm::GemmUniversalMode::kGemm", "float")

## Kernel Schedules
## All
# struct KernelMultistage { };
# struct KernelCpAsyncWarpSpecialized { };
# struct KernelCpAsyncWarpSpecializedPingpong { };
# struct KernelCpAsyncWarpSpecializedCooperative { };
# struct KernelTma { };
# struct KernelTmaWarpSpecialized { };
# struct KernelTmaWarpSpecializedPingpong { };
# struct KernelTmaWarpSpecializedCooperative { };
# struct KernelPtrArrayTmaWarpSpecializedCooperative { };
## FP8
# struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
# struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { }; # noqa
# struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { }; #noqa
# struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };  #noqa

## Epilogue policies
# struct NoSmemWarpSpecialized {};
# struct PtrArrayNoSmemWarpSpecialized {};
# struct TmaWarpSpecialized {};
# struct TmaWarpSpecializedCooperative {};

## Tile scheduler
# struct PersistentScheduler { };
# struct StreamKScheduler { };

## Kgemms
# kGemm
# kGemmSplitKParallel,
# kBatched,
# kArray,
# kGrouped,
# kInvalid

cluster_shapes = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (4, 1, 1),
                  (1, 4, 1), (8, 1, 1), (1, 8, 1), (4, 4, 1)]
tile_shapes_m = [64, 128, 256]
tile_shapes_n = [64, 128, 256]
tile_shapes_k = [32, 64, 128, 256]
tile_shapes = list(product(tile_shapes_m, tile_shapes_n, tile_shapes_k))

kernel_schedules = [
    "cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum",
    "cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum",
    "cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum"
]

epilogue_schedules = [
    "cutlass::epilogue::TmaWarpSpecialized",
    "cutlass::epilogue::TmaWarpSpecializedCooperative"
]

tile_schedules = [
    "cutlass::gemm::PersistentScheduler", "cutlass::gemm::StreamKScheduler"
]

gemm_modes = ["cutlass::gemm::GemmUniversalMode::kGemm"]

acc_types = ["float"]

#epilogue_schedules_v2 = ["cutlass::epilogue::NoSmemWarpSpecialized"]
gemm_modes_v2 = ["cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel"]
acc_types_v2 = ["cutlass::half_t"]

## Make Cutlass3xArgsTest

Cutlass3xArgsTest = []

for ts, cs, ks, es, tile_schedule, gm, at in product(
        tile_shapes, cluster_shapes, kernel_schedules, epilogue_schedules,
        tile_schedules, gemm_modes, acc_types):

    Cutlass3xArgsTest.append(
        DefaultCutlass3xArgsFP8.with_tile_shape(ts).with_cluster_shape(cs).
        with_kernel_schedule(ks).with_epilogue_schedule(es).with_tile_schedule(
            tile_schedule).with_gemm_mode(gm).with_acc_type(at))

Cutlass3xArgsTest = Cutlass3xArgsTest
