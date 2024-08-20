import enum
from typing import Dict, Union

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class VLLMDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecializedMixedInput = enum_auto()
    TmaWarpSpecializedPingpongMixedInput = enum_auto()
    TmaWarpSpecializedCooperativeMixedInput = enum_auto()


VLLMDataTypeNames: Dict[Union[VLLMDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        VLLMDataType.u4b8: "u4b8",
        VLLMDataType.u8b128: "u8b128",
    }
}

VLLMDataTypeTag: Dict[Union[VLLMDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        VLLMDataType.u4b8: "cutlass::vllm_uint4b8_t",
        VLLMDataType.u8b128: "cutlass::vllm_uint8b128_t",
    }
}

VLLMKernelScheduleTag: Dict[Union[
    MixedInputKernelScheduleType, KernelScheduleType], str] = {
        **KernelScheduleTag,  # type: ignore
        **{
            MixedInputKernelScheduleType.TmaWarpSpecializedMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedMixedInput",
            MixedInputKernelScheduleType.TmaWarpSpecializedPingpongMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput",
            MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput",
        }
    }
