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

VLLMDataTypeSize: Dict[Union[VLLMDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        VLLMDataType.u4b8: 4,
        VLLMDataType.u8b128: 8,
    }
}

VLLMDataTypeVLLMScalarTypeTag: Dict[Union[VLLMDataType, DataType], str] = {
    VLLMDataType.u4b8: "vllm::kU4B8",
    VLLMDataType.u8b128: "vllm::kU8B128",
    DataType.u4: "vllm::kU4",
    DataType.u8: "vllm::kU8",
    DataType.s4: "vllm::kS4",
    DataType.s8: "vllm::kS8",
    DataType.f16: "vllm::kFloat16",
    DataType.bf16: "vllm::kBfloat16",
}

VLLMDataTypeTorchDataTypeTag: Dict[Union[VLLMDataType, DataType], str] = {
    DataType.u8: "at::ScalarType::Byte",
    DataType.s8: "at::ScalarType::Char",
    DataType.e4m3: "at::ScalarType::Float8_e4m3fn",
    DataType.s32: "at::ScalarType::Int",
    DataType.f16: "at::ScalarType::Half",
    DataType.bf16: "at::ScalarType::BFloat16",
    DataType.f32: "at::ScalarType::Float",
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
