# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class VLLMDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


VLLMDataTypeNames: dict[VLLMDataType | DataType, str] = {
    **DataTypeNames,  # type: ignore
    **{
        VLLMDataType.u4b8: "u4b8",
        VLLMDataType.u8b128: "u8b128",
    },
}

VLLMDataTypeTag: dict[VLLMDataType | DataType, str] = {
    **DataTypeTag,  # type: ignore
    **{
        VLLMDataType.u4b8: "cutlass::vllm_uint4b8_t",
        VLLMDataType.u8b128: "cutlass::vllm_uint8b128_t",
    },
}

VLLMDataTypeSize: dict[VLLMDataType | DataType, int] = {
    **DataTypeSize,  # type: ignore
    **{
        VLLMDataType.u4b8: 4,
        VLLMDataType.u8b128: 8,
    },
}

VLLMDataTypeVLLMScalarTypeTag: dict[VLLMDataType | DataType, str] = {
    VLLMDataType.u4b8: "vllm::kU4B8",
    VLLMDataType.u8b128: "vllm::kU8B128",
    DataType.u4: "vllm::kU4",
    DataType.u8: "vllm::kU8",
    DataType.s4: "vllm::kS4",
    DataType.s8: "vllm::kS8",
    DataType.f16: "vllm::kFloat16",
    DataType.bf16: "vllm::kBfloat16",
}

VLLMDataTypeTorchDataTypeTag: dict[VLLMDataType | DataType, str] = {
    DataType.u8: "at::ScalarType::Byte",
    DataType.s8: "at::ScalarType::Char",
    DataType.e4m3: "at::ScalarType::Float8_e4m3fn",
    DataType.s32: "at::ScalarType::Int",
    DataType.f16: "at::ScalarType::Half",
    DataType.bf16: "at::ScalarType::BFloat16",
    DataType.f32: "at::ScalarType::Float",
}

VLLMDataTypeStableTorchDataTypeTag: dict[VLLMDataType | DataType, str] = {
    DataType.u8: "torch::headeronly::ScalarType::Byte",
    DataType.s8: "torch::headeronly::ScalarType::Char",
    DataType.e4m3: "torch::headeronly::ScalarType::Float8_e4m3fn",
    DataType.s32: "torch::headeronly::ScalarType::Int",
    DataType.f16: "torch::headeronly::ScalarType::Half",
    DataType.bf16: "torch::headeronly::ScalarType::BFloat16",
    DataType.f32: "torch::headeronly::ScalarType::Float",
}

VLLMKernelScheduleTag: dict[MixedInputKernelScheduleType | KernelScheduleType, str] = {
    **KernelScheduleTag,  # type: ignore
    **{
        MixedInputKernelScheduleType.TmaWarpSpecialized: "cutlass::gemm::KernelTmaWarpSpecialized",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedPingpong: "cutlass::gemm::KernelTmaWarpSpecializedPingpong",  # noqa: E501
        MixedInputKernelScheduleType.TmaWarpSpecializedCooperative: "cutlass::gemm::KernelTmaWarpSpecializedCooperative",  # noqa: E501
    },
}
