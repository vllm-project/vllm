# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes


@dataclass(frozen=True)
class HardwareSchedulingDefaults:
    llm_class_max_num_batched_tokens: int
    api_server_max_num_batched_tokens: int
    max_num_seqs: int
    max_cudagraph_capture_size: int


COMPACT_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=8192,
    api_server_max_num_batched_tokens=2048,
    max_num_seqs=256,
    max_cudagraph_capture_size=128,
)

GRAPH_LIGHT_BALANCED_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=12288,
    api_server_max_num_batched_tokens=4096,
    max_num_seqs=512,
    max_cudagraph_capture_size=128,
)

BALANCED_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=12288,
    api_server_max_num_batched_tokens=4096,
    max_num_seqs=512,
    max_cudagraph_capture_size=256,
)

LARGE_ACCELERATOR_DEFAULTS = HardwareSchedulingDefaults(
    llm_class_max_num_batched_tokens=16384,
    api_server_max_num_batched_tokens=8192,
    max_num_seqs=1024,
    max_cudagraph_capture_size=512,
)


def infer_accelerator_scheduling_defaults(
    device_memory: int,
    device_name: str,
    *,
    is_rocm: bool = False,
    is_xpu: bool = False,
    is_out_of_tree: bool = False,
    device_type: str = "",
    is_data_center_gpu: bool = False,
) -> HardwareSchedulingDefaults:
    device_name = device_name.lower()
    is_a100 = "a100" in device_name
    is_ascend = (
        device_type == "npu"
        or "ascend" in device_name
        or "910b" in device_name
        or "910c" in device_name
        or (is_out_of_tree and "npu" in device_name)
    )

    if is_ascend:
        if "910c" in device_name:
            return GRAPH_LIGHT_BALANCED_ACCELERATOR_DEFAULTS
        if "910b" in device_name or device_memory >= 48 * GiB_bytes:
            return GRAPH_LIGHT_BALANCED_ACCELERATOR_DEFAULTS
        return COMPACT_ACCELERATOR_DEFAULTS

    if is_rocm:
        if any(name in device_name for name in ("mi300", "mi325", "gfx942", "gfx950")):
            return LARGE_ACCELERATOR_DEFAULTS
        if any(name in device_name for name in ("mi250", "mi210", "gfx90a")):
            return BALANCED_ACCELERATOR_DEFAULTS
        if device_memory >= 64 * GiB_bytes:
            return BALANCED_ACCELERATOR_DEFAULTS
        return COMPACT_ACCELERATOR_DEFAULTS

    if is_xpu:
        if "a770" in device_name or not is_data_center_gpu:
            return COMPACT_ACCELERATOR_DEFAULTS
        if any(name in device_name for name in ("1550", "max 1550")) or device_memory >= 96 * GiB_bytes:
            return BALANCED_ACCELERATOR_DEFAULTS
        if any(name in device_name for name in ("1100", "max 1100")) or device_memory >= 48 * GiB_bytes:
            return GRAPH_LIGHT_BALANCED_ACCELERATOR_DEFAULTS
        return COMPACT_ACCELERATOR_DEFAULTS

    if device_memory >= 70 * GiB_bytes and not is_a100:
        return LARGE_ACCELERATOR_DEFAULTS

    if device_memory >= 40 * GiB_bytes and not is_a100:
        return BALANCED_ACCELERATOR_DEFAULTS

    if is_xpu and is_data_center_gpu and device_memory >= 32 * GiB_bytes:
        return BALANCED_ACCELERATOR_DEFAULTS

    return COMPACT_ACCELERATOR_DEFAULTS


def get_current_accelerator_scheduling_defaults() -> HardwareSchedulingDefaults:
    try:
        device_memory = current_platform.get_device_total_memory()
        device_name = current_platform.get_device_name()
    except Exception:
        return COMPACT_ACCELERATOR_DEFAULTS

    is_xpu = current_platform.is_xpu()
    is_rocm = current_platform.is_rocm()
    is_out_of_tree = current_platform.is_out_of_tree()
    device_type = getattr(current_platform, "device_type", "")
    is_data_center_gpu = False
    if is_xpu:
        is_data_center_gpu_fn = getattr(current_platform, "is_data_center_gpu", None)
        if callable(is_data_center_gpu_fn):
            is_data_center_gpu = bool(is_data_center_gpu_fn())

    return infer_accelerator_scheduling_defaults(
        device_memory,
        device_name,
        is_rocm=is_rocm,
        is_xpu=is_xpu,
        is_out_of_tree=is_out_of_tree,
        device_type=device_type,
        is_data_center_gpu=is_data_center_gpu,
    )