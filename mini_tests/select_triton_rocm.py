# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import types

os.environ["VLLM_TARGET_DEVICE"] = "rocm"

# Mock amdsmi to simulate ROCm
amdsmi = types.ModuleType("amdsmi")
amdsmi.amdsmi_init = lambda: None
amdsmi.amdsmi_shut_down = lambda: None
amdsmi.amdsmi_get_processor_handles = lambda: [1]
amdsmi.AmdSmiException = Exception
sys.modules["amdsmi"] = amdsmi
sys.modules["vllm._rocm_C"] = types.ModuleType("_rocm_C")

# Prevent CPU platform from conflicting with ROCm on macOS
import vllm.platforms as platforms_module  # noqa: E402

_orig_cpu = platforms_module.cpu_platform_plugin
platforms_module.cpu_platform_plugin = (
    lambda: None if os.environ.get("VLLM_TARGET_DEVICE") == "rocm" else _orig_cpu()
)
platforms_module.builtin_platform_plugins["cpu"] = platforms_module.cpu_platform_plugin

# Mock torch to look like ROCm
import torch  # noqa: E402

torch.version.hip = "5.7.0"
torch.cuda.get_device_properties = lambda d=0: types.SimpleNamespace(
    gcnArchName="gfx900", major=9, minor=0
)
torch.cuda.get_device_capability = lambda d=0: (9, 0)

# Stub custom ops
_ops = types.ModuleType("_custom_ops")
for op in [
    "cutlass_scaled_mm_supports_fp4",
    "cutlass_scaled_fp4_mm",
    "scaled_fp4_quant",
    "scaled_fp8_quant",
    "apply_repetition_penalties",
    "merge_attn_states",
    "scaled_int8_quant",
]:
    setattr(_ops, op, lambda *a, **k: None)
sys.modules["vllm._custom_ops"] = _ops

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (  # noqa: E402, I001
    ScaledMMLinearLayerConfig,
    choose_scaled_mm_linear_kernel,
)

cfg = ScaledMMLinearLayerConfig(
    is_channelwise=False,
    is_static_input_scheme=True,
    input_symmetric=True,
)

kernel = choose_scaled_mm_linear_kernel(cfg, compute_capability=None)

print("Selected kernel:", kernel.__name__)
assert "TritonScaledMMLinearKernel" in kernel.__name__
print("OK: TritonScaledMMLinearKernel chosen on ROCm fallback.")
