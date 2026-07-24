# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
from collections.abc import Mapping

_ROCM_RUNTIME_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ROCM_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCM_HOME",
    "ROCM_PATH",
)


def environment_looks_like_rocm(
    env: Mapping[str, str] | None = None,
    *,
    amdsmi_present: bool | None = None,
) -> bool:
    env = os.environ if env is None else env
    if any(env.get(var_name) for var_name in _ROCM_RUNTIME_ENV_VARS):
        return True

    if amdsmi_present is None:
        amdsmi_present = importlib.util.find_spec("amdsmi") is not None

    return amdsmi_present


def format_torch_runtime_summary(
    torch_version_cuda: str | None,
    torch_version_hip: str | None,
) -> str:
    return (
        f"torch.version.cuda={torch_version_cuda or 'None'}, "
        f"torch.version.hip={torch_version_hip or 'None'}"
    )


def build_failed_device_type_message(
    torch_version_cuda: str | None,
    torch_version_hip: str | None,
    *,
    environment_looks_like_rocm: bool,
) -> str:
    message = (
        "Failed to infer device type. "
        f"Detected runtime: "
        f"{format_torch_runtime_summary(torch_version_cuda, torch_version_hip)}."
    )

    if environment_looks_like_rocm and not torch_version_hip:
        if torch_version_cuda:
            message += (
                " This environment looks like ROCm, but the installed torch "
                "runtime reports CUDA instead of HIP. On AMD systems this "
                "often means a CUDA wheel was installed for torch or vLLM. "
                "Verify that the installed torch and vLLM packages match "
                "your ROCm environment."
            )
        else:
            message += (
                " This environment looks like ROCm, but the installed torch "
                "runtime does not expose HIP support. On AMD systems this "
                "often means a non-ROCm torch or vLLM build is installed. "
                "Verify that the installed torch and vLLM packages match "
                "your ROCm environment."
            )

    message += (
        " Set the environment variable `VLLM_LOGGING_LEVEL=DEBUG` to turn on "
        "verbose logging to help debug the issue."
    )
    return message
