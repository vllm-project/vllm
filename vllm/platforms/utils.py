# SPDX-License-Identifier: Apache-2.0
import os


def device_id_to_physical_device_id(device_id: int,
                                    device_control_env_var: str) -> int:
    if device_control_env_var in os.environ:
        device_ids = os.environ[device_control_env_var].split(",")
        if device_ids == [""]:
            msg = (
                f"{device_control_env_var} is set to empty string, which means"
                " current platform support is disabled. If you are using ray,"
                f" please unset the environment variable"
                f" `{device_control_env_var}` inside the worker/actor. "
                "Check https://github.com/vllm-project/vllm/issues/8402 for"
                " more information.")
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id
