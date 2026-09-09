# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import requests

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen3-0.6B"
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


def test_dense_dp_world_size():
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        current_platform.device_control_env_var: ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(DP_SIZE * TP_SIZE)
        ),
    }

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        env_dict=env_dict,
    ) as server:
        response = requests.get(server.url_for("get_world_size"))
        response.raise_for_status()
        assert response.json() == {"world_size": TP_SIZE * DP_SIZE}

        response = requests.get(
            server.url_for("get_world_size"), params={"include_dp": "false"}
        )
        response.raise_for_status()
        assert response.json() == {"world_size": TP_SIZE}
