# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.platforms import current_platform

if TYPE_CHECKING:
    import torch

    from vllm.config import ParallelConfig

MODEL_NAME = "Qwen/Qwen3-0.6B"
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


class NumaBindingWorkerExtension:
    parallel_config: "ParallelConfig"
    local_rank: int
    device: "torch.device"

    def check_numa_binding(self):
        import subprocess
        from pathlib import Path

        import psutil
        import torch.distributed as dist

        from vllm.distributed import get_pcp_group, get_world_group
        from vllm.utils.cpu_resource_utils import parse_id_list

        parallel = self.parallel_config
        assert parallel.numa_bind and parallel._numa_bind_auto
        assert parallel.numa_bind_nodes is not None
        gpu_index = self.local_rank
        node = current_platform.get_device_numa_node(self.device.index)
        assert parallel.numa_bind_nodes[gpu_index] == node
        node_cpus = parse_id_list(
            Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
        )
        cpus = set(psutil.Process().cpu_affinity())
        assert cpus and cpus <= set(node_cpus)
        result = subprocess.run(
            ["numactl", "--show"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        policy = dict(line.split(":", 1) for line in result.stdout.splitlines())
        assert policy["policy"].strip() == "preferred"
        assert policy["preferred node"].strip() == str(node)
        parent = psutil.Process(os.getppid())
        parent_cpus = set(parent.cpu_affinity())
        assert cpus <= parent_cpus
        state = {
            "pid": os.getpid(),
            "gpu_index": gpu_index,
            "gpu_uuid": current_platform.get_device_uuid(self.device.index),
            "dp_rank": parallel.data_parallel_rank,
            "pcp_rank": get_pcp_group().rank_in_group,
            "node": node,
            "cpus": sorted(cpus),
            "engine_core_pid": parent.pid,
            "engine_core_cpus": sorted(parent_cpus),
        }
        world = get_world_group()
        states = [None] * world.world_size
        dist.all_gather_object(states, state, group=world.cpu_group)
        return states


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


@pytest.mark.slow_test
def test_pcp_dp_with_auto_numa():
    """Automatic binding must cover both PCP workers in both local DP shards."""
    from shutil import which

    from vllm.utils import numa_utils

    if not current_platform.is_cuda() or current_platform.device_count() < 4:
        pytest.skip("PCP+DP NUMA smoke test requires four NVIDIA GPUs")
    if which("numactl") is None or not numa_utils._is_auto_numa_available(
        verbose=False, require_unbound_process=True
    ):
        pytest.skip("Host cannot engage automatic NUMA binding")

    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--kv-cache-memory-bytes",
        str(256 << 20),
        "--enforce-eager",
        "--distributed-executor-backend",
        "mp",
        "--prefill-context-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--worker-extension-cls",
        "tests.v1.distributed.test_dense_dp_world_size.NumaBindingWorkerExtension",
    ]
    with RemoteOpenAIServer(
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        server_args,
        env_dict={
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(VLLM_PATH), os.environ.get("PYTHONPATH")])
            ),
        },
        max_wait_seconds=1200,
    ) as server:
        completion = server.get_client().completions.create(
            model="deepseek-ai/DeepSeek-V2-Lite-Chat",
            prompt="The first five positive integers are",
            temperature=0,
            max_tokens=16,
        )
        assert completion.choices[0].text.strip()
        assert completion.usage.completion_tokens > 0
        print(f"PCP+DP completion: {completion.choices[0].text!r}")
        response = requests.post(
            server.url_for("collective_rpc"),
            json={"method": "check_numa_binding", "timeout": 30},
            timeout=60,
        )
        response.raise_for_status()
        states = response.json()["results"][0]
        print(f"PCP+DP NUMA worker states: {states}")
        assert sorted(s["gpu_index"] for s in states) == [0, 1, 2, 3]
        assert len({s["gpu_uuid"] for s in states}) == 4
        assert sorted((s["dp_rank"], s["pcp_rank"]) for s in states) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
