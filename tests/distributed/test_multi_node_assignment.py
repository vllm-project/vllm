# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make sure ray assigns GPU workers to the correct node.

Run:
```sh
cd $VLLM_PATH/tests

pytest distributed/test_multi_node_assignment.py
```
"""

import os

import pytest
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import initialize_ray_cluster
from vllm.config import ParallelConfig
from vllm.executor.ray_utils import _wait_until_pg_removed
from vllm.utils import get_ip

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


@pytest.mark.skipif(not VLLM_MULTI_NODE,
                    reason="Need at least 2 nodes to run the test.")
def test_multi_node_assignment() -> None:

    # NOTE: important to keep this class definition here
    # to let ray use cloudpickle to serialize it.
    class Actor:

        def get_ip(self):
            return get_ip()

    for _ in range(10):
        config = ParallelConfig(1, 2)
        initialize_ray_cluster(config)

        current_ip = get_ip()
        workers = []
        for bundle_id, bundle in enumerate(
                config.placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=config.placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
            )(Actor).remote()
            worker_ip = ray.get(worker.get_ip.remote())
            assert worker_ip == current_ip
            workers.append(worker)

        for worker in workers:
            ray.kill(worker)

        _wait_until_pg_removed(config.placement_group)
