# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

import pytest

from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine.utils import EngineZmqAddresses, launch_core_engines


class SelectedCoreEngineLauncher:
    called = False

    @contextlib.contextmanager
    def launch_core_engines(
        self,
        vllm_config,
        executor_class,
        log_stats,
        addresses,
        num_api_servers=1,
    ):
        type(self).called = True
        yield "manager", None, addresses, "tensor_queue"


@pytest.mark.skip_global_cleanup
def test_launch_core_engines_uses_configured_launcher_cls():
    SelectedCoreEngineLauncher.called = False
    addresses = EngineZmqAddresses(inputs=[], outputs=[])
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            engine_core_launcher_cls=f"{__name__}.SelectedCoreEngineLauncher"
        ),
    )

    with launch_core_engines(
        vllm_config=vllm_config,
        executor_class=object,
        log_stats=False,
        addresses=addresses,
    ) as launched:
        manager, coordinator, launched_addresses, tensor_queue = launched

    assert SelectedCoreEngineLauncher.called
    assert manager == "manager"
    assert coordinator is None
    assert launched_addresses is addresses
    assert tensor_queue == "tensor_queue"
