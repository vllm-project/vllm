# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.benchmarks.lib import energy


class NvmlError(Exception):
    pass


@pytest.fixture
def nvml(monkeypatch):
    nvml = Mock()
    nvml.NVMLError = NvmlError
    nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: i
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 3000, 2000, 6000]
    monkeypatch.setattr(energy, "import_pynvml", lambda: nvml)
    monkeypatch.setattr(
        energy, "time", SimpleNamespace(perf_counter=iter([10.0, 12.0]).__next__)
    )
    return nvml


@pytest.mark.parametrize("output_tokens", [0, 8])
def test_energy_aggregates_devices_and_normalizes_units(nvml, output_tokens):
    meter = energy.GpuEnergyMeter([0, 2])
    with meter.measure():
        pass

    assert meter.get_results(output_tokens) == {
        "energy_gpu_ids": [0, 2],
        "gpu_energy_j": 4.0,
        "gpu_energy_duration_s": 2.0,
        "gpu_avg_power_w": 2.0,
        "gpu_energy_per_output_token_j": 0.5 if output_tokens else None,
    }
    assert [call.args for call in nvml.nvmlDeviceGetHandleByIndex.call_args_list] == [
        (0,),
        (2,),
    ]
    nvml.nvmlShutdown.assert_called_once()


@pytest.mark.parametrize("device_ids", [[], [-1], [0, 0]])
def test_energy_rejects_invalid_device_selection(device_ids):
    with pytest.raises(ValueError, match="distinct non-negative"):
        energy.GpuEnergyMeter(device_ids)


@pytest.mark.parametrize("failure", ["init", "initial_read", "final_read", "reset"])
def test_energy_omits_unavailable_or_reset_counters(nvml, failure, caplog):
    if failure == "init":
        nvml.nvmlInit.side_effect = NvmlError("unavailable")
    elif failure == "initial_read":
        nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = NvmlError("unsupported")
    elif failure == "final_read":
        nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 3000, NvmlError()]
    else:
        # One counter resets while the other increases; the sum alone is unsafe.
        nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 3000, 900, 6000]

    meter = energy.GpuEnergyMeter([0, 1])
    workload = Mock()
    with meter.measure():
        workload()

    workload.assert_called_once()
    assert meter.get_results(8) == {}
    assert "energy" in caplog.text
    assert nvml.nvmlShutdown.call_count == (failure != "init")


def test_energy_preserves_workload_exception_and_releases_nvml(nvml):
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 3000, NvmlError()]
    nvml.nvmlShutdown.side_effect = NvmlError("shutdown failed")
    with (
        pytest.raises(ValueError, match="workload failed"),
        energy.GpuEnergyMeter([0, 1]).measure(),
    ):
        raise ValueError("workload failed")
    nvml.nvmlShutdown.assert_called_once()
