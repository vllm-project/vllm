# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import os
from copy import deepcopy

import pytest

from vllm import SamplingParams

MODEL_NAME = "distilbert/distilgpt2"
PROMPTS = ["Hello, my name is"]


@pytest.mark.parametrize("fault", [False, True], ids=["healthy", "handle"])
def test_generate_after_nvml_diagnostic_failure(
    vllm_runner, monkeypatch, tmp_path, caplog_vllm, fault
):
    """A diagnostic-only NVML failure must not prevent real GPU generation."""
    from vllm.logger import DEFAULT_LOGGING_CONFIG, _print_warning_once
    from vllm.platforms import cuda, current_platform

    assert isinstance(current_platform, cuda.NvmlCudaPlatform)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    nvml = cuda.pynvml
    nvml.nvmlInit()
    try:
        missing_index = nvml.nvmlDeviceGetCount()
    finally:
        nvml.nvmlShutdown()
    assert missing_index > 0
    real_count = nvml.nvmlDeviceGetCount
    real_handle = nvml.nvmlDeviceGetHandleByIndex
    injected = []

    def in_diagnostics():
        return any(
            frame.function == "log_warnings" and frame.filename == cuda.__file__
            for frame in inspect.stack(context=0)
        )

    def count():
        return real_count() + int(fault and in_diagnostics())

    def handle(index):
        if fault and index == missing_index and in_diagnostics():
            injected.append(index)
            raise nvml.NVMLError(nvml.NVML_ERROR_UNKNOWN)
        return real_handle(index)

    monkeypatch.setattr(nvml, "nvmlDeviceGetCount", count)
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", handle)
    # A unique public config forces LLM to configure logging, even in a suite.
    config_path = tmp_path / "logging.json"
    config_path.write_text(json.dumps(deepcopy(DEFAULT_LOGGING_CONFIG)))
    _print_warning_once.cache_clear()
    try:
        with vllm_runner(
            os.environ.get("VLLM_NVML_TEST_MODEL", MODEL_NAME),
            trust_remote_code=False,
            dtype="half",
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            gpu_memory_utilization=0.25,
            kv_cache_memory_bytes=256 * 1024 * 1024,
            enforce_eager=True,
            log_config_file=str(config_path),
        ) as runner:
            outputs = runner.llm.generate(
                [PROMPTS[0]],
                SamplingParams(temperature=0, seed=0, max_tokens=4, ignore_eos=True),
                use_tqdm=False,
            )
        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == 4
        assert outputs[0].outputs[0].finish_reason == "length"
        if not fault:
            assert injected == []
        else:
            assert injected
            assert (
                f"Failed to query NVML device {missing_index}: Unknown Error"
                in caplog_vllm.messages
            )
    finally:
        _print_warning_once.cache_clear()
