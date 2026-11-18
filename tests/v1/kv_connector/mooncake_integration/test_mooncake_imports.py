# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake CUDA startup canaries."""

import importlib.metadata as metadata

from vllm import SamplingParams


def test_mooncake_engine_imports():
    """Fail here rather than as "Mooncake is not available" at engine startup.

    The PyPI mooncake-transfer-engine wheel is built against CUDA 12, so the
    CUDA 13 CI image ships the cuda13 variant instead.
    A mismatch surfaces as an ImportError on the compiled extension.
    """
    for package_name in ("mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"):
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package_name}: {version}")

    import mooncake.engine  # noqa: F401


def test_mooncake_starts_with_peermem_and_default_kv_cache(vllm_runner, monkeypatch):
    """Explicit peermem must resolve to a compatible cache before worker startup."""
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "1")
    with vllm_runner(
        "vllm-project/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.04,
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            # Exercise startup on CUDA without depending on a peermem kernel module.
            "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
        },
    ) as runner:
        config = runner.llm.llm_engine.vllm_config.cache_config
        assert config.enable_extensible_kv_cache is False
        [result] = runner.llm.generate(
            ["Hello"], SamplingParams(temperature=0, max_tokens=1)
        )
        assert len(result.outputs[0].token_ids) == 1
