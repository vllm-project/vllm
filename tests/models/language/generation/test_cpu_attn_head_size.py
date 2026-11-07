# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)


def test_cpu_rejects_unsupported_head_size_at_startup(
    vllm_runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run the engine in-process so the ValueError reaches the caller.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "1")
    with (
        pytest.raises(ValueError, match="Head size 56"),
        vllm_runner(
            "Qwen/Qwen2.5-0.5B-Instruct",
            hf_overrides={"num_attention_heads": 16},  # head size 896 / 16 = 56
            load_format="dummy",
            max_model_len=512,
            distributed_executor_backend="uni",
        ) as vllm_model,
    ):
        vllm_model.generate_greedy(["Hello"], max_tokens=1)
