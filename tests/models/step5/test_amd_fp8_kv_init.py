# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 on ROCm initializes with an FP8 E4M3 KV cache at block size 32."""

import pytest

from vllm.platforms import current_platform

from ..registry import HF_EXAMPLE_MODELS
from ..utils import dummy_hf_overrides

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm Step5 FP8 KV path"
)

ARCH = "Step5ForConditionalGeneration"


def test_fp8_kv_block_size_32_initializes(vllm_runner, monkeypatch):
    from vllm.platforms.rocm import on_mi3xx

    if not on_mi3xx():
        pytest.skip("Step5 on ROCm requires MI300 or MI350")
    info = HF_EXAMPLE_MODELS.get_hf_info(ARCH)
    info.check_available_online(on_fail="skip")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        info.default,
        load_format="dummy",
        max_model_len=info.max_model_len,
        hf_overrides=lambda cfg: dummy_hf_overrides(
            cfg,
            model_arch=ARCH,
            exist_overrides=info.hf_overrides,
            use_original_num_layers=True,
        ),
        kv_cache_dtype="fp8",
        block_size=32,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as llm:
        cache = llm.llm.llm_engine.vllm_config.cache_config
        assert cache.cache_dtype == "fp8"
        assert cache.block_size == 32
        out = llm.generate_greedy(["Hello"], max_tokens=4)
        assert len(out[0][0]) > 0
