# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_model

SIMULATED_FORWARD_MODELS = [
    pytest.param("Qwen/Qwen3-235B-A22B", 1, id="qwen3-235b-a22b"),
    pytest.param("allenai/Olmo-3-7B-Think", 2, id="olmo3-hybrid-attention"),
    pytest.param("openai/gpt-oss-120b", 2, id="gpt-oss-120b"),
]


def _check_model_available(model: str) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    except ValueError:
        return
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


def _get_kv_cache_group_count(vllm_model) -> int | None:
    engine_core_client = vllm_model.llm.llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", None)
    if engine_core is None:
        return None

    kv_cache_config = getattr(engine_core.scheduler, "kv_cache_config", None)
    if kv_cache_config is None:
        return None
    return len(kv_cache_config.kv_cache_groups)


@pytest.mark.parametrize(("model", "min_kv_cache_groups"), SIMULATED_FORWARD_MODELS)
def test_simulate_forward_model_matrix(
    vllm_runner,
    model: str,
    min_kv_cache_groups: int,
) -> None:
    if not current_platform.is_cpu():
        pytest.skip("Simulated forward is currently supported on CPU only.")

    _check_model_available(model)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
        detokenize=False,
        extra_args={"simulated_output_token_ids": "501,502"},
    )

    with vllm_runner(
        model,
        simulate_forward=True,
        disable_hybrid_kv_cache_manager=False,
        kv_cache_memory_bytes=16 * 1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate(
            [[1000, 1001, 1002, 1003]],
            sampling_params=sampling_params,
        )
        assert outputs[0][0][0][-2:] == [501, 502]

        group_count = _get_kv_cache_group_count(vllm_model)
        if group_count is not None:
            assert group_count >= min_kv_cache_groups
