import pytest

from vllm import envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


def test_defaults():
    engine_args = EngineArgs(model="facebook/opt-125m")

    # Assert V1 defaults
    assert (engine_args.enable_prefix_caching
            ), "V1 turns on prefix caching by default"


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(
        UsageContext.LLM_CLASS)

    assert vllm_config.scheduler_config.max_num_seqs == 1024
    assert vllm_config.scheduler_config.max_num_batched_tokens == 8192

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(
        UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == 1024
    assert vllm_config.scheduler_config.max_num_batched_tokens == 2048


def test_prefix_cache_disabled_with_multimodel():
    engine_args = EngineArgs(model="llava-hf/llava-1.5-7b-hf")

    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    assert not vllm_config.cache_config.enable_prefix_caching
