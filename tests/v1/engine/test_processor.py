# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.engine.processor import Processor


def make_processor(cache_config: CacheConfig):
    model = "facebook/opt-125m"

    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
    )

    tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
        scheduler_config=vllm_config.scheduler_config,
        parallel_config=vllm_config.parallel_config,
        lora_config=vllm_config.lora_config)
    tokenizer.ping()
    return Processor(vllm_config, tokenizer)


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_process_inputs_prompt_kv_block_hashes(enable_prefix_caching: bool):
    processor = make_processor(
        CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
            enable_prefix_caching=enable_prefix_caching,
        ))
    result = processor.process_inputs(
        request_id="0",
        prompt=TokensPrompt(prompt_token_ids=list(range(49)), ),
        params=SamplingParams(),
    )

    if enable_prefix_caching:
        # Block size is 16, so 49 tokens should result in 3 block hashes.
        assert len(result.prompt_kv_block_hashes) == 3
    else:
        assert result.prompt_kv_block_hashes is None
