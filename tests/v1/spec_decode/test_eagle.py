# SPDX-License-Identifier: Apache-2.0

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SpeculativeConfig, VllmConfig)
from vllm.v1.spec_decode.eagle import EagleProposer


def test_eagle_proposer():
    model_dir = "meta-llama/Llama-3.1-8B-Instruct"
    eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

    def eagle_proposer(k: int) -> EagleProposer:
        model_config = ModelConfig(model=model_dir,
                                   task="generate",
                                   max_model_len=100,
                                   tokenizer=model_dir,
                                   tokenizer_mode="auto",
                                   dtype="auto",
                                   seed=None,
                                   trust_remote_code=False)

        speculative_config = SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(),
            model=eagle_dir,
            method="eagle",
            num_speculative_tokens=k,
        )

        vllm_config = VllmConfig(model_config=model_config,
                                 cache_config=CacheConfig(),
                                 speculative_config=speculative_config)

        return EagleProposer(vllm_config=vllm_config, device='cuda')

    proposer = eagle_proposer(8)
    assert isinstance(proposer, EagleProposer)
    print("EagleProposer instantiated successfully.")
