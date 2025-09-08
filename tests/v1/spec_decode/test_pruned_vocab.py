from typing import Optional
from unittest import mock

import pytest

from vllm.platforms import current_platform
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.v1.spec_decode.eagle import EagleProposer

# NousResearch model is identical to meta-llama/Meta-Llama-3-8B-Instruct but doesn't need Meta to grant permission
model_path: str = "NousResearch/Meta-Llama-3-8B-Instruct"
draft_model_path: str = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
draft_vocab_pruned: str = 'thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt'

def _create_proposer(
    num_speculative_tokens: int,
    speculative_token_tree: Optional[list[tuple[int]]] = None,
    do_prune_vocab: bool = False,
) -> EagleProposer:
    model_config = ModelConfig(model=model_path,
                               runner="generate",
                               max_model_len=100)

    spec_token_tree_str = None
    if speculative_token_tree is not None:
        assert num_speculative_tokens == len(speculative_token_tree)
        spec_token_tree_str = str(speculative_token_tree)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_path,
        method="eagle",
        num_speculative_tokens=num_speculative_tokens,
        speculative_token_tree=spec_token_tree_str,
        draft_vocab_pruned=draft_vocab_pruned if do_prune_vocab else None,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig())

    return EagleProposer(vllm_config=vllm_config,
                         device=current_platform.device_type)


def test_load_pruned_vocab():
    proposer = _create_proposer(2)
    proposer.load_model(target_model)

    proposer_pruned = _create_proposer(2, do_prune_vocab=True)
    proposer_pruned.load_model(target_model)

    assert proposer.model.lm_head.data.shape < proposer_pruned.model.lm_head.data.shape
    assert proposer.model.embed_tokens.data.shape == proposer_pruned.model.embed_tokens.data.shape
