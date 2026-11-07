# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`draft_context_window` windows the draft layers and keeps their full KV."""

import pytest
import torch

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.utils import check_draft_context_window
from vllm.v1.worker.gpu.attn_utils import init_attn_backend

TARGET = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DRAFTER = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@pytest.mark.parametrize("window", [None, 2048])
def test_eagle3_drafter_window(dist_init, window):
    model_config = ModelConfig(TARGET, max_model_len=4096, dtype="bfloat16")
    speculative_config = SpeculativeConfig(
        method="eagle3",
        model=DRAFTER,
        num_speculative_tokens=3,
        draft_context_window=window,
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
    )
    vllm_config = VllmConfig(
        model_config=model_config, speculative_config=speculative_config
    )
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
        Eagle3LlamaForCausalLM(vllm_config=vllm_config)
        layers = get_layers_from_vllm_config(vllm_config, Attention)

    assert [layer.sliding_window for layer in layers.values()] == [window]
    check_draft_context_window(speculative_config, layers, set(layers))

    # A drafter that does not apply the option fails the check.
    speculative_config.draft_context_window = 4096
    with pytest.raises(ValueError, match="model.layers.32"):
        check_draft_context_window(speculative_config, layers, set(layers))


@pytest.mark.parametrize(
    ("is_draft_layer", "option_window", "spec_type"),
    [
        pytest.param(True, 2048, FullAttentionSpec, id="draft-with-option"),
        pytest.param(True, None, SlidingWindowSpec, id="draft-checkpoint-window"),
        pytest.param(False, 2048, SlidingWindowSpec, id="target"),
    ],
)
def test_kv_cache_spec(is_draft_layer, option_window, spec_type):
    speculative_config = SpeculativeConfig(model="ngram", num_speculative_tokens=3)
    vllm_config = VllmConfig(
        model_config=ModelConfig(max_model_len=1024),
        speculative_config=speculative_config,
    )
    # VllmConfig validation only allows the option with method="eagle3".
    speculative_config.draft_context_window = option_window
    with set_current_vllm_config(vllm_config):
        layer = Attention(
            8, 128, 0.1, num_kv_heads=8, per_layer_sliding_window=2048, prefix="layer"
        )
    layer.is_draft_layer = is_draft_layer
    assert type(layer.get_kv_cache_spec(vllm_config)) is spec_type


def test_windowed_draft_layer_gets_its_own_attention_group():
    """FlashInfer needs one window per metadata builder."""
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    with set_current_vllm_config(vllm_config):
        for prefix, window in (("target", None), ("draft", 2048)):
            Attention(
                8,
                128,
                0.1,
                num_kv_heads=8,
                per_layer_sliding_window=window,
                prefix=prefix,
            )
    spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["target", "draft"], spec)],
    )

    attn_groups, _, _ = init_attn_backend(
        kv_cache_config, vllm_config, torch.device("cpu")
    )

    assert [group.layer_names for group in attn_groups[0]] == [["target"], ["draft"]]
