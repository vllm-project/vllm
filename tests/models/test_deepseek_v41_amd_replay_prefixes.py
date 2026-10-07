# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AMD model fills the replay layers' attention metadata keys."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_amd_model_fills_replay_metadata_prefixes(monkeypatch):
    import vllm.models.deepseek_v41.amd.model as amd

    monkeypatch.setattr(
        amd,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(amd.EngramLayout, "from_config", lambda config: None)
    monkeypatch.setattr(amd, "MHCPostOp", lambda: SimpleNamespace())

    def fake_layer(name, compressed, indexer):
        return SimpleNamespace(
            attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(prefix=f"{name}.swa"),
                compressed_cache_prefix=compressed,
                indexer=(
                    None
                    if indexer is None
                    else SimpleNamespace(k_cache=SimpleNamespace(prefix=indexer))
                ),
            ),
            engram=None,
        )

    # Layer 0 is the KV source. Replay starts at layer 1. Layer 2 has no
    # compressed cache and no indexer, so those keys are not added for it.
    layers = [
        fake_layer("layers.0", "layers.0.compressed", "layers.0.indexer"),
        fake_layer("layers.1", "layers.0.compressed", "layers.1.indexer"),
        fake_layer("layers.2", None, None),
    ]
    monkeypatch.setattr(amd, "make_layers", lambda n, factory, prefix: (0, n, layers))

    config = SimpleNamespace(
        vocab_size=32,
        hc_eps=1e-6,
        hc_mult=4,
        hidden_size=16,
        rms_norm_eps=1e-6,
        index_topk=4,
        kv_source_layer_ids=[0],
        num_hidden_layers=3,
        sliding_window=128,
        engram_layer_ids=(),
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
    )
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = config
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.quant_config = None
    vllm_config.cache_config.swa_bounded_replay = True
    vllm_config.use_v2_model_runner = True
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.use_ubatching = False
    vllm_config.parallel_config.enable_expert_parallel = False
    vllm_config.speculative_config = None
    vllm_config.scheduler_config.max_num_batched_tokens = 8

    model = amd.DeepseekV4Model(vllm_config=vllm_config)

    assert model.decoder_replay_start == 1
    assert model.decoder_replay_layers.metadata_prefixes == {
        "layers.1.swa",
        "layers.0.compressed",
        "layers.1.indexer",
        "layers.2.swa",
    }
