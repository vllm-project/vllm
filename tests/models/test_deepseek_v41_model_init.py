# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constructor replay policy for prefixes with and without a KV-source anchor."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("sources", "compress_ratio", "replay"),
    [
        ([], 0, False),
        ([1], 0, True),
        ([1], 1, True),
        ([1], 2, False),
        # Nothing before the cut would run on this rank.
        ([0], 0, False),
    ],
)
def test_prefix_replay_requires_an_uncompressed_kv_source(
    monkeypatch, sources, compress_ratio, replay
):
    import vllm.models.deepseek_v41.nvidia.model as model_module

    monkeypatch.setattr(torch.cuda, "Stream", lambda: None)
    monkeypatch.setattr(model_module, "_use_sequence_parallel", lambda config: False)
    monkeypatch.setattr(model_module, "supports_mhc_overlap", lambda config: False)
    monkeypatch.setattr(model_module, "maybe_init_gemm_rs", lambda *args: None)
    monkeypatch.setattr(model_module.EngramLayout, "from_config", lambda config: None)
    monkeypatch.setattr(model_module.EngramBatch, "create", lambda *args: None)
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    for i, layer in enumerate(layers):
        layer.attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(bounded_replay=True, prefix=f"swa.{i}"),
            compress_ratio=compress_ratio,
            compressed_cache_prefix=None,
            indexer=None,
        )
        layer.write_kv = lambda *args: None
    monkeypatch.setattr(
        model_module, "make_layers", lambda n, factory, prefix: (0, n, layers)
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        hc_mult=4,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        index_topk=512,
        num_hidden_layers=3,
        kv_source_layer_ids=sources,
        sliding_window=128,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="deep_gemm"),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, use_ubatching=False
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        engram_config=None,
        speculative_config=None,
    )
    model = model_module.DeepseekV4Model(vllm_config=vllm_config)
    assert (model.decoder_replay_layers is not None) == replay
    assert model.decoder_replay_start == (1 if replay else 3)
