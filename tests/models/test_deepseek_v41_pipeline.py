# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        mhc_stream=None,
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix


def test_engram_ipc_runtime_state_rebuilds_meta_staging(dist_init, monkeypatch):
    from transformers import PreTrainedConfig

    from vllm.config import (
        EngramConfig,
        KernelConfig,
        SchedulerConfig,
        set_current_vllm_config,
    )
    from vllm.models.deepseek_v41.common.engram import Engram, EngramLayout

    config = PreTrainedConfig(
        hidden_size=16,
        hc_mult=1,
        rms_norm_eps=1e-6,
        engram_layer_ids=[0],
        engram_num_embeddings=[32],
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_compressed_vocab_size=4,
        engram_pad_token_id=0,
        engram_vocab_size=31,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_model_len=8,
            is_encoder_decoder=False,
            max_num_batched_tokens=8,
            max_num_seqs=1,
        ),
        engram_config=EngramConfig(cpu_offload=False),
        kernel_config=KernelConfig(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=1),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    with set_current_vllm_config(vllm_config), torch.device("meta"):
        module = Engram(config, None, EngramLayout(config), 0, False, "engram")

    placeholder = module.staged_rows
    assert placeholder.is_meta
    module.rebuild_runtime_state_after_ipc(8, torch.device("cpu"), None)

    assert module.staged_rows is placeholder
    assert placeholder.device == torch.device("cpu")
    assert placeholder.shape == (8, 1, 32)
    assert module._prefetch_stream is None
    assert module._prefetch_done is None
    assert module._batch is None
