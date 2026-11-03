# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
@torch.inference_mode()
def test_replay_wkv_loads_checkpoint_without_changing_fused_projection(
    dist_init, quantized
):
    """Replay WKV must load weights and block scales using their native loaders."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.models.deepseek_v41.decoder_replay_layers import load_replay_wkv_weight
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    torch.manual_seed(0)
    quant_config = (
        DeepseekV4FP8Config(weight_block_size=[32, 32]) if quantized else None
    )
    fused = MergedColumnParallelLinear(
        256,
        [1024, 512],
        bias=False,
        disable_tp=True,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
    )
    replay = MergedColumnParallelLinear(
        256,
        [512],
        bias=False,
        disable_tp=True,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
    )
    params = dict(replay.named_parameters(prefix="layers.20.attn.replay_wkv"))
    loaded = set()
    for param in fused.parameters():
        param.data.fill_(127 if param.dtype == torch.uint8 else 1)
    for suffix, param in replay.named_parameters():
        if suffix == "weight_scale":
            # The checkpoint has 32x32 E8M0 blocks; the loader expands rows
            # into the runtime's 1x32 uint8 scale layout.
            value = 2.0 ** (torch.arange(16 * 8).reshape(16, 8) % 3)
            value = value.to(torch.float8_e8m0fnu)
            expected_param = value.view(torch.uint8).repeat_interleave(32, dim=0)
        else:
            value = (2.0 ** (torch.arange(param.numel()) % 3)).reshape(param.shape)
            value = value.to(param.dtype)
            expected_param = value
        name = f"layers.20.attn.wkv.{suffix}"
        loaded.add(load_replay_wkv_weight(name, value, params))
        fused_param = dict(fused.named_parameters())[suffix]
        fused_param.weight_loader(fused_param, value, 1)
        torch.testing.assert_close(
            param.float(), expected_param.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused_param[-param.shape[0] :].float(), param.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused_param[: -param.shape[0]].float(),
            torch.full_like(
                fused_param[: -param.shape[0]],
                127 if param.dtype == torch.uint8 else 1,
                dtype=torch.float32,
            ),
            rtol=0,
            atol=0,
        )
        # Other layers, query weights and compressor weights must not alias WKV.
        for other in (
            name.replace("layers.20", "layers.19"),
            name.replace("attn.wkv", "attn.wq_a"),
            name.replace("attn.wkv", "attn.compressor.wkv"),
        ):
            assert (
                load_replay_wkv_weight(other, torch.zeros_like(value), params) is None
            )
        torch.testing.assert_close(
            param.float(), expected_param.float(), rtol=0, atol=0
        )
    assert loaded == set(params)
    device = current_platform.device_type
    for projection in (fused, replay):
        projection.to(device)
        projection.quant_method.process_weights_after_loading(projection)
    hidden = torch.randn(17, 256, device=device, dtype=torch.bfloat16) / 16
    expected = fused(hidden)[0][:, 1024:]
    torch.testing.assert_close(replay(hidden)[0], expected)


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
