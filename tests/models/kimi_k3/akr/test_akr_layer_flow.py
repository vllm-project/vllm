# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control flow of the deferred all-reduce in `KimiDecoderLayer.forward_attn_residual`
(single GPU, TP=1: the all-reduce is the identity): with the `_akr_*_deferred` flags set
(o_proj / MoE return TP partials) the layer must return exactly what the stock flow
returns, on block-write and ordinary layers, with and without an incoming prefix_delta.
"""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

H, NB = 7168, 8


# the repo conftest tears the groups down after every test
@pytest.fixture(autouse=True)
def _dist():
    from tests.utils import (
        ensure_current_vllm_config,
        init_test_distributed_environment,
    )
    from vllm.utils.network_utils import get_open_port

    with ensure_current_vllm_config():
        init_test_distributed_environment(1, 1, 0, str(get_open_port()), local_rank=0)
    yield


def _fake_layer(layer_idx, deferred_attn, prev_deferred, dev):
    from vllm.utils.math_utils import cdiv

    g = torch.Generator(device=dev).manual_seed(layer_idx)
    norm = lambda: types.SimpleNamespace(  # noqa: E731
        weight=(1 + 0.1 * torch.randn(H, device=dev, generator=g)).bfloat16(),
        variance_epsilon=1e-5,
    )
    proj = lambda: types.SimpleNamespace(  # noqa: E731
        weight=(torch.randn(1, H, device=dev, generator=g) * 0.05).bfloat16()
    )
    block = 12
    attn_w = (torch.randn(H, device=dev, generator=g) * 0.3).bfloat16()
    mlp_w = (torch.randn(H, device=dev, generator=g) * 0.2).bfloat16()
    fake = types.SimpleNamespace(
        layer_idx=layer_idx,
        self_attn=object(),
        mlp=lambda x: x * 0.5 + mlp_w,
        is_block_write_layer=layer_idx % block == 0,
        block_write_idx=layer_idx // block,
        prev_valid_blocks=cdiv(layer_idx, block),
        self_attention_res_proj=proj(),
        self_attention_res_norm=norm(),
        mlp_res_proj=proj(),
        mlp_res_norm=norm(),
        input_layernorm=norm(),
        post_attention_layernorm=norm(),
        _akr_attn_deferred=deferred_attn,
        _akr_prev_deferred=prev_deferred,
        _akr_mlp_deferred=False,
    )
    fake._run_self_attn = lambda positions, h: h * 0.25 + attn_w
    return fake


@pytest.mark.parametrize("layer_idx", [0, 1, 11, 12, 13, 24])
@pytest.mark.parametrize("m", [12, 80])
def test_deferred_flow_equals_stock(layer_idx, m):
    from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(5)
    prefix0 = (torch.randn(m, H, device=dev, generator=g) * 0.5).bfloat16()
    bank0 = (torch.randn(m, NB, H, device=dev, generator=g) * 0.5).bfloat16()
    delta = (
        None
        if layer_idx == 0
        else (torch.randn(m, H, device=dev, generator=g) * 0.3).bfloat16()
    )
    outs = []
    for deferred in (False, True):
        fake = _fake_layer(layer_idx, deferred, deferred and delta is not None, dev)
        p, b = prefix0.clone(), bank0.clone()
        outs.append(
            (KimiDecoderLayer.forward_attn_residual(fake, None, p, b, delta), b)
        )
    (a, ba), (c, bc) = outs
    for x, y in zip(a, c):
        assert torch.equal(x, y)
    assert torch.equal(ba, bc)
