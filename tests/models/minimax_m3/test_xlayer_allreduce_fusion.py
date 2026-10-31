# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 (AMD) defers each FFN all-reduce into the next norm.

Every deferred all-reduce must be finished exactly once, by the next layer's
input norm or the final norm; an EAGLE3 capture must never record a per-rank
partial sum; and mono decode, whose kernels expect reduced layer outputs, must
turn deferral off. A tiny model on one rank checks the wiring, and at TP2 its
final hidden states and captures must match the model that reduces every FFN.
"""

import os
import zlib

import pytest
import ray
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 AMD model"
)

LAYERS = 5
DENSE = 2
_CKPT_ENV = "VLLM_TEST_M3_XLAYER_CKPT"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

    path = tmp_path_factory.mktemp("minimax_m3_tiny")
    no_sparse = [0] * LAYERS
    MiniMaxM3Config(
        text_config=dict(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=128,
            dense_intermediate_size=256,
            num_hidden_layers=LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            rotary_dim=64,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_layer_freq=[0] * DENSE + [1] * (LAYERS - DENSE),
            sparse_attention_config=dict(
                use_sparse_attention=True,
                sparse_index_dim=64,
                sparse_num_index_heads=1,
                sparse_topk_blocks=4,
                sparse_block_size=128,
                sparse_disable_index_value=no_sparse,
                sparse_score_type="max",
                sparse_init_block=0,
                sparse_local_block=1,
                sparse_attention_freq=no_sparse,
            ),
        ),
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
    ).save_pretrained(path)
    return str(path)


def _build(checkpoint):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    vllm_config = VllmConfig(model_config=ModelConfig(checkpoint, max_model_len=1024))
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        return MiniMaxM3Model(vllm_config=vllm_config, prefix="model")


def test_each_deferred_all_reduce_completes_once(dist_init, checkpoint):
    model = _build(checkpoint)
    deferred = [layer.ffn_all_reduce_deferred for layer in model.layers]
    assert all(deferred)
    assert [layer.fuse_input_allreduce for layer in model.layers] == [
        False,
        *deferred[:-1],
    ]
    assert model.fuse_final_norm_allreduce


def test_eagle3_captures_after_a_deferring_layer_are_reduced(dist_init, checkpoint):
    model = _build(checkpoint)
    assert not model._aux_needs_reduce(0)
    assert all(model._aux_needs_reduce(idx) for idx in range(1, LAYERS + 1))


@ray.remote(num_gpus=1, max_calls=1)
def _run_deferred_matches_reduced(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model
    from vllm.models.minimax_m3.amd.mono.envs import envs
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)

        def build(defer: bool):
            # Mono decode is the switch that turns deferral off; at TP2 it then
            # refuses the deployment, so both models run the same layer loop.
            m.setattr(envs, "MONO_ENABLE", not defer)
            model_config = ModelConfig(os.environ[_CKPT_ENV], max_model_len=1024)
            vllm_config = VllmConfig(model_config=model_config)
            with (
                set_current_vllm_config(vllm_config),
                set_default_torch_dtype(torch.bfloat16),
                device,
            ):
                model = MiniMaxM3Model(vllm_config=vllm_config, prefix="model")
            assert model.fuse_final_norm_allreduce is defer
            with torch.no_grad():
                for name, p in model.named_parameters():
                    g = torch.Generator(device=device).manual_seed(
                        zlib.crc32(f"{name}/{rank}".encode())
                    )
                    std = 0.1 if p.ndim == 1 else p.shape[-1] ** -0.5
                    p.copy_(torch.randn(p.shape, generator=g, device=device) * std)
            process_weights_after_loading(model, model_config, device)
            model._set_aux_hidden_state_layers(tuple(range(1, LAYERS + 1)))
            return model, vllm_config

        input_ids = torch.arange(37, device=device) * 7 % 256
        positions = torch.arange(37, device=device)
        outputs = []
        for defer in (True, False):
            model, vllm_config = build(defer)
            with torch.inference_mode(), set_forward_context(None, vllm_config):
                outputs.append(model(input_ids, positions, None))
        (got, got_aux), (ref, ref_aux) = outputs
        for a, b in zip([got, *got_aux], [ref, *ref_aux], strict=True):
            torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("use_aiter", [False, True])
def test_deferred_matches_reduced_at_tp2(checkpoint, monkeypatch, use_aiter):
    """The final hidden states and every EAGLE3 capture match the all-reduce-
    per-FFN model: a dropped or doubled deferred reduce is off by a TP factor."""
    tp_size = 2
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs.")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
    monkeypatch.setenv(_CKPT_ENV, checkpoint)
    multi_process_parallel(monkeypatch, tp_size, 1, _run_deferred_matches_reduced)


def test_mono_decode_keeps_every_ffn_reduced(dist_init, checkpoint, monkeypatch):
    pytest.importorskip("flydsl")
    from vllm.models.minimax_m3.amd.mono.envs import envs

    monkeypatch.setattr(envs, "MONO_ENABLE", True)
    model = _build(checkpoint)
    assert not any(
        layer.ffn_all_reduce_deferred or layer.fuse_input_allreduce
        for layer in model.layers
    )
    assert not model.fuse_final_norm_allreduce
