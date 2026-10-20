# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct Beam from a synthetic config and generate with dummy weights.

Exercises NoPE + SSMax, hybrid sliding-window attention, the sigmoid output
gate and its per-head bias, sandwich norm, depth-scaled residuals, grouped
sigmoid routing, and logit softcapping without downloading a checkpoint.
"""

import json
from dataclasses import replace
from operator import attrgetter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import LLM, SamplingParams
from vllm.models.beam.nvidia import model as beam
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)

# `moe_intermediate_size` must be a multiple of 128 for the BF16 MoE kernel.
TINY_CONFIG = {
    "architectures": ["BeamForCausalLM"],
    "model_type": "beam",
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "moe_intermediate_size": 256,
    "num_experts_per_tok": 2,
    "num_experts": 8,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": True,
    # Layer 1 is NoPE, so it takes the SSMax path and full attention while
    # layer 0 stays RoPE + sliding. Exercises both branches in one model.
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    },
    "mlp_layer_types": ["dense", "sparse"],
    "final_logit_softcapping": 30.0,
    "torch_dtype": "bfloat16",
    "bos_token_id": 1,
    "eos_token_id": 2,
}


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Beam targets CUDA")
@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize(
    "enforce_eager,tp_size,num_experts,enable_ep",
    [
        (True, 1, 8, False),
        (False, 1, 8, False),
        (True, 4, 2, False),
        (True, 4, 8, True),
    ],
)
def test_beam_builds_and_generates(
    tmp_path, enforce_eager, tp_size, num_experts, enable_ep
):
    """Release-style metadata must load and generate in both execution modes."""
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    config = {
        **TINY_CONFIG,
        "num_experts": num_experts,
        "moe_intermediate_size": max(256, 128 * tp_size),
        "n_group": 1 if num_experts == 2 else 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    llm = LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=16 * 1024**2,
        gpu_memory_utilization=0.25,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
    )
    outputs = llm.generate(
        {"prompt_token_ids": [5, 6, 7, 8]},
        SamplingParams(max_tokens=8, ignore_eos=True),
    )

    assert len(outputs[0].outputs[0].token_ids) == 8
    # TP shards every expert; only EP partitions the expert set across ranks.
    local_experts = num_experts // tp_size if enable_ep else num_experts
    assert llm.apply_model(attrgetter("num_local_physical_experts")) == (
        [local_experts] * tp_size
    )


def test_eplb_is_rejected_until_validated(monkeypatch):
    """Expert parallelism is validated; expert rebalancing is not."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=BeamConfig()),
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=True, use_sequence_parallel_moe=False
        ),
    )
    with pytest.raises(NotImplementedError, match="EPLB"):
        beam.BeamForCausalLM(vllm_config=config)


def test_sequence_parallel_moe_is_rejected_until_validated():
    """DP attention with EP MoE takes the sequence-parallel path; unvalidated."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=BeamConfig()),
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False, use_sequence_parallel_moe=True
        ),
    )
    with pytest.raises(NotImplementedError, match="sequence-parallel"):
        beam.BeamForCausalLM(vllm_config=config)


def test_dense_only_pipeline_stage_initializes(monkeypatch):
    """A stage with dense layers and remote MoE layers needs no local experts."""
    layer = beam.BeamDecoderLayer.__new__(beam.BeamDecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = nn.Identity()
    stage = nn.Module()
    stage.layers = nn.ModuleList([layer, beam.PPMissingLayer()])
    stage.make_empty_intermediate_tensors = lambda *args, **kwargs: None
    monkeypatch.setattr(beam, "BeamModel", lambda **kwargs: stage)
    monkeypatch.setattr(
        beam, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(beam, "LogitsProcessor", lambda *args: nn.Identity())
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=BeamConfig()),
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False, use_sequence_parallel_moe=False
        ),
    )

    model = beam.BeamForCausalLM(vllm_config=config)

    assert model.moe_layers == []
    assert model.num_moe_layers == 0
    assert model.num_logical_experts == 0
    assert model.num_local_physical_experts == 0


class _AddOneLayer(nn.Module):
    """Adds one to the residual stream and hands the next layer double that."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()

    def forward(self, positions, hidden_states, normalized_states, next_norm_weight):
        hidden_states = hidden_states + 1
        return hidden_states, hidden_states * 2


@pytest.mark.cpu_test
def test_dflash_target_layers_capture_layer_outputs(monkeypatch):
    """DFlash `target_layer_ids` name the layer whose output is captured."""
    model = beam.BeamModel.__new__(beam.BeamModel)
    nn.Module.__init__(model)
    model.start_layer, model.end_layer = 0, 3
    model.layers = nn.ModuleList([_AddOneLayer() for _ in range(3)])
    model.norm = nn.Identity()
    target = beam.BeamForCausalLM.__new__(beam.BeamForCausalLM)
    nn.Module.__init__(target)
    target.model = model
    monkeypatch.setattr(
        beam,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [0, 1]})
        )
    )

    set_eagle3_aux_hidden_state_layers(target, spec_config)
    output, aux_hidden_states = target(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    torch.testing.assert_close(output, torch.tensor([[8.0, 10.0]]))
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[3.0, 4.0]]))


def _deferred_moe_config(**parallel_overrides):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    parallel = replace(FusedMoEParallelConfig.make_no_parallel(), **parallel_overrides)
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=1024,
        intermediate_size=1024,
        num_local_experts=8 // parallel.ep_size,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        skip_final_all_reduce=True,
    )


@pytest.mark.parametrize(
    "allow_ep,parallel_overrides,padded,expected",
    [
        (False, {}, False, False),
        (True, {}, False, True),
        (True, {"dp_size": 2}, False, False),
        (True, {"pcp_size": 2}, False, False),
        (True, {"sp_size": 4}, False, False),
        (True, {}, True, False),
    ],
)
def test_moe_finalize_deferral_requires_replicated_tokens(
    allow_ep, parallel_overrides, padded, expected
):
    """EP deferral requires explicit consumer support and unpadded replicated tokens."""
    config = _deferred_moe_config(ep_size=4, use_ep=True, **parallel_overrides)
    if padded:
        config.hidden_dim_unpadded -= 1

    config.defer_moe_finalize(4095, allow_ep=allow_ep)

    assert config.should_defer_moe_finalize(64) is expected
    assert config.should_defer_moe_finalize(4095) is expected
    assert not config.should_defer_moe_finalize(4096)
    assert not config.should_defer_moe_finalize(0)


@pytest.mark.parametrize(
    "world_size,ep_size,max_tokens,finalize_limit,backend_limit,expected_limit,enabled",
    [
        (2, 2, 16384, 8192, -1, 8192, True),
        (4, 4, 131072, 65536, -1, 65536, True),
        (8, 8, 16384, 8192, -1, 8192, True),
        (4, 4, 1024, 65536, -1, 1024, True),
        (4, 4, 16384, 65536, 128, 128, True),
        (4, 1, 131072, 65536, -1, 131072, True),
        (4, 8, 8192, 65536, -1, 8192, False),
    ],
)
def test_decoder_finalizes_large_ep_batches_before_fused_allreduce(
    monkeypatch,
    world_size,
    ep_size,
    max_tokens,
    finalize_limit,
    backend_limit,
    expected_limit,
    enabled,
):
    """Large EP batches return tensors; TP and smaller workspace limits are retained."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    moe_config = _deferred_moe_config(
        tp_size=world_size if ep_size == 1 else 1,
        ep_size=ep_size,
        use_ep=ep_size > 1,
    )
    moe_config.limit_deferred_moe_finalize(backend_limit)

    def init_moe(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.experts = SimpleNamespace(
            moe_config=moe_config,
            routed_scaling_factor=2.5,
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(experts_cls=TrtLlmNvFp4ExpertsMonolithic)
            ),
        )

    monkeypatch.setattr(beam.BeamSparseMoeBlock, "__init__", init_moe)
    monkeypatch.setattr(beam, "BeamAttention", lambda *args, **kwargs: nn.Identity())
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=BeamConfig(**TINY_CONFIG)),
        cache_config=None,
        quant_config=None,
    )
    workspace = SimpleNamespace(
        world_size=world_size,
        max_tokens=max_tokens,
        ep_finalize_max_tokens=finalize_limit,
        ep_finalize_oneshot_max_tokens=64,
    )

    beam.BeamDecoderLayer(config, "model.layers.1", workspace)

    assert moe_config.should_defer_moe_finalize(expected_limit) is enabled
    assert not moe_config.should_defer_moe_finalize(expected_limit + 1)
