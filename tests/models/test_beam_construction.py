# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct Beam from a synthetic config and generate with dummy weights.

Exercises NoPE + SSMax, hybrid sliding-window attention, the sigmoid output
gate and its per-head bias, sandwich norm, depth-scaled residuals, grouped
sigmoid routing, and logit softcapping without downloading a checkpoint.
"""

import json
from operator import attrgetter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import LLM, SamplingParams
from vllm.models.beam import model as beam
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
    # Keep a supported attention head size on both NVIDIA and AMD backends.
    "hidden_size": 512,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Beam targets CUDA and ROCm"
)
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
