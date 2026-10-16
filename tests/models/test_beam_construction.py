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
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.beam.nvidia import model as beam
from vllm.models.beam.nvidia import moe as beam_moe
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
    "rope_theta": 10000.0,
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


@pytest.mark.parametrize("scale,expected", [(None, 1.0), (0.0, 0.0), (2.5, 2.5)])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_moe_preserves_explicit_routed_scaling(
    monkeypatch, scale, expected, sequence_parallel
):
    """Only an absent scale defaults to one; zero must disable routed output."""

    def gate(*args, out_dtype, **kwargs):
        result = nn.Module()
        result.out_dtype = out_dtype
        return result

    def experts(**kwargs):
        assert kwargs["routed_scaling_factor"] == expected
        assert kwargs["apply_routed_scale_to_output"] is True
        assert kwargs["is_sequence_parallel"] is sequence_parallel
        assert kwargs["reduce_results"] is sequence_parallel
        return nn.Identity()

    monkeypatch.setattr(beam_moe, "GateLinear", gate)
    monkeypatch.setattr(beam_moe, "FusedMoEFactory", experts)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=BeamConfig(**{**TINY_CONFIG, "routed_scaling_factor": scale})
        ),
        quant_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=sequence_parallel,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
    )
    beam_moe.BeamSparseMoeBlock(config, reduce_results=False)


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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "parallel_overrides,supported,padded,expected",
    [
        ({}, True, False, True),
        ({"dp_size": 4, "ep_size": 4}, True, False, True),
        ({"dp_size": 1, "ep_size": 1}, True, False, False),
        ({"tp_size": 2}, True, False, False),
        ({"pcp_size": 2}, True, False, False),
        ({}, False, False, False),
        ({}, True, True, False),
    ],
)
def test_dp_moe_boundary_fusion_requires_supported_topology(
    monkeypatch, parallel_overrides, supported, padded, expected
):
    """DP owns its returned tokens and defers only with a full boundary consumer."""
    from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL
    from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
        FlashInferMoeEpExperts,
    )

    parallel = {"dp_size": 2, "ep_size": 2, "use_ep": True, **parallel_overrides}
    moe = replace(
        _deferred_moe_config(**parallel),
        moe_backend=FLASHINFER_MOE_EP_CUTEDSL,
        skip_final_all_reduce=False,
    )
    if padded:
        moe.hidden_dim_unpadded -= 1
    block = beam_moe.BeamSparseMoeBlock.__new__(beam_moe.BeamSparseMoeBlock)
    nn.Module.__init__(block)
    block.is_sequence_parallel = False
    block.gather_workspace = None
    block.output_boundary_fused = False
    block._boundary_layer_name = "moe.finalize"
    block._compilation_config = SimpleNamespace(static_forward_context={})
    block.experts = SimpleNamespace(
        moe_config=moe,
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(experts_cls=FlashInferMoeEpExperts),
            apply_router_weight_on_input=False,
        ),
        routed_scaling_factor=2.5,
        shared_experts=None,
        routed_input_transform=None,
        routed_output_transform=None,
    )
    monkeypatch.setattr(
        flashinfer_moe_ep,
        "supports_flashinfer_moe_ep_deferred_finalize",
        lambda: supported,
    )

    assert block.fuse_output_post_norm(None, 1e-6) is expected
    assert block.output_boundary_fused is expected
    assert moe.use_deferred_moe_finalize is expected
    assert block.experts.routed_scaling_factor == 2.5
    assert block.gather_workspace is None
    if expected:
        assert block._compilation_config.static_forward_context["moe.finalize"] is block
    else:
        assert not block._compilation_config.static_forward_context

    moe.defer_moe_finalize(allow_ep=True)
    assert moe.use_deferred_moe_finalize is expected
    assert moe.should_defer_moe_finalize(1) is expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("next_norm", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("rows", [0, 5])
def test_moe_boundary_consumes_borrowed_outputs_in_opaque_op(
    monkeypatch, next_norm, sequence_parallel, rows
):
    """The boundary op preserves raw-output ownership and explicit residual inputs."""
    from vllm.models.beam.nvidia import moe_finalize

    block = beam_moe.BeamSparseMoeBlock.__new__(beam_moe.BeamSparseMoeBlock)
    nn.Module.__init__(block)
    block.is_sequence_parallel = sequence_parallel
    block.output_boundary_fused = True
    block.gather_workspace = (
        SimpleNamespace(world_size=2) if sequence_parallel else None
    )
    local_rows = (rows + 1) // 2 if sequence_parallel else rows
    partials = torch.randn(local_rows * 2, 16)
    weights = torch.randn(local_rows, 2)
    mapping = torch.arange(local_rows * 2, dtype=torch.int32).view(local_rows, 2)
    routed_inputs = []
    calls = []

    def routed(hidden_states, router_logits, shared_experts_input):
        assert hidden_states is router_logits
        assert shared_experts_input is None
        routed_inputs.append(hidden_states)
        return UnfinalizedMoEOutput(partials, weights, mapping)

    encoded_name = beam_moe._layer_name_type("moe")
    block.experts = SimpleNamespace(
        layer_name="moe",
        _encode_layer_name=lambda: encoded_name,
        _forward_impl=routed,
        routed_scaling_factor=2.5,
    )
    monkeypatch.setattr(beam_moe, "get_layer_from_name", lambda name: block.experts)
    monkeypatch.setattr(
        beam_moe,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"moe.finalize": block}),
    )
    monkeypatch.setattr(
        beam_moe, "sequence_parallel_chunk", lambda x: x[: (x.shape[0] + 1) // 2]
    )

    def finalize(x, residual, *args, **kwargs):
        assert x.gemm2_permuted is partials and x.expert_weights is weights
        assert x.expanded_idx_to_permuted_idx is mapping
        calls.append((args, kwargs))
        return residual + 1, residual + 2 if args[3] is not None else None

    finalizer = (
        "beam_moe_finalize_gather_norm_residual_norm"
        if sequence_parallel
        else "beam_moe_finalize_norm_residual_norm"
    )
    monkeypatch.setattr(moe_finalize, finalizer, finalize)
    forward = block.forward_boundary
    hidden_states, residual = torch.randn(rows, 16), torch.randn(rows, 16)
    post_weight = torch.randn(16)
    next_weight = torch.randn(16) if next_norm else None
    with torch.library._scoped_library("vllm", "IMPL") as lib:
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "vllm::beam_moe_forward_boundary", "CPU"
        ):
            lib.impl(
                "beam_moe_forward_boundary",
                beam_moe._beam_moe_forward_boundary,
                "CPU",
            )
        result, normalized = forward(
            hidden_states,
            residual,
            post_weight,
            1e-6,
            next_weight,
            1e-6 if next_norm else None,
            (0.25, 0.75),
        )
    torch.testing.assert_close(result, residual + 1)
    if next_norm:
        torch.testing.assert_close(normalized, residual + 2)
    else:
        assert normalized is None
    torch.testing.assert_close(routed_inputs[0], hidden_states[:local_rows])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is post_weight and args[2] is next_weight
    assert args[4] == (0.25, 0.75)
    if sequence_parallel:
        assert args[5] is block.gather_workspace
    else:
        assert len(args) == 5
    assert kwargs == {"routed_scale": 2.5}


@pytest.mark.cpu_test
@pytest.mark.parametrize("defer_output", [False, True])
def test_sp_moe_combine_stays_reduced_when_finalization_is_deferred(defer_output):
    """Deferring the local expert sum must never introduce an EP all-reduce."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
        PassThroughPrepareAndFinalize,
    )

    prepare_finalize = PassThroughPrepareAndFinalize()
    prepare_finalize.post_init_setup(
        SimpleNamespace(
            moe_config=SimpleNamespace(use_deferred_moe_finalize=defer_output)
        )
    )
    assert prepare_finalize.output_is_reduced()
    assert prepare_finalize.supports_deferred_moe_finalize() is defer_output


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
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_decoder_finalizes_large_ep_batches_before_flashinfer_allreduce(
    monkeypatch,
    world_size,
    ep_size,
    max_tokens,
    finalize_limit,
    backend_limit,
    expected_limit,
    enabled,
    sequence_parallel,
):
    """Large EP batches return tensors; TP and smaller workspace limits are retained."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    moe_config = _deferred_moe_config(
        tp_size=world_size if ep_size == 1 else 1,
        ep_size=ep_size,
        use_ep=ep_size > 1,
        sp_size=world_size if sequence_parallel else 1,
    )
    moe_config.limit_deferred_moe_finalize(backend_limit)

    def init_moe(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.is_sequence_parallel = sequence_parallel
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

    layer = beam.BeamDecoderLayer(config, "model.layers.1", workspace)

    assert moe_config.should_defer_moe_finalize(expected_limit) is (
        enabled and not sequence_parallel
    )
    assert not moe_config.should_defer_moe_finalize(expected_limit + 1)
    if sequence_parallel:
        assert layer.mlp_boundary is beam.beam_norm_residual_norm
        assert layer.finalize_boundary is None
    assert layer.attention_boundary.func is beam.beam_allreduce_norm_residual_norm
