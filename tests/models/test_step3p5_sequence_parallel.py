# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel MoE tests for Step3.5/Step3.7 and Step3p5MTP."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.models import step3p5, step3p5_mtp
from vllm.sequence import IntermediateTensors

HIDDEN, INTERMEDIATE, MOE_INTERMEDIATE, SHARE_INTERMEDIATE = 8, 16, 12, 16
TP_SIZE = 2


class _Attention(nn.Module):
    def __init__(self, *args, reduce_results: bool = True, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=reduce_results)
        self.num_tokens = 0

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        del positions
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _LocalExperts(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_num_tokens = 0

    def forward(
        self, *, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        assert hidden_states.shape == router_logits.shape
        self.last_num_tokens = hidden_states.shape[0]
        return self.proj(hidden_states)


def _make_step3p5_config(mlp_layer_types: list[str]) -> PreTrainedConfig:
    return PreTrainedConfig(
        model_type="step3p5",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        share_expert_dim=SHARE_INTERMEDIATE,
        moe_num_experts=4,
        moe_top_k=2,
        norm_expert_weight=True,
        moe_router_scaling_factor=1.0,
        use_moe_router_bias=True,
        need_fp32_gate=True,
        swiglu_limits=None,
        swiglu_limits_shared=None,
        num_hidden_layers=len(mlp_layer_types),
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        vocab_size=32,
        tie_word_embeddings=False,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
        },
        layer_types=["full_attention"] * len(mlp_layer_types),
        mlp_layer_types=mlp_layer_types,
    )


def _fake_tensor_parallel_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate rank 1 of TP=2 where TP all-reduce must never run on SP shards."""

    def _forbidden_all_reduce(*args, **kwargs):
        raise AssertionError("tensor-parallel all-reduce on a sequence-parallel shard")

    for mod in (linear, parameter, step3p5):
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
        )
    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        linear, "tensor_model_parallel_all_reduce", _forbidden_all_reduce
    )


def _reference_mlp(
    mlp: step3p5.Step3p5MLP, x: torch.Tensor, intermediate_size: int
) -> torch.Tensor:
    gate_up = x @ mlp.gate_up_proj.weight.T
    gate, up = gate_up.split(intermediate_size, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.T


def test_step3p5_mlp_runs_locally_on_sequence_parallel_shard(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
) -> None:
    _fake_tensor_parallel_world(monkeypatch)
    config = _make_step3p5_config(["sparse"])

    sharded = step3p5.Step3p5MLP(
        config=config,
        hidden_size=HIDDEN,
        intermediate_size=SHARE_INTERMEDIATE,
        hidden_act="silu",
        reduce_results=False,
        prefix="model.layers.0.moe.share_expert",
    )
    assert sharded.down_proj.tp_size == TP_SIZE
    assert sharded.gate_up_proj.weight.shape == (
        2 * SHARE_INTERMEDIATE // TP_SIZE,
        HIDDEN,
    )

    replicated = step3p5.Step3p5MLP(
        config=config,
        hidden_size=HIDDEN,
        intermediate_size=SHARE_INTERMEDIATE,
        hidden_act="silu",
        reduce_results=False,
        is_sequence_parallel=True,
        prefix="model.layers.0.moe.share_expert",
    )
    assert replicated.down_proj.tp_size == 1
    assert replicated.gate_up_proj.weight.shape == (2 * SHARE_INTERMEDIATE, HIDDEN)
    assert replicated.down_proj.weight.shape == (HIDDEN, SHARE_INTERMEDIATE)

    torch.manual_seed(0)
    for param in replicated.parameters():
        param.data.normal_()
    shard = torch.randn(3, HIDDEN)
    expected = _reference_mlp(replicated, shard, SHARE_INTERMEDIATE)
    for submod in (replicated.gate_up_proj, replicated.down_proj):
        assert submod.quant_method is not None
        submod.quant_method.process_weights_after_loading(submod)
    torch.testing.assert_close(replicated(shard), expected)


@pytest.mark.parametrize("use_sequence_parallel_moe", [False, True])
def test_step3p5_decoder_layer_constructor_wires_sequence_parallel_moe(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    use_sequence_parallel_moe: bool,
) -> None:
    _fake_tensor_parallel_world(monkeypatch)
    monkeypatch.setattr(step3p5, "Step3p5Attention", _Attention)
    monkeypatch.setattr(
        step3p5,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: TP_SIZE)),
    )
    monkeypatch.setattr(step3p5, "get_dp_group", lambda: SimpleNamespace(world_size=2))

    captured_moe_kwargs: dict[str, object] = {}

    def _fake_moe_factory(**kwargs):
        captured_moe_kwargs.update(kwargs)
        return _LocalExperts(HIDDEN)

    monkeypatch.setattr(step3p5, "FusedMoEFactory", _fake_moe_factory)

    config = _make_step3p5_config(["sparse", "dense"])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
    )

    moe_layer = step3p5.Step3p5DecoderLayer(vllm_config, prefix="model.layers.0")
    dense_layer = step3p5.Step3p5DecoderLayer(vllm_config, prefix="model.layers.1")

    assert moe_layer.use_sequence_parallel_moe is use_sequence_parallel_moe
    assert moe_layer.self_attn.o_proj.reduce_results is (not use_sequence_parallel_moe)
    assert moe_layer.moe.is_sequence_parallel is use_sequence_parallel_moe
    assert captured_moe_kwargs.get("is_sequence_parallel") is use_sequence_parallel_moe
    assert moe_layer.moe.share_expert.down_proj.tp_size == (
        1 if use_sequence_parallel_moe else TP_SIZE
    )

    assert dense_layer.use_sequence_parallel_moe is False
    assert dense_layer.self_attn.o_proj.reduce_results is True
    assert dense_layer.mlp.down_proj.tp_size == TP_SIZE


def test_step3p5_model_and_mtp_sequence_parallel_forward_equivalence(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
) -> None:
    """Two-rank TP=2 lockstep check across MoE->Dense->MoE, PP boundary, and MTP."""
    rank_local = threading.local()
    barrier = threading.Barrier(TP_SIZE)
    exchange: list[torch.Tensor | None] = [None, None]

    def _tp_rank() -> int:
        return rank_local.rank

    def _all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        r = _tp_rank()
        exchange[r] = tensor.clone()
        barrier.wait()
        out = exchange[0] + exchange[1]  # type: ignore[operator]
        barrier.wait()
        return out

    def _all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        r = _tp_rank()
        exchange[r] = tensor.clone()
        barrier.wait()
        out = torch.cat([exchange[0], exchange[1]], dim=dim)  # type: ignore[list-item]
        barrier.wait()
        return out

    def _reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        total = _all_reduce(tensor)
        return total.chunk(TP_SIZE, dim=dim)[_tp_rank()].contiguous()

    def _sp_chunk(tensor: torch.Tensor) -> torch.Tensor:
        pad = (-tensor.shape[0]) % TP_SIZE
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
        return padded.chunk(TP_SIZE, dim=0)[_tp_rank()].contiguous()

    for mod in (step3p5, step3p5_mtp):
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda: TP_SIZE, raising=False
        )
        monkeypatch.setattr(
            mod, "tensor_model_parallel_all_gather", _all_gather, raising=False
        )
        monkeypatch.setattr(
            mod,
            "tensor_model_parallel_reduce_scatter",
            _reduce_scatter,
            raising=False,
        )
        monkeypatch.setattr(mod, "sequence_parallel_chunk", _sp_chunk, raising=False)

    torch.manual_seed(7)
    num_tokens = 5  # Odd token count exercises SP pad + trim.
    positions = torch.arange(num_tokens)
    inputs_embeds = torch.randn(num_tokens, HIDDEN)
    prev_hidden = torch.randn(num_tokens, HIDDEN)

    attn_weights = [torch.randn(HIDDEN, HIDDEN) for _ in range(3)]
    dense_w = torch.randn(HIDDEN, HIDDEN)
    moe_weights = [torch.randn(HIDDEN, HIDDEN), None, torch.randn(HIDDEN, HIDDEN)]

    def _build_layer(
        layer_idx: int, is_moe: bool, use_sp_moe: bool
    ) -> step3p5.Step3p5DecoderLayer:
        layer = object.__new__(step3p5.Step3p5DecoderLayer)
        nn.Module.__init__(layer)
        layer.hidden_size = HIDDEN
        layer.layer_idx = layer_idx
        layer.use_moe = is_moe
        layer.use_sequence_parallel_moe = use_sp_moe and is_moe
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()

        w_attn_shard = attn_weights[layer_idx].chunk(TP_SIZE, dim=0)[_tp_rank()]
        reduce_attn = not layer.use_sequence_parallel_moe

        class _ShardAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen_tokens = 0

            def forward(
                self, positions: torch.Tensor, hidden_states: torch.Tensor
            ) -> torch.Tensor:
                del positions
                self.seen_tokens = hidden_states.shape[0]
                x_shard = hidden_states.chunk(TP_SIZE, dim=-1)[_tp_rank()]
                partial = x_shard @ w_attn_shard
                return _all_reduce(partial) if reduce_attn else partial

        layer.self_attn = _ShardAttn()
        if is_moe:
            moe_block = object.__new__(step3p5.FusedMoEBlock)
            nn.Module.__init__(moe_block)
            moe_block.is_sequence_parallel = use_sp_moe
            experts = _LocalExperts(HIDDEN)
            experts.proj.weight = nn.Parameter(
                moe_weights[layer_idx].clone(),  # type: ignore[union-attr]
                requires_grad=False,
            )
            moe_block.experts = experts
            layer.moe = moe_block
        else:
            w_dense_shard = dense_w.chunk(TP_SIZE, dim=0)[_tp_rank()]

            class _ShardDense(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.seen_tokens = 0

                def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                    self.seen_tokens = hidden_states.shape[0]
                    x_shard = hidden_states.chunk(TP_SIZE, dim=-1)[_tp_rank()]
                    return _all_reduce(x_shard @ w_dense_shard)

            layer.mlp = _ShardDense()
        return layer

    def _run_two_ranks(use_sp_moe: bool, is_last_pp_rank: bool):
        results: list[tuple[torch.Tensor, list[int], torch.Tensor] | None] = [
            None,
            None,
        ]
        errors: list[BaseException] = []

        eh_weight = torch.eye(HIDDEN, HIDDEN * 2)

        def _worker(rank: int) -> None:
            rank_local.rank = rank
            try:
                model = object.__new__(step3p5.Step3p5Model)
                nn.Module.__init__(model)
                model.do_not_compile = True
                model.start_layer = 0
                model.end_layer = 3
                model.layers = nn.ModuleList(
                    [
                        _build_layer(0, is_moe=True, use_sp_moe=use_sp_moe),
                        _build_layer(1, is_moe=False, use_sp_moe=use_sp_moe),
                        _build_layer(2, is_moe=True, use_sp_moe=use_sp_moe),
                    ]
                )
                out = model(
                    input_ids=torch.zeros(num_tokens, dtype=torch.long),
                    positions=positions,
                    intermediate_tensors=None,
                    inputs_embeds=inputs_embeds.clone(),
                )
                if is_last_pp_rank:
                    assert isinstance(out, torch.Tensor)
                    model_tensor = out
                else:
                    assert isinstance(out, IntermediateTensors)
                    model_tensor = out["hidden_states"]

                seen = [
                    model.layers[0].moe.experts.last_num_tokens,
                    model.layers[1].mlp.seen_tokens,
                    model.layers[2].moe.experts.last_num_tokens,
                ]

                mtp_layer = object.__new__(step3p5_mtp.Step3p5AMultiTokenPredictorLayer)
                nn.Module.__init__(mtp_layer)
                mtp_layer.enorm = nn.Identity()
                mtp_layer.hnorm = nn.Identity()
                mtp_layer.eh_proj = nn.Linear(HIDDEN * 2, HIDDEN, bias=False)
                mtp_layer.eh_proj.weight = nn.Parameter(
                    eh_weight.clone(), requires_grad=False
                )
                mtp_layer.mtp_block = _build_layer(
                    0, is_moe=True, use_sp_moe=use_sp_moe
                )
                mtp_out = mtp_layer(
                    input_ids=torch.zeros(num_tokens, dtype=torch.long),
                    positions=positions,
                    previous_hidden_states=prev_hidden.clone(),
                    inputs_embeds=inputs_embeds.clone(),
                )
                results[rank] = (model_tensor, seen, mtp_out)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                barrier.abort()

        monkeypatch.setattr(
            step3p5,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=is_last_pp_rank),
        )
        threads = [
            threading.Thread(target=_worker, args=(rank,)) for rank in range(TP_SIZE)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        assert results[0] is not None and results[1] is not None
        return results[0], results[1]

    for is_last_pp_rank in (True, False):
        barrier.reset()
        (ref_r0, ref_seen, ref_mtp0), (ref_r1, _, ref_mtp1) = _run_two_ranks(
            use_sp_moe=False, is_last_pp_rank=is_last_pp_rank
        )
        assert ref_seen == [5, 5, 5]
        torch.testing.assert_close(ref_r0, ref_r1)

        barrier.reset()
        (sp_r0, sp_seen, sp_mtp0), (sp_r1, _, sp_mtp1) = _run_two_ranks(
            use_sp_moe=True, is_last_pp_rank=is_last_pp_rank
        )
        assert sp_seen == [3, 5, 3]
        assert sp_r0.shape == (num_tokens, HIDDEN)
        assert sp_mtp0.shape == (num_tokens, HIDDEN)
        torch.testing.assert_close(sp_r0, ref_r0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_r1, ref_r1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_mtp0, ref_mtp0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_mtp1, ref_mtp1, atol=1e-5, rtol=1e-5)
