# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.minimax_m3.amd.model as minimax_m3
import vllm.models.minimax_m3.amd.mtp as minimax_m3_mtp


class _Attention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states


class _Norm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return hidden_states
        residual = hidden_states + residual
        return residual + 100, residual


class _FFN(nn.Module):
    def __init__(self, *, reduce_results: bool, **kwargs) -> None:
        super().__init__()
        self.down_proj = SimpleNamespace(reduce_results=reduce_results)
        self.experts = SimpleNamespace(
            moe_config=SimpleNamespace(skip_final_all_reduce=not reduce_results)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _ReplicatedLinear(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _DeferredLayer(nn.Module):
    def __init__(self, deferred: bool) -> None:
        super().__init__()
        self.fuse_input_allreduce = False
        self.ffn_all_reduce_deferred = deferred


class _FinalDeferredLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        capture_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not capture_aux
        return torch.ones_like(hidden_states), torch.full_like(hidden_states, 10)


@pytest.mark.parametrize("force_moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize(
    ("pipeline_parallel_size", "is_mtp_block", "expected_deferred"),
    [
        pytest.param(1, False, True, id="target-tp-only"),
        pytest.param(2, False, False, id="pipeline-boundary"),
        pytest.param(1, True, False, id="mtp-output"),
    ],
)
def test_decoder_defers_only_when_the_next_layer_can_consume_it(
    monkeypatch,
    force_moe: bool,
    pipeline_parallel_size: int,
    is_mtp_block: bool,
    expected_deferred: bool,
) -> None:
    monkeypatch.setattr(minimax_m3, "MiniMaxM3Attention", _Attention)
    monkeypatch.setattr(minimax_m3, "MiniMaxM3SparseAttention", _Attention)
    monkeypatch.setattr(minimax_m3, "MiniMaxM3MLP", _FFN)
    monkeypatch.setattr(minimax_m3, "MiniMaxM3MoE", _FFN)
    monkeypatch.setattr(minimax_m3, "MiniMAXGemmaRMSNorm", _Norm)
    monkeypatch.setattr(minimax_m3, "_sparse_attention_layer_ids", lambda _: ())
    monkeypatch.setattr(minimax_m3, "_is_moe_layer", lambda *_: False)
    config = SimpleNamespace(
        hidden_size=8,
        dense_intermediate_size=16,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
    )
    layer = minimax_m3.MiniMaxM3DecoderLayer(
        vllm_config=vllm_config,
        prefix="model.layers.0",
        force_moe=force_moe,
        is_mtp_block=is_mtp_block,
    )

    assert layer.ffn_all_reduce_deferred is expected_deferred


def test_mtp_decoder_keeps_its_ffn_output_reduced(monkeypatch) -> None:
    decoder_kwargs = {}

    class _Decoder(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            decoder_kwargs.update(kwargs)

    monkeypatch.setattr(minimax_m3_mtp, "MiniMaxM3DecoderLayer", _Decoder)
    monkeypatch.setattr(minimax_m3_mtp, "MiniMAXGemmaRMSNorm", _Norm)
    monkeypatch.setattr(minimax_m3_mtp, "ReplicatedLinear", _ReplicatedLinear)
    config = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        quant_config=None,
    )

    minimax_m3_mtp.MiniMaxM3MultiTokenPredictorLayer(vllm_config, "mtp.layers.0")

    assert decoder_kwargs["vllm_config"] is vllm_config
    assert decoder_kwargs["is_mtp_block"] is True


@pytest.mark.parametrize(
    ("deferred", "expected_input_fusion", "expected_final_fusion"),
    [
        pytest.param((True, True), (False, True), True, id="both-deferred"),
        pytest.param((True, False), (False, True), False, id="first-deferred"),
        pytest.param((False, True), (False, False), True, id="last-deferred"),
        pytest.param((False, False), (False, False), False, id="none-deferred"),
    ],
)
def test_model_fusion_schedule_follows_effective_ffn_reduction(
    deferred: tuple[bool, bool],
    expected_input_fusion: tuple[bool, bool],
    expected_final_fusion: bool,
) -> None:
    layers = nn.ModuleList([_DeferredLayer(value) for value in deferred])
    final_fusion = minimax_m3._configure_cross_layer_allreduce(layers, 0, len(layers))

    assert tuple(layer.fuse_input_allreduce for layer in layers) == (
        expected_input_fusion
    )
    assert final_fusion is expected_final_fusion


def test_eagle_aux_capture_is_full_and_rank_invariant(monkeypatch) -> None:
    layer = minimax_m3.MiniMaxM3DecoderLayer.__new__(minimax_m3.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.fuse_input_allreduce = True
    layer.is_moe_layer = False
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    layer.self_attn = _Attention()
    layer.mlp = _FFN(reduce_results=False)

    input_residuals: list[torch.Tensor] = []
    all_rank_sum = torch.full((2, 3), 4.0)

    def fused_allreduce(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = all_rank_sum if norm is layer.input_layernorm else hidden_states * 4
        normed, combined = norm(reduced, residual)
        if norm is layer.input_layernorm:
            input_residuals.append(combined)
        return normed, combined

    monkeypatch.setattr(minimax_m3, "fused_allreduce_gemma_rms_norm", fused_allreduce)
    residual = torch.full((2, 3), 10.0)
    rank_partials = [torch.ones(2, 3), torch.full((2, 3), 3.0)]
    aux_hidden_states = []
    for partial in rank_partials:
        _, _, aux_hidden_state = layer(
            torch.arange(2), partial, residual, capture_aux=True
        )
        aux_hidden_states.append(aux_hidden_state)

    expected = torch.full((2, 3), 14.0)
    for aux_hidden_state, input_residual in zip(
        aux_hidden_states, input_residuals, strict=True
    ):
        torch.testing.assert_close(aux_hidden_state, expected)
        assert aux_hidden_state.data_ptr() != input_residual.data_ptr()
    torch.testing.assert_close(aux_hidden_states[0], aux_hidden_states[1])


def test_eagle_aux_capture_at_final_deferred_boundary(monkeypatch) -> None:
    model = minimax_m3.MiniMaxM3Model.__new__(minimax_m3.MiniMaxM3Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_FinalDeferredLayer()])
    model.start_layer = 0
    model.end_layer = 1
    model.aux_hidden_state_layers = (1,)
    model.fuse_final_norm_allreduce = True
    model.norm = _Norm()

    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(minimax_m3, "get_pp_group", lambda: pp_group)
    full_residuals: list[torch.Tensor] = []

    def fused_allreduce(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, full_residual = norm(hidden_states * 4, residual)
        full_residuals.append(full_residual)
        return normed, full_residual

    monkeypatch.setattr(minimax_m3, "fused_allreduce_gemma_rms_norm", fused_allreduce)
    inputs_embeds = torch.zeros(2, 3)
    hidden_states, aux_hidden_states = model(
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    expected_residual = torch.full((2, 3), 14.0)
    torch.testing.assert_close(hidden_states, expected_residual + 100)
    assert len(aux_hidden_states) == 1
    torch.testing.assert_close(aux_hidden_states[0], expected_residual)
    assert aux_hidden_states[0].data_ptr() != full_residuals[0].data_ptr()
