# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.xpu import kda as kimi_xpu_kda
from vllm.models.kimi_k3.xpu import linear as kimi_xpu
from vllm.models.kimi_k3.xpu.ops.attn_res import attn_res
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class _ResidualNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return hidden_states + 1
        return hidden_states + residual + 1, residual + hidden_states


class _Scale(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


class _WeightedNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-5


class _Projection(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, hidden_size))


class _TupleIdentity(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states, None


class _GatedAdd(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states + gate.unsqueeze(0)


class _ConstantProjection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.output, None


class _FakeMLA(nn.Module):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.full((1, 1), 4.0)


class _SingleRankPPGroup:
    is_first_rank = True
    is_last_rank = True


class _StandardModelLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        del positions
        if residual is None:
            residual = hidden_states * 10
        return hidden_states + 1, prefix_sum, residual


class _AttnResModelLayer(nn.Module):
    def __init__(self, expected_delta: torch.Tensor | None, increment: float) -> None:
        super().__init__()
        self.expected_delta = expected_delta
        self.increment = increment

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del positions
        if self.expected_delta is None:
            assert hidden_states is None
        else:
            torch.testing.assert_close(hidden_states, self.expected_delta)
        delta = prefix_sum.new_full(prefix_sum.shape, self.increment)
        return delta, prefix_sum + self.increment * 10, residual


class _FusedWeightModule(nn.Module):
    def __init__(self, calls: list[tuple[torch.Tensor, int]]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 2))

        def weight_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            shard_id: int,
        ) -> None:
            assert param is self.weight
            calls.append((loaded_weight, shard_id))

        self.weight.weight_loader = weight_loader  # type: ignore[attr-defined]


class _WeightLoaderLayer(nn.Module):
    def __init__(self, calls: list[tuple[torch.Tensor, int]]) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = _FusedWeightModule(calls)


@pytest.mark.parametrize(
    ("attention_cls", "gate_lower_bound"),
    [
        (kimi_xpu_kda.KimiK3DeltaAttention, -5.0),
        (kimi_xpu_kda.KimiLinearDeltaAttention, None),
    ],
)
def test_xpu_kda_adapter_dispatches_native_op(
    monkeypatch, attention_cls, gate_lower_bound
) -> None:
    attention = object.__new__(attention_cls)
    nn.Module.__init__(attention)
    attention.prefix = "model.layers.0.self_attn"
    attention.local_num_heads = 2
    attention.head_dim = 2
    attention.local_projection_size = 4
    attention.conv1d = nn.Linear(2, 12, bias=False)
    attention.conv1d.weight.data = attention.conv1d.weight.data.unsqueeze(1)
    attention.A_log = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    attention.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    attention.gate_lower_bound = gate_lower_bound
    attention.o_norm = _GatedAdd()
    conv_state = torch.zeros(1, 12, 1)
    recurrent_state = torch.zeros(1, 2, 2, 2)
    attention.kv_cache = (conv_state, recurrent_state)

    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        has_initial_state=torch.tensor([False]),
        non_spec_query_start_loc=torch.tensor([0, 2]),
        non_spec_state_indices_tensor=torch.tensor([0]),
    )
    forward_context = type(
        "ForwardContext",
        (),
        {"attn_metadata": {attention.prefix: metadata}},
    )()
    monkeypatch.setattr(kimi_xpu_kda, "get_forward_context", lambda: forward_context)

    captured: dict[str, object] = {}

    def fake_kda_attention(*args: object) -> None:
        captured["args"] = args
        output = args[0]
        assert isinstance(output, torch.Tensor)
        output.fill_(3)

    monkeypatch.setattr(
        torch.ops._xpu_C, "kda_attention", fake_kda_attention, raising=False
    )
    mixed_qkv = torch.arange(24, dtype=torch.bfloat16).view(2, 12)
    raw_gate = torch.ones(1, 2, 2, 2, dtype=torch.bfloat16)
    raw_beta = torch.ones(1, 2, 2, dtype=torch.bfloat16)
    output_gate = torch.full((2, 2, 2), 2, dtype=torch.bfloat16)
    output = torch.empty(1, 2, 2, 2, dtype=torch.bfloat16)

    attention._forward(mixed_qkv, raw_gate, output_gate, raw_beta, output)

    args = captured["args"]
    assert isinstance(args, tuple)
    q_proj, k_proj, v_proj = args[1:4]
    assert isinstance(q_proj, torch.Tensor)
    assert isinstance(k_proj, torch.Tensor)
    assert isinstance(v_proj, torch.Tensor)
    assert q_proj.shape == k_proj.shape == v_proj.shape == (2, 4)
    assert q_proj.stride() == k_proj.stride() == v_proj.stride() == (12, 1)
    assert args[5].dtype == torch.float32
    assert args[5].is_contiguous()
    assert args[6] is conv_state
    assert args[7] is recurrent_state
    assert args[13:16] == (1, 0, 0)
    assert args[16] is metadata.has_initial_state
    assert args[17] is metadata.non_spec_query_start_loc
    assert args[19] is metadata.non_spec_state_indices_tensor
    assert args[24:] == (2, gate_lower_bound)
    torch.testing.assert_close(output, torch.full_like(output, 5))


def test_xpu_decoder_layer_selects_full_rank_kda(monkeypatch) -> None:
    class _FakeLayer(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    class _FakeKDA(nn.Module):
        def __init__(
            self,
            config: KimiLinearConfig,
            vllm_config: object,
            prefix: str,
        ) -> None:
            super().__init__()
            self.config = config
            self.vllm_config = vllm_config
            self.prefix = prefix

    monkeypatch.setattr(kimi_xpu, "KimiK3DeltaAttention", _FakeKDA)
    monkeypatch.setattr(kimi_xpu, "KimiMLP", _FakeLayer)
    monkeypatch.setattr(kimi_xpu, "RMSNorm", _FakeLayer)
    config = KimiLinearConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "use_full_rank_gate": True,
        },
    )
    vllm_config = SimpleNamespace(quant_config=None)

    layer = kimi_xpu.KimiDecoderLayer(
        config,
        vllm_config,  # type: ignore[arg-type]
        prefix="model.layers.0",
    )

    assert isinstance(layer.self_attn, _FakeKDA)
    assert layer.self_attn.prefix == "model.layers.0.self_attn"


def test_xpu_decoder_layer_selects_low_rank_kimi_linear_kda(monkeypatch) -> None:
    class _FakeLayer(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    class _FakeKDA(nn.Module):
        def __init__(
            self,
            config: KimiLinearConfig,
            vllm_config: object,
            prefix: str,
        ) -> None:
            super().__init__()
            self.config = config
            self.vllm_config = vllm_config
            self.prefix = prefix

    monkeypatch.setattr(kimi_xpu, "KimiK3DeltaAttention", _FakeLayer)
    monkeypatch.setattr(kimi_xpu, "KimiLinearDeltaAttention", _FakeKDA)
    monkeypatch.setattr(kimi_xpu, "KimiMLP", _FakeLayer)
    monkeypatch.setattr(kimi_xpu, "RMSNorm", _FakeLayer)
    config = KimiLinearConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "use_full_rank_gate": False,
        },
    )
    vllm_config = SimpleNamespace(quant_config=None)

    layer = kimi_xpu.KimiDecoderLayer(
        config,
        vllm_config,  # type: ignore[arg-type]
        prefix="model.layers.0",
    )

    assert isinstance(layer.self_attn, _FakeKDA)
    assert layer.self_attn.prefix == "model.layers.0.self_attn"


def test_mla_output_gate_is_applied_before_output_projection() -> None:
    wrapper = object.__new__(MultiHeadLatentAttentionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.q_lora_rank = None
    wrapper.kv_lora_rank = 1
    wrapper.qk_rope_head_dim = 1
    wrapper.qk_nope_head_dim = 1
    wrapper.qk_head_dim = 2
    wrapper.v_head_dim = 1
    wrapper.num_heads = 1
    wrapper.kv_a_proj_with_mqa = _ConstantProjection(torch.ones(1, 2))
    wrapper.q_proj = _ConstantProjection(torch.ones(1, 2))
    wrapper.kv_a_layernorm = nn.Identity()
    wrapper.rotary_emb = None
    wrapper.indexer = None
    wrapper.is_sparse = False
    wrapper.skip_topk = False
    wrapper.dcp_q_replicate = False
    wrapper.mla_attn = _FakeMLA()
    wrapper.g_proj = _ConstantProjection(torch.zeros(1, 1))
    wrapper.o_proj = _TupleIdentity()

    output = wrapper(torch.tensor([0]), torch.ones(1, 2))

    torch.testing.assert_close(output, torch.full((1, 1), 2.0))


def test_xpu_decoder_layer_runs_attention_then_mlp() -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = False
    layer.input_layernorm = _ResidualNorm()
    layer.post_attention_layernorm = _ResidualNorm()
    layer.mlp = _Scale()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states + 3,
        layer,
    )

    hidden_states = torch.tensor([[1.0, 2.0]])
    output, prefix_sum, residual = layer(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        residual=None,
    )

    torch.testing.assert_close(output, torch.tensor([[14.0, 18.0]]))
    assert prefix_sum is None
    torch.testing.assert_close(residual, torch.tensor([[6.0, 8.0]]))


def test_xpu_kimi_linear_model_combines_standard_residual(monkeypatch) -> None:
    monkeypatch.setattr(kimi_xpu, "get_pp_group", lambda: _SingleRankPPGroup())
    model = object.__new__(kimi_xpu.KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=None)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([_StandardModelLayer(), _StandardModelLayer()])
    inputs_embeds = torch.tensor([[1.0, 2.0]])

    output = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.tensor([[13.0, 24.0]]))


def test_xpu_kimi_linear_model_preserves_attn_res_states(monkeypatch) -> None:
    monkeypatch.setattr(kimi_xpu, "get_pp_group", lambda: _SingleRankPPGroup())
    model = object.__new__(kimi_xpu.KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=2)
    model.start_layer = 0
    model.end_layer = 2
    first_delta = torch.ones(1, 2)
    model.layers = nn.ModuleList(
        [
            _AttnResModelLayer(None, 1.0),
            _AttnResModelLayer(first_delta, 2.0),
        ]
    )
    model.output_attn_res_norm = _WeightedNorm(2)
    model.output_attn_res_proj = _Projection(2)
    calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Size]] = []

    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        calls.append((prefix, delta, blocks.shape))
        assert delta is not None
        return prefix + delta

    monkeypatch.setattr(kimi_xpu, "attn_res", fake_attn_res)
    inputs_embeds = torch.tensor([[1.0, 2.0]])

    output = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    assert isinstance(output, torch.Tensor)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([[31.0, 32.0]]))
    torch.testing.assert_close(calls[0][1], torch.full((1, 2), 2.0))
    assert calls[0][2] == torch.Size([1, 1, 2])
    torch.testing.assert_close(output, torch.tensor([[33.0, 34.0]]))


def test_xpu_kimi_linear_model_allocates_attn_res_intermediates() -> None:
    model = object.__new__(kimi_xpu.KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=2, hidden_size=4)
    model.start_layer = 4

    intermediate_tensors = model.make_empty_intermediate_tensors(
        batch_size=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert set(intermediate_tensors.tensors) == {
        "hidden_states",
        "prefix_sum",
        "residual",
    }
    assert intermediate_tensors["hidden_states"].shape == (3, 4)
    assert intermediate_tensors["prefix_sum"].shape == (3, 4)
    assert intermediate_tensors["residual"].shape == (3, 2, 4)


def test_xpu_kimi_linear_model_loads_fused_mlp_shards() -> None:
    calls: list[tuple[torch.Tensor, int]] = []
    model = object.__new__(kimi_xpu.KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config=None,
        q_lora_rank=None,
        is_moe=False,
        num_nextn_predict_layers=0,
    )
    model.layers = nn.ModuleList([_WeightLoaderLayer(calls)])
    gate_weight = torch.ones(2, 2)
    up_weight = torch.full((2, 2), 2.0)

    loaded = model.load_weights(
        [
            ("layers.0.mlp.gate_proj.weight", gate_weight),
            ("layers.0.mlp.up_proj.weight", up_weight),
        ]
    )

    assert loaded == {"layers.0.mlp.gate_up_proj.weight"}
    assert len(calls) == 2
    assert calls[0][0] is gate_weight and calls[0][1] == 0
    assert calls[1][0] is up_weight and calls[1][1] == 1


def test_xpu_decoder_layer_uses_three_attn_res_states(monkeypatch) -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.prev_valid_blocks = 1
    layer.block_write_idx = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = _WeightedNorm(2)
    layer.post_attention_layernorm = _WeightedNorm(2)
    layer.self_attention_res_norm = _WeightedNorm(2)
    layer.mlp_res_norm = _WeightedNorm(2)
    layer.self_attention_res_proj = _Projection(2)
    layer.mlp_res_proj = _Projection(2)
    layer.mlp = _Scale()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states + 3,
        layer,
    )
    calls: list[tuple[torch.Tensor, torch.Tensor | None, int, int]] = []

    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        *args: object,
        num_blocks: int,
        block_write_idx: int,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        calls.append((prefix, delta, num_blocks, block_write_idx))
        if delta is not None:
            prefix.add_(delta)
        return prefix + 100

    monkeypatch.setattr(kimi_xpu, "attn_res", fake_attn_res)
    hidden_states = torch.tensor([[1.0, 2.0]])
    prefix_sum = torch.tensor([[10.0, 20.0]])
    residual = torch.tensor([[[7.0, 8.0]]])

    output, updated_prefix, updated_residual = layer(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        prefix_sum=prefix_sum,
        residual=residual,
    )

    assert len(calls) == 2
    assert calls[0][1] is hidden_states
    assert calls[0][2:] == (1, -1)
    assert calls[1][1] is not None
    assert calls[1][2:] == (1, -1)
    torch.testing.assert_close(updated_prefix, torch.tensor([[125.0, 147.0]]))
    torch.testing.assert_close(output, torch.tensor([[450.0, 494.0]]))
    assert updated_residual is residual


def test_xpu_decoder_layer_resets_prefix_after_block_write(monkeypatch) -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.prev_valid_blocks = 0
    layer.is_block_write_layer = True
    layer.post_attention_layernorm = _WeightedNorm(2)
    layer.mlp_res_norm = _WeightedNorm(2)
    layer.mlp_res_proj = _Projection(2)
    call: dict[str, object] = {}

    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        *args: object,
        num_blocks: int,
        block_write_idx: int,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        call.update(
            prefix=prefix,
            delta=delta,
            blocks=blocks,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
        )
        return prefix + 100

    monkeypatch.setattr(kimi_xpu, "attn_res", fake_attn_res)
    attention_output = torch.tensor([[3.0, 4.0]])
    old_prefix = torch.tensor([[1.0, 2.0]])
    residual = torch.zeros(1, 1, 2)

    output, prefix_sum, updated_residual = layer._post_attn_norm(
        attention_output, residual, old_prefix
    )

    assert call["prefix"] is attention_output
    assert call["delta"] is None
    assert call["num_blocks"] == 1
    assert call["block_write_idx"] == -1
    assert prefix_sum is attention_output
    assert updated_residual is residual
    torch.testing.assert_close(output, torch.tensor([[103.0, 104.0]]))


def _reference_attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    if delta is not None:
        prefix.add_(delta)
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(prefix)
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if output_norm_weight is not None:
        output = F.rms_norm(
            output, (hidden_size,), output_norm_weight, output_norm_eps
        )
    return output


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU AttnRes requires XPU",
)
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "block_capacity", "hidden_size"),
    [
        pytest.param(1, 1, 2, 128, id="decode-single"),
        pytest.param(17, 4, 6, 1024, id="decode-multiple-blocks"),
        pytest.param(320, 8, 10, 7168, id="prefill-full"),
    ],
)
def test_xpu_attn_res_matches_reference(
    num_tokens: int,
    num_blocks: int,
    block_capacity: int,
    hidden_size: int,
) -> None:
    eps = 1e-5
    device = torch.device("xpu")
    prefix = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    blocks = torch.randn(
        num_tokens,
        block_capacity,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device=device, dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    delta = torch.randn_like(prefix)
    output_norm_weight = 1 + 0.1 * torch.randn_like(norm_weight)
    block_write_idx = num_blocks
    expected_prefix = prefix.clone()
    expected_blocks = blocks.clone()
    expected = _reference_attn_res(
        expected_prefix,
        delta,
        expected_blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        eps,
    )

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        eps,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()