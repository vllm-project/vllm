# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.models.k2_horizon as k2_horizon
from tests.quantization.utils import is_quant_method_supported
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.k2_horizon import (
    K2HorizonDecoderLayer,
    K2HorizonModel,
    K2HorizonRMSNorm,
    _rope_weight_perm,
    apply_partial_rope,
    is_rope_weights_folding_supported,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DEVICE = current_platform.device_type
DTYPE = torch.bfloat16
SHAPES = [
    (128, 64),
    (128, 128),
    (192, 128),
    (96, 32),
]


def get_gptj_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_emb = get_rope(
        rope_head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": base},
        dtype=dtype,
    )
    return apply_partial_rope(
        rotary_emb,
        positions,
        q,
        k,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        fold_rope_weights=False,
    )


def _apply_head_perm(
    x: torch.Tensor, num: int, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    return (
        x.reshape(*x.shape[:-1], num, head_dim)[..., idx].reshape(*x.shape).contiguous()
    )


def get_neox_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_emb = get_rope(
        head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": base, "rope_dim": rope_head_dim},
        dtype=dtype,
    )
    return apply_partial_rope(
        rotary_emb,
        positions,
        q,
        k,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        fold_rope_weights=True,
    )


def fold_qk_proj_weight(
    weight: torch.Tensor, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    hidden = weight.shape[-1]
    return weight.view(-1, head_dim, hidden)[:, idx, :].reshape(-1, hidden).contiguous()


def fold_qk_channels(
    vec: torch.Tensor, head_dim: int, idx: torch.Tensor
) -> torch.Tensor:
    return vec.view(-1, head_dim)[:, idx].reshape(-1).contiguous()


@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
def test_rope_fold_matches_permute_then_rope(
    head_dim, rope_head_dim, default_vllm_config
):
    dtype = DTYPE
    set_random_seed(0)
    num_tokens = 17
    num_heads = 8
    num_kv_heads = 2
    hidden_size = 512
    max_position = 4096
    base = 10000.0

    torch.set_default_dtype(dtype)

    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE)

    scale = hidden_size**-0.5
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
    w_q = (
        torch.randn(num_heads * head_dim, hidden_size, dtype=dtype, device=DEVICE)
        * scale
    )
    w_k = (
        torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype, device=DEVICE)
        * scale
    )

    idx = _rope_weight_perm(head_dim, rope_head_dim).to(DEVICE)

    q = torch.nn.functional.linear(x, w_q)
    k = torch.nn.functional.linear(x, w_k)
    q_gptj, k_gptj = get_gptj_rope(
        positions,
        q,
        k,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        max_position,
        base,
        dtype,
    )

    q_folded = torch.nn.functional.linear(x, fold_qk_proj_weight(w_q, head_dim, idx))
    k_folded = torch.nn.functional.linear(x, fold_qk_proj_weight(w_k, head_dim, idx))
    q_neox, k_neox = get_neox_rope(
        positions,
        q_folded,
        k_folded,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        max_position,
        base,
        dtype,
    )

    q_gptj_perm = _apply_head_perm(q_gptj, num_heads, head_dim, idx)
    k_gptj_perm = _apply_head_perm(k_gptj, num_kv_heads, head_dim, idx)

    atol, rtol = 1e-2, 1e-2

    # Two complementary checks. First, an element-wise check that the folded
    # NeoX path equals the GPT-J path up to the channel permutation P
    # (neox_rope(Pq, Pk) == P @ gptj_rope(q, k)), hence permuting q_gptj/k_gptj.
    # Second, the scores check below: P cancels in q @ k.T.
    torch.testing.assert_close(q_neox, q_gptj_perm, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_neox, k_gptj_perm, atol=atol, rtol=rtol)

    def scores(qt, kt):
        qh = qt.reshape(num_tokens, num_heads, head_dim)
        kh = kt.reshape(num_tokens, num_kv_heads, head_dim)
        g = num_heads // num_kv_heads
        qh = qh.reshape(num_tokens, num_kv_heads, g, head_dim).float()
        kh = kh.float()
        return torch.einsum("tkgd,skd->tksg", qh, kh)

    torch.testing.assert_close(
        scores(q_neox, k_neox), scores(q_gptj, k_gptj), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
@pytest.mark.parametrize("num_heads", [8, 2])
def test_qk_bias_fold_matches_permute(
    head_dim, rope_head_dim, num_heads, default_vllm_config
):
    dtype = DTYPE
    set_random_seed(0)
    num_tokens = 17
    hidden_size = 512
    torch.set_default_dtype(dtype)

    scale = hidden_size**-0.5
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
    w = (
        torch.randn(num_heads * head_dim, hidden_size, dtype=dtype, device=DEVICE)
        * scale
    )
    b = torch.randn(num_heads * head_dim, dtype=dtype, device=DEVICE)

    idx = _rope_weight_perm(head_dim, rope_head_dim).to(DEVICE)

    ref = _apply_head_perm(
        torch.nn.functional.linear(x, w, b), num_heads, head_dim, idx
    )
    folded = torch.nn.functional.linear(
        x,
        fold_qk_proj_weight(w, head_dim, idx),
        fold_qk_channels(b, head_dim, idx),
    )

    atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(folded, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a GPU (CUDA/ROCm)"
)
@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
@pytest.mark.parametrize("num_heads", [8, 2])
def test_qk_norm_scale_fold_matches_permute(
    head_dim, rope_head_dim, num_heads, default_vllm_config
):
    dtype = DTYPE
    set_random_seed(0)
    num_tokens = 13
    torch.set_default_dtype(dtype)

    hidden = num_heads * head_dim
    idx = _rope_weight_perm(head_dim, rope_head_dim).to(DEVICE)

    x = torch.randn(num_tokens, hidden, dtype=dtype, device=DEVICE)

    norm = K2HorizonRMSNorm(hidden_size=hidden, n_groups=num_heads).to(DEVICE)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden, dtype=dtype, device=DEVICE))

    y_ref = _apply_head_perm(norm(x.clone()), num_heads, head_dim, idx)

    folded_norm = K2HorizonRMSNorm(hidden_size=hidden, n_groups=num_heads).to(DEVICE)
    with torch.no_grad():
        folded_norm.weight.copy_(fold_qk_channels(norm.weight, head_dim, idx))
    # Permute x to emulate the folded q_proj/k_proj: at runtime the norm sees the
    # already-permuted projection output (P @ x), not the raw activations.
    y_folded = folded_norm(_apply_head_perm(x.clone(), num_heads, head_dim, idx))

    atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(y_folded, y_ref, atol=atol, rtol=rtol)


class _Fp8QKVOnlyConfig(Fp8Config):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and not (
            prefix.endswith("qkv_proj") or prefix.endswith("qk_proj")
        ):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)


class _DummyAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, q, k, v):
        return q


@pytest.fixture
def dist_init():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


def _make_hf_config(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_head_dim: int,
    attention_bias: bool,
    query_key_norm: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        max_position_embeddings=4096,
        dual_chunk_attention_config=None,
        mlp_only_layers=[],
        num_experts=0,  # dense layer: avoids the MoE path
        decoder_sparse_step=1,
        mova_num_experts=0,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_parameters={"rope_theta": 10000.0},
        rope_head_dim=rope_head_dim,
        query_key_norm=query_key_norm,
        rms_norm_eps=1e-6,
        attention_bias=attention_bias,
        head_dim=head_dim,
        attention_gate_func=None,
        intermediate_size=2 * hidden_size,
        hidden_act="silu",
        layernorm_num_groups=1,
        vocab_size=1000,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )


def _build_single_layer_model(
    monkeypatch, hf_config: SimpleNamespace, *, quant_config=None
) -> K2HorizonModel:
    monkeypatch.setattr(k2_horizon, "Attention", _DummyAttention)

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=hf_config, dtype=torch.get_default_dtype()
    )
    vllm_config.quant_config = quant_config

    with set_current_vllm_config(vllm_config):
        layer = K2HorizonDecoderLayer(vllm_config=vllm_config, prefix="model.layers.0")

    model = K2HorizonModel.__new__(K2HorizonModel)
    nn.Module.__init__(model)
    model.config = hf_config
    model.quant_config = quant_config
    model.num_redundant_experts = 0
    model.layers = nn.ModuleList([layer])
    return model


def _fold_rows(weight: torch.Tensor, head_dim: int, idx: torch.Tensor) -> torch.Tensor:
    hidden = weight.shape[-1]
    return weight.view(-1, head_dim, hidden)[:, idx, :].reshape(-1, hidden).contiguous()


def _fold_channels(vec: torch.Tensor, head_dim: int, idx: torch.Tensor) -> torch.Tensor:
    return vec.view(-1, head_dim)[:, idx].reshape(-1).contiguous()


def _non_qk_weights(prefix, *, hidden_size, intermediate_size, q_size, device=None):
    return [
        (
            f"{prefix}.self_attn.o_proj.weight",
            torch.randn(hidden_size, q_size, device=device),
        ),
        (
            f"{prefix}.mlp.gate_proj.weight",
            torch.randn(intermediate_size, hidden_size, device=device),
        ),
        (
            f"{prefix}.mlp.up_proj.weight",
            torch.randn(intermediate_size, hidden_size, device=device),
        ),
        (
            f"{prefix}.mlp.down_proj.weight",
            torch.randn(hidden_size, intermediate_size, device=device),
        ),
        (
            f"{prefix}.input_layernorm.weight",
            torch.randn(hidden_size, device=device),
        ),
        (
            f"{prefix}.post_attention_layernorm.weight",
            torch.randn(hidden_size, device=device),
        ),
    ]


@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
@pytest.mark.parametrize("attention_bias", [False, True])
def test_load_weights_dense_folds_qk(
    head_dim,
    rope_head_dim,
    attention_bias,
    monkeypatch,
):
    dtype = DTYPE
    set_random_seed(0)
    num_heads = 8
    num_kv_heads = 2
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    prefix = "layers.0"

    with set_default_torch_dtype(dtype):
        hf_config = _make_hf_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            attention_bias=attention_bias,
            query_key_norm=True,
        )
        model = _build_single_layer_model(monkeypatch, hf_config, quant_config=None)

        assert model.layers[0].self_attn.fold_rope_weights is True

        raw = {
            "q_proj.weight": torch.randn(q_size, hidden_size),
            "k_proj.weight": torch.randn(kv_size, hidden_size),
            "v_proj.weight": torch.randn(kv_size, hidden_size),
            "q_norm.weight": torch.randn(q_size),
            "k_norm.weight": torch.randn(kv_size),
        }
        if attention_bias:
            raw.update(
                {
                    "q_proj.bias": torch.randn(q_size),
                    "k_proj.bias": torch.randn(kv_size),
                    "v_proj.bias": torch.randn(kv_size),
                    "o_proj.bias": torch.randn(hidden_size),
                }
            )

        weights: list[tuple[str, torch.Tensor]] = [
            (f"{prefix}.self_attn.{name}", tensor) for name, tensor in raw.items()
        ]
        weights += _non_qk_weights(
            prefix,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            q_size=q_size,
        )

        loaded = model.load_weights(iter(weights))

    params = dict(model.named_parameters())
    assert loaded == set(params)

    idx = _rope_weight_perm(head_dim, rope_head_dim)

    qkv = params[f"{prefix}.self_attn.qkv_proj.weight"]
    q_part = qkv[:q_size]
    k_part = qkv[q_size : q_size + kv_size]
    v_part = qkv[q_size + kv_size :]

    q_norm = params[f"{prefix}.self_attn.q_norm.weight"]
    k_norm = params[f"{prefix}.self_attn.k_norm.weight"]

    assert torch.equal(q_part, _fold_rows(raw["q_proj.weight"], head_dim, idx))
    assert torch.equal(k_part, _fold_rows(raw["k_proj.weight"], head_dim, idx))
    assert torch.equal(v_part, raw["v_proj.weight"])  # v never folded
    assert torch.equal(q_norm, _fold_channels(raw["q_norm.weight"], head_dim, idx))
    assert torch.equal(k_norm, _fold_channels(raw["k_norm.weight"], head_dim, idx))
    assert qkv.dtype == dtype

    if attention_bias:
        qkv_bias = params[f"{prefix}.self_attn.qkv_proj.bias"]
        q_bias = qkv_bias[:q_size]
        k_bias = qkv_bias[q_size : q_size + kv_size]
        v_bias = qkv_bias[q_size + kv_size :]
        assert torch.equal(q_bias, _fold_channels(raw["q_proj.bias"], head_dim, idx))
        assert torch.equal(k_bias, _fold_channels(raw["k_proj.bias"], head_dim, idx))
        assert torch.equal(v_bias, raw["v_proj.bias"])  # v bias never folded


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="fp8 is not supported.",
)
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("head_dim,rope_head_dim", SHAPES)
@pytest.mark.parametrize("attention_bias", [False, True])
def test_load_weights_fp8_qk_no_fold(
    head_dim,
    rope_head_dim,
    attention_bias,
    monkeypatch,
):
    set_random_seed(0)
    num_heads = 8
    num_kv_heads = 2
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    prefix = "layers.0"

    def rand_fp8(*shape):
        return (torch.randn(*shape, device=DEVICE) * 32.0).to(torch.float8_e4m3fn)

    def rand_scale():
        return torch.rand(1, device=DEVICE, dtype=torch.float32) + 0.1

    with set_default_torch_dtype(torch.bfloat16), torch.device(DEVICE):
        hf_config = _make_hf_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            attention_bias=attention_bias,
            query_key_norm=True,
        )
        quant_config = _Fp8QKVOnlyConfig(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        model = _build_single_layer_model(
            monkeypatch, hf_config, quant_config=quant_config
        )

        qkv_proj = model.layers[0].self_attn.qkv_proj
        assert isinstance(qkv_proj.quant_method, Fp8LinearMethod)
        assert model.layers[0].self_attn.fold_rope_weights is False

        raw_w = {
            "q_proj.weight": rand_fp8(q_size, hidden_size),
            "k_proj.weight": rand_fp8(kv_size, hidden_size),
            "v_proj.weight": rand_fp8(kv_size, hidden_size),
        }
        raw_s = {
            "q_proj.weight_scale": rand_scale(),
            "k_proj.weight_scale": rand_scale(),
            "v_proj.weight_scale": rand_scale(),
        }
        q_norm = torch.randn(q_size, device=DEVICE)
        k_norm = torch.randn(kv_size, device=DEVICE)

        weights: list[tuple[str, torch.Tensor]] = [
            (f"{prefix}.self_attn.{name}", tensor)
            for name, tensor in {**raw_w, **raw_s}.items()
        ]
        weights += [
            (f"{prefix}.self_attn.q_norm.weight", q_norm),
            (f"{prefix}.self_attn.k_norm.weight", k_norm),
        ]
        biases = {}
        if attention_bias:
            biases = {
                "q_proj.bias": torch.randn(q_size, device=DEVICE),
                "k_proj.bias": torch.randn(kv_size, device=DEVICE),
                "v_proj.bias": torch.randn(kv_size, device=DEVICE),
                "o_proj.bias": torch.randn(hidden_size, device=DEVICE),
            }
            weights += [
                (f"{prefix}.self_attn.{name}", tensor)
                for name, tensor in biases.items()
            ]
        weights += _non_qk_weights(
            prefix,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            q_size=q_size,
            device=DEVICE,
        )

        loaded = model.load_weights(iter(weights))

    params = dict(model.named_parameters())
    assert loaded == set(params)

    qkv = params[f"{prefix}.self_attn.qkv_proj.weight"]
    weight_scale = params[f"{prefix}.self_attn.qkv_proj.weight_scale"]

    assert qkv.dtype == torch.float8_e4m3fn
    assert torch.equal(qkv[:q_size], raw_w["q_proj.weight"])
    assert torch.equal(qkv[q_size : q_size + kv_size], raw_w["k_proj.weight"])
    assert torch.equal(qkv[q_size + kv_size :], raw_w["v_proj.weight"])

    assert torch.equal(
        weight_scale,
        torch.cat(
            [
                raw_s["q_proj.weight_scale"],
                raw_s["k_proj.weight_scale"],
                raw_s["v_proj.weight_scale"],
            ]
        ),
    )

    assert torch.equal(params[f"{prefix}.self_attn.q_norm.weight"], q_norm)
    assert torch.equal(params[f"{prefix}.self_attn.k_norm.weight"], k_norm)

    if attention_bias:
        qkv_bias = params[f"{prefix}.self_attn.qkv_proj.bias"]
        assert torch.equal(qkv_bias[:q_size], biases["q_proj.bias"])
        assert torch.equal(qkv_bias[q_size : q_size + kv_size], biases["k_proj.bias"])
        assert torch.equal(qkv_bias[q_size + kv_size :], biases["v_proj.bias"])


def test_is_rope_weights_folding_supported(default_vllm_config):
    dense = SimpleNamespace(quant_method=UnquantizedLinearMethod())
    quantized = SimpleNamespace(quant_method=object())

    assert is_rope_weights_folding_supported(dense, None) is True
    assert is_rope_weights_folding_supported(quantized, None) is False
    assert is_rope_weights_folding_supported(dense, {"chunk_size": 8}) is False
    assert is_rope_weights_folding_supported(SimpleNamespace(), None) is False
