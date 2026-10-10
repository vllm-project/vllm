# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch
import torch.nn.functional as F

from vllm.config import KernelConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DEVICE = current_platform.device_type


def _make_gdn_decode(
    backend: Literal["auto", "triton", "flashinfer"],
    state_dtype: torch.dtype,
    model_dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    num_v_heads: int = 8,
    head_v_dim: int = 128,
):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import GDNDecode

    config = VllmConfig(kernel_config=KernelConfig(gdn_decode_backend=backend))
    config.model_config = cast(ModelConfig, SimpleNamespace(dtype=model_dtype))
    with set_current_vllm_config(config):
        return GDNDecode(2, num_v_heads, head_dim, head_v_dim, state_dtype)


def _decode_inputs(
    state_dtype: torch.dtype,
    strided_qkv: bool,
    batch: int = 4,
    num_v_heads: int = 8,
):
    torch.manual_seed(0)
    h, hv, dim = 2, num_v_heads, 128
    qkv_width = 2 * h * dim + hv * dim
    projection = torch.randn(
        batch, qkv_width + hv * dim, device="cuda", dtype=torch.bfloat16
    )
    mixed_qkv = projection[:, :qkv_width]
    if not strided_qkv:
        mixed_qkv = mixed_qkv.contiguous()
    gates = torch.randn(batch, 2 * hv, device="cuda", dtype=torch.bfloat16)
    b, a = gates.chunk(2, dim=-1)
    inputs = {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": torch.randn(hv, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(hv, device="cuda", dtype=torch.float32),
    }
    # The extra head simulates conv data sharing a padded cache page.
    backing = torch.randn(3, 9, hv + 1, dim, dim, device="cuda", dtype=state_dtype)
    backing[1:].copy_(backing[:1].expand_as(backing[1:]))
    return inputs, backing


def _assert_decode_matches(out_fi, out_ref, backing, indices, state_dtype):
    valid = indices > 0
    atol = 2e-2 if state_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(out_fi[valid], out_ref[valid], atol=atol, rtol=1e-2)
    torch.testing.assert_close(backing[2], backing[1], atol=atol, rtol=1e-2)
    untouched = torch.ones(backing.shape[1], device="cuda", dtype=torch.bool)
    untouched[indices[valid].long()] = False
    torch.testing.assert_close(
        backing[2, untouched], backing[0, untouched], atol=0, rtol=0
    )
    torch.testing.assert_close(backing[2, :, -1], backing[0, :, -1], atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdn_decode_default_does_not_require_flashinfer(
    backend, state_dtype, monkeypatch
):
    """The default decode remains usable when FlashInfer is unavailable."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    def unavailable():
        raise AssertionError("The Triton backend must not import FlashInfer")

    monkeypatch.setattr(qwen_gdn_linear_attn, "_get_flashinfer_gdn_decode", unavailable)
    assert _make_gdn_decode(backend, state_dtype).backend == "triton"


def test_gdn_decode_explicit_flashinfer_rejects_non_cuda(monkeypatch):
    """An explicit unsupported backend must fail before serving starts."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match=r"FlashInfer GDN decode requires CUDA SM80\+"):
        _make_gdn_decode("flashinfer", torch.float32)


@pytest.mark.parametrize(
    "overrides,error_match",
    [
        pytest.param(
            {"model_dtype": torch.float32},
            "FlashInfer GDN decode currently supports only BF16 model inputs",
            id="fp32-input",
        ),
        pytest.param(
            {"state_dtype": torch.float16},
            "FlashInfer GDN decode requires FP32 SSM state",
            id="fp16-state",
        ),
        pytest.param(
            {"head_dim": 64},
            "FlashInfer GDN decode currently supports key head dimension 128",
            id="unsupported-key-head-dim",
        ),
        pytest.param(
            {"head_v_dim": 64},
            "FlashInfer GDN decode currently supports value head dimension 128",
            id="unsupported-value-head-dim",
        ),
        pytest.param(
            {"num_v_heads": 3},
            "FlashInfer GDN decode requires value heads to be a multiple of key heads",
            id="non-divisible-head-group",
        ),
        pytest.param(
            {"state_dtype": torch.bfloat16},
            "FlashInfer GDN decode requires FP32 SSM state",
            id="unsupported-bf16-state",
        ),
    ],
)
def test_gdn_decode_explicit_flashinfer_rejects_unsupported_config(
    overrides, error_match, monkeypatch
):
    """Unsupported layouts fail before importing or launching FlashInfer."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    def unavailable():
        raise AssertionError("Unsupported configurations must fail before API lookup")

    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        qwen_gdn_linear_attn.current_platform, "has_device_capability", lambda _: True
    )
    monkeypatch.setattr(qwen_gdn_linear_attn, "_get_flashinfer_gdn_decode", unavailable)
    kwargs = {"state_dtype": torch.float32, **overrides}
    with pytest.raises(ValueError, match=error_match):
        _make_gdn_decode("flashinfer", **kwargs)


@pytest.mark.parametrize("backend", ["auto", "triton", "flashinfer"])
def test_gdn_layer_preserves_loaded_bias_with_flashinfer_dtype(backend, monkeypatch):
    """BF16 checkpoint bias values survive loading into the selected backend."""
    from transformers import Qwen3NextConfig

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=2, rank_in_group=1)
    )
    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        qwen_gdn_linear_attn.current_platform, "has_device_capability", lambda _: True
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn.current_platform,
        "current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn, "_get_flashinfer_gdn_decode", lambda: lambda: None
    )
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    hf_config = Qwen3NextConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config = VllmConfig(kernel_config=KernelConfig(gdn_decode_backend=backend))
    config.model_config = cast(
        ModelConfig, SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf_config)
    )
    config.cache_config.mamba_ssm_cache_dtype = "float32"
    config.additional_config = {"gdn_prefill_backend": "triton"}
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        layer = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention(
            hf_config, config, prefix="model.layers.0.linear_attn"
        )
    checkpoint_bias = torch.linspace(-4.3, 2.8, 16).to(torch.bfloat16)
    layer.dt_bias.weight_loader(layer.dt_bias, checkpoint_bias)
    expected_dtype = torch.float32 if backend == "flashinfer" else torch.bfloat16
    assert layer.dt_bias.dtype == expected_dtype
    torch.testing.assert_close(
        layer.dt_bias.float(), checkpoint_bias[8:].float(), atol=0, rtol=0
    )


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="FlashInfer GDN decode requires CUDA SM80+.",
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided_qkv", [False, True])
@pytest.mark.parametrize("batch,num_v_heads", [(4, 8), (1, 4), (4, 4)])
def test_flashinfer_decode_preserves_indexed_state(
    index_dtype: torch.dtype, strided_qkv: bool, batch: int, num_v_heads: int
):
    """Packed views and inference-mode Parameters preserve indexed state."""
    pytest.importorskip("flashinfer.gdn_decode")
    state_dtype = torch.float32
    inputs, backing = _decode_inputs(state_dtype, strided_qkv, batch, num_v_heads)
    if num_v_heads == 4:
        inputs["A_log"] = torch.nn.Parameter(inputs["A_log"])
        inputs["dt_bias"] = torch.nn.Parameter(inputs["dt_bias"])
    indices = torch.tensor([6, 2, 0, -1][:batch], device="cuda", dtype=index_dtype)
    fi_indices = torch.tensor([6, 2, -1, -1][:batch], device="cuda", dtype=index_dtype)
    out_ref = torch.empty(
        batch, 1, num_v_heads, 128, device="cuda", dtype=torch.bfloat16
    )
    out_fi = torch.empty_like(out_ref)
    with torch.inference_mode():
        _make_gdn_decode("triton", state_dtype, num_v_heads=num_v_heads)(
            **inputs,
            initial_state=backing[1, :, :num_v_heads],
            ssm_state_indices=indices,
            out=out_ref,
        )
        _make_gdn_decode("flashinfer", state_dtype, num_v_heads=num_v_heads)(
            **inputs,
            initial_state=backing[2, :, :num_v_heads],
            ssm_state_indices=fi_indices,
            out=out_fi,
        )
    _assert_decode_matches(out_fi, out_ref, backing, indices, state_dtype)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="FlashInfer GDN decode requires CUDA SM80+.",
)
@pytest.mark.parametrize("num_v_heads", [4, 8])
def test_flashinfer_decode_graph_reads_remapped_slots(num_v_heads: int):
    """Graph replay reads current pool indices instead of capture-time values."""
    pytest.importorskip("flashinfer.gdn_decode")
    state_dtype = torch.float32
    inputs, backing = _decode_inputs(
        state_dtype, strided_qkv=True, num_v_heads=num_v_heads
    )
    indices = torch.tensor([6, 2, 0, -1], device="cuda", dtype=torch.int32)
    fi_indices = torch.tensor([6, 2, -1, -1], device="cuda", dtype=torch.int32)
    out_ref = torch.empty(4, 1, num_v_heads, 128, device="cuda", dtype=torch.bfloat16)
    out_fi = torch.empty_like(out_ref)
    flashinfer = _make_gdn_decode("flashinfer", state_dtype, num_v_heads=num_v_heads)
    kwargs = dict(
        **inputs,
        initial_state=backing[2, :, :num_v_heads],
        ssm_state_indices=fi_indices,
        out=out_fi,
    )
    flashinfer(**kwargs)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer(**kwargs)
    inputs["a"].add_(1)
    inputs["b"].neg_()
    backing[2].copy_(backing[0])
    indices.copy_(torch.tensor([5, 3, 0, -1], device="cuda", dtype=torch.int32))
    fi_indices.copy_(torch.tensor([5, 3, -1, -1], device="cuda", dtype=torch.int32))
    _make_gdn_decode("triton", state_dtype, num_v_heads=num_v_heads)(
        **inputs,
        initial_state=backing[1, :, :num_v_heads],
        ssm_state_indices=indices,
        out=out_ref,
    )
    graph.replay()
    torch.accelerator.synchronize()
    _assert_decode_matches(out_fi, out_ref, backing, indices, state_dtype)


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[
        :num_tokens
    ].view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )
