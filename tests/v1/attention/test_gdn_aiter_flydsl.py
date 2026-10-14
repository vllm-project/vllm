# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    _resolve_gdn_prefill_backend,
)
from vllm.platforms import current_platform


def _make_config(
    *,
    backend: str = "aiter_flydsl",
    head_k_dim: int | None = 128,
    head_v_dim: int | None = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": backend},
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_text_config=SimpleNamespace(
                linear_key_head_dim=head_k_dim,
                linear_value_head_dim=head_v_dim,
            ),
        ),
    )


def _resolve_on_rocm(config, *, kernels_available: bool = True):
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=True
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "is_gdn_flydsl_prefill_available",
            return_value=kernels_available,
        ),
    ):
        return _resolve_gdn_prefill_backend(config)


@pytest.mark.parametrize(
    "head_k_dim,head_v_dim,dtype,expected",
    [
        (128, 128, torch.bfloat16, "aiter_flydsl"),
        (64, 128, torch.bfloat16, "triton"),
        (128, 64, torch.bfloat16, "triton"),
        (128, 128, torch.float16, "triton"),
    ],
)
def test_resolve_aiter_flydsl_gdn_prefill_backend(
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    expected: str,
):
    config = _make_config(head_k_dim=head_k_dim, head_v_dim=head_v_dim, dtype=dtype)

    requested, active = _resolve_on_rocm(config)

    assert requested == "aiter_flydsl"
    assert active == expected


def test_kda_style_model_cannot_select_flydsl():
    """Kimi K3 KDA reports head dims through linear_attn_config, not these.

    Its builder overrides _build_chunk_metadata, which the AITER path skips,
    so this pins the reason the two never meet: the resolver turns the model
    down before the builder is ever constructed.
    """
    config = _make_config(head_k_dim=None, head_v_dim=None)

    assert _resolve_on_rocm(config) == ("aiter_flydsl", "triton")


@pytest.mark.parametrize("on_rocm", [True, False], ids=["kernels_missing", "not_rocm"])
def test_explicit_aiter_flydsl_fails_closed(on_rocm: bool):
    """An explicit request that cannot be honoured is a configuration error.

    Falling back silently would report the run as healthy while the requested
    kernels were never used.
    """
    config = _make_config()
    if on_rocm:
        with pytest.raises(RuntimeError, match="not importable"):
            _resolve_on_rocm(config, kernels_available=False)
        return
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=False
        ),
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_cuda", return_value=True
        ),
        pytest.raises(RuntimeError, match="aiter_flydsl"),
    ):
        _resolve_gdn_prefill_backend(config)


def _flydsl_prefill_unavailable() -> bool:
    return (
        not current_platform.is_rocm()
        or not rocm_aiter_ops.is_gdn_flydsl_prefill_available()
    )


@pytest.mark.skipif(
    _flydsl_prefill_unavailable(),
    reason="needs ROCm with the AITER FlyDSL GDN prefill kernels installed",
)
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"]
)
def test_aiter_flydsl_prefill_matches_triton_reference(state_dtype: torch.dtype):
    """Run both prefill paths on device and compare.

    The reference is the same AITER entry point with the FlyDSL prepare and K5
    kernels switched off, so the only thing that differs between the two calls
    is the kernels this backend exists to select. The state dtype follows
    --mamba-ssm-cache-dtype, which may be float32 or bfloat16.
    """
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_lens = [70, 130]
    total_tokens = sum(seq_lens)
    num_k_heads, num_v_heads, head_dim = 2, 4, 128

    # use_qk_l2norm_in_kernel=False matches the prefill path, where the model
    # has already normalised q and k. Feeding raw normals instead lets the
    # delta rule run away to NaN, which says nothing about either kernel.
    q = torch.nn.functional.normalize(
        torch.randn(1, total_tokens, num_k_heads, head_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(1, total_tokens, num_k_heads, head_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    # g is a log-domain decay, so keep it negative or the recurrence diverges
    # and the comparison stops saying anything about the kernels.
    g = -torch.rand(1, total_tokens, num_v_heads, dtype=torch.float32, device=device)
    beta = torch.rand(1, total_tokens, num_v_heads, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 70, 200], dtype=torch.int32, device=device)
    initial_state = torch.randn(
        len(seq_lens),
        num_v_heads,
        head_dim,
        head_dim,
        device=device,
    ).to(state_dtype)

    shared = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "output_final_state": True,
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": False,
    }

    reference_out, reference_state = chunk_gated_delta_rule_opt_vk(
        initial_state=initial_state.clone(),
        state_dtype=state_dtype,
        use_chunk_flydsl=False,
        use_prepare_flydsl=False,
        **shared,
    )
    flydsl_out, flydsl_state = ChunkGatedDeltaRule.forward_aiter_flydsl(
        None,
        initial_state=initial_state.clone(),
        aiter_prefill_metadata=rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
            seq_lens, cu_seqlens=cu_seqlens
        ),
        **shared,
    )

    assert flydsl_state.dtype == state_dtype
    torch.testing.assert_close(
        flydsl_out.float(), reference_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        flydsl_state.float(), reference_state.float(), rtol=2e-2, atol=2e-2
    )
