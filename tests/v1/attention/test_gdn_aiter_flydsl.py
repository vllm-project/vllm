# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    _resolve_gdn_prefill_backend,
)
from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadataBuilder
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


def _make_config(
    *,
    backend: str = "aiter_flydsl",
    head_k_dim: int = 128,
    head_v_dim: int = 128,
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


def test_cli_accepts_aiter_flydsl_gdn_prefill_backend():
    with patch.object(qwen_gdn_linear_attn.current_platform, "device_type", "cpu"):
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--gdn-prefill-backend", "aiter_flydsl"])
    assert args.gdn_prefill_backend == "aiter_flydsl"


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
    config = _make_config(
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        dtype=dtype,
    )
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=True
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "is_gdn_flydsl_prefill_available",
            return_value=True,
        ),
    ):
        requested, active = _resolve_gdn_prefill_backend(config)

    assert requested == "aiter_flydsl"
    assert active == expected


def test_explicit_aiter_flydsl_unavailable_kernels_fail_closed():
    config = _make_config()
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=True
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "is_gdn_flydsl_prefill_available",
            return_value=False,
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "gdn_flydsl_prefill_unavailable_reason",
            return_value=(
                "ImportError: No module named 'aiter.ops.flydsl.kernels.gdr_prefill'"
            ),
        ),
        pytest.raises(RuntimeError, match="kernels.gdr_prefill"),
    ):
        _resolve_gdn_prefill_backend(config)


def test_aiter_flydsl_dispatch_arguments(monkeypatch: pytest.MonkeyPatch):
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 8, 4, 128, dtype=torch.bfloat16)
    g = torch.empty(1, 8, 4, dtype=torch.float32)
    beta = torch.empty_like(g)
    initial_state = torch.empty(2, 4, 128, 128, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    prefill_metadata = object()
    expected_o = torch.empty_like(v)
    captured = {}

    def fake_aiter(**kwargs):
        captured.update(kwargs)
        return expected_o, initial_state

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_aiter_flydsl_chunk_gated_delta_rule",
        fake_aiter,
    )
    output, final_state = ChunkGatedDeltaRule.forward_aiter_flydsl(
        None,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        prefill_metadata=prefill_metadata,
    )

    assert output is expected_o
    assert final_state is initial_state
    assert captured["use_chunk_flydsl"] is True
    assert captured["use_prepare_flydsl"] is True
    assert captured["state_dtype"] is torch.float32
    assert captured["prefill_metadata"] is prefill_metadata
    assert captured["inplace_final_state"] is False
    assert captured["use_qk_l2norm_in_kernel"] is False


def test_prefill_metadata_is_built_for_the_kernels_chunk_size(monkeypatch):
    """The chunk size belongs to the AITER kernel, not to the caller."""
    cu_seqlens = torch.tensor([0, 70, 200], dtype=torch.int32)
    expected_metadata = object()
    captured = {}

    def fake_build(seq_lens_cpu, *, cu_seqlens, chunk_size):
        captured.update(
            seq_lens_cpu=seq_lens_cpu, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )
        return expected_metadata

    gated_delta_net = SimpleNamespace(
        build_gated_delta_rule_prefill_metadata=fake_build
    )
    monkeypatch.setitem(
        sys.modules, "aiter.ops.triton.gated_delta_net", gated_delta_net
    )

    metadata = rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
        [70, 130], cu_seqlens=cu_seqlens
    )

    assert metadata is expected_metadata
    assert captured == {
        "seq_lens_cpu": [70, 130],
        "cu_seqlens": cu_seqlens,
        "chunk_size": 64,
    }


def test_chunk_metadata_keeps_its_two_tensor_contract():
    """Subclasses such as KimiK3ROCmKDAMetadataBuilder override this method.

    The AITER backend carries its own metadata object and must not widen the
    return value to smuggle it through, or every override breaks.
    """
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.gdn_prefill_backend = "triton"
    cu_seqlens_cpu = torch.tensor([0, 70, 200], dtype=torch.int32)

    result = builder._build_chunk_metadata(
        cu_seqlens_cpu.clone(),
        cu_seqlens_cpu,
        torch.device("cpu"),
    )

    assert len(result) == 2
    assert all(isinstance(t, torch.Tensor) for t in result)

    base = inspect.signature(GDNAttentionMetadataBuilder._build_chunk_metadata)
    override = inspect.signature(KimiK3ROCmKDAMetadataBuilder._build_chunk_metadata)
    assert base.return_annotation == override.return_annotation


def test_kda_style_model_cannot_select_flydsl():
    """Kimi K3 KDA reports head dims through linear_attn_config, not these.

    Its builder overrides _build_chunk_metadata, which the AITER path skips,
    so this pins the reason the two never meet: the resolver turns the model
    down before the builder is ever constructed.
    """
    config = _make_config(head_k_dim=None, head_v_dim=None)
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=True
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "is_gdn_flydsl_prefill_available",
            return_value=True,
        ),
    ):
        requested, active = _resolve_gdn_prefill_backend(config)

    assert (requested, active) == ("aiter_flydsl", "triton")


def test_builder_with_custom_chunk_metadata_is_rejected_for_flydsl():
    """And if some future builder could select it, say so at startup."""
    check = GDNAttentionMetadataBuilder._check_chunk_metadata_override

    # The KDA builder is the in-tree example of an override.
    with pytest.raises(RuntimeError, match="KimiK3ROCmKDAMetadataBuilder"):
        check(KimiK3ROCmKDAMetadataBuilder, "aiter_flydsl")

    # It is only the AITER path that skips the override.
    check(KimiK3ROCmKDAMetadataBuilder, "triton")
    check(GDNAttentionMetadataBuilder, "aiter_flydsl")


def test_flydsl_availability_respects_the_aiter_switch(monkeypatch):
    """VLLM_ROCM_USE_AITER stays the one switch that turns AITER kernels off.

    Selecting the backend opts in to it specifically, but it cannot opt back
    in to AITER as a whole once the user has disabled it.
    """
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", False)
    rocm_aiter_ops.is_gdn_flydsl_prefill_available.cache_clear()
    try:
        assert rocm_aiter_ops.is_gdn_flydsl_prefill_available() is False
        assert (
            "VLLM_ROCM_USE_AITER"
            in rocm_aiter_ops.gdn_flydsl_prefill_unavailable_reason()
        )
    finally:
        rocm_aiter_ops.is_gdn_flydsl_prefill_available.cache_clear()


def _flydsl_prefill_unavailable() -> bool:
    return (
        not current_platform.is_rocm()
        or not rocm_aiter_ops.is_gdn_flydsl_prefill_available()
    )


@pytest.mark.skipif(
    _flydsl_prefill_unavailable(),
    reason="needs ROCm with the AITER FlyDSL GDN prefill kernels installed",
)
def test_aiter_flydsl_prefill_matches_triton_reference():
    """Run both prefill paths on device and compare.

    The reference is the same AITER entry point with the FlyDSL prepare and K5
    kernels switched off, so the only thing that differs between the two calls
    is the kernels this backend exists to select.
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
        dtype=torch.float32,
        device=device,
    )

    shared = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    reference_out, reference_state = chunk_gated_delta_rule_opt_vk(
        initial_state=initial_state.clone(),
        state_dtype=torch.float32,
        use_chunk_flydsl=False,
        use_prepare_flydsl=False,
        **shared,
    )
    flydsl_out, flydsl_state = ChunkGatedDeltaRule.forward_aiter_flydsl(
        None,
        initial_state=initial_state.clone(),
        prefill_metadata=rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
            seq_lens, cu_seqlens=cu_seqlens
        ),
        **shared,
    )

    torch.testing.assert_close(
        flydsl_out.float(), reference_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        flydsl_state.float(), reference_state.float(), rtol=2e-2, atol=2e-2
    )


def test_explicit_aiter_flydsl_off_rocm_fails_closed():
    """Asking for a ROCm backend on another platform is a configuration error.

    Falling back silently would report the run as healthy while the requested
    kernels were never used.
    """
    config = _make_config()
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
