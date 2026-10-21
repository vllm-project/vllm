# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing of the FA4 SM90 FP8 MMA path (head_size=512 with an FP8 KV cache).

The device capability is mocked, so no GPU is needed.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms.interface import DeviceCapability

fa_interface = pytest.importorskip("vllm.vllm_flash_attn.flash_attn_interface")

from vllm.v1.attention.backends.fa_utils import get_flash_attn_version  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
)

SM80 = DeviceCapability(8, 0)
SM90 = DeviceCapability(9, 0)
SM100 = DeviceCapability(10, 0)


@contextmanager
def _device(capability, *, head_size, kv_cache_dtype, fa_version=None, dcp=1):
    platform = MagicMock()
    platform.is_xpu.return_value = False
    platform.is_rocm.return_value = False
    platform.is_cuda.return_value = True
    platform.get_device_capability.return_value = capability
    platform.is_device_capability_family.side_effect = (
        lambda family, device_id=0: capability.to_int() // 10 == family // 10
    )
    config = MagicMock()
    config.attention_config.flash_attn_version = fa_version
    config.model_config.is_diffusion = False
    config.model_config.get_head_size.return_value = head_size
    config.cache_config.cache_dtype = kv_cache_dtype
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.dcp_comm_backend = "ag_rs"
    backends = "vllm.v1.attention.backends"
    with (
        patch(f"{backends}.fa_utils.current_platform", platform),
        patch(f"{backends}.flash_attn.current_platform", platform),
        patch.object(fa_interface, "is_fa_version_supported", return_value=True),
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch(
            f"{backends}.flash_attn.get_current_vllm_config_or_none",
            return_value=config,
        ),
        patch(
            "vllm.distributed.parallel_state.get_dcp_world_size_and_rank",
            return_value=(dcp, 0),
        ),
    ):
        yield


def _supports_combination(capability, head_size, kv_cache_dtype, block_size=64):
    return FlashAttentionBackend.supports_combination(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=capability,
    )


def _impl(head_size, kv_cache_dtype):
    return FlashAttentionImpl(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize(
    "head_size,kv_cache_dtype,fa_version,sm90_fp8_mma",
    [
        (512, "fp8", 4, True),
        (512, "fp8_e4m3", 4, True),
        # Gemma-4 sliding-window layers stay on FA3 FP8 with an FP8 query.
        (256, "fp8", 3, False),
        (512, "auto", 4, False),
    ],
)
def test_sm90_routing(head_size, kv_cache_dtype, fa_version, sm90_fp8_mma):
    with _device(SM90, head_size=head_size, kv_cache_dtype=kv_cache_dtype):
        assert get_flash_attn_version(head_size=head_size) == fa_version
        assert _supports_combination(SM90, head_size, kv_cache_dtype) is None
        impl = _impl(head_size, kv_cache_dtype)
    assert impl.vllm_flash_attn_version == fa_version
    # The FP8 MMA kernel quantizes Q in-kernel, so vLLM must not quantize it.
    assert impl.supports_quant_query_input is not sm90_fp8_mma


def test_sm90_fp8_mma_block_size():
    with _device(SM90, head_size=512, kv_cache_dtype="fp8"):
        assert FlashAttentionBackend.get_supported_kernel_block_sizes() == [64]
        assert "block size" in _supports_combination(SM90, 512, "fp8", 16)
    with _device(SM90, head_size=512, kv_cache_dtype="auto"):
        assert _supports_combination(SM90, 512, "auto", 16) is None


@pytest.mark.parametrize(
    "capability,head_size,fa_version",
    [
        (SM90, 256, 4),  # forced FA4 with an FP8 KV cache needs head_size=512
        (SM100, 512, None),  # FA4 on Blackwell does not support head_size=512
        (SM80, 512, None),  # FA2 has no FP8 KV cache
    ],
)
def test_fp8_kv_rejected(capability, head_size, fa_version):
    with _device(
        capability, head_size=head_size, kv_cache_dtype="fp8", fa_version=fa_version
    ):
        assert _supports_combination(capability, head_size, "fp8") is not None
        with pytest.raises(NotImplementedError, match="fp8 kv-cache"):
            _impl(head_size, "fp8")


@pytest.mark.parametrize(
    "head_size,kv_cache_dtype,rejected",
    [(512, "fp8", True), (512, "auto", False), (256, "fp8", False)],
)
def test_sm90_fp8_mma_rejects_dcp(head_size, kv_cache_dtype, rejected):
    with _device(SM90, head_size=head_size, kv_cache_dtype=kv_cache_dtype, dcp=2):
        if rejected:
            with pytest.raises(NotImplementedError, match="context parallelism"):
                _impl(head_size, kv_cache_dtype)
        else:
            assert _impl(head_size, kv_cache_dtype).dcp_world_size == 2


@pytest.mark.parametrize(
    "head_size,q_dtype,causal,match",
    [
        # FA4 + FP8 KV below d512 raises; no silent switch to FA3.
        (256, torch.float8_e4m3fn, True, "q=torch.float8_e4m3fn/256"),
        (128, torch.float8_e4m3fn, False, "head_size=512"),
        (256, torch.bfloat16, True, "q=torch.bfloat16/256"),
        (512, torch.bfloat16, False, "noncausal"),
    ],
)
def test_wrapper_rejects_unsupported_sm90_fp8(head_size, q_dtype, causal, match):
    q = torch.zeros(4, 8, head_size, dtype=q_dtype)
    k = torch.zeros(1, 64, 1, head_size, dtype=torch.float8_e4m3fn)
    fa3 = MagicMock()
    with (
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        patch.object(torch.ops, "_vllm_fa3_C", fa3, create=True),
        pytest.raises(NotImplementedError, match=match),
    ):
        fa_interface.flash_attn_varlen_func(
            q,
            k,
            k,
            max_seqlen_q=4,
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            max_seqlen_k=4,
            seqused_k=torch.tensor([4], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            causal=causal,
            fa_version=4,
        )
    fa3.fwd.assert_not_called()
