# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing of the FA4 SM90 FP8 MMA path (head_size=512 with an FP8 KV cache).

The device capability is mocked, so no GPU is needed.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms.interface import DeviceCapability

fa_interface = pytest.importorskip(
    "vllm.vllm_flash_attn.flash_attn_interface",
    reason="requires compiled CUDA flash-attention extensions",
    exc_type=ImportError,
)

from vllm.v1.attention.backends.fa_utils import get_flash_attn_version  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

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
    platform.is_device_capability_family.side_effect = lambda family, device_id=0: (
        capability.to_int() // 10 == family // 10
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
        yield config


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


def _impl(head_size, kv_cache_dtype, logits_soft_cap=None):
    return FlashAttentionImpl(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=logits_soft_cap,
    )


@pytest.mark.parametrize(
    "head_size,kv_cache_dtype,fa_version,sm90_fp8_mma",
    [
        (512, "fp8", 4, True),
        (512, "fp8_e4m3", 4, True),
        # Gemma-4 sliding-window layers run FA4 native FP8 with an FP8 query.
        (256, "fp8", 4, False),
        (512, "auto", 4, False),
    ],
)
def test_sm90_routing(head_size, kv_cache_dtype, fa_version, sm90_fp8_mma):
    with _device(SM90, head_size=head_size, kv_cache_dtype=kv_cache_dtype):
        fp8_kv_cache = kv_cache_dtype != "auto"
        assert (
            get_flash_attn_version(head_size=head_size, fp8_kv_cache=fp8_kv_cache)
            == fa_version
        )
        assert _supports_combination(SM90, head_size, kv_cache_dtype) is None
        impl = _impl(head_size, kv_cache_dtype)
    assert impl.vllm_flash_attn_version == fa_version
    # The FP8 MMA kernel quantizes Q in-kernel, so vLLM must not quantize it.
    assert impl.supports_quant_query_input is not sm90_fp8_mma


@pytest.mark.parametrize(
    "case,fa_version",
    [
        ("upgrade", 4),
        ("softcap", 3),
        ("explicit_fa3", 3),
        ("batch_invariant", 3),
        ("head_size_v", 3),
    ],
)
def test_sm90_fp8_kv_fa4_upgrade(case, fa_version):
    """FP8 KV on SM90 upgrades FA3 to FA4 only where FA4 runs native FP8.

    Batch invariance stays on FA3: FA4 would fall back to FA2, which has no FP8 KV.
    """
    with (
        _device(
            SM90,
            head_size=128,
            kv_cache_dtype="fp8",
            fa_version=3 if case == "explicit_fa3" else None,
        ),
        patch("vllm.envs.VLLM_BATCH_INVARIANT", case == "batch_invariant"),
    ):
        version = get_flash_attn_version(
            head_size=128,
            head_size_v=64 if case == "head_size_v" else None,
            requires_softcap=case == "softcap",
            fp8_kv_cache=True,
        )
    assert version == fa_version


@pytest.mark.parametrize(
    "kv_cache_dtype,logits_soft_cap,fa_version",
    [("fp8", None, 4), ("fp8", 50.0, 3), ("auto", None, 3)],
)
def test_sm90_builder_matches_impl(kv_cache_dtype, logits_soft_cap, fa_version):
    """The metadata builder schedules for the FA version its layers run.

    A softcap layer with an FP8 KV cache stays on FA3, so its builder must keep
    the FA3 AOT scheduler metadata.
    """
    fp8 = kv_cache_dtype == "fp8"
    with _device(SM90, head_size=128, kv_cache_dtype=kv_cache_dtype) as config:
        impl = _impl(128, kv_cache_dtype, logits_soft_cap)
        config.model_config.rswa_window = None
        config.model_config.is_mm_prefix_lm = False
        config.parallel_config.cp_kv_cache_interleave_size = 1
        config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = (
            False
        )
        config.compilation_config.max_cudagraph_capture_size = None
        kv_cache_spec = SimpleNamespace(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR if fp8 else KVQuantMode.NONE,
            dcp_sharded=False,
        )
        layers = {"layer": SimpleNamespace(impl=impl)}
        with patch(
            "vllm.v1.attention.backends.flash_attn.get_layers_from_vllm_config",
            return_value=layers,
        ):
            builder = FlashAttentionMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=list(layers),
                vllm_config=config,
                device=torch.device("cpu"),
            )
    assert impl.vllm_flash_attn_version == fa_version
    assert builder.aot_schedule == (fa_version == 3)


def test_sm90_fp8_mma_block_size():
    with _device(SM90, head_size=512, kv_cache_dtype="fp8"):
        assert FlashAttentionBackend.get_supported_kernel_block_sizes() == [64]
        assert "block size" in _supports_combination(SM90, 512, "fp8", 16)
    with _device(SM90, head_size=512, kv_cache_dtype="auto"):
        assert _supports_combination(SM90, 512, "auto", 16) is None


@pytest.mark.parametrize(
    "capability,head_size,fa_version",
    [
        (SM90, 80, 4),  # SM90 FA4 FP8 KV needs head_size 64, 96, 128, 192, 256 or 512
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
        # FA4 + FP8 KV with a BF16 query below d512 raises; no silent switch to FA3.
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
