# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate common DSv4 attention JIT dispatch."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.models.common.ops.fused_qk_rmsnorm import FusedQKVRMSNormKernel
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    FusedKVCompressNormRopeInsertIndexerTritonKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    FusedInvRopeFP8QuantKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
    FusedMTPInputRMSNormKernel,
    MTPSharedHeadRMSNormKernel,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    SavePartialStatesKernel,
)


def test_deepseek_v4_c128a_topk_metadata_warmup_keys() -> None:
    from vllm.models.deepseek_v4.sparse_mla import (
        _BUILD_C128A_TOPK_METADATA_KERNEL,
    )

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=65536,
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
        )
    )

    assert _BUILD_C128A_TOPK_METADATA_KERNEL.get_warmup_keys(vllm_config) == [
        _BUILD_C128A_TOPK_METADATA_KERNEL.CompileKey(
            compress_ratio=128,
            max_compressed_tokens=512,
            block_size=2,
            triton_block_size=1024,
        )
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                use_fp4_cache=False,
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=4,
                cache_block_size=64,
                cache_alignment=1,
                runtime_state_width=1024,
                runtime_quant_block=64,
                runtime_token_stride=576,
                runtime_scale_dim=8,
                runtime_kv_block_stride=9216,
            ),
            (False, 512, 512, 1024, 4, True, 64, 448.0, 64, 576, 8, 64, 16, 9216),
        ),
        (
            dict(
                use_fp4_cache=True,
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=128,
                cache_block_size=64,
                cache_alignment=1,
                runtime_state_width=512,
                runtime_quant_block=32,
                runtime_token_stride=256,
                runtime_scale_dim=16,
                runtime_kv_block_stride=576,
            ),
            (True, 512, 512, 512, 128, False, 64, 448.0, 32, 256, 16, 64, 1, 576),
        ),
    ],
)
def test_fused_compress_quant_dispatch_matches_legacy_runtime_meta(
    kwargs: dict[str, Any],
    expected: tuple[Any, ...],
) -> None:
    kernel = FusedKVCompressNormRopeInsertIndexerTritonKernel()

    assert kernel.dispatch(**kwargs) == kernel.CompileKey(*expected)


def test_fused_mtp_input_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = FusedMTPInputRMSNormKernel()

    assert kernel.dispatch(hidden=7168, hc_mult=4, eps=1.0e-6) == kernel.CompileKey(
        hidden=7168,
        hc_mult=4,
        block_size=8192,
        eps=1.0e-6,
    )


def test_mtp_shared_head_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = MTPSharedHeadRMSNormKernel()

    assert kernel.dispatch(hidden=7168, eps=1.0e-6) == kernel.CompileKey(
        hidden=7168,
        block_size=8192,
        eps=1.0e-6,
    )


def test_fused_qkv_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = FusedQKVRMSNormKernel()

    assert kernel.dispatch(
        q_size=1536,
        kv_size=512,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
        launch_pdl=False,
    ) == kernel.CompileKey(
        q_size=1536,
        kv_size=512,
        block_size=2048,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
        launch_pdl=False,
    )


def test_fused_inv_rope_warmup_uses_runtime_stride_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = FusedInvRopeFP8QuantKernel()
    warmup_kwargs: dict[str, Any] = {}

    def capture_warmup(*args: Any, **kwargs: Any) -> None:
        warmup_kwargs.update(kwargs)

    monkeypatch.setattr(kernel.kernel, "warmup", capture_warmup)
    kernel.compile(
        kernel.CompileKey(
            heads_per_group=16,
            fp8_max=448.0,
            quant_group_size=128,
            chunks_per_head=4,
            rope_start=64,
            half_rope=32,
            tma_aligned_scales=True,
            use_gdc=True,
        )
    )

    stride_names = (
        "o_stride_token",
        "o_stride_head",
        "cache_stride_pos",
        "fp8_stride_group",
        "fp8_stride_token",
        "scale_stride_group",
        "scale_stride_k",
    )
    assert {warmup_kwargs[name] for name in stride_names} == {16}


def test_save_partial_states_dispatch_matches_legacy_meta() -> None:
    kernel = SavePartialStatesKernel()

    assert kernel.dispatch(
        head_size=512,
        state_width=1024,
        compress_ratio=4,
        kv_stride=512,
        score_stride=8,
        ape_stride=64,
        state_cache_stride0=32768,
        state_cache_stride1=2048,
        block_size=16,
        launch_pdl=True,
    ) == kernel.CompileKey(
        head_size=512,
        triton_block_size=512,
        state_width=1024,
        compress_ratio=4,
        kv_stride=512,
        score_stride=8,
        ape_stride=64,
        state_cache_stride0=32768,
        state_cache_stride1=2048,
        block_size=16,
        launch_pdl=True,
    )


def test_save_partial_states_warmup_filters_zipped_compression_cases() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                head_dim=512,
                compress_ratios=(128,),
            )
        )
    )

    warmup_keys = SavePartialStatesKernel().get_warmup_keys(vllm_config)

    assert len(warmup_keys) == 1
    assert warmup_keys[0].compress_ratio == 128
