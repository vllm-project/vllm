# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate common DSv4 attention JIT dispatch."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.models.common.ops.fused_qk_rmsnorm import FusedQKVRMSNormKernel
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    BuildFlashinferMixedSparseIndicesKernel,
    CombineTopkSwaIndicesKernel,
    ComputeGlobalTopkIndicesAndLensKernel,
    DequantizeAndGatherKCacheKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    FusedKVCompressNormRopeInsertIndexerTritonKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    FusedIndexerQRopeMxFp4TritonKernel,
    FusedIndexerQRopeQuantTritonKernel,
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
                dtype=torch.bfloat16,
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
            (
                torch.bfloat16,
                False,
                512,
                512,
                1024,
                4,
                True,
                64,
                448.0,
                64,
                576,
                8,
                4,
                16,
                9216,
            ),
        ),
        (
            dict(
                dtype=torch.bfloat16,
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
            (
                torch.bfloat16,
                True,
                512,
                512,
                512,
                128,
                False,
                64,
                448.0,
                32,
                256,
                16,
                8,
                1,
                576,
            ),
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

    assert kernel.dispatch(
        dtype=torch.bfloat16, hidden=7168, hc_mult=4, eps=1.0e-6
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
        hidden=7168,
        hc_mult=4,
        block_size=8192,
        eps=1.0e-6,
    )


def test_mtp_shared_head_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = MTPSharedHeadRMSNormKernel()

    assert kernel.dispatch(
        dtype=torch.bfloat16, hidden=7168, eps=1.0e-6
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
        hidden=7168,
        block_size=8192,
        eps=1.0e-6,
    )


@pytest.mark.parametrize(
    ("kernel", "compile_key"),
    [
        (
            FusedMTPInputRMSNormKernel(),
            FusedMTPInputRMSNormKernel.CompileKey(
                dtype=torch.bfloat16,
                hidden=7168,
                hc_mult=4,
                block_size=8192,
                eps=1.0e-6,
            ),
        ),
        (
            MTPSharedHeadRMSNormKernel(),
            MTPSharedHeadRMSNormKernel.CompileKey(
                dtype=torch.bfloat16,
                hidden=7168,
                block_size=8192,
                eps=1.0e-6,
            ),
        ),
    ],
)
def test_mtp_rmsnorm_compile_is_launch_only(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Any,
    compile_key: Any,
) -> None:
    warmup_calls = 0

    def capture_warmup(**kwargs: Any) -> None:
        nonlocal warmup_calls
        warmup_calls += 1

    monkeypatch.setattr(kernel.kernel, "warmup", capture_warmup)
    kernel.compile(compile_key)

    assert warmup_calls == 1


def test_fused_qkv_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = FusedQKVRMSNormKernel()

    assert kernel.dispatch(
        dtype=torch.bfloat16,
        q_size=1536,
        kv_size=512,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
        launch_pdl=False,
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
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


def test_attention_warmup_covers_pdl_variants() -> None:
    qkv_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                q_lora_rank=1536,
                head_dim=512,
                rms_norm_eps=1.0e-6,
            ),
        )
    )
    inv_rope_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_attention_heads=128,
                o_groups=8,
                head_dim=512,
                qk_rope_head_dim=64,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )

    assert {
        key.launch_pdl for key in FusedQKVRMSNormKernel().get_warmup_keys(qkv_config)
    } == {False, True}
    assert {
        key.launch_pdl
        for key in FusedInvRopeFP8QuantKernel().get_warmup_keys(inv_rope_config)
    } == {False, True}


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
            launch_pdl=True,
        )
    )

    # Specialized strides must warm as div-16 to match the runtime compile key.
    specialized_stride_names = (
        "o_stride_token",
        "o_stride_head",
        "cache_stride_pos",
        "fp8_stride_group",
        "fp8_stride_token",
        "scale_stride_group",
    )
    assert all(warmup_kwargs[name] % 16 == 0 for name in specialized_stride_names)

    # scale_stride_k's int class varies with batch size, so it is not specialized.
    assert "scale_stride_k" in kernel.kernel.do_not_specialize


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
    )


def test_save_partial_states_warmup_uses_instance_geometry() -> None:
    kernel = SavePartialStatesKernel()

    assert kernel.get_warmup_keys(head_dim=128, compress_ratio=4) == [
        kernel.CompileKey(
            head_size=256,
            triton_block_size=256,
            state_width=256,
            compress_ratio=4,
            kv_stride=512,
            score_stride=512,
            ape_stride=256,
            state_cache_stride0=2048,
            state_cache_stride1=512,
            block_size=4,
        )
    ]


def test_save_partial_states_forwards_runtime_pdl_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = SavePartialStatesKernel()
    launch = Mock()
    monkeypatch.setattr(kernel, "launch", launch)

    kernel(
        torch.empty(2, 128),
        torch.empty(2, 1),
        torch.empty(2, 64),
        torch.empty(2, dtype=torch.int64),
        torch.empty(1, 4, 256),
        torch.empty(2, dtype=torch.int64),
        block_size=4,
        state_width=256,
        compress_ratio=4,
        pdl_kwargs={"launch_pdl": True},
    )

    assert launch.call_args.kwargs["launch_pdl"] is True


def _cache_warmup_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=65536,
            hf_config=SimpleNamespace(
                sliding_window=128,
                index_topk=1024,
                compress_ratios=(0, 4, 128),
                index_head_dim=128,
                qk_rope_head_dim=64,
                vision_n_layers=0,
            ),
        ),
        cache_config=SimpleNamespace(block_size=256, cache_dtype="fp8_ds_mla"),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: default
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        speculative_config=None,
    )


def test_fused_compressor_warmup_uses_runtime_cache_config() -> None:
    kernel = FusedKVCompressNormRopeInsertIndexerTritonKernel()
    keys = kernel.get_warmup_keys(_cache_warmup_config())

    assert len(keys) == 1
    key = keys[0]
    assert key.dtype is torch.bfloat16
    assert key.compress_ratio == 4
    assert key.use_fp4_cache is False
    assert key.kv_cache_block_size == 64
    assert key.kv_block_stride == 8640


def test_dequant_gather_warmup_matches_runtime_cache_geometry() -> None:
    keys = DequantizeAndGatherKCacheKernel().get_warmup_keys(_cache_warmup_config())

    assert {key.max_blocks_per_seq for key in keys} == {256}
    assert {(key.cache_block_size, key.has_gather_lens) for key in keys} == {
        (256, True),
        (64, False),
        (2, False),
    }
    assert {key.offset for key in keys if key.cache_block_size == 256} == {1, 2, 16}


def test_compute_global_topk_warmup_covers_runtime_geometries() -> None:
    kernel = ComputeGlobalTopkIndicesAndLensKernel()

    assert set(kernel.get_warmup_keys(_cache_warmup_config())) == {
        kernel.CompileKey(
            global_topk_indices_stride=1024,
            topk_indices_stride=1024,
            topk=1024,
            block_table_stride=256,
            block_size=64,
        ),
        *(
            kernel.CompileKey(
                global_topk_indices_stride=topk_width,
                topk_indices_stride=topk_width,
                topk=topk_width,
                block_table_stride=256,
                block_size=2,
            )
            for topk_width in (128, 256, 512)
        ),
    }


def test_combine_topk_swa_warmup_covers_runtime_widths() -> None:
    keys = CombineTopkSwaIndicesKernel().get_warmup_keys(_cache_warmup_config())
    compile_classes = {
        (key.COMPRESS_RATIO, key.TOP_K, key.PADDED_TOP_K) for key in keys
    }

    assert (1, 0, 1024) in compile_classes
    assert (4, 1024, 1024) in compile_classes
    assert {
        compile_class for compile_class in compile_classes if compile_class[0] == 128
    } == {
        (128, 128, 128),
        (128, 256, 256),
        (128, 512, 512),
    }


def test_flashinfer_sparse_indices_warmup_matches_runtime_modes() -> None:
    keys = BuildFlashinferMixedSparseIndicesKernel().get_warmup_keys(
        _cache_warmup_config()
    )

    c4_keys = [key for key in keys if key.compress_ratio == 4]
    assert {key.top_k for key in c4_keys} == {0, 1024}
    assert {key.decode_compressed_topk for key in c4_keys} == {1024}
    assert all(key.decode_compressed_indices_are_local for key in c4_keys)
    assert all(not key.has_decode_compressed_lens for key in c4_keys)

    c128_keys = [key for key in keys if key.compress_ratio == 128]
    assert {(key.decode_compressed_topk, key.top_k) for key in c128_keys} == {
        (128, 0),
        (128, 128),
        (256, 0),
        (256, 256),
        (512, 0),
        (512, 512),
    }
    assert all(not key.decode_compressed_indices_are_local for key in c128_keys)
    assert all(key.has_decode_compressed_lens for key in c128_keys)


@pytest.mark.parametrize(
    "kernel",
    [FusedIndexerQRopeQuantTritonKernel(), FusedIndexerQRopeMxFp4TritonKernel()],
)
def test_indexer_q_warmup_uses_model_dtype(kernel: Any) -> None:
    dispatch_kwargs = dict(
        dtype=torch.bfloat16,
        num_heads=16,
        head_dim=128,
        rope_dim=64,
    )
    if isinstance(kernel, FusedIndexerQRopeQuantTritonKernel):
        dispatch_kwargs["use_fnuz"] = False
    inputs = kernel.warmup_inputs(kernel.dispatch(**dispatch_kwargs))

    assert inputs["index_q"].dtype is torch.bfloat16
    assert inputs["index_weights"].dtype is torch.bfloat16


def test_rmsnorm_warmup_uses_model_dtype() -> None:
    kernel = FusedQKVRMSNormKernel()
    key = kernel.dispatch(
        dtype=torch.bfloat16,
        q_size=1536,
        kv_size=512,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
        launch_pdl=False,
    )
    inputs = kernel.warmup_inputs(key)

    assert inputs["q_weight"].dtype is torch.bfloat16
    assert inputs["kv_weight"].dtype is torch.bfloat16
